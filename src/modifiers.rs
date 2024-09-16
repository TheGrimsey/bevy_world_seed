use bevy::{
    math::{IVec2, Vec2, Vec3, Vec3Swizzles}, prelude::{
        Bundle, Changed, Component, CubicCurve, Entity, EventReader, EventWriter, GlobalTransform,
        Or, Query, ReflectComponent, Res, ResMut, Resource, TransformBundle,
    }, reflect::Reflect, utils::HashMap
};

use crate::{noise::TerrainNoiseDetailLayer, RebuildTile, TerrainSettings};

/// Bundle containing all the base components required for a Shape Modifier to function.
/// 
/// It additionally needs an Operation and optionally properties.
#[derive(Bundle)]
pub struct ShapeModifierBundle {
    pub aabb: ModifierAabb,
    pub shape: ShapeModifier,
    pub properties: ModifierHeightProperties,
    pub priority: ModifierPriority,
    pub transform_bundle: TransformBundle,
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub enum ShapeModifier {
    Circle { radius: f32 },
    // Half-size.
    Rectangle { x: f32, z: f32 },
}

/// Determines the falloff distance for operations.
/// 
/// Affects the strength falloff of height & texture operators 
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct ModifierFalloffProperty(pub f32);

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct ModifierHeightProperties {
    // TODO: These should be bitflags. They are only bools for testing in editor.
    pub allow_raising: bool,
    pub allow_lowering: bool,
}
impl Default for ModifierHeightProperties {
    fn default() -> Self {
        Self { allow_raising: true, allow_lowering: true }
    }
}

/// Operation for modifying the height of terrain.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub enum ModifierHeightOperation {
    /// Set the height within the modifier's bounds equal to the modifiers global Y coordinate
    #[default]
    Set,
    /// Change the height within the modifier's bounds by the entered value.
    Change(f32),
    Step {
        step: f32,
        smoothing: f32,
    },
    Noise {
        noise: TerrainNoiseDetailLayer,
    },
}

/// Operation for creating or removing holes in terrain.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct ModifierHoleOperation {
    /// When true, fill in holes instead of creating them.
    pub invert: bool
}


/// Clamps the max strength of a modifier to this value.
/// 
/// Use if you want to only blend the modifier in a bit. 
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct ModifierStrengthLimitProperty(pub f32);

#[derive(Bundle)]
pub struct TerrainSplineBundle {
    pub tile_aabb: ModifierAabb,
    pub spline: TerrainSplineShape,
    pub properties: TerrainSplineProperties,
    pub spline_cached: TerrainSplineCached,
    pub priority: ModifierPriority,
    pub transform_bundle: TransformBundle,
}

/// Defines the order in which to apply the modifier where lower values are applied earlier.
#[derive(Component, Reflect, Default, PartialEq, Eq, PartialOrd, Ord)]
#[reflect(Component)]
pub struct ModifierPriority(pub i32);

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct ModifierAabb {
    pub(super) min: IVec2,
    pub(super) max: IVec2,
}

/// Defines the shape of a spline modifier.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct TerrainSplineShape {
    // TODO: This should be expecting the generic Curve trait when it's implemented.
    pub curve: CubicCurve<Vec3>,
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct TerrainSplineProperties {
    pub width: f32,
    pub falloff: f32,
}

/// Cache of points used when updating tiles.
/// 
/// Automatically updated when the terrain spline changes. 
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct TerrainSplineCached {
    pub(super) points: Vec<Vec3>,
}

pub(super) struct TileModifierEntry {
    pub(super) entity: Entity,
    /// Acts as a 8x8 map telling us where in the tile this modifier has an effect.
    /// 
    /// Allows us to skip checking modifiers for points that don't overlap, giving speed ups depending on how big the modifier is relative to the tile.
    pub(super) overlap_bits: u64,
}

#[derive(Resource, Default)]
pub(super) struct TileToModifierMapping {
    pub(super) shape: HashMap<IVec2, Vec<TileModifierEntry>>,
    pub(super) splines: HashMap<IVec2, Vec<TileModifierEntry>>,
}

pub(super) fn update_terrain_spline_cache(
    mut query: Query<
        (
            &mut TerrainSplineCached,
            &TerrainSplineShape,
            &TerrainSplineProperties,
            &GlobalTransform,
        ),
        Or<(
            Changed<TerrainSplineProperties>,
            Changed<TerrainSplineShape>,
            Changed<GlobalTransform>,
        )>,
    >,
    terrain_settings: Res<TerrainSettings>,
) {
    query.par_iter_mut().for_each(
        |(mut spline_cached, spline, spline_properties, global_transform)| {
            spline_cached.points.clear();

            // Filter points that are very close together.
            let dedup_distance = (spline_properties.width * spline_properties.width)
                .min(terrain_settings.max_spline_simplification_distance);

            // We need to figure out a subdivision amount that gives us close enough points.
            // Can't put this too low or some winding paths might break.
            let mut subdivisions = 20;

            // Check so all points are close enough to the next.
            while spline.curve.iter_positions(subdivisions).zip(spline.curve.iter_positions(subdivisions).skip(1)).any(|(a,b)| a.distance_squared(b) > dedup_distance * 1.5) {
                subdivisions = (subdivisions as f32 * 1.2) as usize; 
            }

            spline_cached.points.extend(
                spline
                    .curve
                    .iter_positions(subdivisions)
                    .map(|point| global_transform.transform_point(point)),
            );
            
            spline_cached
                .points
                .dedup_by(|a, b| a.distance_squared(*b) < dedup_distance);
        
        },
    );
}

pub(super) fn update_terrain_spline_aabb(
    mut query: Query<
        (
            Entity,
            &TerrainSplineCached,
            &TerrainSplineProperties,
            &mut ModifierAabb,
        ),
        (
            Changed<TerrainSplineCached>,
            Changed<TerrainSplineProperties>,
        ),
    >,
    terrain_settings: Res<TerrainSettings>,
    mut tile_to_modifier_mapping: ResMut<TileToModifierMapping>,
    mut rebuild_tiles_event: EventWriter<RebuildTile>,
) {
    let tile_size = terrain_settings.tile_size();

    query.iter_mut().for_each(
        |(entity, spline_cached, spline_properties, mut tile_aabb)| {
            for x in tile_aabb.min.x..=tile_aabb.max.x {
                for y in tile_aabb.min.y..=tile_aabb.max.y {
                    let tile = IVec2::new(x, y);
                    if let Some(entries) = tile_to_modifier_mapping.splines.get_mut(&tile) {
                        if let Some(index) = entries.iter().position(|entry| entity == entry.entity)
                        {
                            entries.swap_remove(index);
                            rebuild_tiles_event.send(RebuildTile(tile));
                        }
                    }
                }
            }

            let (min, max) = if spline_cached.points.is_empty() {
                (IVec2::ZERO, IVec2::ZERO)
            } else {
                let (min, max) = spline_cached.points.iter().fold(
                    (spline_cached.points[0].xz(), spline_cached.points[0].xz()),
                    |(min, max), point| (min.min(point.xz()), max.max(point.xz())),
                );

                let total_width = spline_properties.falloff + spline_properties.width;

                (
                    (min - total_width).as_ivec2() >> terrain_settings.tile_size_power.get(),
                    (max + total_width).as_ivec2() >> terrain_settings.tile_size_power.get(),
                )
            };

            for x in min.x..=max.x {
                for y in min.y..=max.y {
                    let tile = IVec2::new(x, y);
                    let tile_world = (tile << terrain_settings.tile_size_power.get()).as_vec2();

                    let mut overlap_bits = 0;

                    for (a, b) in spline_cached
                        .points
                        .iter()
                        .zip(spline_cached.points.iter().skip(1))
                    {
                        let a_2d = a.xz() - tile_world;
                        let b_2d = b.xz() - tile_world;

                        let min =
                            a_2d.min(b_2d) - spline_properties.width - spline_properties.falloff.max(f32::EPSILON);
                        let max =
                            a_2d.max(b_2d) + spline_properties.width + spline_properties.falloff.max(f32::EPSILON);

                        let min_scaled = ((min / tile_size) * 7.0).as_ivec2();
                        let max_scaled = ((max / tile_size) * 7.0).as_ivec2();

                        if min_scaled.x < 8
                            && min_scaled.y < 8
                            && max_scaled.x >= 0
                            && max_scaled.y >= 0
                        {
                            for y in min_scaled.y.max(0)..=max_scaled.y.min(7) {
                                let i = y * 8;
                                for x in min_scaled.x.max(0)..=max_scaled.x.min(7) {
                                    let bit = i + x;

                                    overlap_bits |= 1 << bit;
                                }
                            }
                        }
                    }

                    if overlap_bits != 0 {
                        let entry = TileModifierEntry {
                            entity,
                            overlap_bits,
                        };

                        if let Some(entries) = tile_to_modifier_mapping.splines.get_mut(&tile) {
                            entries.push(entry);
                        } else {
                            tile_to_modifier_mapping.splines.insert(tile, vec![entry]);
                        }

                        rebuild_tiles_event.send(RebuildTile(tile));
                    }
                }
            }

            tile_aabb.min = min;
            tile_aabb.max = max;
        },
    );
}

pub(super) fn update_shape_modifier_aabb(
    mut query: Query<
        (
            Entity,
            &ShapeModifier,
            Option<&ModifierFalloffProperty>,
            &mut ModifierAabb,
            &GlobalTransform,
        ),
        Or<(
            Changed<ShapeModifier>,
            Changed<ModifierHeightOperation>,
            Changed<ModifierHeightProperties>,
            Changed<ModifierFalloffProperty>,
            Changed<GlobalTransform>,
        )>,
    >,
    terrain_settings: Res<TerrainSettings>,
    mut tile_to_modifier_mapping: ResMut<TileToModifierMapping>,
    mut rebuild_tiles_event: EventWriter<RebuildTile>,
) {
    let tile_size = terrain_settings.tile_size();

    query
        .iter_mut()
        .for_each(|(entity, shape, modifier_falloff, mut tile_aabb, global_transform)| {
            for x in tile_aabb.min.x..=tile_aabb.max.x {
                for y in tile_aabb.min.y..=tile_aabb.max.y {
                    let tile = IVec2::new(x, y);

                    if let Some(entries) = tile_to_modifier_mapping.shape.get_mut(&tile) {
                        if let Some(index) = entries
                            .iter()
                            .position(|existing_entity| entity == existing_entity.entity)
                        {
                            entries.swap_remove(index);

                            rebuild_tiles_event.send(RebuildTile(tile));
                        }
                    }
                }
            }

            let (min, max) = match shape {
                ShapeModifier::Circle { radius } => (
                    global_transform.translation().xz() + Vec2::splat(-radius),
                    global_transform.translation().xz() + Vec2::splat(*radius),
                ),
                ShapeModifier::Rectangle { x, z } => {
                    let min = global_transform.transform_point(Vec3::new(-x, 0.0, -z));
                    let max = global_transform.transform_point(Vec3::new(*x, 0.0, *z));

                    (min.min(max).xz(), max.max(min).xz())
                }
            };

            let falloff = modifier_falloff.map_or(0.0, |falloff| falloff.0).max(f32::EPSILON);
            let min = min - falloff;
            let max = max + falloff;

            let tile_min = min.as_ivec2() >> terrain_settings.tile_size_power.get();
            let tile_max = max.as_ivec2() >> terrain_settings.tile_size_power.get();

            for x in tile_min.x..=tile_max.x {
                for y in tile_min.y..=tile_max.y {
                    let tile = IVec2::new(x, y);
                    let tile_world = (tile << terrain_settings.tile_size_power.get()).as_vec2();

                    let mut overlap_bits = 0;

                    let min = min - tile_world;
                    let max = max - tile_world;

                    let min_scaled = ((min / tile_size) * 7.0).as_ivec2();
                    let max_scaled = ((max / tile_size) * 7.0).as_ivec2();

                    if min_scaled.x < 8
                        && min_scaled.y < 8
                        && max_scaled.x >= 0
                        && max_scaled.y >= 0
                    {
                        for y in min_scaled.y.max(0)..=max_scaled.y.min(7) {
                            let i = y * 8;
                            for x in min_scaled.x.max(0)..=max_scaled.x.min(7) {
                                let bit = i + x;

                                overlap_bits |= 1 << bit;
                            }
                        }
                    }

                    if overlap_bits != 0 {
                        let entry = TileModifierEntry {
                            entity,
                            overlap_bits,
                        };

                        if let Some(entries) = tile_to_modifier_mapping.shape.get_mut(&tile) {
                            entries.push(entry);
                        } else {
                            tile_to_modifier_mapping.shape.insert(tile, vec![entry]);
                        }

                        rebuild_tiles_event.send(RebuildTile(tile));
                    }
                }
            }

            tile_aabb.min = tile_min;
            tile_aabb.max = tile_max;
        });
}

pub(super) fn update_tile_modifier_priorities(
    mut tile_to_modifier_mapping: ResMut<TileToModifierMapping>,
    mut event_reader: EventReader<RebuildTile>,
    priority_query: Query<&ModifierPriority>,
) {
    for RebuildTile(tile) in event_reader.read() {
        if let Some(entries) = tile_to_modifier_mapping.shape.get_mut(tile) {
            entries.sort_unstable_by(|a, b| {
                priority_query
                    .get(a.entity)
                    .ok()
                    .cmp(&priority_query.get(b.entity).ok())
            });
        }

        if let Some(entries) = tile_to_modifier_mapping.splines.get_mut(tile) {
            entries.sort_unstable_by(|a, b| {
                priority_query
                    .get(a.entity)
                    .ok()
                    .cmp(&priority_query.get(b.entity).ok())
            });
        }
    }
}
