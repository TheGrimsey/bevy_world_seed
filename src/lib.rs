#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::num::NonZeroU8;

use bevy::{
    app::{App, Plugin, PostUpdate},
    log::info_span,
    math::{FloatExt, IVec2, Vec2, Vec3, Vec3Swizzles},
    prelude::{
        resource_changed, AnyOf, Component, Deref, Event, EventReader, EventWriter,
        IntoSystemConfigs, Local, Query, ReflectResource, Res, Resource, SystemSet,
        TransformSystem,
    },
    reflect::Reflect,
    transform::components::GlobalTransform,
};
#[cfg(feature = "rendering")]
use debug_draw::TerrainDebugDrawPlugin;
#[cfg(feature = "rendering")]
use material::{TerrainTexturingPlugin, TerrainTexturingSettings};
#[cfg(feature = "rendering")]
use meshing::TerrainMeshingPlugin;
use modifiers::{
    update_shape_modifier_aabb, update_terrain_spline_aabb, update_terrain_spline_cache,
    update_tile_modifier_priorities, ModifierAabb, ModifierFalloffProperty,
    ModifierHeightOperation, ModifierHeightProperties, ModifierHoleOperation, ModifierPriority,
    ModifierStrengthLimitProperty, ShapeModifier, TerrainSplineCached, TerrainSplineProperties,
    TerrainSplineShape, TileToModifierMapping,
};
use noise::{NoiseCache, TerrainNoiseDetailLayer, TerrainNoiseSettings};
use snap_to_terrain::TerrainSnapToTerrainPlugin;
use terrain::{insert_components, update_tiling, Holes, Terrain, TileToTerrain};
use utils::{distance_squared_to_line_segment, index_to_x_z};

pub mod modifiers;
pub mod terrain;

#[cfg(feature = "rendering")]
mod debug_draw;
#[cfg(feature = "rendering")]
pub mod material;
#[cfg(feature = "rendering")]
mod meshing;
pub mod snap_to_terrain;

pub mod noise;

pub mod utils;

/// System sets containing the crate's systems.
#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
pub enum TerrainSets {
    Modifiers,
    Heights,
}
pub struct TerrainPlugin {
    pub noise_settings: Option<TerrainNoiseSettings>,
    pub terrain_settings: TerrainSettings,
    #[cfg(feature = "rendering")]
    pub texturing_settings: Option<TerrainTexturingSettings>,
    #[cfg(feature = "rendering")]
    pub debug_draw: bool,
}
impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "rendering")]
        if let Some(texturing_settings) = &self.texturing_settings {
            app.add_plugins((
                TerrainMeshingPlugin,
                TerrainTexturingPlugin(texturing_settings.clone()),
                TerrainSnapToTerrainPlugin,
            ));

            if self.debug_draw {
                app.add_plugins(TerrainDebugDrawPlugin);
            }
        }

        app.add_systems(
            PostUpdate,
            (
                (insert_components, update_tiling).before(update_terrain_heights),
                (
                    (
                        (
                            (update_terrain_spline_cache, update_terrain_spline_aabb).chain(),
                            update_shape_modifier_aabb,
                        ),
                        update_tile_modifier_priorities
                            .run_if(resource_changed::<TileToModifierMapping>),
                    )
                        .chain()
                        .in_set(TerrainSets::Modifiers),
                    update_terrain_heights.in_set(TerrainSets::Heights),
                )
                    .chain(),
            )
                .after(TransformSystem::TransformPropagate),
        );

        app.insert_resource(self.noise_settings.clone().unwrap_or_default());
        app.insert_resource(self.terrain_settings.clone());

        app.init_resource::<TileToModifierMapping>()
            .init_resource::<TileToTerrain>()
            .register_type::<TerrainSplineShape>()
            .register_type::<TerrainSplineCached>()
            .register_type::<ModifierAabb>()
            .register_type::<TerrainSplineProperties>()
            .register_type::<Terrain>()
            .register_type::<ShapeModifier>()
            .register_type::<ModifierHeightOperation>()
            .register_type::<ModifierPriority>()
            .register_type::<ModifierStrengthLimitProperty>()
            .register_type::<ModifierFalloffProperty>()
            .register_type::<ModifierHeightProperties>()
            .add_event::<RebuildTile>()
            .add_event::<TileHeightsRebuilt>();

        {
            app.register_type::<TerrainNoiseDetailLayer>()
                .register_type::<TerrainNoiseSettings>();
        }
    }
}

#[derive(Resource, Reflect, Clone)]
#[reflect(Resource)]
pub struct TerrainSettings {
    /// The size of a tile on one side expressed as the power in a left-shift.
    ///
    /// Ex. a value of `2` would be `1 << 2 == 4`.
    ///
    /// This enforces the size of a tile to be a power of 2.
    pub tile_size_power: NonZeroU8,
    /// How many points are on one edge of a terrain tile.
    pub edge_points: u16,
    /// The max amount of tile height updates to do per frame.
    pub max_tile_updates_per_frame: NonZeroU8,
    /// Points closer than this square distance are removed.
    pub max_spline_simplification_distance: f32,
}
impl TerrainSettings {
    pub fn tile_size(&self) -> f32 {
        (1usize << self.tile_size_power.get()) as f32
    }
}

#[derive(Event)]
struct RebuildTile(IVec2);

/// Emitted when the heights of a tile has been updated.
#[derive(Event)]
pub struct TileHeightsRebuilt(pub IVec2);

#[derive(Component, Deref, Debug)]
pub struct Heights(Box<[f32]>);

fn update_terrain_heights(
    terrain_noise_layers: Option<Res<TerrainNoiseSettings>>,
    shape_modifier_query: Query<(
        &ShapeModifier,
        &ModifierHeightProperties,
        Option<&ModifierStrengthLimitProperty>,
        Option<&ModifierFalloffProperty>,
        AnyOf<(&ModifierHeightOperation, &ModifierHoleOperation)>,
        &GlobalTransform,
    )>,
    spline_query: Query<(
        &TerrainSplineCached,
        &TerrainSplineProperties,
        Option<&ModifierStrengthLimitProperty>,
    )>,
    mut heights: Query<(&mut Heights, &mut Holes)>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    tile_to_terrain: Res<TileToTerrain>,
    mut tile_generate_queue: Local<Vec<IVec2>>,
    mut noise_cache: Local<NoiseCache>,
    mut event_reader: EventReader<RebuildTile>,
    mut tile_rebuilt_events: EventWriter<TileHeightsRebuilt>,
) {
    for RebuildTile(tile) in event_reader.read() {
        if !tile_generate_queue.contains(tile) {
            tile_generate_queue.push(*tile);
        }
    }

    if tile_generate_queue.is_empty() {
        return;
    }

    let tile_size = terrain_settings.tile_size();
    let scale = tile_size / (terrain_settings.edge_points - 1) as f32;
    let inv_tile_size_scale = scale * (7.0 / tile_size);

    let tiles_to_generate = tile_generate_queue
        .len()
        .min(terrain_settings.max_tile_updates_per_frame.get() as usize);

    for tile in tile_generate_queue.drain(..tiles_to_generate) {
        let Some(tiles) = tile_to_terrain.0.get(&tile) else {
            continue;
        };
        let terrain_translation = (tile << terrain_settings.tile_size_power.get()).as_vec2();
        let shape_modifiers = tile_to_modifier.shape.get(&tile);
        let splines = tile_to_modifier.splines.get(&tile);

        let mut iter = heights.iter_many_mut(tiles.iter());
        while let Some((mut heights, mut holes)) = iter.fetch_next() {
            // Clear heights.
            heights.0.fill(0.0);
            holes.0.clear();

            // First, set by noise.
            if let Some(terrain_noise_layers) = terrain_noise_layers.as_ref() {
                let _span = info_span!("Apply noise").entered();
                for (i, val) in heights.0.iter_mut().enumerate() {
                    let (x, z) = index_to_x_z(i, terrain_settings.edge_points as usize);

                    let vertex_position =
                        terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);

                    *val += terrain_noise_layers.sample_position(&mut noise_cache, vertex_position);
                }
            }

            // Secondly, set by shape-modifiers.
            if let Some(shapes) = shape_modifiers {
                let _span = info_span!("Apply shape modifiers").entered();

                for entry in shapes.iter() {
                    if let Ok((
                        modifier,
                        modifier_properties,
                        modifier_strength_limit,
                        modifier_falloff,
                        (operation, hole_punch),
                        global_transform,
                    )) = shape_modifier_query.get(entry.entity)
                    {
                        let shape_translation = global_transform.translation().xz();
                        let falloff = modifier_falloff
                            .map_or(0.0, |falloff| falloff.0)
                            .max(f32::EPSILON);

                        match modifier {
                            ShapeModifier::Circle { radius } => {
                                for (i, val) in heights.0.iter_mut().enumerate() {
                                    let (x, z) =
                                        index_to_x_z(i, terrain_settings.edge_points as usize);

                                    let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                                    let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                                    let overlap_index = overlap_y * 8 + overlaps_x;
                                    if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                        continue;
                                    }

                                    let vertex_position = terrain_translation
                                        + Vec2::new(x as f32 * scale, z as f32 * scale);

                                    let strength = 1.0
                                        - ((vertex_position.distance(shape_translation) - radius)
                                            / falloff)
                                            .clamp(
                                                0.0,
                                                modifier_strength_limit
                                                    .map_or(1.0, |modifier| modifier.0),
                                            );

                                    if let Some(operation) = operation {
                                        *val = apply_modifier(
                                            modifier_properties,
                                            operation,
                                            vertex_position,
                                            shape_translation,
                                            *val,
                                            global_transform,
                                            strength,
                                            &mut noise_cache,
                                            false,
                                        );
                                    }
                                    if let Some(hole_punch) = hole_punch.filter(|_| strength >= 1.0)
                                    {
                                        holes.0.set(i, !hole_punch.invert);
                                    }
                                }
                            }
                            ShapeModifier::Rectangle { x, z } => {
                                let rect_min = Vec2::new(-x, -z);
                                let rect_max = Vec2::new(*x, *z);

                                for (i, val) in heights.0.iter_mut().enumerate() {
                                    let (x, z) =
                                        index_to_x_z(i, terrain_settings.edge_points as usize);

                                    let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                                    let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                                    let overlap_index = overlap_y * 8 + overlaps_x;
                                    if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                        continue;
                                    }

                                    let vertex_position = terrain_translation
                                        + Vec2::new(x as f32 * scale, z as f32 * scale);
                                    let vertex_local = global_transform
                                        .affine()
                                        .inverse()
                                        .transform_point3(Vec3::new(
                                            vertex_position.x,
                                            0.0,
                                            vertex_position.y,
                                        ))
                                        .xz();

                                    let d_x = (rect_min.x - vertex_local.x)
                                        .max(vertex_local.x - rect_max.x)
                                        .max(0.0);
                                    let d_y = (rect_min.y - vertex_local.y)
                                        .max(vertex_local.y - rect_max.y)
                                        .max(0.0);
                                    let d_d = (d_x * d_x + d_y * d_y).sqrt();

                                    let strength = 1.0
                                        - (d_d / falloff).clamp(
                                            0.0,
                                            modifier_strength_limit
                                                .map_or(1.0, |modifier| modifier.0),
                                        );

                                    if let Some(operation) = operation {
                                        *val = apply_modifier(
                                            modifier_properties,
                                            operation,
                                            vertex_position,
                                            shape_translation,
                                            *val,
                                            global_transform,
                                            strength,
                                            &mut noise_cache,
                                            false,
                                        );
                                    }
                                    if let Some(hole_punch) = hole_punch.filter(|_| strength >= 1.0)
                                    {
                                        holes.0.set(i, !hole_punch.invert);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Finally, set by splines.
            if let Some(splines) = splines {
                let _span = info_span!("Apply splines").entered();

                for entry in splines.iter() {
                    if let Ok((spline, spline_properties, modifier_strength_limit)) =
                        spline_query.get(entry.entity)
                    {
                        for (i, val) in heights.0.iter_mut().enumerate() {
                            let (x, z) = index_to_x_z(i, terrain_settings.edge_points as usize);

                            let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                            let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                            let overlap_index = overlap_y * 8 + overlaps_x;
                            if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                continue;
                            }

                            let vertex_position =
                                terrain_translation + Vec2::new(x as f32, z as f32) * scale;
                            let mut distance = f32::INFINITY;
                            let mut height = None;

                            for points in spline.points.windows(2) {
                                let a_2d = points[0].xz();
                                let b_2d = points[1].xz();

                                let (new_distance, t) =
                                    distance_squared_to_line_segment(a_2d, b_2d, vertex_position);

                                if new_distance < distance {
                                    distance = new_distance;
                                    height = Some(points[0].lerp(points[1], t).y);
                                }
                            }

                            if let Some(height) = height {
                                let strength = 1.0
                                    - ((distance.sqrt() - spline_properties.width)
                                        / spline_properties.falloff.max(f32::EPSILON))
                                    .clamp(
                                        0.0,
                                        modifier_strength_limit.map_or(1.0, |modifier| modifier.0),
                                    );
                                *val = val.lerp(height, strength);
                            }
                        }
                    }
                }
            }
        }

        tile_rebuilt_events.send(TileHeightsRebuilt(tile));
    }
}

fn apply_modifier(
    modifier_properties: &ModifierHeightProperties,
    operation: &ModifierHeightOperation,
    vertex_position: Vec2,
    shape_translation: Vec2,
    val: f32,
    global_transform: &GlobalTransform,
    strength: f32,
    noise_cache: &mut NoiseCache,
    set_with_position_y: bool,
) -> f32 {
    let mut new_val = match operation {
        ModifierHeightOperation::Set => {
            // Relative position so we can apply the rotation from the shape modifier. This gets us tilted circles.
            let position = vertex_position - shape_translation;

            let height = if set_with_position_y {
                position.y
            } else {
                global_transform
                    .transform_point(Vec3::new(position.x, 0.0, position.y))
                    .y
            };

            val.lerp(height, strength)
        }
        ModifierHeightOperation::Change(change) => val + *change * strength,
        ModifierHeightOperation::Step { step, smoothing } => val.lerp(
            (((val / *step) * smoothing).round() / smoothing) * *step,
            strength,
        ),
        ModifierHeightOperation::Noise { noise } => val.lerp(
            noise.sample(
                vertex_position.x,
                vertex_position.y,
                noise_cache.get(noise.seed),
            ),
            strength,
        ),
    };

    if !modifier_properties.allow_raising {
        new_val = new_val.min(val);
    }
    if !modifier_properties.allow_lowering {
        new_val = new_val.max(val);
    }

    new_val
}
