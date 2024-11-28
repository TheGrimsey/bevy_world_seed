#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::num::NonZeroU8;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::Assets;
use bevy_math::{FloatExt, IVec2, Vec2, Vec3, Vec3Swizzles};
use bevy_ecs::prelude::{any_with_component, resource_changed, AnyOf, Component, DetectChanges, Event, EventReader, EventWriter, Query, Res, ResMut, Resource, SystemSet, IntoSystemConfigs, ReflectResource};
use bevy_transform::prelude::{GlobalTransform, TransformSystem};
use bevy_log::info_span;
use bevy_reflect::Reflect;
use bevy_derive::Deref;

use bevy_lookup_curve::{LookupCurve, LookupCurvePlugin};
#[cfg(feature = "debug_draw")]
use debug_draw::TerrainDebugDrawPlugin;
use easing::EasingFunction;
use feature_placement::FeaturePlacementPlugin;
#[cfg(feature = "rendering")]
use material::{TerrainTexturingPlugin, TerrainTexturingSettings};
#[cfg(feature = "rendering")]
use meshing::TerrainMeshingPlugin;
use modifiers::{
    update_shape_modifier_aabb, update_terrain_spline_aabb, update_terrain_spline_cache,
    update_tile_modifier_priorities, ModifierFalloffNoiseProperty, ModifierFalloffProperty,
    ModifierHeightOperation, ModifierHeightProperties, ModifierHoleOperation,
    ModifierNoiseOperation, ModifierPriority, ModifierStrengthLimitProperty, ModifierTileAabb,
    ShapeModifier, TerrainSplineCached, TerrainSplineProperties, TerrainSplineShape,
    TileToModifierMapping,
};
use noise::{apply_noise_simd, LayerNoiseSettings, NoiseCache, NoiseIndexCache, TerrainNoiseSettings};
use snap_to_terrain::TerrainSnapToTerrainPlugin;
use terrain::{insert_components, update_tiling, Holes, Terrain, TileToTerrain};
use utils::{distance_squared_to_line_segment, index_to_x_z};

pub mod easing;
pub mod modifiers;
pub mod terrain;

#[cfg(feature = "debug_draw")]
pub mod debug_draw;
#[cfg(feature = "rendering")]
pub mod material;
#[cfg(feature = "rendering")]
pub mod meshing;
pub mod snap_to_terrain;

pub mod feature_placement;

pub mod noise;

pub mod utils;

/// System sets containing the crate's systems.
#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
pub enum TerrainSets {
    /// Initialize components for Terrain & modifier entities.
    Init,
    Modifiers,
    Heights,
    Meshing,
    Material,
}
pub struct TerrainPlugin {
    pub noise_settings: Option<TerrainNoiseSettings>,
    pub terrain_settings: TerrainSettings,
    #[cfg(feature = "rendering")]
    pub texturing_settings: Option<TerrainTexturingSettings>,
    #[cfg(feature = "debug_draw")]
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

            #[cfg(feature = "debug_draw")]
            if self.debug_draw {
                app.add_plugins(TerrainDebugDrawPlugin);
            }
        }

        if !app.is_plugin_added::<LookupCurvePlugin>() {
            app.add_plugins(LookupCurvePlugin);
        }

        app.add_plugins(FeaturePlacementPlugin);

        app.add_systems(
            PostUpdate,
            (
                (insert_components, update_tiling)
                    .run_if(any_with_component::<Terrain>)
                    .before(update_terrain_heights)
                    .in_set(TerrainSets::Init),
                (
                    (
                        (
                            (update_terrain_spline_cache, update_terrain_spline_aabb)
                                .chain()
                                .run_if(any_with_component::<TerrainSplineCached>),
                            update_shape_modifier_aabb.run_if(any_with_component::<ShapeModifier>),
                        ),
                        update_tile_modifier_priorities
                            .run_if(resource_changed::<TileToModifierMapping>),
                    )
                        .chain()
                        .in_set(TerrainSets::Modifiers),
                    update_terrain_heights
                        .run_if(any_with_component::<Heights>)
                        .in_set(TerrainSets::Heights),
                )
                    .chain(),
            )
                .after(TransformSystem::TransformPropagate),
        );

        app.insert_resource(self.noise_settings.clone().unwrap_or_default());
        app.insert_resource(self.terrain_settings.clone());

        app.init_resource::<TileToModifierMapping>()
            .init_resource::<TileToTerrain>()
            .init_resource::<NoiseCache>()
            .init_resource::<NoiseIndexCache>()
            .register_type::<TerrainSplineShape>()
            .register_type::<TerrainSplineCached>()
            .register_type::<ModifierTileAabb>()
            .register_type::<TerrainSplineProperties>()
            .register_type::<Terrain>()
            .register_type::<TerrainSettings>()
            .register_type::<ShapeModifier>()
            .register_type::<ModifierHeightOperation>()
            .register_type::<ModifierPriority>()
            .register_type::<ModifierStrengthLimitProperty>()
            .register_type::<ModifierNoiseOperation>()
            .register_type::<ModifierFalloffProperty>()
            .register_type::<ModifierHeightProperties>()
            .register_type::<ModifierFalloffNoiseProperty>()
            .add_event::<RebuildTile>()
            .add_event::<TileHeightsRebuilt>();

        {
            app.register_type::<LayerNoiseSettings>()
                .register_type::<TerrainNoiseSettings>();
        }

        app.init_resource::<TerrainHeightRebuildQueue>();
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
    ///
    /// The distance between vertices is equal to `(edge_points - 1) / tile_size`.
    pub edge_points: u16,
    /// The max amount of tile height updates to do per frame.
    pub max_tile_updates_per_frame: NonZeroU8,
    /// Spline points which are closer to the previous point than this square distance are removed.
    ///
    /// Used to reduce the amount of line segments to compare against when applying spline modifiers.
    ///
    /// Values less than the spacing between vertices in the terrain will have little effect. Greater values will show themselves in slopes & when the curve curves on the XZ-plane.
    pub max_spline_simplification_distance_squared: f32,
}
impl TerrainSettings {
    pub fn tile_size(&self) -> f32 {
        (1usize << self.tile_size_power.get()) as f32
    }
}

/// Event emitted to mark a tile to be rebuilt.
///
/// Sent when modifiers are changed.
#[derive(Event)]
pub struct RebuildTile(pub IVec2);

/// Emitted when the heights of a tile has been updated.
#[derive(Event)]
pub struct TileHeightsRebuilt(pub IVec2);

/// Container for the height of points in a terrain tile.
#[derive(Component, Deref, Debug)]
pub struct Heights(Box<[f32]>);

/// Queue of terrain tiles which [`Heights`] are to be rebuilt.
#[derive(Resource, Default)]
pub struct TerrainHeightRebuildQueue(Vec<IVec2>);
impl TerrainHeightRebuildQueue {
    pub fn get(&self) -> &[IVec2] {
        &self.0
    }
    pub fn count(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

fn update_terrain_heights(
    terrain_noise_settings: Option<Res<TerrainNoiseSettings>>,
    shape_modifier_query: Query<(
        &ShapeModifier,
        &ModifierHeightProperties,
        Option<&ModifierStrengthLimitProperty>,
        (
            Option<&ModifierFalloffProperty>,
            Option<&ModifierFalloffNoiseProperty>,
        ),
        AnyOf<(
            &ModifierHeightOperation,
            &ModifierNoiseOperation,
            &ModifierHoleOperation,
        )>,
        &GlobalTransform,
    )>,
    spline_query: Query<(
        &TerrainSplineCached,
        &TerrainSplineProperties,
        Option<&ModifierFalloffProperty>,
        Option<&ModifierStrengthLimitProperty>,
    )>,
    mut heights: Query<(&mut Heights, &mut Holes)>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    tile_to_terrain: Res<TileToTerrain>,
    mut tile_generate_queue: ResMut<TerrainHeightRebuildQueue>,
    mut noise_cache: ResMut<NoiseCache>,
    mut event_reader: EventReader<RebuildTile>,
    mut tile_rebuilt_events: EventWriter<TileHeightsRebuilt>,
    lookup_curves: Res<Assets<LookupCurve>>,
    mut noise_index_cache: ResMut<NoiseIndexCache>
) {
    // Cache indexes into the noise cache for each terrain noise.
    // Saves having to do the checks and insertions for every iteration when applying the noise.
    if let Some(terrain_noise_settings) = terrain_noise_settings
        .as_ref()
        .filter(|noise_settings| noise_settings.is_changed())
    {
        noise_index_cache.fill_cache(terrain_noise_settings, &mut noise_cache);
    }

    for RebuildTile(tile) in event_reader.read() {
        if !tile_generate_queue.0.contains(tile) {
            tile_generate_queue.0.push(*tile);
        }
    }

    if tile_generate_queue.is_empty() {
        return;
    }

    // Don't generate if we are waiting for splines to load.
    if terrain_noise_settings.as_ref().is_some_and(|layers| {
        layers
            .splines
            .iter()
            .any(|layer| !lookup_curves.contains(&layer.amplitude_curve))
    }) {
        return;
    }

    let tile_size = terrain_settings.tile_size();
    let scale = tile_size / (terrain_settings.edge_points - 1) as f32;
    let inv_tile_size_scale = scale * (7.0 / tile_size);

    let tiles_to_generate = tile_generate_queue
        .count()
        .min(terrain_settings.max_tile_updates_per_frame.get() as usize);

    for tile in tile_generate_queue.0.drain(..tiles_to_generate) {
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
            if let Some(terrain_noise_layers) = terrain_noise_settings.as_ref() {
                let _span = info_span!("Apply noise").entered();
                apply_noise_simd(
                    &mut heights.0,
                    &terrain_settings,
                    terrain_translation,
                    scale,
                    &noise_cache,
                    &noise_index_cache,
                    &lookup_curves,
                    terrain_noise_layers,
                );
            }

            // Secondly, set by shape-modifiers.
            if let Some(shapes) = shape_modifiers {
                let _span = info_span!("Apply shape modifiers").entered();

                for entry in shapes.iter() {
                    if let Ok((
                        modifier,
                        modifier_properties,
                        modifier_strength_limit,
                        (modifier_falloff, modifier_falloff_noise),
                        (operation, noise_operation, hole_punch),
                        global_transform,
                    )) = shape_modifier_query.get(entry.entity)
                    {
                        let (_, _, shape_translation) =
                            global_transform.to_scale_rotation_translation();
                        let (falloff, easing_function) = modifier_falloff
                            .map_or((f32::EPSILON, EasingFunction::Linear), |falloff| {
                                (falloff.falloff, falloff.easing_function)
                            });

                        // Cache the noise index.
                        let modifier_falloff_noise = modifier_falloff_noise.map(|falloff_noise| {
                            (
                                falloff_noise,
                                noise_cache.get_simplex_index(falloff_noise.noise.seed),
                            )
                        });
                        let noise_operation = noise_operation.map(|noise_operation| {
                            (
                                noise_operation,
                                noise_cache.get_simplex_index(noise_operation.noise.seed)
                            )
                        });

                        for (i, val) in heights.0.iter_mut().enumerate() {
                            let (x, z) = index_to_x_z(i, terrain_settings.edge_points as usize);

                            let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                            let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                            let overlap_index = overlap_y * 8 + overlaps_x;
                            if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                continue;
                            }

                            let vertex_position =
                                terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);

                            let vertex_local = global_transform
                                .affine()
                                .inverse()
                                .transform_point3(Vec3::new(
                                    vertex_position.x,
                                    0.0,
                                    vertex_position.y,
                                ))
                                .xz();

                            let falloff = modifier_falloff_noise.map_or(falloff, |(falloff_noise, noise_index)| {
                                let normalized_vertex =
                                    shape_translation.xz() + vertex_local.normalize_or_zero();

                                falloff
                                    + falloff_noise.noise.sample(
                                        normalized_vertex.x,
                                        normalized_vertex.y,
                                        unsafe { noise_cache.get_by_index(noise_index) },
                                    )
                            });

                            let strength = match modifier {
                                ShapeModifier::Circle { radius } => {
                                    1.0 - ((vertex_local.distance(Vec2::ZERO) - radius) / falloff)
                                }
                                ShapeModifier::Rectangle { x, z } => {
                                    let rect_min = Vec2::new(-x, -z);
                                    let rect_max = Vec2::new(*x, *z);

                                    let d_x = (rect_min.x - vertex_local.x)
                                        .max(vertex_local.x - rect_max.x)
                                        .max(0.0);
                                    let d_y = (rect_min.y - vertex_local.y)
                                        .max(vertex_local.y - rect_max.y)
                                        .max(0.0);
                                    let d_d = (d_x * d_x + d_y * d_y).sqrt();

                                    1.0 - (d_d / falloff)
                                }
                            };

                            let clamped_strength = strength.clamp(
                                0.0,
                                modifier_strength_limit.map_or(1.0, |modifier| modifier.0),
                            );
                            let eased_strength = easing_function.ease(clamped_strength);

                            if let Some(operation) = operation {
                                *val = apply_modifier(
                                    modifier_properties,
                                    operation,
                                    vertex_position,
                                    shape_translation.xz(),
                                    *val,
                                    global_transform,
                                    eased_strength,
                                    false,
                                );
                            }
                            if let Some((noise_operation, noise_index)) = noise_operation {
                                *val += noise_operation.noise.sample(
                                    vertex_position.x,
                                    vertex_position.y,
                                    unsafe { noise_cache.get_by_index(noise_index) },
                                ) * eased_strength;
                            }
                            if let Some(hole_punch) = hole_punch.filter(|_| strength >= 1.0) {
                                holes.0.set(i, !hole_punch.invert);
                            }
                        }
                    }
                }
            }

            // Finally, set by splines.
            if let Some(splines) = splines {
                let _span = info_span!("Apply splines").entered();

                for entry in splines.iter() {
                    if let Ok((
                        spline,
                        spline_properties,
                        modifier_falloff,
                        modifier_strength_limit,
                    )) = spline_query.get(entry.entity)
                    {
                        let (falloff, easing_function) = modifier_falloff
                            .map_or((f32::EPSILON, EasingFunction::Linear), |falloff| {
                                (falloff.falloff, falloff.easing_function)
                            });

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
                                    - ((distance.sqrt() - spline_properties.half_width) / falloff);

                                let clamped_strength = strength.clamp(
                                    0.0,
                                    modifier_strength_limit.map_or(1.0, |modifier| modifier.0),
                                );
                                let eased_strength = easing_function.ease(clamped_strength);

                                *val = val.lerp(height, eased_strength);
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
    };

    if !modifier_properties.allow_raising {
        new_val = new_val.min(val);
    }
    if !modifier_properties.allow_lowering {
        new_val = new_val.max(val);
    }

    new_val
}
