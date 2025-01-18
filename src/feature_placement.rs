use core::f32;
use std::f32::consts::TAU;
use bevy_transform::components::{Transform, GlobalTransform};
use bevy_app::{App, Plugin, PostUpdate};
use bevy_math::{IVec2, Quat, Vec2, Vec3, Vec3Swizzles};
use bevy_log::info_span;
use bevy_ecs::prelude::{
    on_event, Commands, Component, Entity, EventReader, IntoSystemConfigs,
    Query, ReflectComponent, Res, ResMut, Resource
};
use bevy_hierarchy::DespawnRecursiveExt;
use bevy_reflect::Reflect;

use turborand::{rng::Rng, SeededCore, TurboRand};

use crate::{
    modifiers::{ModifierFalloffNoiseProperty, ShapeModifier, TerrainSplineCached, TileToModifierMapping}, noise::{NoiseCache, NoiseFilterCondition, NoiseIndexCache, TerrainNoiseSettings, TileBiomes}, terrain::TileToTerrain, utils::{distance_squared_to_line_segment, get_flat_normal_at_position_in_tile, get_height_at_position_in_tile}, Heights, TerrainSets, TerrainSettings, TileHeightsRebuilt
};

pub struct FeaturePlacementPlugin;
impl Plugin for FeaturePlacementPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            update_features_on_tile_built
                .run_if(on_event::<TileHeightsRebuilt>())
                .after(TerrainSets::Heights),
        );

        app.init_resource::<TerrainFeatures>();

        app.register_type::<SpawnedFeatures>().register_type::<ShapeBlocksFeaturePlacement>();
    }
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq, Debug)]
pub enum FeaturePlacementCondition {
    HeightBetween {
        min: f32,
        max: f32,
    },
    SlopeBetween {
        min_angle_radians: f32,
        max_angle_radians: f32,
    },
    InBiome {
        biome: u32
    },
    InNoise {
        noise: u32,
        condition: NoiseFilterCondition,
        falloff: f32
    }
    /*HeightDeltaInRadiusLessThan {
        radius: f32,
        max_increase: f32,
        max_decrease: f32
    }*/

}

pub enum FeatureSpawnStrategy {
    Custom(Box<dyn Fn(&mut Commands, Entity, &[FeaturePlacement], &mut Vec<Entity>) + Sync + Send>),
}

pub enum FeatureDespawnStrategy {
    /// Calls `despawn_recursive` on the entity.
    Default,
    Custom(Box<dyn Fn(&mut Commands, &[Entity]) + Sync + Send>),
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum FeatureScaleRandomization {
    /// Scale will stay as Vec3::ONE,
    #[default]
    None,
    Uniform {
        min: f32,
        max: f32,
    },
    Independent {
        min: Vec3,
        max: Vec3
    }
}

pub struct Feature {
    pub weight: f32,

    pub collision_radius: f32,
    pub placement_conditions: Vec<FeaturePlacementCondition>,

    pub randomize_yaw_rotation: bool,
    pub align_to_terrain_normal: bool,
    pub scale_randomization: FeatureScaleRandomization,

    pub spawn_strategy: FeatureSpawnStrategy,
    pub despawn_strategy: FeatureDespawnStrategy,
}
impl Feature {
    pub fn check_conditions(
        &self,
        tile_translation: Vec2,
        placement_position: Vec3,
        // If not provided, generated in condition.
        terrain_normal: Option<Vec3>,
        heights: &Heights,
        biome_values: &[f32],
        terrain_settings: &TerrainSettings,
        terrain_noise_layers: &TerrainNoiseSettings,
        noise_cache: &NoiseCache,
        noise_index_cache: &NoiseIndexCache,
        rng: &Rng
    ) -> bool {
        self.placement_conditions
            .iter()
            .all(|condition| match condition {
                FeaturePlacementCondition::HeightBetween { min, max } => {
                    *min <= placement_position.y && placement_position.y <= *max
                }
                FeaturePlacementCondition::SlopeBetween {
                    min_angle_radians,
                    max_angle_radians,
                } => {
                    let normal = terrain_normal.unwrap_or_else(|| get_flat_normal_at_position_in_tile(
                        placement_position.xz(),
                        heights,
                        terrain_settings,
                    ));
                    let angle = normal.dot(Vec3::Y).acos();

                    *min_angle_radians <= angle && angle <= *max_angle_radians
                }
                FeaturePlacementCondition::InBiome { biome } => {
                    if biome_values.get(*biome as usize).is_some_and(|biome| *biome > f32::EPSILON) {
                        let pos = tile_translation + placement_position.xz(); 
                        // TODO: Cache this...
                        let mut data = Vec::with_capacity(terrain_noise_layers.data.len());
                        terrain_noise_layers.sample_data(noise_cache, noise_index_cache, pos, &mut data);

                        let mut biomes = Vec::with_capacity(terrain_noise_layers.biome.len());
                        terrain_noise_layers.sample_biomes(noise_cache, noise_index_cache, pos, &data, &mut biomes);

                        biomes.get(*biome as usize).is_some_and(|biomeness| rng.f32() <= *biomeness)
                    } else {
                        false
                    }
                },
                FeaturePlacementCondition::InNoise { noise, condition, falloff } => {
                    if let Some(noise_layer) = terrain_noise_layers.data.get(*noise as usize) {
                        let pos = tile_translation + placement_position.xz(); 
                        let sample = noise_layer.sample_scaled_raw(pos.x, pos.y, unsafe { noise_cache.get_by_index(noise_index_cache.data_index_cache[*noise as usize] as usize) });

                        rng.f32() <= condition.evaluate_condition(sample, *falloff)
                    } else {
                        false
                    }
                },
            })
    }
}

/// The shape blocks placing features within it's bounds.
/// 
/// Use to remove features from for example buildings.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct ShapeBlocksFeaturePlacement {
    /// Feature layers which this shape blocks. 
    pub layer: u64
}

pub struct FeatureGroup {
    pub feature_seed: u32,
    pub placements_per_tile: u32,

    pub belongs_to_layers: u64,
    pub removes_layers: u64,

    pub features: Vec<Feature>,
}
impl FeatureGroup {
    pub fn generate_placements(
        &self,
        rng: &Rng,
        tile_translation: Vec2,
        heights: &Heights,
        tile_biomes: &TileBiomes,
        terrain_settings: &TerrainSettings,
        terrain_noise_layers: &TerrainNoiseSettings,
        noise_cache: &NoiseCache,
        noise_index_cache: &NoiseIndexCache,
    ) -> Vec<FeaturePlacement> {
        let tile_size = terrain_settings.tile_size();
        let total_weight = self.features.iter().fold(0.0, |acc, feature| acc + feature.weight);

        (0..self.placements_per_tile)
            .filter_map(|index| {
                let mut weight_i = rng.f32() * total_weight;

                let (feature_index, feature) = self.features.iter().enumerate().find(|(_, entry)| {
                    weight_i -= entry.weight;

                    weight_i <= 0.0
                }).unwrap();

                let position = Vec2::new(rng.f32() * tile_size, rng.f32() * tile_size);
                let height = get_height_at_position_in_tile(position, heights, terrain_settings);

                let position = Vec3::new(position.x, height, position.y);
                
                let scale = match &feature.scale_randomization {
                    FeatureScaleRandomization::None => Vec3::ONE,
                    FeatureScaleRandomization::Uniform { min, max } => {
                        let x = rng.f32();
                        
                        Vec3::splat(*min + x * (*max - *min))
                    } ,
                    FeatureScaleRandomization::Independent { min, max } => {
                        let rng = Vec3::new(
                            rng.f32(),
                            rng.f32(),
                            rng.f32(),
                        );

                        *min + rng * (*max - *min)
                    },
                };

                let (terrain_normal, mut rotation) = if feature.align_to_terrain_normal {
                    let normal = get_flat_normal_at_position_in_tile(
                        position.xz(),
                        heights,
                        terrain_settings,
                    );

                    // Calculate the angle between Y axis and the normal
                    let dot = Vec3::Y.dot(normal);
                    let angle = dot.acos();

                    // Calculate the rotation axis as the cross product of Y axis and the normal
                    let axis = Vec3::Y.cross(normal).normalize();

                    (Some(normal), Quat::from_axis_angle(axis, angle))
                } else {
                    (None, Quat::IDENTITY)
                };

                if feature.randomize_yaw_rotation {
                    let yaw = rng.f32() * TAU;

                    rotation *= Quat::from_rotation_y(yaw);
                }

                if feature.check_conditions(
                    tile_translation,
                    position,
                    terrain_normal,
                    heights,
                    &tile_biomes.0,
                    terrain_settings,
                    terrain_noise_layers,
                    noise_cache,
                    noise_index_cache,
                    rng
                ) {
                    Some(FeaturePlacement {
                        index,
                        feature: feature_index as u32,
                        transform: Transform {
                            translation: position,
                            rotation,
                            scale,
                        },
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

fn filter_features_by_collision(
    terrain_features: &TerrainFeatures,
    feature_placements: Vec<Vec<FeaturePlacement>>,
    adjacent_placements: Vec<Vec<FeaturePlacement>>,
) -> Vec<Vec<FeaturePlacement>> {
    let _span = info_span!("Filter features by collisions").entered();

    feature_placements
        .iter()
        .enumerate()
        .map(|(i, placements_to_filter)| {
            let feature_group = &terrain_features.feature_groups[i];

            placements_to_filter
                .iter()
                .filter_map(|placement| {
                    let feature = &feature_group.features[placement.feature as usize];

                    let placement_radius = feature.collision_radius * placement.transform.scale.xz().max_element();

                    let can_place =
                        feature_placements
                            .iter()
                            .enumerate()
                            .all(|(j, other_placements)| {
                                let other_feature_group = &terrain_features.feature_groups[j];
                                let features_can_overlap = feature_group.belongs_to_layers
                                    & other_feature_group.removes_layers
                                    != 0;

                                if features_can_overlap {
                                    let same_feature_group = i == j;

                                    other_placements
                                        .iter()
                                        .filter(|other_placement| {
                                            !same_feature_group
                                                || placement.index != other_placement.index
                                        })
                                        .all(|other_placement| {
                                            let other_feature: &Feature = &other_feature_group
                                                .features
                                                [other_placement.feature as usize];

                                            let radii_sum = placement_radius + (other_feature.collision_radius * other_placement.transform.scale.xz().max_element());
                                            let min_distance_squared = radii_sum * radii_sum;

                                            placement
                                                .transform
                                                .translation
                                                .distance_squared(other_placement.transform.translation)
                                                >= min_distance_squared
                                        })
                                } else {
                                    true
                                }
                            })
                            && adjacent_placements.iter().enumerate().all(
                                |(j, other_placements)| {
                                    let other_feature_group = &terrain_features.feature_groups[j];
                                    let features_can_overlap = feature_group.belongs_to_layers
                                        & other_feature_group.removes_layers
                                        != 0;

                                    if !features_can_overlap {
                                        other_placements.iter().all(|other_placement| {
                                            let other_feature: &Feature = &other_feature_group
                                                .features
                                                [other_placement.feature as usize];
                                            let min_distance_squared = (feature.collision_radius
                                                + feature.collision_radius)
                                                * (other_feature.collision_radius
                                                    + other_feature.collision_radius);

                                            placement
                                                .transform
                                                .translation
                                                .distance_squared(other_placement.transform.translation)
                                                >= min_distance_squared
                                        })
                                    } else {
                                        true
                                    }
                                },
                            );

                    if can_place {
                        Some(placement.clone())
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect()
}

fn filter_features_by_blocking_shapes(
    tile: IVec2,
    tile_size: f32,
    tile_translation: Vec2,
    tile_to_modifier: &TileToModifierMapping,
    shape_modifiers_query: &Query<(&ShapeBlocksFeaturePlacement, &ShapeModifier, &GlobalTransform, Option<&ModifierFalloffNoiseProperty>)>,
    spline_modifiers_query: &Query<(&ShapeBlocksFeaturePlacement, &TerrainSplineCached)>,
    feature_placements: &mut [Vec<FeaturePlacement>],
    terrain_features: &TerrainFeatures,
    noise_cache: &mut NoiseCache,
) {
    let _span = info_span!("Filter features by blocking shapes").entered();
    
    let shape_modifiers_in_tile = tile_to_modifier.shape.get(&tile);
    let spline_modifiers_in_tile = tile_to_modifier.splines.get(&tile);

    feature_placements.iter_mut().enumerate().for_each(|(feature_group, placements)| {
        let feature_group = &terrain_features.feature_groups[feature_group];

        if let Some(shape_modifiers_in_tile) = shape_modifiers_in_tile {
            for entry in shape_modifiers_in_tile {
                if let Ok((blocks_features, shape, global_transform, modifier_falloff_noise)) = shape_modifiers_query.get(entry.entity) {
                    if blocks_features.layer & feature_group.belongs_to_layers != 0 {
                        let modifier_falloff_noise = modifier_falloff_noise.map(|falloff_noise| {
                            (
                                falloff_noise,
                                noise_cache.get_simplex_index(falloff_noise.noise.seed)
                            )
                        });
    
                        let mut offset_transform = Transform::from(*global_transform);
                        offset_transform.translation -= Vec3::new(tile_translation.x, 0.0, tile_translation.y);
                        let global_transform = GlobalTransform::from(offset_transform);
                        
                        placements.retain(|placement| {
                            let overlaps_x = ((placement.transform.translation.x / tile_size) * 7.0) as u32;
                            let overlap_y = ((placement.transform.translation.z / tile_size) * 7.0) as u32;
                            let overlap_index = overlap_y * 8 + overlaps_x;
                            
                            if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                return false;
                            }

                            let feature = &feature_group.features[placement.feature as usize];
                            let feature_radius = feature.collision_radius * placement.transform.scale.xz().max_element();
                            
                            let vertex_local = global_transform
                                .affine()
                                .inverse()
                                .transform_point3(placement.transform.translation.with_y(0.0))
                                .xz();
    
                            let distance_offset = modifier_falloff_noise.map_or(0.0, |(falloff_noise, noise_index)| {
                                let normalized_vertex = global_transform.translation().xz() + vertex_local;
    
                                falloff_noise.noise.sample(
                                    normalized_vertex.x,
                                    normalized_vertex.y,
                                    unsafe { noise_cache.get_by_index(noise_index) },
                                )
                            });
    
                            let distance = match shape {
                                ShapeModifier::Circle { radius } => {
                                    vertex_local.distance(Vec2::ZERO) - radius
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
                                    
                                    (d_x * d_x + d_y * d_y).sqrt()
                                }
                            };
                            
                            (distance - distance_offset - feature_radius) > 0.0
                        });
                    }
                } 
            }
        }

        if let Some(splines_in_tile) = spline_modifiers_in_tile {
            for entry in splines_in_tile {
                if let Ok((blocks_features, spline)) = spline_modifiers_query.get(entry.entity) {
                    if blocks_features.layer & feature_group.belongs_to_layers != 0 {
                        placements.retain(|placement| {
                            let feature = &feature_group.features[placement.feature as usize];
                            let feature_radius = feature.collision_radius * placement.transform.scale.xz().max_element();

                            let placement_position = tile_translation + placement.transform.translation.xz();
                            let mut distance = f32::INFINITY;

                            for points in spline.points.windows(2) {
                                let a_2d = points[0].xz();
                                let b_2d = points[1].xz();

                                let (new_distance, _) =
                                    distance_squared_to_line_segment(a_2d, b_2d, placement_position);

                                if new_distance < distance {
                                    distance = new_distance;
                                }
                            }

                            (distance.sqrt() - feature_radius) > 0.0
                        });
                    }
                }
            }
        }
    });
}

#[derive(Resource, Default)]
pub struct TerrainFeatures {
    pub feature_groups: Vec<FeatureGroup>,
}

#[derive(Debug, Clone)]
pub struct FeaturePlacement {
    pub index: u32,
    pub feature: u32,
    /// Tile local transform of the feature. 
    pub transform: Transform,
}

fn cantor_hash(a: u64, b: u64) -> u64 {
    (a + b + 1).wrapping_mul(a + b) / 2 + b
}
pub fn seed_hash(tile: IVec2, feature_seed: u64) -> u64 {
    let x = if tile.x < 0 {
        tile.x.abs() as i64 + i32::MAX as i64
    } else {
        tile.x as i64
    };
    let y = if tile.y < 0 {
        tile.y.abs() as i64 + i32::MAX as i64
    } else {
        tile.y as i64
    };

    cantor_hash(x as u64, y as u64).wrapping_mul(feature_seed)
}

#[test]
fn no_seed_collisions() {
    let feature_seed = 5;

    let mut seeds: std::collections::HashMap<u64, Vec<IVec2>> = std::collections::HashMap::new();

    let range = 500;

    for x in -range..range {
        for y in -range..range {
            let seed = seed_hash(IVec2::new(x, y), feature_seed);

            if let Some(entries) = seeds.get_mut(&seed) {
                entries.push(IVec2::new(x, y));
            } else {
                seeds.insert(seed, vec![IVec2::new(x, y)]);
            }
        }
    }

    assert!(seeds.values().all(|entries| entries.len() == 1));
}

#[derive(Reflect)]
struct SpawnedFeature {
    group: u32,
    feature: u32,
    instances: Vec<Entity>,
}

/// Holds all features belonging to this tile.
#[derive(Reflect, Component, Default)]
#[reflect(Component)]
pub struct SpawnedFeatures {
    spawned: Vec<SpawnedFeature>,
}

fn update_features_on_tile_built(
    mut commands: Commands,
    mut events: EventReader<TileHeightsRebuilt>,
    tile_to_terrain: Res<TileToTerrain>,
    terrain_features: Res<TerrainFeatures>,
    terrain_settings: Res<TerrainSettings>,
    terrain_noise_layers: Res<TerrainNoiseSettings>,
    mut noise_cache: ResMut<NoiseCache>,
    noise_index_cache: Res<NoiseIndexCache>,
    mut query: Query<(&Heights, &mut SpawnedFeatures, &TileBiomes)>,
    blocking_shape_modifiers_query: Query<(&ShapeBlocksFeaturePlacement, &ShapeModifier, &GlobalTransform, Option<&ModifierFalloffNoiseProperty>)>,
    blocking_spline_modifiers_query: Query<(&ShapeBlocksFeaturePlacement, &TerrainSplineCached)>,
    tile_to_modifier: Res<TileToModifierMapping>
) {
    for TileHeightsRebuilt(tile) in events.read() {
        let _span = info_span!("Spawn features for tile").entered();
        let Some(tile_entity) = tile_to_terrain
            .get(tile)
            .and_then(|tiles| tiles.first().cloned())
        else {
            continue;
        };
        let Ok((heights, mut spawned_features, tile_biomes)) = query.get_mut(tile_entity) else {
            continue;
        };

        let tile_translation = (*tile << terrain_settings.tile_size_power.get() as u32).as_vec2();

        for spawned_feature in spawned_features.spawned.drain(..) {
            let feature_group = &terrain_features.feature_groups[spawned_feature.group as usize];
            let feature = &feature_group.features[spawned_feature.feature as usize];

            match &feature.despawn_strategy {
                FeatureDespawnStrategy::Default => {
                    for entity in spawned_feature.instances {
                        commands.entity(entity).despawn_recursive();
                    }
                }
                FeatureDespawnStrategy::Custom(despawn_fn) => {
                    despawn_fn(&mut commands, &spawned_feature.instances);
                }
            }
        }

        let feature_placements = {
            let _span = info_span!("Generate feature placements").entered();
            terrain_features
                .feature_groups
                .iter()
                .map(|feature_group| {
                    let seed = seed_hash(*tile, feature_group.feature_seed as u64);
                    let rng = Rng::with_seed(seed);

                    feature_group.generate_placements(&rng, tile_translation, heights, tile_biomes, &terrain_settings, &terrain_noise_layers, &noise_cache, &noise_index_cache)
                })
                .collect::<Vec<_>>()
        };

        let mut filtered_feature_placements =
            filter_features_by_collision(&terrain_features, feature_placements, Vec::default());

        filter_features_by_blocking_shapes(
            *tile,
            terrain_settings.tile_size(),
            tile_translation,
            &tile_to_modifier,
            &blocking_shape_modifiers_query,
            &blocking_spline_modifiers_query,
            &mut filtered_feature_placements,
            &terrain_features,
            &mut noise_cache,
        );

        for (group_i, mut feature_placements) in filtered_feature_placements
            .into_iter()
            .enumerate()
            .filter(|(_, placements)| !placements.is_empty())
        {
            let _span = info_span!("Spawn feature group").entered();
            let feature_group = &terrain_features.feature_groups[group_i];

            // Sort placements so that placements are grouped by the feature they are for.
            // This allows us to send slices of the same feature's placements to user code.
            feature_placements
                .sort_unstable_by(|a, b| a.feature.cmp(&b.feature).then(a.index.cmp(&b.index)));

            // Now that we have it sorted we just need to separate out each feature & spawn them.
            let mut start_index = 0;
            for i in 1..feature_placements.len() {
                let feature_i = feature_placements[start_index].feature;
                if feature_i != feature_placements[i].feature {
                    let feature = &feature_group.features[feature_i as usize];
                    let mut spawned_feature = SpawnedFeature {
                        group: group_i as u32,
                        feature: feature_i,
                        instances: Vec::with_capacity(i - start_index),
                    };

                    {
                        let _span = info_span!("Spawn feature").entered();
                        match &feature.spawn_strategy {
                            FeatureSpawnStrategy::Custom(spawn_function) => {
                                spawn_function(
                                    &mut commands,
                                    tile_entity,
                                    &feature_placements[start_index..i],
                                    &mut spawned_feature.instances,
                                );
                            }
                        }
                    }

                    spawned_features.spawned.push(spawned_feature);

                    // This is a new slice of features, so we move up the index.
                    start_index = i;
                }
            }

            // Deal with the remaining grouping.
            let feature_i = feature_placements[start_index].feature;
            let feature = &feature_group.features[feature_i as usize];
            let mut spawned_feature = SpawnedFeature {
                group: group_i as u32,
                feature: feature_i,
                instances: Vec::with_capacity(feature_placements.len() - start_index),
            };

            {
                let _span = info_span!("Spawn feature").entered();
                match &feature.spawn_strategy {
                    FeatureSpawnStrategy::Custom(spawn_function) => {
                        spawn_function(
                            &mut commands,
                            tile_entity,
                            &feature_placements[start_index..],
                            &mut spawned_feature.instances,
                        );
                    }
                }
            }
            spawned_features.spawned.push(spawned_feature);
        }
    }
}
