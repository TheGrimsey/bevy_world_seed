use std::f32::consts::TAU;

use bevy::{
    app::{App, Plugin, PostUpdate},
    log::info_span,
    math::{IVec2, Vec2, Vec3, Vec3Swizzles},
    prelude::{
        on_event, Commands, Component, DespawnRecursiveExt, Entity, EventReader, IntoSystemConfigs,
        Query, ReflectComponent, Res, Resource,
    },
    reflect::Reflect,
};
use turborand::{rng::Rng, SeededCore, TurboRand};

use crate::{
    terrain::TileToTerrain,
    utils::{get_flat_normal_at_position_in_tile, get_height_at_position_in_tile},
    Heights, TerrainSets, TerrainSettings, TileHeightsRebuilt,
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

        app.register_type::<SpawnedFeatures>();
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
    pub collision_radius: f32,
    pub placement_conditions: Vec<FeaturePlacementCondition>,

    pub randomize_yaw_rotation: bool,
    pub scale_randomization: FeatureScaleRandomization,

    pub spawn_strategy: FeatureSpawnStrategy,
    pub despawn_strategy: FeatureDespawnStrategy,
}
impl Feature {
    pub fn check_conditions(
        &self,
        placement_position: Vec3,
        heights: &Heights,
        terrain_settings: &TerrainSettings,
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
                    let normal = get_flat_normal_at_position_in_tile(
                        placement_position.xz(),
                        heights,
                        terrain_settings,
                    );
                    let angle = normal.dot(Vec3::Y).acos();

                    *min_angle_radians <= angle && angle <= *max_angle_radians
                }
            })
    }
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
        heights: &Heights,
        terrain_settings: &TerrainSettings,
    ) -> Vec<FeaturePlacement> {
        let tile_size = terrain_settings.tile_size();

        (0..self.placements_per_tile)
            .filter_map(|index| {
                let feature_index = rng.u32(0..self.features.len() as u32);
                let position = Vec2::new(rng.f32() * tile_size, rng.f32() * tile_size);
                let height = get_height_at_position_in_tile(position, heights, terrain_settings);

                let position = Vec3::new(position.x, height, position.y);

                let feature = &self.features[feature_index as usize]; 
                
                if feature.check_conditions(
                    position,
                    heights,
                    terrain_settings,
                ) {
                    let scale = match &feature.scale_randomization {
                        FeatureScaleRandomization::None => Vec3::ONE,
                        FeatureScaleRandomization::Uniform { min, max } => Vec3::splat(*min + rng.f32() * (*max - *min)),
                        FeatureScaleRandomization::Independent { min, max } => {
                            let rng = Vec3::new(rng.f32(), rng.f32(), rng.f32());

                            *min + rng * (*max - *min)
                        },
                    };
                    let yaw_rotation_radians = if feature.randomize_yaw_rotation {
                        rng.f32() * TAU
                    } else {
                        0.0
                    };

                    Some(FeaturePlacement {
                        index,
                        feature: feature_index,
                        position,
                        scale,
                        yaw_rotation_radians
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

                    let placement_radius = feature.collision_radius * placement.scale.xz().max_element();

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

                                            let radii_sum = placement_radius + (other_feature.collision_radius * other_placement.scale.xz().max_element());
                                            let min_distance_squared = radii_sum * radii_sum;

                                            placement
                                                .position
                                                .distance_squared(other_placement.position)
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
                                                .position
                                                .distance_squared(other_placement.position)
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

#[derive(Resource, Default)]
pub struct TerrainFeatures {
    pub feature_groups: Vec<FeatureGroup>,
}

#[derive(Debug, Clone)]
pub struct FeaturePlacement {
    pub index: u32,
    pub feature: u32,
    pub position: Vec3,
    pub scale: Vec3,
    pub yaw_rotation_radians: f32
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
    mut query: Query<(&Heights, &mut SpawnedFeatures)>,
) {
    for TileHeightsRebuilt(tile) in events.read() {
        let _span = info_span!("Spawn features for tile").entered();
        let Some(tile_entity) = tile_to_terrain
            .get(tile)
            .and_then(|tiles| tiles.first().cloned())
        else {
            continue;
        };
        let Ok((heights, mut spawned_features)) = query.get_mut(tile_entity) else {
            continue;
        };

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

                    feature_group.generate_placements(&rng, heights, &terrain_settings)
                })
                .collect::<Vec<_>>()
        };

        let mut filtered_feature_placements =
            filter_features_by_collision(&terrain_features, feature_placements, Vec::default());

        for (i, mut feature_placements) in filtered_feature_placements
            .drain(..)
            .enumerate()
            .filter(|(_, placements)| !placements.is_empty())
        {
            let _span = info_span!("Spawn feature group").entered();
            let feature_group = &terrain_features.feature_groups[i];

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
                        group: i as u32,
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
                group: i as u32,
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
