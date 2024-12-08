use std::num::NonZeroU8;

use bevy::{
    app::{App, Startup},
    asset::{AssetMode, AssetPlugin, AssetServer, Assets},
    color::{Color, Srgba},
    core::Name,
    diagnostic::FrameTimeDiagnosticsPlugin,
    math::{Vec3, Vec3Swizzles},
    pbr::{DirectionalLight, DirectionalLightBundle, PbrBundle, StandardMaterial},
    prelude::{
        default, BuildChildren, Commands, Cuboid, GlobalTransform, Mesh, PluginGroup, Res,
        ResMut, Transform, TransformBundle, VisibilityBundle, With, World,
    },
    DefaultPlugins,
};
use bevy_editor_pls::{
    default_windows::cameras::ActiveEditorCamera,
    editor_window::{EditorWindow, EditorWindowContext},
    egui::{self}, AddEditorWindow, EditorPlugin,
};
use bevy_lookup_curve::{
    editor::{LookupCurveEditor, LookupCurveEguiEditor},
    LookupCurve,
};
use bevy_world_seed::{
    easing::EasingFunction,
    feature_placement::{
        Feature, FeatureDespawnStrategy, FeatureGroup, FeaturePlacementCondition, FeatureScaleRandomization, FeatureSpawnStrategy, TerrainFeatures
    },
    material::{
        GlobalTexturingRules, TerrainTextureRebuildQueue, TerrainTexturingSettings, TexturingRule,
        TexturingRuleEvaluator,
    },
    meshing::TerrainMeshRebuildQueue,
    noise::{
        calc_filter_strength, BiomeSettings, DomainWarping, StrengthCombinator, FilterComparingTo, LayerNoiseSettings, LayerOperation, NoiseCache, NoiseFilter, NoiseFilterCondition, NoiseGroup, NoiseIndexCache, NoiseLayer, NoiseScaling, TerrainNoiseSettings, TerrainNoiseSplineLayer
    },
    terrain::{Terrain, TileToTerrain},
    RebuildTile, TerrainHeightRebuildQueue, TerrainPlugin, TerrainSettings,
};

fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins.set(AssetPlugin {
            mode: AssetMode::Processed,
            ..default()
        }),
        EditorPlugin::default(),
        FrameTimeDiagnosticsPlugin,
    ));

    app.add_plugins(TerrainPlugin {
        noise_settings: Some(TerrainNoiseSettings {
            noise_groups: vec![
                NoiseGroup {
                    layers: vec![
                        NoiseLayer {
                            operation: LayerOperation::Noise {
                                noise: LayerNoiseSettings {
                                    amplitude: 128.0,
                                    frequency: 0.002,
                                    seed: 281139797,
                                    domain_warp: vec![DomainWarping {
                                        amplitude: 30.0,
                                        frequency: 0.001,
                                        z_offset: 3.55
                                    }],
                                    scaling: NoiseScaling::Ridged
                                }
                            },
                            filters: vec![NoiseFilter {
                                condition: NoiseFilterCondition::Above(0.9),
                                falloff: 0.3,
                                falloff_easing_function: EasingFunction::SmoothStep,
                                compare_to: FilterComparingTo::Spline { index: 0 },
                            }],
                            filter_combinator: StrengthCombinator::Max
                        },
                        NoiseLayer {
                            operation: LayerOperation::Noise {
                                noise: LayerNoiseSettings {
                                    amplitude: 64.0,
                                    frequency: 0.002,
                                    seed: 3,
                                    domain_warp: vec![],
                                    scaling: NoiseScaling::Ridged
                                },
                            },
                            filters: vec![],
                            filter_combinator: StrengthCombinator::Max
                        },
                        NoiseLayer {
                            operation: LayerOperation::Noise {
                                noise: LayerNoiseSettings {
                                    amplitude: 32.0,
                                    frequency: 0.002,
                                    seed: 891670187,
                                    domain_warp: vec![],
                                    scaling: NoiseScaling::Ridged
                                }
                            },
                            filters: vec![],
                            filter_combinator: StrengthCombinator::Max
                        },
                        NoiseLayer {
                            operation: LayerOperation::Noise {
                                noise: LayerNoiseSettings {
                                    amplitude: 8.0,
                                    frequency: 0.006,
                                    seed: 2,
                                    domain_warp: vec![],
                                    scaling: NoiseScaling::Ridged
                                }
                            },
                            filters: vec![],
                            filter_combinator: StrengthCombinator::Max
                        },
                        NoiseLayer {
                            operation: LayerOperation::Noise {
                                noise: LayerNoiseSettings {
                                    amplitude: 16.0,
                                    frequency: 0.004,
                                    seed: 3,
                                    domain_warp: vec![],
                                    scaling: NoiseScaling::Ridged
                                },
                            },
                            filters: vec![],
                            filter_combinator: StrengthCombinator::Max
                        },
                        NoiseLayer {
                            operation: LayerOperation::Noise {
                                noise: LayerNoiseSettings {
                                    amplitude: 128.0,
                                    frequency: 0.0005,
                                    seed: 3,
                                    domain_warp: vec![],
                                    scaling: NoiseScaling::Ridged
                                },
                            },
                            filters: vec![NoiseFilter {
                                condition: NoiseFilterCondition::Above(0.8),
                                falloff: 0.25,
                                falloff_easing_function: EasingFunction::SmoothStep,
                                compare_to: FilterComparingTo::Spline { index: 0 },
                            }],
                            filter_combinator: StrengthCombinator::Max
                        },
                        NoiseLayer {
                            operation: LayerOperation::Noise {
                                noise: LayerNoiseSettings {
                                    amplitude: 64.0,
                                    frequency: 0.002,
                                    seed: 3,
                                    domain_warp: vec![],
                                    scaling: NoiseScaling::Ridged
                                },
                            },
                            filters: vec![],
                            filter_combinator: StrengthCombinator::Max
                        },
                    ],
                    filters: vec![NoiseFilter {
                        condition: NoiseFilterCondition::Above(0.6),
                        falloff: 0.15,
                        falloff_easing_function: EasingFunction::SmoothStep,
                        compare_to: FilterComparingTo::Spline { index: 0 },
                    }],
                    filter_combinator: StrengthCombinator::Min
                }
            ],
            data: vec![
                LayerNoiseSettings {
                    amplitude: 1.0,
                    frequency: 0.0003,
                    seed: 3110758200,
                    domain_warp: vec![],
                    scaling: NoiseScaling::Unitized
                },
                LayerNoiseSettings {
                    amplitude: 1.0,
                    frequency: 0.0001,
                    seed: 2400218420,
                    domain_warp: vec![],
                    scaling: NoiseScaling::Unitized
                },
                LayerNoiseSettings {
                    amplitude: 1.0,
                    frequency: 0.0001,
                    seed: 1228950654,
                    domain_warp: vec![],
                    scaling: NoiseScaling::Unitized
                },
            ],
            biome: vec![
                BiomeSettings {
                    filters: vec![NoiseFilter {
                        condition: NoiseFilterCondition::Above(0.5),
                        falloff: 0.1,
                        falloff_easing_function: EasingFunction::SmoothStep,
                        compare_to: FilterComparingTo::Data { index: 0 }
                    },
                    NoiseFilter {
                        condition: NoiseFilterCondition::Above(0.5),
                        falloff: 0.1,
                        falloff_easing_function: EasingFunction::SmoothStep,
                        compare_to: FilterComparingTo::Data { index: 1 }
                    },
                    NoiseFilter {
                        condition: NoiseFilterCondition::Above(0.5),
                        falloff: 0.1,
                        falloff_easing_function: EasingFunction::SmoothStep,
                        compare_to: FilterComparingTo::Data { index: 2 }
                    }],
                    filter_combinator: StrengthCombinator::Min
                }
            ],
            ..default()
        }),
        terrain_settings: TerrainSettings {
            tile_size_power: NonZeroU8::new(7).unwrap(),
            edge_points: 129,
            max_tile_updates_per_frame: NonZeroU8::new(4).unwrap(),
            max_spline_simplification_distance_squared: 6.0,
        },
        texturing_settings: Some(TerrainTexturingSettings {
            texture_resolution_power: NonZeroU8::new(6).unwrap(),
            max_tile_updates_per_frame: NonZeroU8::new(2).unwrap(),
        }),
        debug_draw: true,
    });

    app.add_systems(Startup, spawn_terrain);
    app.add_systems(Startup, insert_rules);

    app.add_editor_window::<NoiseDebugWindow>();

    app.run();
}

/// Defines the rules for procedural texturing
fn insert_rules(
    mut texturing_rules: ResMut<GlobalTexturingRules>,
    mut terrain_noise_settings: ResMut<TerrainNoiseSettings>,
    mut terrain_features: ResMut<TerrainFeatures>,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut material: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    let continentallness = asset_server.load("curves/continentallness.curve.ron");
    let peaks_and_valleys = asset_server.load("curves/peaks_and_valleys.curve.ron");

    terrain_noise_settings.splines.extend([
        TerrainNoiseSplineLayer {
            amplitude_curve: continentallness.clone(),
            frequency: 0.0005,
            seed: 5,
            domain_warp: vec![],
            filters: vec![],
            filter_combinator: StrengthCombinator::Min
        },
        TerrainNoiseSplineLayer {
            amplitude_curve: peaks_and_valleys.clone(),
            frequency: 0.002,
            seed: 14085,
            domain_warp: vec![
                DomainWarping {
                    amplitude: 40.0,
                    frequency: 0.009,
                    z_offset: 93.0
                }
            ],
            filters: vec![
                NoiseFilter {
                    condition: NoiseFilterCondition::Above(0.3),
                    falloff: 0.2,
                    falloff_easing_function: EasingFunction::SmoothStep,
                    compare_to: FilterComparingTo::Spline { index: 0 }
                },
                NoiseFilter {
                    condition: NoiseFilterCondition::Above(1.0),
                    falloff: 1.0,
                    falloff_easing_function: EasingFunction::SmoothStep,
                    compare_to: FilterComparingTo::Data { index: 0 }
                }
            ],
            filter_combinator: StrengthCombinator::Min
        },
    ]);

    commands.spawn(LookupCurveEditor {
        sample: Some(0.0),
        egui_editor: LookupCurveEguiEditor {
            ron_path: Some("./assets/curves/continentallness.curve.ron".to_string()),
            grid_step_y: 10.0,
            ..Default::default()
        },
        ..LookupCurveEditor::new(continentallness)
    });
    commands.spawn(LookupCurveEditor {
        sample: Some(0.0),
        egui_editor: LookupCurveEguiEditor {
            ron_path: Some("./assets/curves/peaks_and_valleys.curve.ron".to_string()),
            grid_step_y: 10.0,
            ..Default::default()
        },
        ..LookupCurveEditor::new(peaks_and_valleys)
    });

    texturing_rules.rules.extend([
        TexturingRule {
            evaluators: vec![TexturingRuleEvaluator::AngleGreaterThan {
                angle_radians: 40.0_f32.to_radians(),
                falloff_radians: 2.5_f32.to_radians(),
            }],
            evaulator_combinator: StrengthCombinator::Min,
            texture: asset_server.load("textures/cracked_concrete_diff_1k.dds"),
            normal_texture: Some(asset_server.load("textures/cracked_concrete_nor_gl_1k.dds")),
            units_per_texture: 4.0,
        },
        TexturingRule {
            evaluators: vec![TexturingRuleEvaluator::AngleLessThan {
                angle_radians: 40.0_f32.to_radians(),
                falloff_radians: 2.5_f32.to_radians(),
            }],
            evaulator_combinator: StrengthCombinator::Min,
            texture: asset_server.load("textures/brown_mud_leaves.dds"),
            normal_texture: Some(asset_server.load("textures/brown_mud_leaves_01_nor_gl_2k.dds")),
            units_per_texture: 4.0,
        },
    ]);

    let cube_mesh_handle = meshes.add(Cuboid::from_length(1.0));
    let red_cube_mesh_handle = cube_mesh_handle.clone();
    let blue_material_handle = material.add(StandardMaterial::from_color(Srgba::BLUE));
    
    let red_material_handle = material.add(StandardMaterial::from_color(Srgba::RED));

    let blue_spawn_strategy = FeatureSpawnStrategy::Custom(Box::new(
        move |commands, terrain_entity, placements, spawned_entities| {
            spawned_entities.extend(placements.iter().map(|placement| {
                commands
                    .spawn(PbrBundle {
                        mesh: cube_mesh_handle.clone(),
                        material: blue_material_handle.clone(),
                        transform: Transform {
                            translation: placement.position + (Vec3::new(0.0, 0.5, 0.0) * placement.scale),
                            rotation: placement.rotation,
                            scale: placement.scale,
                        },
                        ..default()
                    })
                    .set_parent(terrain_entity)
                    .id()
            }));
        },
    ));

    let red_spawn_strategy = FeatureSpawnStrategy::Custom(Box::new(
        move |commands, terrain_entity, placements, spawned_entities| {
            spawned_entities.extend(placements.iter().map(|placement| {
                commands
                    .spawn(PbrBundle {
                        mesh: red_cube_mesh_handle.clone(),
                        material: red_material_handle.clone(),
                        transform: Transform {
                            translation: placement.position + (Vec3::new(0.0, 0.5, 0.0) * placement.scale),
                            rotation: placement.rotation,
                            scale: placement.scale,
                        },
                        ..default()
                    })
                    .set_parent(terrain_entity)
                    .id()
            }));
        },
    ));

    terrain_features.feature_groups.extend([FeatureGroup {
        feature_seed: 5,
        placements_per_tile: 512,
        belongs_to_layers: 1,
        removes_layers: 1,
        features: vec![Feature {
            weight: 1.0,
            collision_radius: 1.0,
            placement_conditions: vec![FeaturePlacementCondition::SlopeBetween {
                min_angle_radians: 0.0,
                max_angle_radians: 45.0_f32.to_radians(),
            },
            FeaturePlacementCondition::InBiome {
                biome: 0
            }],
            randomize_yaw_rotation: true,
            align_to_terrain_normal: true,
            scale_randomization: FeatureScaleRandomization::Uniform { min: 1.0, max: 2.0 },
            spawn_strategy: blue_spawn_strategy,
            despawn_strategy: FeatureDespawnStrategy::Default,
        },
        Feature {
            weight: 0.1,
            collision_radius: 1.0,
            placement_conditions: vec![FeaturePlacementCondition::SlopeBetween {
                min_angle_radians: 0.0,
                max_angle_radians: 45.0_f32.to_radians(),
            }],
            randomize_yaw_rotation: true,
            align_to_terrain_normal: false,
            scale_randomization: FeatureScaleRandomization::Uniform { min: 1.0, max: 2.0 },
            spawn_strategy: red_spawn_strategy,
            despawn_strategy: FeatureDespawnStrategy::Default,
        }],
    }]);
}

fn spawn_terrain(mut commands: Commands, terrain_settings: Res<TerrainSettings>) {
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            illuminance: 1000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_translation(Vec3::new(32.0, 25.0, 16.0))
            .looking_at(Vec3::ZERO, Vec3::Y)
            .with_translation(Vec3::ZERO),
        ..default()
    });

    let terrain_range = 15;

    for x in -terrain_range..terrain_range {
        for z in -terrain_range..terrain_range {
            commands.spawn((
                Terrain::default(),
                TransformBundle::from_transform(Transform::from_translation(Vec3::new(
                    x as f32 * terrain_settings.tile_size(),
                    0.0,
                    z as f32 * terrain_settings.tile_size(),
                ))),
                VisibilityBundle::default(),
                Name::new(format!("Terrain ({x},{z}")),
            ));
        }
    }
}

pub struct NoiseDebugWindow;
#[derive(Default)]
pub struct NoiseDebugWindowState {}
impl EditorWindow for NoiseDebugWindow {
    type State = NoiseDebugWindowState;
    const NAME: &'static str = "Noise Debug";

    fn ui(world: &mut World, _cx: EditorWindowContext, ui: &mut egui::Ui) {
        if ui.button("Regenerate Terrain").clicked() {
            let mut tiles = world
                .resource::<TileToTerrain>()
                .keys()
                .cloned()
                .collect::<Vec<_>>();

            tiles.sort_unstable_by(|a, b| a.x.cmp(&b.x).then(a.y.cmp(&b.y)));

            world.send_event_batch(tiles.into_iter().map(RebuildTile));
        }

        let heights_queue = world.resource::<TerrainHeightRebuildQueue>();
        let mesh_queue = world.resource::<TerrainMeshRebuildQueue>();
        let texture_queue = world.resource::<TerrainTextureRebuildQueue>();

        if !heights_queue.is_empty() || !mesh_queue.is_empty() || !texture_queue.is_empty() {
            ui.heading("Queued");

            ui.columns(3, |ui| {
                ui[0].label("Heights");
                ui[1].label("Meshes");
                ui[2].label("Textures");

                ui[0].label(heights_queue.count().to_string());
                ui[1].label(mesh_queue.count().to_string());
                ui[2].label(texture_queue.count().to_string());
            });
        }

        let mut query_state =
            world.query_filtered::<&GlobalTransform, With<ActiveEditorCamera>>();
        let lookup_curves = world.resource::<Assets<LookupCurve>>();

        if let Ok(transform) = query_state.get_single(world) {
            let noise_cache = world.resource::<NoiseCache>();
            let noise_settings = world.resource::<TerrainNoiseSettings>();
            let noise_index_cache = world.resource::<NoiseIndexCache>();

            let translation = transform.translation();

            let mut data = Vec::with_capacity(noise_settings.data.len());
            noise_settings.sample_data(noise_cache, noise_index_cache, translation.xz(), &mut data);

            let height = noise_settings.sample_position(
                noise_cache,
                noise_index_cache,
                translation.xz(),
                lookup_curves,
                &data,
                &[]
            );
            ui.heading(format!("Height: {height}"));

            ui.heading("Data");
            for (i, data_layer) in noise_settings.data.iter().enumerate() {
                let noise = data_layer.sample_scaled_raw(translation.x, translation.z, unsafe { noise_cache.get_by_index(noise_index_cache.data_index_cache[i] as usize) });
                
                ui.label(format!("- {i}: {noise:.3}"));
            }

            ui.heading("Splines");

            for (i, spline) in noise_settings.splines.iter().enumerate() {
                let cached_noise = unsafe { noise_cache.get_by_index(noise_index_cache.spline_index_cache[i] as usize) };
                let noise_raw = spline.sample_raw(
                    translation.x,
                    translation.z,
                    cached_noise,
                );
                let noise = spline.sample(
                    translation.x,
                    translation.z,
                    noise_settings,
                    noise_cache,
                    &data,
                    &[],
                    &noise_index_cache.spline_index_cache,
                    cached_noise,
                    lookup_curves,
                );

                let strength = calc_filter_strength(translation.xz(), &spline.filters, spline.filter_combinator, noise_settings, noise_cache, &data, &[], &noise_index_cache.spline_index_cache);

                if let Some(lookup_curve) = lookup_curves.get(&spline.amplitude_curve) {
                    ui.label(format!(
                        "- {}: {noise:.3} (Noise: {noise_raw:.3}, Strength: {strength:.3})",
                        lookup_curve.name.as_ref().map_or("", |name| name.as_str())
                    ));
                }
            }

            ui.heading("Noise Groups");

            for (i, group) in noise_settings.noise_groups.iter().enumerate() {
                let group_noises = &noise_index_cache.group_index_cache[noise_index_cache.group_offset_cache[i] as usize..];

                let noise = group.sample(
                    noise_settings,
                    noise_cache,
                    &data,
                    &[],
                    &noise_index_cache.spline_index_cache,
                    group_noises,
                    translation.xz()
                );

                let strength = calc_filter_strength(translation.xz(), &group.filters, group.filter_combinator, noise_settings, noise_cache, &data, &[], &noise_index_cache.spline_index_cache);

                egui::CollapsingHeader::new(format!("GROUP {i}: {noise:.3} ({strength:.3})")).id_source(format!("group_{i}")).show(ui, |ui| {
                    unsafe {
                        for (i,layer) in group.layers.iter().enumerate() {
                            let strength = calc_filter_strength(translation.xz(), &layer.filters, layer.filter_combinator, noise_settings, noise_cache, &data, &[], &noise_index_cache.spline_index_cache);
                            
                            match &layer.operation {
                                LayerOperation::Noise { noise } => {
                                    let cached_noise = noise_cache.get_by_index(group_noises[i] as usize);
    
                                    let noise_raw = noise.sample_scaled_raw(
                                        translation.x,
                                        translation.z,
                                        cached_noise,
                                    );
                                    let noise = noise.sample(
                                        translation.x,
                                        translation.z,
                                        cached_noise,
                                    );
        
                                    ui.label(format!("- {i}: {noise:.3} (Noise: {noise_raw:.3}, Strength: {strength:.3})"));
                                },
                                LayerOperation::Step { step } => {
                                    ui.label(format!("- {i}: Step {step} (Strength: {strength:.3})"));
                                },
                            }
                        
                        }
                    }
                });
            }
        }
    }
}
