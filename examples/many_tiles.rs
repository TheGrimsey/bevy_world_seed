use std::num::NonZeroU8;

use bevy::{
    app::{App, Startup},
    asset::{AssetMode, AssetPlugin, AssetServer, Assets},
    color::{Color, Srgba},
    core::Name,
    diagnostic::FrameTimeDiagnosticsPlugin,
    math::{Quat, Vec3, Vec3Swizzles},
    pbr::{DirectionalLight, DirectionalLightBundle, PbrBundle, StandardMaterial},
    prelude::{
        default, BuildChildren, Commands, Cuboid, GlobalTransform, Mesh, Mut, PluginGroup, Res,
        ResMut, Transform, TransformBundle, VisibilityBundle, With, World,
    },
    DefaultPlugins,
};
use bevy_editor_pls::{
    default_windows::cameras::ActiveEditorCamera,
    editor_window::{EditorWindow, EditorWindowContext},
    egui, AddEditorWindow, EditorPlugin,
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
        FilterComparingTo, FilteredTerrainNoiseDetailLayer, NoiseCache, NoiseFilter,
        NoiseFilterCondition, TerrainNoiseDetailLayer, TerrainNoiseSettings,
        TerrainNoiseSplineLayer,
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
            splines: vec![],
            layers: vec![
                FilteredTerrainNoiseDetailLayer {
                    layer: TerrainNoiseDetailLayer {
                        amplitude: 16.0,
                        frequency: 0.005,
                        seed: 3,
                    },
                    filter: Some(NoiseFilter {
                        condition: NoiseFilterCondition::Above(0.4),
                        falloff: 0.1,
                        falloff_easing_function: EasingFunction::CubicInOut,
                        compare_to: FilterComparingTo::Spline { index: 0 },
                    }),
                },
                FilteredTerrainNoiseDetailLayer {
                    layer: TerrainNoiseDetailLayer {
                        amplitude: 8.0,
                        frequency: 0.01,
                        seed: 1,
                    },
                    filter: None,
                },
                FilteredTerrainNoiseDetailLayer {
                    layer: TerrainNoiseDetailLayer {
                        amplitude: 4.0,
                        frequency: 0.02,
                        seed: 2,
                    },
                    filter: None,
                },
                FilteredTerrainNoiseDetailLayer {
                    layer: TerrainNoiseDetailLayer {
                        amplitude: 2.0,
                        frequency: 0.04,
                        seed: 3,
                    },
                    filter: None,
                },
            ],
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
            frequency: 0.001,
            seed: 5,
        },
        TerrainNoiseSplineLayer {
            amplitude_curve: peaks_and_valleys.clone(),
            frequency: 0.005,
            seed: 6,
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
            evaluator: TexturingRuleEvaluator::AngleGreaterThan {
                angle_radians: 40.0_f32.to_radians(),
                falloff_radians: 2.5_f32.to_radians(),
            },
            texture: asset_server.load("textures/cracked_concrete_diff_1k.dds"),
            normal_texture: Some(asset_server.load("textures/cracked_concrete_nor_gl_1k.dds")),
            units_per_texture: 4.0,
        },
        TexturingRule {
            evaluator: TexturingRuleEvaluator::AngleLessThan {
                angle_radians: 40.0_f32.to_radians(),
                falloff_radians: 2.5_f32.to_radians(),
            },
            texture: asset_server.load("textures/brown_mud_leaves.dds"),
            normal_texture: Some(asset_server.load("textures/brown_mud_leaves_01_nor_gl_2k.dds")),
            units_per_texture: 4.0,
        },
    ]);

    let mesh_handle = meshes.add(Cuboid::from_length(1.0));
    let material_handle = material.add(StandardMaterial::from_color(Srgba::BLUE));

    let spawn_strategy = FeatureSpawnStrategy::Custom(Box::new(
        move |commands, terrain_entity, placements, spawned_entities| {
            spawned_entities.extend(placements.iter().map(|placement| {
                commands
                    .spawn(PbrBundle {
                        mesh: mesh_handle.clone(),
                        material: material_handle.clone(),
                        transform: Transform {
                            translation: placement.position + (Vec3::new(0.0, 0.5, 0.0) * placement.scale),
                            rotation: Quat::from_rotation_y(placement.yaw_rotation_radians),
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
            collision_radius: 1.0,
            placement_conditions: vec![FeaturePlacementCondition::SlopeBetween {
                min_angle_radians: 0.0,
                max_angle_radians: 45.0_f32.to_radians(),
            }],
            randomize_yaw_rotation: true,
            scale_randomization: FeatureScaleRandomization::Uniform { min: 0.25, max: 2.5 },
            spawn_strategy,
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

    let terrain_range = 1;

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
        world.resource_scope(|world, mut noise_cache: Mut<NoiseCache>| {
            if ui.button("Regenerate Terrain").clicked() {
                let mut tiles = world
                    .resource::<TileToTerrain>()
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>();

                tiles.sort_by(|a, b| a.x.cmp(&b.x).then(a.y.cmp(&b.y)));

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
                let noise_settings = world.resource::<TerrainNoiseSettings>();
                let translation = transform.translation();

                let height = noise_settings.sample_position(
                    &mut noise_cache,
                    translation.xz(),
                    lookup_curves,
                );
                ui.heading(format!("Height: {height}"));

                ui.heading("Spline Noise");

                for spline in noise_settings.splines.iter() {
                    let noise_raw = spline.sample_raw(
                        translation.x,
                        translation.z,
                        noise_cache.get(spline.seed),
                    );
                    let noise = spline.sample(
                        translation.x,
                        translation.z,
                        noise_cache.get(spline.seed),
                        lookup_curves,
                    );

                    if let Some(lookup_curve) = lookup_curves.get(&spline.amplitude_curve) {
                        ui.label(format!(
                            "- {}: {noise:.5} ({noise_raw:.2})",
                            lookup_curve.name.as_ref().map_or("", |name| name.as_str())
                        ));
                    }
                }

                ui.heading("Detail Noise");

                for (i, detail_layer) in noise_settings.layers.iter().enumerate() {
                    let noise_raw = detail_layer.layer.sample_raw(
                        translation.x,
                        translation.z,
                        noise_cache.get(detail_layer.layer.seed),
                    );
                    let noise = detail_layer.layer.sample(
                        translation.x,
                        translation.z,
                        noise_cache.get(detail_layer.layer.seed),
                    );

                    ui.label(format!("- {i}: {noise:.5} ({noise_raw:.2})"));
                }
            }
        });
    }
}
