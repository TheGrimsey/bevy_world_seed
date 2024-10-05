use std::num::NonZeroU8;

use bevy::{
    app::{App, Startup},
    asset::{AssetMode, AssetPlugin, AssetServer, Assets},
    color::Color,
    core::Name,
    diagnostic::FrameTimeDiagnosticsPlugin,
    math::{Vec3, Vec3Swizzles},
    pbr::{DirectionalLight, DirectionalLightBundle},
    prelude::{
        default, Commands, GlobalTransform, Mut, PluginGroup, Res, ResMut, Transform, TransformBundle, VisibilityBundle, With, World
    },
    DefaultPlugins,
};
use bevy_editor_pls::{default_windows::cameras::ActiveEditorCamera, editor_window::{EditorWindow, EditorWindowContext}, egui, AddEditorWindow, EditorPlugin};
use bevy_lookup_curve::{editor::{LookupCurveEditor, LookupCurveEguiEditor}, LookupCurve};
use bevy_world_seed::{
    material::{
        GlobalTexturingRules, TerrainTexturingSettings, TexturingRule, TexturingRuleEvaluator,
    }, noise::{NoiseCache, TerrainNoiseDetailLayer, TerrainNoiseSettings, TerrainNoiseSplineLayer}, terrain::{Terrain, TileToTerrain}, RebuildTile, TerrainPlugin, TerrainSettings
};

fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins.set(AssetPlugin {
            mode: AssetMode::Processed,
            ..default()
        }),
        EditorPlugin::default(),
        FrameTimeDiagnosticsPlugin
    ));

    app.add_plugins(TerrainPlugin {
        noise_settings: Some(TerrainNoiseSettings {
            splines: vec![
            ],
            layers: vec![
                TerrainNoiseDetailLayer {
                    amplitude: 16.0,
                    frequency: 0.005,
                    seed: 3,
                },
                TerrainNoiseDetailLayer {
                    amplitude: 8.0,
                    frequency: 0.01,
                    seed: 1,
                },
                TerrainNoiseDetailLayer {
                    amplitude: 4.0,
                    frequency: 0.02,
                    seed: 2,
                },
                TerrainNoiseDetailLayer {
                    amplitude: 2.0,
                    frequency: 0.04,
                    seed: 3,
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
    asset_server: Res<AssetServer>,
    mut commands: Commands
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
            ron_path: Some( "./assets/curves/peaks_and_valleys.curve.ron".to_string()),
            grid_step_y: 10.0,
            ..Default::default()
        },
        ..LookupCurveEditor::new(peaks_and_valleys)
    });

    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::AngleGreaterThan {
            angle_radians: 40.0_f32.to_radians(),
            falloff_radians: 2.5_f32.to_radians(),
        },
        texture: asset_server.load("textures/cracked_concrete_diff_1k.dds"),
        normal_texture: Some(asset_server.load("textures/cracked_concrete_nor_gl_1k.dds")),
        units_per_texture: 4.0,
    });

    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::AngleLessThan {
            angle_radians: 40.0_f32.to_radians(),
            falloff_radians: 2.5_f32.to_radians(),
        },
        texture: asset_server.load("textures/brown_mud_leaves.dds"),
        normal_texture: Some(asset_server.load("textures/brown_mud_leaves_01_nor_gl_2k.dds")),
        units_per_texture: 4.0,
    });
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

    let terrain_range = 5;

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
pub struct NoiseDebugWindowState {
}
impl EditorWindow for NoiseDebugWindow {
    type State = NoiseDebugWindowState;
    const NAME: &'static str = "Noise Debug";

    fn ui(world: &mut World, _cx: EditorWindowContext, ui: &mut egui::Ui) {
        world.resource_scope(|world, mut noise_cache: Mut<NoiseCache>| {
            if ui.button("Regenerate Terrain").clicked() {
                let mut tiles = world.resource::<TileToTerrain>().keys().cloned().collect::<Vec<_>>();

                tiles.sort_by(|a, b| a.x.cmp(&b.x).then(a.y.cmp(&b.y)));

                world.send_event_batch(tiles.into_iter().map(RebuildTile));
            }

            let mut query_state = world.query_filtered::<&GlobalTransform, With<ActiveEditorCamera>>();
            let lookup_curves = world.resource::<Assets<LookupCurve>>();
        
            if let Ok(transform) = query_state.get_single(world) {
                let noise_settings = world.resource::<TerrainNoiseSettings>();
                let translation = transform.translation();
    
                let height = noise_settings.sample_position(&mut noise_cache, translation.xz(), lookup_curves);
                ui.heading(format!("Height: {height}"));

                ui.heading("Spline Noise");
    
                for spline in noise_settings.splines.iter() {
                    let noise = spline.sample(translation.x, translation.z, noise_cache.get(spline.seed), lookup_curves);

                    if let Some(lookup_curve) = lookup_curves.get(&spline.amplitude_curve) {
                        ui.label(format!("- {}: {noise}", lookup_curve.name.as_ref().map_or("", |name| name.as_str())));
                    }
                }

                ui.heading("Detail Noise");
                
                for (i, detail_layer) in noise_settings.layers.iter().enumerate() {
                    let noise = detail_layer.sample(translation.x, translation.z, noise_cache.get(detail_layer.seed));

                    ui.label(format!("- Detail {i}: {noise}"));
                }
            }
        });

    }
}