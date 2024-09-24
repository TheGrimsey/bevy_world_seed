use std::num::NonZeroU8;

use bevy::{
    app::{App, Startup},
    asset::{AssetMode, AssetPlugin, AssetServer},
    color::Color,
    core::Name,
    diagnostic::FrameTimeDiagnosticsPlugin,
    math::Vec3,
    pbr::{DirectionalLight, DirectionalLightBundle},
    prelude::{
        default, Commands, PluginGroup, Res, ResMut, Transform, TransformBundle, VisibilityBundle,
    },
    DefaultPlugins,
};
use bevy_editor_pls::EditorPlugin;
use bevy_terrain_test::{
    material::{
        GlobalTexturingRules, TerrainTexturingSettings, TexturingRule, TexturingRuleEvaluator,
    },
    noise::{TerrainNoiseDetailLayer, TerrainNoiseSettings, TerrainNoiseSplineLayer},
    terrain::Terrain,
    TerrainPlugin, TerrainSettings,
};
use splines::{Interpolation, Key, Spline};

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

    // Continental-ness.
    let spline = Spline::from_iter(
        [
            Key::new(0.0, 45.0, Interpolation::Cosine),
            Key::new(0.1, 0.0, Interpolation::Cosine),
            Key::new(0.2, 0.0, Interpolation::Cosine),
            Key::new(0.35, 25.0, Interpolation::Cosine),
            Key::new(0.45, 40.0, Interpolation::Cosine),
            Key::new(0.50, 45.0, Interpolation::Cosine),
            Key::new(0.70, 54.0, Interpolation::Cosine),
            Key::new(0.80, 60.0, Interpolation::Cosine),
            Key::new(1.0, 60.0, Interpolation::default()),
        ]
        .into_iter(),
    );

    // Peaks & Valleys
    let peaks_n_valleys = Spline::from_iter(
        [
            Key::new(0.0, -40.0, Interpolation::Cosine),
            Key::new(0.15, -20.0, Interpolation::Cosine),
            Key::new(0.3, 00.0, Interpolation::Cosine),
            Key::new(0.60, 40.0, Interpolation::Cosine),
            Key::new(0.90, 60.0, Interpolation::Cosine),
            Key::new(1.0, 100.0, Interpolation::default()),
        ]
        .into_iter(),
    );

    app.add_plugins(TerrainPlugin {
        noise_settings: Some(TerrainNoiseSettings {
            splines: vec![
                TerrainNoiseSplineLayer {
                    amplitude_spline: spline,
                    frequency: 0.001,
                    seed: 5,
                },
                TerrainNoiseSplineLayer {
                    amplitude_spline: peaks_n_valleys,
                    frequency: 0.001,
                    seed: 6,
                },
            ],
            layers: vec![
                TerrainNoiseDetailLayer {
                    amplitude: 4.0,
                    frequency: 0.01,
                    seed: 3,
                },
                TerrainNoiseDetailLayer {
                    amplitude: 2.0,
                    frequency: 0.02,
                    seed: 1,
                },
                TerrainNoiseDetailLayer {
                    amplitude: 1.0,
                    frequency: 0.04,
                    seed: 2,
                },
            ],
        }),
        terrain_settings: TerrainSettings {
            tile_size_power: NonZeroU8::new(7).unwrap(),
            edge_points: 129,
            max_tile_updates_per_frame: NonZeroU8::new(16).unwrap(),
            max_spline_simplification_distance: 3.0,
        },
        texturing_settings: Some(TerrainTexturingSettings {
            texture_resolution_power: NonZeroU8::new(6).unwrap(),
            max_tile_updates_per_frame: NonZeroU8::new(4).unwrap(),
        }),
        debug_draw: true,
    });

    app.add_systems(Startup, spawn_terrain);
    app.add_systems(Startup, insert_texturing_rules);

    app.run();
}

/// Defines the rules for procedural texturing
fn insert_texturing_rules(
    mut texturing_rules: ResMut<GlobalTexturingRules>,
    asset_server: Res<AssetServer>,
) {
    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::AngleGreaterThan {
            angle_radians: 40.0_f32.to_radians(),
            falloff_radians: 2.5_f32.to_radians(),
        },
        texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
        normal_texture: None,
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
