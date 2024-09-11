use std::num::{NonZeroU32, NonZeroU8};

use bevy::{app::{App, Startup}, asset::{AssetMode, AssetPlugin, AssetServer}, color::Color, core::Name, diagnostic::FrameTimeDiagnosticsPlugin, math::Vec3, pbr::{DirectionalLight, DirectionalLightBundle}, prelude::{default, PluginGroup, Commands, Res, ResMut, Transform, TransformBundle, VisibilityBundle}, DefaultPlugins};
use bevy_editor_pls::EditorPlugin;
use bevy_terrain_test::{material::{GlobalTexturingRules, TerrainTexturingSettings, TexturingRule, TexturingRuleEvaluator}, terrain::Terrain, TerrainNoiseLayer, TerrainNoiseLayers, TerrainPlugin, TerrainSettings};


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
        noise_settings: Some(TerrainNoiseLayers {
            layers: vec![
                TerrainNoiseLayer { amplitude: 8.0, frequency: 0.01, seed: 3 },
                TerrainNoiseLayer { amplitude: 4.0, frequency: 0.02, seed: 1 },
                TerrainNoiseLayer { amplitude: 2.0, frequency: 0.04, seed: 2 },
            ],
        }),
        terrain_settings: TerrainSettings {
            tile_size_power: NonZeroU8::new(6).unwrap(),
            edge_points: 65,
            max_tile_updates_per_frame: NonZeroU8::new(16).unwrap(),
            max_spline_simplification_distance: 3.0
        },
        texturing_settings: TerrainTexturingSettings {
            texture_resolution_power: 6,
            max_tile_updates_per_frame: NonZeroU32::new(4).unwrap(),
        },
        debug_draw: true
    });

    app.add_systems(Startup, spawn_terrain);
    app.add_systems(Startup, insert_texturing_rules);

    app.run();
}

/// Defines the rules for procedural texturing
fn insert_texturing_rules(mut texturing_rules: ResMut<GlobalTexturingRules>, asset_server: Res<AssetServer>) {
    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::AngleGreaterThan {
            angle_radians: 40.0_f32.to_radians(),
            falloff_radians: 2.5_f32.to_radians()
        },
        texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
        units_per_texture: 4.0
    });
    
    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::AngleLessThan {
            angle_radians: 40.0_f32.to_radians(),
            falloff_radians: 2.5_f32.to_radians()
        },
        texture: asset_server.load("textures/brown_mud_leaves.dds"),
        units_per_texture: 4.0
    });
}

fn spawn_terrain(
    mut commands: Commands,
    terrain_settings: Res<TerrainSettings>,
) {
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            illuminance: 1000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_translation(Vec3::new(32.0, 25.0, 16.0)).looking_at(Vec3::ZERO, Vec3::Y).with_translation(Vec3::ZERO),
        ..default()
    });

    let terrain_range = 15;

    for x in -terrain_range..terrain_range {
        for z in -terrain_range..terrain_range {
            commands.spawn((
                Terrain::default(),
                TransformBundle::from_transform(Transform::from_translation(Vec3::new(x as f32 * terrain_settings.tile_size(), 0.0,  z as f32 * terrain_settings.tile_size()))),
                VisibilityBundle::default(),
                Name::new(format!("Terrain ({x},{z}"))
            ));
        }
    }
}
