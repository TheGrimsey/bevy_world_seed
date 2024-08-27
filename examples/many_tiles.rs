use std::num::{NonZeroU32, NonZeroU8};

use bevy::{app::{App, Startup}, asset::{AssetMode, AssetPlugin, AssetServer}, color::Color, core::Name, diagnostic::FrameTimeDiagnosticsPlugin, math::Vec3, pbr::{DirectionalLight, DirectionalLightBundle}, prelude::{default, PluginGroup, Commands, Res, ResMut, Transform, TransformBundle, VisibilityBundle}, DefaultPlugins};
use bevy_editor_pls::EditorPlugin;
use bevy_terrain_test::{material::{GlobalTexturingRules, TerrainTexturingSettings, TexturingRule, TexturingRuleEvaluator}, terrain::TerrainCoordinate, Heights, TerrainNoiseLayer, TerrainNoiseLayers, TerrainPlugin, TerrainSettings};


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
                TerrainNoiseLayer { height_scale: 2.0, planar_scale: 1.0 / 40.0, seed: 1 },
                TerrainNoiseLayer { height_scale: 8.0, planar_scale: 1.0 / 60.0, seed: 3 }
            ],
        }),
        terrain_settings: TerrainSettings {
            tile_size_power: NonZeroU8::new(5).unwrap(),
            edge_points: 65,
            max_tile_updates_per_frame: NonZeroU8::new(4).unwrap(),
            max_spline_simplification_distance: 3.0
        },
        texturing_settings: TerrainTexturingSettings {
            texture_resolution_power: 6,
            max_tile_updates_per_frame: NonZeroU32::new(4).unwrap(),
        },
        debug_draw: true
    });

    app.add_systems(Startup, spawn_terrain);
    app.add_systems(Startup, insert_rules);

    app.run();
}

fn insert_rules(mut texturing_rules: ResMut<GlobalTexturingRules>, asset_server: Res<AssetServer>) {
    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::AngleGreaterThan {
            angle_radians: 40.0_f32.to_radians()
        },
        texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
        tiling_factor: 2.0
    });
    
    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::AngleLessThan {
            angle_radians: 40.0_f32.to_radians()
        },
        texture: asset_server.load("textures/brown_mud_leaves.dds"),
        tiling_factor: 1.0
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

    let size = terrain_settings.edge_points as usize * terrain_settings.edge_points as usize;
    let flat_heights = vec![0.0; size].into_boxed_slice();

    let terrain_range = 20;

    for x in -terrain_range..terrain_range {
        for z in -terrain_range..terrain_range {
            commands.spawn((
                TerrainCoordinate::default(),
                Heights(flat_heights.clone()),
                TransformBundle::from_transform(Transform::from_translation(Vec3::new(x as f32 * terrain_settings.tile_size(), 0.0,  z as f32 * terrain_settings.tile_size()))),
                VisibilityBundle::default(),
                Name::new(format!("Terrain ({x},{z})"))
            ));
        }
    }
}
