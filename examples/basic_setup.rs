use std::num::{NonZeroU32, NonZeroU8};

use bevy::{app::{App, Startup}, asset::AssetServer, color::Color, core::Name, math::Vec3, pbr::{DirectionalLight, DirectionalLightBundle}, prelude::{default, Commands, CubicCardinalSpline, CubicCurve, CubicGenerator, Res, ResMut, Transform, TransformBundle, VisibilityBundle}, DefaultPlugins};
use bevy_editor_pls::EditorPlugin;
use bevy_terrain_test::{material::{GlobalTexturingRules, TerrainTexturingSettings, TextureModifierFalloffProperty, TextureModifierOperation, TexturingRule, TexturingRuleEvaluator}, modifiers::{ModifierAabb, ModifierFalloffProperty, ModifierHeightOperation, ModifierHeightProperties, ModifierHoleOperation, ModifierPriority, ShapeModifier, ShapeModifierBundle, TerrainSplineBundle, TerrainSplineCached, TerrainSplineProperties, TerrainSplineShape}, terrain::Terrain,TerrainNoiseLayer, TerrainNoiseLayers, TerrainPlugin, TerrainSettings};

fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins,
        EditorPlugin::default()
    ));

    app.add_plugins(TerrainPlugin {
        noise_settings: Some(TerrainNoiseLayers {
            layers: vec![
                TerrainNoiseLayer { amplitude: 6.0, frequency: 1.0 / 30.0, seed: 1 }
            ],
        }),
        terrain_settings: TerrainSettings {
            tile_size_power: NonZeroU8::new(5).unwrap(),
            edge_points: 65,
            max_tile_updates_per_frame: NonZeroU8::new(2).unwrap(),
            max_spline_simplification_distance: 3.0
        },
        texturing_settings: TerrainTexturingSettings {
            texture_resolution_power: 6,
            max_tile_updates_per_frame: NonZeroU32::new(2).unwrap(),
        },
        debug_draw: true
    });

    app.add_systems(Startup, spawn_terrain);
    app.add_systems(Startup, insert_texturing_rules);

    app.run();
}

fn insert_texturing_rules(mut texturing_rules: ResMut<GlobalTexturingRules>, asset_server: Res<AssetServer>) {
    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::AngleGreaterThan {
            angle_radians: 30.0_f32.to_radians(),
            falloff_radians: 2.5_f32.to_radians()
        },
        texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
        units_per_texture: 4.0
    });
    
    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::AngleLessThan {
            angle_radians: 30.0_f32.to_radians(),
            falloff_radians: 2.5_f32.to_radians()
        },
        texture: asset_server.load("textures/brown_mud_leaves.dds"),
        units_per_texture: 4.0
    });
}

fn spawn_terrain(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
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

    let spline: CubicCurve<Vec3> = CubicCardinalSpline::new_catmull_rom(vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(8.0, 0.0, 8.0),
        Vec3::new(16.0, 0.0, 16.0),
        Vec3::new(20.0, 1.0, 20.0),
    ]).to_curve();

    commands.spawn((
        TerrainSplineBundle {
            tile_aabb: ModifierAabb::default(),
            spline: TerrainSplineShape {
                curve: spline
            },
            properties: TerrainSplineProperties {
                width: 4.0,
                falloff: 4.0
            },
            spline_cached: TerrainSplineCached::default(),
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::default()
        },
        TextureModifierOperation {
            texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
            max_strength: 0.95,
            units_per_texture: 4.0
        },
        TextureModifierFalloffProperty(1.0),
        Name::new("Spline"),
    ));

    commands.spawn((
        ShapeModifierBundle {
            aabb: ModifierAabb::default(),
            shape: ShapeModifier::Circle {
                radius: 4.0
            },
            properties: ModifierHeightProperties {
                allow_lowering: true,
                allow_raising: true,
            },
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(Vec3::new(10.0, 5.0, 48.0))),
        },
        ModifierFalloffProperty(4.0),
        ModifierHeightOperation::Set,
        TextureModifierOperation {
            texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
            max_strength: 0.95,
            units_per_texture: 4.0
        },
        Name::new("Modifier (Circle)")
    ));

    
    commands.spawn((
        ShapeModifierBundle {
            aabb: ModifierAabb::default(),
            shape: ShapeModifier::Circle {
                radius: 2.9
            },
            properties: ModifierHeightProperties {
                allow_lowering: true,
                allow_raising: true,
            },
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(Vec3::new(40.0, 2.0, 6.0))),
        },
        ModifierHoleOperation {
            invert: false
        },
        Name::new("Modifier (Circle Hole)")
    ));

    commands.spawn((
        ShapeModifierBundle {
            aabb: ModifierAabb::default(),
            shape: ShapeModifier::Rectangle {
                x: 2.5,
                z: 5.0
            },
            properties: ModifierHeightProperties {
                allow_lowering: true,
                allow_raising: true,
            },
            priority: ModifierPriority(2),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(Vec3::new(32.0, 5.0, 50.0))),
        },
        ModifierFalloffProperty(4.0),
        ModifierHeightOperation::Set,
        TextureModifierOperation {
            texture: asset_server.load("textures/brown_mud_leaves.dds"),
            max_strength: 0.95,
            units_per_texture: 4.0
        },
        Name::new("Modifier (Rectangle)")
    ));

    commands.spawn((
        Terrain::default(),
        TransformBundle::default(),
        VisibilityBundle::default(),
        Name::new("Terrain")
    ));

    commands.spawn((
        Terrain::default(),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(terrain_settings.tile_size(), 0.0, 0.0))),
        VisibilityBundle::default(),
        Name::new("Terrain (1, 0))")
    ));
    
    commands.spawn((
        Terrain::default(),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(0.0, 0.0, terrain_settings.tile_size()))),
        VisibilityBundle::default(),
        Name::new("Terrain (0, 1)")
    ));
    
    commands.spawn((
        Terrain::default(),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(terrain_settings.tile_size(), 0.0, terrain_settings.tile_size()))),
        VisibilityBundle::default(),
        Name::new("Terrain (1, 1)")
    ));
}
