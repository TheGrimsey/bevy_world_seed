use std::num::NonZeroU8;

use bevy::{
    app::{App, Startup},
    asset::{AssetServer, Assets},
    color::Color,
    core::Name,
    math::Vec3,
    pbr::{DirectionalLight, StandardMaterial},
    prelude::{
        default, BuildChildren, Commands, CubicCardinalSpline, CubicCurve, CubicGenerator, Cuboid, Mesh, Mesh3d, Res, ResMut, Transform,
    },
    DefaultPlugins,
};
use bevy_editor_pls::EditorPlugin;
use bevy_hierarchy::ChildBuild;
use bevy_pbr::MeshMaterial3d;
use bevy_world_seed::{
    easing::EasingFunction,
    material::{
        GlobalTexturingRules, TerrainTexturingSettings, TextureModifierFalloffProperty,
        TextureModifierOperation, TexturingRule, TexturingRuleEvaluator,
    },
    modifiers::{
        ModifierFalloffNoiseProperty, ModifierFalloffProperty, ModifierHeightOperation,
        ModifierHeightProperties, ModifierHoleOperation, ModifierNoiseOperation, ModifierPriority,
        ModifierStrengthLimitProperty, ShapeModifier, ShapeModifierBundle,
        TerrainSplineBundle, TerrainSplineProperties, TerrainSplineShape,
    },
    noise::{LayerNoiseSettings, LayerOperation, NoiseGroup, NoiseLayer, NoiseScaling, StrengthCombinator, TerrainNoiseSettings},
    snap_to_terrain::SnapToTerrain,
    terrain::Terrain,
    TerrainPlugin, TerrainSettings,
};

fn main() {
    let mut app = App::new();

    app.add_plugins((DefaultPlugins, EditorPlugin::default()));

    app.add_plugins(TerrainPlugin {
        noise_settings: Some(TerrainNoiseSettings {
            noise_groups: vec![
                NoiseGroup {
                    layers: vec![NoiseLayer {
                        operation: LayerOperation::Noise {
                            noise: LayerNoiseSettings {
                                amplitude: 6.0,
                                frequency: 1.0 / 30.0,
                                seed: 1,
                                domain_warp: vec![],
                                scaling: NoiseScaling::Normalized
                            }
                        },
                        ..default()
                    }],
                    ..default()
                }
            ],
            ..default()
        }),
        terrain_settings: TerrainSettings {
            tile_size_power: NonZeroU8::new(5).unwrap(),
            edge_points: 65,
            max_tile_updates_per_frame: NonZeroU8::new(2).unwrap(),
            max_spline_simplification_distance_squared: 6.0,
        },
        texturing_settings: Some(TerrainTexturingSettings {
            texture_resolution_power: NonZeroU8::new(7).unwrap(),
            max_tile_updates_per_frame: NonZeroU8::new(2).unwrap(),
        }),
        debug_draw: true,
    });

    app.add_systems(Startup, spawn_terrain);
    app.add_systems(Startup, insert_texturing_rules);

    app.run();
}

fn insert_texturing_rules(
    mut texturing_rules: ResMut<GlobalTexturingRules>,
    asset_server: Res<AssetServer>,
) {
    texturing_rules.rules.push(TexturingRule {
        evaluators: vec![TexturingRuleEvaluator::AngleGreaterThan {
            angle_radians: 30.0_f32.to_radians(),
            falloff_radians: 2.5_f32.to_radians(),
        }],
        evaulator_combinator: StrengthCombinator::Min,
        texture: asset_server.load("textures/cracked_concrete_diff_1k.dds"),
        normal_texture: Some(asset_server.load("textures/cracked_concrete_nor_gl_1k.dds")),
        units_per_texture: 4.0,
    });

    texturing_rules.rules.push(TexturingRule {
        evaluators: vec![TexturingRuleEvaluator::AngleLessThan {
            angle_radians: 30.0_f32.to_radians(),
            falloff_radians: 2.5_f32.to_radians(),
        }],
        evaulator_combinator: StrengthCombinator::Min,
        texture: asset_server.load("textures/brown_mud_leaves.dds"),
        normal_texture: Some(asset_server.load("textures/brown_mud_leaves_01_nor_gl_2k.dds")),
        units_per_texture: 4.0,
    });
}

fn spawn_terrain(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    terrain_settings: Res<TerrainSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        DirectionalLight {
            color: Color::WHITE,
            illuminance: 1000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_translation(Vec3::new(32.0, 25.0, 16.0))
                .looking_at(Vec3::ZERO, Vec3::Y)
                .with_translation(Vec3::ZERO),
        Name::new("Directional Light"),
    ));

    // Create a simple curve from a few points.
    let curve: CubicCurve<Vec3> = CubicCardinalSpline::new_catmull_rom(vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(8.0, 0.0, 8.0),
        Vec3::new(10.0, 0.0, 20.0),
        Vec3::new(30.0, 1.0, 30.0),
        Vec3::new(40.0, 1.0, 48.0),
    ])
    .to_curve().expect("Couldn't create spline curve.");

    // Spawn a spline modifier that also applies a texture.
    commands.spawn((
        TerrainSplineBundle {
            spline: TerrainSplineShape { curve },
            properties: TerrainSplineProperties { half_width: 3.0 },
            priority: ModifierPriority(1),
            transform: Transform::default(),
        },
        // Adding `TextureModifierOperation` so this spline applies a texture.
        TextureModifierOperation {
            texture: asset_server.load("textures/cracked_concrete_diff_1k.dds"),
            normal_texture: Some(asset_server.load("textures/cracked_concrete_nor_gl_1k.dds")),
            max_strength: 0.95,
            units_per_texture: 4.0,
        },
        ModifierFalloffProperty {
            falloff: 8.8,
            easing_function: EasingFunction::BackInOut,
        },
        TextureModifierFalloffProperty {
            falloff: 1.0,
            easing_function: EasingFunction::CubicIn,
        },
        Name::new("Spline"),
    ));

    // Spawn a circle modifier that also applies a texture.
    commands.spawn((
        ShapeModifierBundle {
            shape: ShapeModifier::Circle { radius: 4.0 },
            properties: ModifierHeightProperties {
                allow_lowering: true,
                allow_raising: true,
            },
            priority: ModifierPriority(1),
            transform: Transform::from_translation(
                Vec3::new(10.0, 5.0, 48.0),
            ),
        },
        ModifierStrengthLimitProperty(0.9),
        ModifierFalloffProperty {
            falloff: 4.0,
            easing_function: EasingFunction::CubicInOut,
        },
        ModifierFalloffNoiseProperty {
            noise: LayerNoiseSettings {
                amplitude: 2.0,
                frequency: 1.0,
                seed: 5,
                domain_warp: vec![],
                scaling: NoiseScaling::Normalized
            },
        },
        ModifierHeightOperation::Set,
        TextureModifierOperation {
            texture: asset_server.load("textures/cracked_concrete_diff_1k.dds"),
            normal_texture: Some(asset_server.load("textures/cracked_concrete_nor_gl_1k.dds")),
            max_strength: 0.95,
            units_per_texture: 4.0,
        },
        Name::new("Modifier (Circle)"),
    ));

    // Spawn a circle hole punching modifier.
    commands.spawn((
        ShapeModifierBundle {
            shape: ShapeModifier::Circle { radius: 2.9 },
            properties: ModifierHeightProperties {
                allow_lowering: true,
                allow_raising: true,
            },
            priority: ModifierPriority(1),
            transform: Transform::from_translation(
                Vec3::new(40.0, 2.0, 6.0),
            ),
        },
        ModifierHoleOperation { invert: false },
        Name::new("Modifier (Circle Hole)"),
    ));

    // Spawn a rectangle modifier that also applies a texture.
    commands.spawn((
        ShapeModifierBundle {
            shape: ShapeModifier::Rectangle { x: 2.5, z: 5.0 },
            properties: ModifierHeightProperties {
                allow_lowering: true,
                allow_raising: true,
            },
            priority: ModifierPriority(2),
            transform: Transform::from_translation(
                Vec3::new(32.0, 5.0, 50.0),
            ),
        },
        ModifierFalloffProperty {
            falloff: 4.0,
            easing_function: EasingFunction::CubicInOut,
        },
        ModifierHeightOperation::Set,
        ModifierNoiseOperation {
            noise: LayerNoiseSettings {
                amplitude: 2.0,
                frequency: 0.1,
                seed: 5,
                domain_warp: vec![],
                scaling: NoiseScaling::Normalized
            },
        },
        TextureModifierOperation {
            texture: asset_server.load("textures/brown_mud_leaves.dds"),
            normal_texture: Some(asset_server.load("textures/brown_mud_leaves_01_nor_gl_2k.dds")),
            max_strength: 0.95,
            units_per_texture: 4.0,
        },
        Name::new("Modifier (Rectangle)"),
    ));

    let mesh = meshes.add(Cuboid::from_size(Vec3::ONE));
    let material = materials.add(Color::srgb(0.3, 0.3, 0.9));
    // Spawn a cube that snaps to terrain height.
    commands
        .spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(Vec3::new(16.0, 0.0, 16.0)),
            SnapToTerrain { y_offset: 0.5, align_to_terrain_normal: true },
            Name::new("Snap To Terrain"),
        ))
        .with_children(|child_builder| {
            child_builder.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(Vec3::new(4.0, 0.0, 0.0)),
                SnapToTerrain { y_offset: 0.5, align_to_terrain_normal: false },
                Name::new("Snap To Terrain (Child 1)"),
            ));

            /*child_builder.spawn((
                PbrBundle {
                    mesh: mesh.clone(),
                    material: material.clone(),
                    transform: Transform::from_translation(Vec3::new(-3.0, 0.0, 3.0)),
                    ..default()
                },
                SnapToTerrain { y_offset: 0.5 },
                Name::new("Snap To Terrain (Child 2)"),
            ));

            child_builder.spawn((
                PbrBundle {
                    mesh: mesh.clone(),
                    material: material.clone(),
                    transform: Transform::from_translation(Vec3::new(0.5, 0.0, -3.0)),
                    ..default()
                },
                SnapToTerrain { y_offset: 0.5 },
                Name::new("Snap To Terrain (Child 2)"),
            ));*/
        });

    // Spawn terrain tiles.
    commands.spawn((
        Terrain::default(),
        Transform::default(),
        Name::new("Terrain"),
    ));

    commands.spawn((
        Terrain::default(),
        Transform::from_translation(Vec3::new(
            terrain_settings.tile_size(),
            0.0,
            0.0,
        )),
        Name::new("Terrain (1, 0))"),
    ));

    commands.spawn((
        Terrain::default(),
        Transform::from_translation(Vec3::new(
            0.0,
            0.0,
            terrain_settings.tile_size(),
        )),
        Name::new("Terrain (0, 1)"),
    ));

    commands.spawn((
        Terrain::default(),
        Transform::from_translation(Vec3::new(
            terrain_settings.tile_size(),
            0.0,
            terrain_settings.tile_size(),
        )),
        Name::new("Terrain (1, 1)"),
    ));
}
