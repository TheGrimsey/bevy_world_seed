use std::num::NonZeroU32;

use bevy::{app::{App, Startup}, asset::AssetServer, color::Color, core::Name, math::Vec3, pbr::{DirectionalLight, DirectionalLightBundle}, prelude::{default, Commands, CubicCardinalSpline, CubicCurve, CubicGenerator, Res, Transform, TransformBundle, VisibilityBundle}, DefaultPlugins};
use bevy_editor_pls::EditorPlugin;
use bevy_terrain_test::{material::{TerrainTexturingSettings, TextureModifier}, modifiers::{ModifierOperation, ModifierPriority, Shape, ShapeModifier, ShapeModifierBundle, TerrainSpline, TerrainSplineBundle, TerrainSplineCached, TerrainSplineProperties, TerrainTileAabb}, terrain::TerrainCoordinate, Heights, TerrainNoiseLayer, TerrainNoiseLayers, TerrainPlugin, TerrainSettings};


fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins,
        EditorPlugin::default()
    ));

    app.add_plugins(TerrainPlugin {
        noise_settings: Some(TerrainNoiseLayers {
            layers: vec![
                TerrainNoiseLayer { height_scale: 3.0, planar_scale: 1.0 / 20.0, seed: 1 }
            ],
        }),
        terrain_settings: TerrainSettings {
            tile_size_power: 5,
            edge_length: 65,
            max_tile_updates_per_frame: NonZeroU32::new(2).unwrap(),
            max_spline_simplification_distance: 3.0
        },
        texturing_settings: TerrainTexturingSettings {
            texture_resolution_power: 6,
            max_tile_updates_per_frame: NonZeroU32::new(2).unwrap(),
        },
        debug_draw: true
    });

    app.add_systems(Startup, spawn_terrain);

    app.run();
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
        Vec3::new(32.0, 1.0, 32.0),
    ]).to_curve();

    commands.spawn((
        TerrainSplineBundle {
            tile_aabb: TerrainTileAabb::default(),
            spline: TerrainSpline {
                curve: spline
            },
            properties: TerrainSplineProperties {
                width: 8.0,
                falloff: 4.0
            },
            spline_cached: TerrainSplineCached::default(),
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::default()
        },
        TextureModifier {
            texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
            max_texture_strength: 0.95
        },
        Name::new("Spline")
    ));

    commands.spawn((
        ShapeModifierBundle {
            aabb: TerrainTileAabb::default(),
            modifier: ShapeModifier {
                shape: Shape::Circle {
                    radius: 4.0
                },
                falloff: 4.0,
                allow_lowering: true,
                allow_raising: true
            },
            operation: ModifierOperation::Set,
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(Vec3::new(10.0, 5.0, 48.0))),
        },
        TextureModifier {
            texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
            max_texture_strength: 0.95
        },
        Name::new("Modifier (Circle)")
    ));

    commands.spawn((
        ShapeModifierBundle {
            aabb: TerrainTileAabb::default(),
            modifier: ShapeModifier {
                shape: Shape::Rectangle {
                    x: 2.5,
                    z: 5.0
                },
                falloff: 12.0,
                allow_raising: true,
                allow_lowering: true
            },
            operation: ModifierOperation::Set,
            priority: ModifierPriority(2),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(Vec3::new(32.0, 5.0, 50.0))),
        },
        TextureModifier {
            texture: asset_server.load("textures/brown_mud_leaves.jpg"),
            max_texture_strength: 0.95
        },
        Name::new("Modifier (Rectangle)")
    ));

    let size = terrain_settings.edge_length as usize * terrain_settings.edge_length as usize;
    let flat_heights = vec![0.0; size].into_boxed_slice();

    commands.spawn((
        TerrainCoordinate::default(),
        Heights(flat_heights.clone()),
        TransformBundle::default(),
        VisibilityBundle::default(),
        Name::new("Terrain")
    ));

    commands.spawn((
        TerrainCoordinate::default(),
        Heights(flat_heights.clone()),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(terrain_settings.tile_size(), 0.0, 0.0))),
        VisibilityBundle::default(),
        Name::new("Terrain (1, 0))")
    ));
    
    commands.spawn((
        TerrainCoordinate::default(),
        Heights(flat_heights.clone()),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(0.0, 0.0, terrain_settings.tile_size()))),
        VisibilityBundle::default(),
        Name::new("Terrain (0, 1)")
    ));
    
    commands.spawn((
        TerrainCoordinate::default(),
        Heights(flat_heights),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(terrain_settings.tile_size(), 0.0, terrain_settings.tile_size()))),
        VisibilityBundle::default(),
        Name::new("Terrain (1, 1)")
    ));
}
