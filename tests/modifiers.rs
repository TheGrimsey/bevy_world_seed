use std::num::NonZeroU8;

use bevy::{
    app::{App, First, Last, Startup},
    math::{IVec2, Vec2, Vec3},
    prelude::{Commands, Query, Res, Transform, TransformPlugin},
    MinimalPlugins,
};
use bevy_world_seed::{
    modifiers::{
        ModifierHeightOperation, ModifierHeightProperties, ModifierPriority,
        ShapeModifier, ShapeModifierBundle,
    },
    terrain::{Terrain, TileToTerrain},
    utils::get_height_at_position_in_tile,
    Heights, TerrainPlugin, TerrainSettings,
};

fn setup_app(app: &mut App) {
    app.add_plugins((MinimalPlugins, TransformPlugin));

    let terrain_settings = TerrainSettings {
        tile_size_power: NonZeroU8::new(5).unwrap(),
        edge_points: 65,
        max_tile_updates_per_frame: NonZeroU8::MAX,
        max_spline_simplification_distance_squared: 6.0,
    };

    #[cfg(feature = "rendering")]
    app.add_plugins(TerrainPlugin {
        noise_settings: None,
        terrain_settings,
        texturing_settings: None,
        debug_draw: false,
    });

    #[cfg(not(feature = "rendering"))]
    app.add_plugins(TerrainPlugin {
        noise_settings: None,
        terrain_settings,
    });

    app.add_systems(First, spawn_terrain_tiles);
}

fn spawn_terrain_tiles(mut commands: Commands, terrain_settings: Res<TerrainSettings>) {
    let terrain_range = 1;

    for x in -terrain_range..terrain_range {
        for z in -terrain_range..terrain_range {
            commands.spawn((
                Terrain::default(),
                Transform::from_translation(Vec3::new(
                    x as f32 * terrain_settings.tile_size(),
                    0.0,
                    z as f32 * terrain_settings.tile_size(),
                )),
            ));
        }
    }
}

#[test]
fn test_circle_modifier_applies() {
    let mut app = App::new();

    setup_app(&mut app);

    let circle_height = 5.0;

    app.add_systems(
        Startup,
        move |mut commands: Commands, terrain_settings: Res<TerrainSettings>| {
            let tile_size = terrain_settings.tile_size();

            // Does not have falloff.
            commands.spawn((
                ShapeModifierBundle {
                    shape: ShapeModifier::Circle {
                        // Size of the tile.
                        radius: tile_size / 2.0,
                    },
                    properties: ModifierHeightProperties {
                        allow_lowering: true,
                        allow_raising: true,
                    },
                    priority: ModifierPriority(1),
                    transform: Transform::from_translation(
                        Vec3::new(tile_size / 2.0, circle_height, tile_size / 2.0),
                    ),
                },
                ModifierHeightOperation::Set,
            ));
        },
    );

    app.add_systems(
        Last,
        move |terrain_settings: Res<TerrainSettings>,
              tile_to_terrain: Res<TileToTerrain>,
              tiles_query: Query<&Heights>| {
            let tile = tile_to_terrain
                .get(&IVec2::ZERO)
                .expect("Missing terrain tile in tile to terrain")
                .first()
                .unwrap();

            let heights = tiles_query.get(*tile).expect("Couldn't get tile entity.");

            assert_eq!(
                circle_height,
                get_height_at_position_in_tile(
                    Vec2::splat(terrain_settings.tile_size() / 2.0),
                    heights,
                    &terrain_settings
                ),
                "Center of circle modifier should set the height to {circle_height}."
            );

            assert_eq!(
                0.0,
                get_height_at_position_in_tile(Vec2::splat(0.0), heights, &terrain_settings),
                "Circle modifier shouldn't affect corner of tile."
            );
        },
    );

    app.update();
}

#[test]
fn test_rectangle_modifier_applies() {
    let mut app = App::new();

    setup_app(&mut app);

    let modifier_height = 5.0;

    app.add_systems(
        Startup,
        move |mut commands: Commands, terrain_settings: Res<TerrainSettings>| {
            let tile_size = terrain_settings.tile_size();

            // Does not have falloff.
            commands.spawn((
                ShapeModifierBundle {
                    shape: ShapeModifier::Rectangle {
                        // Size of the tile.
                        x: tile_size / 2.0,
                        z: tile_size / 2.0,
                    },
                    properties: ModifierHeightProperties {
                        allow_lowering: true,
                        allow_raising: true,
                    },
                    priority: ModifierPriority(1),
                    transform: Transform::from_translation(
                        Vec3::new(tile_size / 2.0, modifier_height, tile_size / 2.0),
                    ),
                },
                ModifierHeightOperation::Set,
            ));
        },
    );

    app.add_systems(
        Last,
        move |terrain_settings: Res<TerrainSettings>,
              tile_to_terrain: Res<TileToTerrain>,
              tiles_query: Query<&Heights>| {
            let tile = tile_to_terrain
                .get(&IVec2::ZERO)
                .expect("Missing terrain tile in tile to terrain")
                .first()
                .unwrap();

            let heights = tiles_query.get(*tile).expect("Couldn't get tile entity.");

            assert_eq!(
                modifier_height,
                get_height_at_position_in_tile(
                    Vec2::splat(terrain_settings.tile_size() / 2.0),
                    heights,
                    &terrain_settings
                ),
                "Center of rectangle should set the height to {modifier_height}."
            );

            assert_eq!(
                modifier_height,
                get_height_at_position_in_tile(Vec2::splat(0.0), heights, &terrain_settings),
                "Rectangle modifier should affect corner of tile."
            );
        },
    );

    app.update();
}
