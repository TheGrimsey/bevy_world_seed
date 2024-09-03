use std::num::{NonZeroU32, NonZeroU8};

use bevy::{app::{App, Startup}, asset::{AssetMode, AssetPlugin}, color::Color, core::Name, diagnostic::FrameTimeDiagnosticsPlugin, math::Vec3, pbr::{DirectionalLight, DirectionalLightBundle}, prelude::{default, BuildChildren, Commands, Component, Entity, EventReader, PluginGroup, Query, Res, Transform, TransformBundle, VisibilityBundle}, DefaultPlugins};
use bevy_editor_pls::EditorPlugin;
use bevy_rapier3d::{parry::shape::{HeightField, HeightFieldCellStatus, SharedShape}, prelude::Collider};
use bevy_terrain_test::{material::TerrainTexturingSettings, terrain::{Holes, Terrain, TileToTerrain}, Heights, TerrainNoiseLayers, TerrainPlugin, TerrainSettings, TileHeightsRebuilt};


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
            ],
        }),
        terrain_settings: TerrainSettings {
            tile_size_power: NonZeroU8::new(7).unwrap(),
            edge_points: 129,
            max_tile_updates_per_frame: NonZeroU8::new(16).unwrap(),
            max_spline_simplification_distance: 3.0
        },
        texturing_settings: TerrainTexturingSettings {
            texture_resolution_power: 1,
            max_tile_updates_per_frame: NonZeroU32::new(4).unwrap(),
        },
        debug_draw: true
    });

    app.add_systems(Startup, spawn_terrain);

    app.run();
}

/// Holds the child entity which has the Heightfield on it
/// 
/// Need a separate entity because heightfields are anchored at the center. 
#[derive(Component)]
struct HeightfieldEntity(Entity);

fn update_heightfield(
    mut commands: Commands,
    mut tile_rebuilt_events: EventReader<TileHeightsRebuilt>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_terrain: Res<TileToTerrain>,
    query: Query<(Entity, &Heights, &Holes, Option<&HeightfieldEntity>)>,
) {
    let tile_size = terrain_settings.tile_size();

    for TileHeightsRebuilt(tile) in tile_rebuilt_events.read() {
        let Some(tiles) = tile_to_terrain.get(tile) else {
            continue;
        };

        query.iter_many(tiles.iter()).for_each(|(entity, heights, holes, height_field)| {
            let collider_entity = if let Some(HeightfieldEntity(entity)) = height_field {
                *entity
            } else {
                let collider_entity = commands.spawn(TransformBundle::from_transform(Transform::from_translation(Vec3::new(tile_size / 2.0, 0.0, tile_size / 2.0)))).set_parent(entity).id();

                commands.entity(entity).insert(HeightfieldEntity(collider_entity));

                collider_entity
            };

            let heights = bevy_rapier3d::na::DMatrix::from_vec(terrain_settings.edge_points.into(), terrain_settings.edge_points.into(), heights.to_vec());

            let mut collider = HeightField::new(heights, Vec3::new(tile_size, 1.0, tile_size).into());

            for hole in holes.iter_holes(terrain_settings.edge_points) {
                let cell_status = &mut collider.cells_statuses_mut()[(hole.x, hole.z)];

                if hole.is_left {
                    cell_status.set(HeightFieldCellStatus::LEFT_TRIANGLE_REMOVED, true);
                } else {
                    cell_status.set(HeightFieldCellStatus::RIGHT_TRIANGLE_REMOVED, true);    
                }
            }

            commands.entity(collider_entity).insert(Collider::from(SharedShape::new(collider)));
        });
    }
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

    let terrain_range = 2;

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
