use std::num::NonZeroU8;

use bevy::{
    app::{App, Startup, Update},
    asset::{AssetMode, AssetPlugin},
    color::Color,
    core::Name,
    diagnostic::FrameTimeDiagnosticsPlugin,
    math::Vec3,
    pbr::{DirectionalLight, DirectionalLightBundle},
    prelude::{
        default, on_event, BuildChildren, Commands, Component, Entity, EventReader,
        IntoSystemConfigs, PluginGroup, Query, Res, Transform, TransformBundle, VisibilityBundle,
    },
    DefaultPlugins,
};
use bevy_editor_pls::EditorPlugin;
use bevy_rapier3d::{
    parry::shape::{HeightField, HeightFieldCellStatus, SharedShape},
    plugin::{NoUserData, RapierPhysicsPlugin},
    prelude::Collider,
    render::RapierDebugRenderPlugin,
};
use bevy_world_seed::{
    material::TerrainTexturingSettings,
    modifiers::{
        ModifierHeightProperties, ModifierHoleOperation, ModifierPriority, ModifierTileAabb,
        ShapeModifier, ShapeModifierBundle,
    },
    noise::{FilterCombinator, FilteredTerrainNoiseDetailLayer, NoiseScaling, TerrainNoiseDetailLayer, TerrainNoiseSettings},
    terrain::{Holes, Terrain, TileToTerrain},
    utils::index_to_x_z,
    Heights, TerrainPlugin, TerrainSettings, TileHeightsRebuilt,
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
        RapierPhysicsPlugin::<NoUserData>::default(),
        RapierDebugRenderPlugin::default(),
    ));

    app.add_plugins(TerrainPlugin {
        noise_settings: Some(TerrainNoiseSettings {
            data: vec![],
            splines: vec![],
            layers: vec![FilteredTerrainNoiseDetailLayer {
                layer: TerrainNoiseDetailLayer {
                    amplitude: 4.0,
                    frequency: 1.0 / 30.0,
                    seed: 2,
                    domain_warp: vec![],
                    scaling: NoiseScaling::Normalized
                },
                filter: vec![],
                filter_combinator: FilterCombinator::Max
            }],
        }),
        terrain_settings: TerrainSettings {
            tile_size_power: NonZeroU8::new(6).unwrap(),
            edge_points: 65,
            max_tile_updates_per_frame: NonZeroU8::new(16).unwrap(),
            max_spline_simplification_distance_squared: 6.0,
        },
        texturing_settings: Some(TerrainTexturingSettings {
            texture_resolution_power: NonZeroU8::new(1).unwrap(),
            max_tile_updates_per_frame: NonZeroU8::new(4).unwrap(),
        }),
        debug_draw: true,
    });

    app.add_systems(Startup, spawn_terrain);

    app.add_systems(
        Update,
        update_heightfield.run_if(on_event::<TileHeightsRebuilt>()),
    );

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

        query
            .iter_many(tiles.iter())
            .for_each(|(entity, heights, holes, height_field)| {
                let collider_entity = if let Some(HeightfieldEntity(entity)) = height_field {
                    *entity
                } else {
                    let collider_entity = commands
                        .spawn(TransformBundle::from_transform(
                            Transform::from_translation(Vec3::new(
                                tile_size / 2.0,
                                0.0,
                                tile_size / 2.0,
                            )),
                        ))
                        .set_parent(entity)
                        .id();

                    commands
                        .entity(entity)
                        .insert(HeightfieldEntity(collider_entity));

                    collider_entity
                };

                // Heightfield expects a row to be progressing on Z. But the mesh is laid out with row on X.
                // So we need to rotate the heights.
                let mut rotated_heights = vec![0.0; heights.len()];
                for (i, height) in heights.iter().enumerate() {
                    let (x, z) = index_to_x_z(i, terrain_settings.edge_points as usize);

                    let new_i = x * terrain_settings.edge_points as usize + z;
                    rotated_heights[new_i] = *height;
                }

                let heights = bevy_rapier3d::na::DMatrix::from_vec(
                    terrain_settings.edge_points.into(),
                    terrain_settings.edge_points.into(),
                    rotated_heights,
                );

                let mut collider =
                    HeightField::new(heights, Vec3::new(tile_size, 1.0, tile_size).into());

                for hole in holes.iter_holes(terrain_settings.edge_points) {
                    // Hole z & x are reversed in heightfield.
                    let cell_status =
                        &mut collider.cells_statuses_mut()[(hole.z as usize, hole.x as usize)];

                    // We aren't overwriting the cell status completely because cells may be returned multiple times.
                    if hole.left_triangle_removed {
                        *cell_status |= HeightFieldCellStatus::LEFT_TRIANGLE_REMOVED;
                    }
                    if hole.right_triangle_removed {
                        *cell_status |= HeightFieldCellStatus::RIGHT_TRIANGLE_REMOVED;
                    }
                }

                commands
                    .entity(collider_entity)
                    .insert(Collider::from(SharedShape::new(collider)));
            });
    }
}

fn spawn_terrain(mut commands: Commands) {
    commands.spawn((
        ShapeModifierBundle {
            aabb: ModifierTileAabb::default(),
            shape: ShapeModifier::Circle { radius: 2.9 },
            properties: ModifierHeightProperties::default(),
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(
                Vec3::new(40.0, 2.0, 6.0),
            )),
        },
        ModifierHoleOperation::default(),
        Name::new("Modifier (Circle Hole)"),
    ));

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

    commands.spawn((
        Terrain::default(),
        TransformBundle::default(),
        VisibilityBundle::default(),
        Name::new("Terrain"),
    ));
}
