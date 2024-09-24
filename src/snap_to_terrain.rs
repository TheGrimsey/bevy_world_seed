use bevy::{
    app::{App, Plugin, PostUpdate}, ecs::entity::EntityHashSet, log::info, math::{IVec2, Vec2, Vec3, Vec3Swizzles}, prelude::{
        any_with_component, on_event, Changed, Commands, Component, Entity, EventReader, GlobalTransform, IntoSystemConfigs, Mut, Or, Parent, Query, ReflectComponent, Res, ResMut, Resource, Transform, TransformSystem, With, Without
    }, reflect::Reflect, utils::HashMap
};

use crate::{
    terrain::TileToTerrain, utils::get_height_at_position_in_tile, Heights, TerrainSettings,
    TileHeightsRebuilt,
};

/// Contains the component & systems which allow entities to automatically snap to the height of a tile.
///
///

pub(super) struct TerrainSnapToTerrainPlugin;
impl Plugin for TerrainSnapToTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            (
                snap_on_added,
                update_snap_entity_tile,
                snap_on_tile_rebuilt.run_if(on_event::<TileHeightsRebuilt>()),
                update_snap_on_transform,
            )
                .chain()
                .run_if(any_with_component::<SnapToTerrain>)
                .after(TransformSystem::TransformPropagate),
        );

        app.init_resource::<TileToSnapEntities>();

        app.register_type::<SnapToTerrain>()
            .register_type::<SnapEntityTile>();
    }
}

/// Causes the entity to snap to the height of terrain.
///
/// Entities are snapped when their `GlobalTransform`, `SnapToTerrain` component, or the tile is updated.
/// 
/// The snapping is done on the global Y, meaning a child of a rotated entity will snap to the terrain below it in global space.
/// This changes the entity's `Transform::translation` but the system tries to keep track of the original position, keeping the offset from the parent (excepting the global Y axis).
/// This will be notable for entities with rotated parents.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SnapToTerrain {
    /// Offset to apply to the y of the entity.
    ///
    /// Modify this to push an entity further into the ground or above it.
    pub y_offset: f32,
}

#[derive(Component, Reflect)]
#[reflect(Component)]
struct SnapEntityTile {
    /// Tile the entity is contained in.
    tile: IVec2,
    /// Cached offset applied to the entity.
    /// Used to "recover" an entity's position after rotated.
    offset: Vec3
}

#[derive(Resource, Default)]
struct TileToSnapEntities(HashMap<IVec2, EntityHashSet>);

fn snap_on_added(
    mut commands: Commands,
    terrain_settings: Res<TerrainSettings>,
    mut query: Query<
        (Entity, &GlobalTransform),
        (With<SnapToTerrain>, Without<SnapEntityTile>),
    >,
    mut tile_to_snap_entities: ResMut<TileToSnapEntities>,
) {
    query
        .iter_mut()
        .for_each(|(entity, global_transform)| {
            let tile_coordinate = global_transform.translation().as_ivec3().xz()
                >> terrain_settings.tile_size_power.get();

            if let Some(entities) = tile_to_snap_entities.0.get_mut(&tile_coordinate) {
                entities.insert(entity);
            } else {
                tile_to_snap_entities
                    .0
                    .insert(tile_coordinate, EntityHashSet::from_iter([entity]));
            }

            commands
                .entity(entity)
                .insert(SnapEntityTile {
                    tile: tile_coordinate,
                    offset: Vec3::ZERO
                });
        });
}

fn snap_on_tile_rebuilt(
    terrain_settings: Res<TerrainSettings>,
    tile_to_snap_entities: Res<TileToSnapEntities>,
    mut snap_entity_query: Query<(&mut Transform, Option<&Parent>, &SnapToTerrain, &mut SnapEntityTile)>,
    parent_query: Query<&GlobalTransform>,
    mut rebuilt_heights_events: EventReader<TileHeightsRebuilt>,
    tile_to_terrain: Res<TileToTerrain>,
    tiles_query: Query<&Heights>,
) {
    for TileHeightsRebuilt(tile_coordinate) in rebuilt_heights_events.read() {
        let Some(snap_entities) = tile_to_snap_entities.0.get(tile_coordinate) else {
            continue;
        };
        if snap_entities.is_empty() {
            continue;
        }

        let Some(tile) = tile_to_terrain
            .get(tile_coordinate)
            .and_then(|tiles| tiles.first())
            .and_then(|entity| tiles_query.get(*entity).ok())
        else {
            continue;
        };
        let tile_translation =
            (*tile_coordinate << terrain_settings.tile_size_power.get()).as_vec2();

        let mut iter = snap_entity_query.iter_many_mut(snap_entities.iter());
        while let Some((transform, parent, snap_to_terrain, snap_entity_tile)) = iter.fetch_next() {
            update_snap_position(
                parent.and_then(|parent| parent_query.get(parent.get()).ok()),
                tile_translation,
                tile,
                &terrain_settings,
                transform,
                snap_entity_tile,
                snap_to_terrain,
            );
        }
    }
}

fn update_snap_position(
    parent_transform: Option<&GlobalTransform>,
    tile_translation: Vec2,
    tile: &Heights,
    terrain_settings: &TerrainSettings,
    mut transform: Mut<'_, Transform>,
    mut snap_entity_tile: Mut<'_, SnapEntityTile>,
    snap_to_terrain: &SnapToTerrain,
) {
    let true_transform = transform.with_translation(transform.translation - snap_entity_tile.offset);
    info!("True Translation: {:?}, Offset: {:?}", true_transform.translation, snap_entity_tile.offset);
    
    let true_global_transform = if let Some(parent_transform) = parent_transform {
        parent_transform.mul_transform(true_transform)
    } else {
        GlobalTransform::from(true_transform)
    };
    
    let relative_location = true_global_transform.translation().xz() - tile_translation;

    let height = get_height_at_position_in_tile(relative_location, tile, terrain_settings);

    let offset_height = height + snap_to_terrain.y_offset;

    let target_translation = true_global_transform.translation().with_y(offset_height);

    let localized_translation = if let Some(parent_transform) = parent_transform {
        parent_transform.affine().inverse().transform_point3(target_translation)
    } else {
        target_translation
    };
    snap_entity_tile.offset = localized_translation - true_transform.translation;

    // Don't update if we wouldn't make a difference.
    // Prevents us from ending up in a loop where GlobalTransform keeps being updated.
    if localized_translation.distance_squared(transform.translation) > 0.001 {
        transform.translation = localized_translation;
    }
}

fn update_snap_entity_tile(
    mut tile_to_snap_entities: ResMut<TileToSnapEntities>,
    terrain_settings: Res<TerrainSettings>,
    mut query: Query<(Entity, &GlobalTransform, &mut SnapEntityTile), Changed<GlobalTransform>>,
) {
    query
        .iter_mut()
        .for_each(|(entity, global_transform, mut snap_entity_tile)| {
            let tile_coordinate = global_transform.translation().as_ivec3().xz()
                >> terrain_settings.tile_size_power.get();

            let old_tile = snap_entity_tile.tile;
            if old_tile != tile_coordinate {
                snap_entity_tile.tile = tile_coordinate;

                if let Some(entities) = tile_to_snap_entities.0.get_mut(&old_tile) {
                    entities.remove(&entity);

                    if entities.is_empty() {
                        tile_to_snap_entities.0.remove(&old_tile);
                    }
                }

                if let Some(entities) = tile_to_snap_entities.0.get_mut(&tile_coordinate) {
                    entities.insert(entity);
                } else {
                    tile_to_snap_entities
                        .0
                        .insert(tile_coordinate, EntityHashSet::from_iter([entity]));
                }
            }
        });
}

fn update_snap_on_transform(
    terrain_settings: Res<TerrainSettings>,
    tile_to_terrain: Res<TileToTerrain>,
    mut transform_query: Query<
        (
            &mut Transform,
            Option<&Parent>,
            &SnapToTerrain,
            &mut SnapEntityTile,
        ),
        Or<(
            Changed<GlobalTransform>,
            Changed<SnapToTerrain>
        )>
    >,
    parent_query: Query<&GlobalTransform>,
    tiles_query: Query<&Heights>,
) {
    transform_query.par_iter_mut().for_each(
        |(transform, parent, snap_to_terrain, snap_entity_tile)| {
            let Some(tile) = tile_to_terrain
                .get(&snap_entity_tile.tile)
                .and_then(|tiles| tiles.first())
                .and_then(|entity| tiles_query.get(*entity).ok())
            else {
                return;
            };
            let tile_translation =
                (snap_entity_tile.tile << terrain_settings.tile_size_power.get()).as_vec2();

            update_snap_position(
                parent.and_then(|parent| parent_query.get(parent.get()).ok()),
                tile_translation,
                tile,
                &terrain_settings,
                transform,
                snap_entity_tile,
                snap_to_terrain,
            );
        },
    );
}
