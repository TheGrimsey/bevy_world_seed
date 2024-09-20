use bevy::{app::{App, Plugin, PostUpdate}, ecs::entity::EntityHashSet, math::{IVec2, Vec3Swizzles}, prelude::{any_with_component, on_event, Changed, Commands, Component, Entity, EventReader, GlobalTransform, IntoSystemConfigs, Query, Res, ResMut, Resource, Transform, TransformSystem, Without}, utils::HashMap};

use crate::{terrain::TileToTerrain, utils::get_height_at_position_in_tile, Heights, TerrainSettings, TileHeightsRebuilt};

/// Contains the component & systems which allow entities to automatically snap to the height of a tile.
/// 
/// 

pub(super) struct TerrainSnapToTerrainPlugin;
impl Plugin for TerrainSnapToTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, (
            snap_on_added,
            update_snap_entity_tile,
            snap_on_tile_rebuilt.run_if(on_event::<TileHeightsRebuilt>()),
        ).chain().run_if(any_with_component::<SnapToTerrain>).after(TransformSystem::TransformPropagate));
        
        app.init_resource::<TileToSnapEntities>();
    }
}

/// Causes the entity to snap to the height of terrain.
/// 
/// Entities are snapped when the component is added or the tile is updated.
#[derive(Component)]
pub struct SnapToTerrain {
    /// Offset to apply to the y of the entity.
    /// 
    /// Modify this to push an entity further into the ground or above it.
    pub y_offset: f32
}

#[derive(Component)]
struct SnapEntityTile(IVec2);

#[derive(Resource, Default)]
struct TileToSnapEntities(HashMap<IVec2, EntityHashSet>);

fn snap_on_added(
    mut commands: Commands,
    terrain_settings: Res<TerrainSettings>,
    mut query: Query<(Entity, &mut Transform, &GlobalTransform, &SnapToTerrain), Without<SnapEntityTile>>,
    tile_to_terrain: Res<TileToTerrain>,
    tiles_query: Query<&Heights>,
    mut tile_to_snap_entities: ResMut<TileToSnapEntities>
) {
    query.iter_mut().for_each(|(entity, mut transform, global_transform, snap_to_terrain)| {
        let tile_coordinate = global_transform.translation().as_ivec3().xz() >> terrain_settings.tile_size_power.get();

        commands.entity(entity).insert(SnapEntityTile(tile_coordinate));

        if let Some(entities) = tile_to_snap_entities.0.get_mut(&tile_coordinate) {
            entities.insert(entity);
        } else {
            tile_to_snap_entities.0.insert(tile_coordinate, EntityHashSet::from_iter([entity]));
        }

        let Some(tile) = tile_to_terrain.get(&tile_coordinate).and_then(|tiles| tiles.first()).and_then(|entity| tiles_query.get(*entity).ok()) else {
            return;
        };

        let relative_location = global_transform.translation().xz() - (tile_coordinate << terrain_settings.tile_size_power.get()).as_vec2();

        let height = get_height_at_position_in_tile(relative_location, tile, &terrain_settings);

        transform.translation.y = height + snap_to_terrain.y_offset;
    });
}

fn snap_on_tile_rebuilt(
    terrain_settings: Res<TerrainSettings>,
    tile_to_snap_entities: ResMut<TileToSnapEntities>,
    mut snap_entity_query: Query<(&mut Transform, &GlobalTransform, &SnapToTerrain)>,
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

        let Some(tile) = tile_to_terrain.get(tile_coordinate).and_then(|tiles| tiles.first()).and_then(|entity| tiles_query.get(*entity).ok()) else {
            continue;
        };
        let tile_translation = (*tile_coordinate << terrain_settings.tile_size_power.get()).as_vec2();

        let mut iter = snap_entity_query.iter_many_mut(snap_entities.iter());
        while let Some((mut transform, global_transform, snap_to_terrain)) = iter.fetch_next() {
            let relative_location = global_transform.translation().xz() - tile_translation;

            let height = get_height_at_position_in_tile(relative_location, tile, &terrain_settings);

            transform.translation.y = height + snap_to_terrain.y_offset;
        }
    }
}

fn update_snap_entity_tile(
    terrain_settings: Res<TerrainSettings>,
    mut query: Query<(&GlobalTransform, &mut SnapEntityTile), Changed<GlobalTransform>>
) {
    query.par_iter_mut().for_each(|(global_transform, mut snap_entity_tile)| {
        let tile_coordinate = global_transform.translation().as_ivec3().xz() >> terrain_settings.tile_size_power.get();

        if snap_entity_tile.0 != tile_coordinate {
            snap_entity_tile.0 = tile_coordinate;
        }
    });
}