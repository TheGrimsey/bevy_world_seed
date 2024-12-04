use bevy_derive::Deref;
use bevy_transform::prelude::GlobalTransform;
use bevy_ecs::{
    prelude::{Changed, Commands, Component, DetectChanges, Entity, EventWriter, Query, Res, ResMut, Resource, With, Without, ReflectComponent},
    component::{ComponentHooks, StorageType}
};
use bevy_math::{IVec2, Vec3Swizzles};
use bevy_reflect::Reflect;
use bevy_utils::HashMap;
use fixedbitset::FixedBitSet;

use crate::{
    feature_placement::SpawnedFeatures, noise::TileBiomes, utils::index_to_x_z, Heights, RebuildTile, TerrainSettings
};

/// Bitset marking which points are holes.
/// Size should equal the amount of vertices in a terrain tile.
#[derive(Component, Debug)]
pub struct Holes(pub(super) FixedBitSet);
impl Holes {
    /// Returns an iterator of every hole cell in the terrain.
    ///
    /// Cells may be returned multiple times.
    ///
    /// This can be used to set the holes in a parry Heightfield.
    pub fn iter_holes(&self, edge_points: u16) -> impl Iterator<Item = HoleEntry> + '_ {
        self.0.ones().flat_map(move |i| {
            let (x, z) = index_to_x_z(i, edge_points as usize);

            [
                Some(HoleEntry {
                    x: x as u16,
                    z: z as u16,
                    left_triangle_removed: true,
                    right_triangle_removed: false,
                }),
                (x > 0).then(|| HoleEntry {
                    x: (x - 1) as u16,
                    z: z as u16,
                    left_triangle_removed: true,
                    right_triangle_removed: true,
                }),
                (x > 0 && z > 0).then(|| HoleEntry {
                    x: (x - 1) as u16,
                    z: (z - 1) as u16,
                    left_triangle_removed: false,
                    right_triangle_removed: true,
                }),
                (z > 0).then(|| HoleEntry {
                    x: x as u16,
                    z: (z - 1) as u16,
                    left_triangle_removed: true,
                    right_triangle_removed: true,
                }),
            ]
            .into_iter()
            .flatten()
        })
    }
}

#[derive(Debug)]
pub struct HoleEntry {
    pub x: u16,
    pub z: u16,
    pub left_triangle_removed: bool,
    pub right_triangle_removed: bool,
}

/// Marker component for the entity being a terrain tile.
///
/// Internal `IVec2` is updated based on `GlobalTransform` to the tile this terrain corresponds to.
#[derive(Reflect, Debug, Default, Clone, Copy)]
#[reflect(Component)]
pub struct Terrain(pub(super) IVec2);
impl Terrain {
    /// Manually set the tile.
    ///
    /// It is not necessary to use this as the tile will be updated after the `GlobalTransform`.
    /// But depending on when in a schedule you spawn the Terrain, it may be useful to set this manually.
    pub fn new_with_tile(tile: IVec2) -> Self {
        Self(tile)
    }
}
impl Component for Terrain {
    const STORAGE_TYPE: StorageType = StorageType::Table;

    fn register_component_hooks(hooks: &mut ComponentHooks) {
        hooks.on_remove(|mut world, entity, _id| {
            let terrain = *world.get::<Terrain>(entity).unwrap();

            // Remove ourselves from the tiles to terrain list.
            let mut tile_to_terrain = world.resource_mut::<TileToTerrain>();
            if let Some(tiles) = tile_to_terrain.0.get_mut(&terrain.0) {
                if let Some(i) = tiles.iter().position(|entry| *entry == entity) {
                    tiles.swap_remove(i);
                }
            }
        });
    }
}

/// Mapping tile coordinate to all terrain tiles on that tile.
// Using a Vec<Entity> to prevent accidental overlaps from breaking the previous tile.
#[derive(Resource, Default, Deref)]
pub struct TileToTerrain(pub(super) HashMap<IVec2, Vec<Entity>>);

pub(super) fn insert_components(
    mut commands: Commands,
    terrain_settings: Res<TerrainSettings>,
    query: Query<Entity, (With<Terrain>, Without<Heights>, Without<Holes>)>,
) {
    let heights = terrain_settings.edge_points as usize * terrain_settings.edge_points as usize;

    query.iter().for_each(|entity| {
        commands.entity(entity).insert((
            Heights(vec![0.0; heights].into_boxed_slice()),
            Holes(FixedBitSet::with_capacity(heights)),
            SpawnedFeatures::default(),
            TileBiomes::default()
        ));
    });
}

pub(super) fn update_tiling(
    mut tile_to_terrain: ResMut<TileToTerrain>,
    mut rebuild_tiles_event: EventWriter<RebuildTile>,
    mut query: Query<(Entity, &mut Terrain, &GlobalTransform), Changed<GlobalTransform>>,
    terrain_setttings: Res<TerrainSettings>,
) {
    query
        .iter_mut()
        .for_each(|(entity, mut terrain_coordinate, global_transform)| {
            let coordinate = global_transform.translation().xz().as_ivec2()
                >> terrain_setttings.tile_size_power.get();

            if terrain_coordinate.is_added() || terrain_coordinate.0 != coordinate {
                if let Some(entries) = tile_to_terrain.0.get_mut(&terrain_coordinate.0) {
                    if let Some(index) = entries.iter().position(|e| *e == entity) {
                        entries.swap_remove(index);
                    }
                }

                if let Some(entries) = tile_to_terrain.0.get_mut(&coordinate) {
                    entries.push(entity);
                } else {
                    tile_to_terrain.0.insert(coordinate, vec![entity]);
                }

                terrain_coordinate.0 = coordinate;
                rebuild_tiles_event.send(RebuildTile(coordinate));
            }
        });
}
