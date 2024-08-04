use bevy::{math::{IVec2, Vec2, Vec3, Vec3Swizzles}, prelude::{Changed, Component, DetectChanges, Entity, GlobalTransform, Query, ReflectComponent, Res, ResMut, Resource}, reflect::Reflect, render::{mesh::{Indices, Mesh, PrimitiveTopology}, render_asset::RenderAssetUsages}, utils::HashMap};

use crate::{DirtyTiles, TerrainSettings};

pub fn create_terrain_mesh(size: Vec2, edge_length: u16, heights: &[f32]) -> Mesh {
    let z_vertex_count = edge_length;
    let x_vertex_count = edge_length;

    assert_eq!(z_vertex_count*x_vertex_count, heights.len() as u16);

    let num_vertices = (z_vertex_count * x_vertex_count) as usize;
    let num_indices = ((z_vertex_count - 1) * (x_vertex_count - 1) * 6) as usize;

    let mut positions: Vec<Vec3> = Vec::with_capacity(num_vertices);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(num_vertices);
    let mut indices: Vec<u16> = Vec::with_capacity(num_indices);

    for z in 0..z_vertex_count {
        let tz = z as f32 / (z_vertex_count - 1) as f32;
        
        for x in 0..x_vertex_count {
            let tx = x as f32 / (x_vertex_count - 1) as f32;

            let index = z as usize * edge_length as usize + x as usize;

            let pos = Vec3::new(tx * size.x, heights[index], tz * size.y);
            positions.push(pos);
            uvs.push([tx, tz]);
        }
    }

    for z in 0..z_vertex_count - 1 {
        for x in 0..x_vertex_count - 1 {
            let quad = z * x_vertex_count + x;
            indices.push(quad + x_vertex_count + 1);
            indices.push(quad + 1);
            indices.push(quad + x_vertex_count);
            indices.push(quad);
            indices.push(quad + x_vertex_count);
            indices.push(quad + 1);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_indices(Indices::U16(indices))
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

    mesh.compute_smooth_normals();

    mesh
}

#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
pub struct TerrainCoordinate(pub(super) IVec2); 

/// Using a Vec<Entity> to prevent accidental overlaps from breaking the previous tile.
#[derive(Resource, Default)]
pub(super) struct TileToTerrain(pub(super) HashMap<IVec2, Vec<Entity>>);

pub(super) fn update_tiling(
    mut tile_to_terrain: ResMut<TileToTerrain>,
    mut dirty_tiles: ResMut<DirtyTiles>,
    mut query: Query<(Entity, &mut TerrainCoordinate, &GlobalTransform), Changed<GlobalTransform>>,
    terrain_setttings: Res<TerrainSettings>
) {
    query.iter_mut().for_each(|(entity, mut terrain_coordinate, global_transform)| {
        let coordinate = global_transform.translation().as_ivec3().xz() >> terrain_setttings.tile_size_power;

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
            dirty_tiles.0.insert(coordinate);
        }
    });
}