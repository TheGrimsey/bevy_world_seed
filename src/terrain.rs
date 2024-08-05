use bevy::{log::info, math::{IVec2, Vec3, Vec3Swizzles}, prelude::{Changed, Component, DetectChanges, Entity, GlobalTransform, Query, ReflectComponent, Res, ResMut, Resource}, reflect::Reflect, render::{mesh::{Indices, Mesh, PrimitiveTopology}, render_asset::RenderAssetUsages}, utils::HashMap};

use crate::{DirtyTiles, TerrainSettings};

/*
*   0: -X,
*   1: +X,
*   2: -Y,
*   3: +Y
*/

fn face_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a).normalize()
}

pub fn create_terrain_mesh(size: f32, edge_length: u16, heights: &[f32], neighbours: &[Option<&[f32]>; 4]) -> Mesh {
    let vertex_count = edge_length;

    assert_eq!(vertex_count*vertex_count, heights.len() as u16);

    let num_vertices = (vertex_count * vertex_count) as usize;
    let num_indices = ((vertex_count - 1) * (vertex_count - 1) * 6) as usize;

    let mut positions: Vec<Vec3> = Vec::with_capacity(num_vertices);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(num_vertices);
    let mut indices: Vec<u16> = Vec::with_capacity(num_indices);

    for z in 0..vertex_count {
        let tz = z as f32 / (vertex_count - 1) as f32;
        let z_i = z as usize * edge_length as usize;
        
        for x in 0..vertex_count {
            let tx = x as f32 / (vertex_count - 1) as f32;

            let index = z_i + x as usize;

            let pos = Vec3::new(tx * size, heights[index], tz * size);
            positions.push(pos);
            uvs.push([tx, tz]);
        }
    }

    info!("Pos: {positions:?}");

    // Create triangles.
    for z in 0..vertex_count - 1 {
        for x in 0..vertex_count - 1 {
            let quad = z * vertex_count + x;
            indices.push(quad + vertex_count + 1);
            indices.push(quad + 1);
            indices.push(quad + vertex_count);
            indices.push(quad);
            indices.push(quad + vertex_count);
            indices.push(quad + 1);
        }
    }

    // Generate normals
    let mut normals = vec![Vec3::ZERO; positions.len()];
    let mut adjacency_counts = vec![0_u8; positions.len()];

    indices
        .chunks_exact(3)
        .for_each(|face| {
            let [a, b, c] = [face[0], face[1], face[2]];
            let normal = face_normal(positions[a as usize], positions[b as usize], positions[c as usize]);
            
            [a, b, c].iter().for_each(|pos| {
                normals[*pos as usize] += normal;
                adjacency_counts[*pos as usize] += 1;
            });
        });

    // Add neighbors.
    
    let vertex_edge = (vertex_count - 1) as f32;
    let step = (1.0 / vertex_edge) * size;
    // Bottom row.
    if let Some(neighbors) = neighbours[2] {
        let neighbor_row = &neighbors[edge_length as usize * (edge_length as usize -1)..];

        info!("BOT");

        // Ignoring corners.
        for x in 1..(edge_length-1) as usize {
            let s = positions[x];
            // 3 bottom triangles.
            let a = Vec3::new(s.x - step, heights[x - 1], s.z);
            let b = Vec3::new(s.x, neighbor_row[x], s.z - step);
            let c = Vec3::new(s.x + step, neighbor_row[x + 1], s.z - step);
            let d = Vec3::new(s.x + step, heights[x + 1], s.z);
            
            let face_a = face_normal(b, a, s);
            let face_b = face_normal(c, b, s);
            let face_c = face_normal(d, c, s);

            info!("A: {a}, B: {b}, C: {c}, D: {d}, S: {s}");
            info!("F_A: {face_a}, F_B: {face_b}, F_C: {face_c}");

            adjacency_counts[x] += 3;

            normals[x] += face_a + face_b + face_c;
        }
    }
    if let Some(neighbors) = neighbours[3] {
        let neighbor_row = &neighbors[..edge_length as usize];

        info!("TOP");
        // Ignoring corners.
        for x in 1..(edge_length-1) as usize {
            let s_x = (edge_length as usize * (edge_length as usize - 1)) + x;

            let s: Vec3 = positions[s_x];
            // 3 top triangles.
            let a = Vec3::new(s.x + step, heights[x + 1], s.z);
            let b = Vec3::new(s.x, neighbor_row[x], s.z + step);
            let c = Vec3::new(s.x - step, neighbor_row[x - 1], s.z + step);
            let d = Vec3::new(s.x - step, heights[x - 1], s.z);
            
            let face_a = face_normal(b, a, s);
            let face_b = face_normal(c, b, s);
            let face_c = face_normal(d, c, s);

            info!("A: {a}, B: {b}, C: {c}, D: {d}, S: {s}");
            info!("F_A: {face_a}, F_B: {face_b}, F_C: {face_c}");

            adjacency_counts[s_x] += 3;

            normals[s_x] += face_a + face_b + face_c;
        }
    }

    // average (smooth) normals for shared vertices...
    for i in 0..normals.len() {
        let count = adjacency_counts[i];
        normals[i] = (normals[i] / (count as f32)).normalize();
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_indices(Indices::U16(indices))
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
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