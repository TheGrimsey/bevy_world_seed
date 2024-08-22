use bevy::{
    app::{App, Plugin, PostUpdate}, asset::{Assets, Handle}, log::info, math::{IVec2, Vec3}, prelude::{Changed, Commands, Entity, Event, EventWriter, IntoSystemConfigs, Mesh, Query, Res, ResMut}, render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    }
};
use bevy_rapier3d::prelude::Collider;

use crate::{
    terrain::{TerrainCoordinate, TileToTerrain}, update_terrain_heights, Heights, TerrainSettings
};

pub struct TerrainMeshingPlugin;
impl Plugin for TerrainMeshingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            update_mesh_from_heights.after(update_terrain_heights),
        );
        
        app.add_event::<TerrainMeshRebuilt>();
    }
}


#[derive(Event)]
pub struct TerrainMeshRebuilt(pub IVec2);

/*
*   0: -X,
*   1: +X,
*   2: -Y,
*   3: +Y
*/

fn update_mesh_from_heights(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(Entity, &Heights, Option<&Handle<Mesh>>, &TerrainCoordinate), Changed<Heights>>,
    heights_query: Query<&Heights>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_terrain: Res<TileToTerrain>,
    mut repaint_texture_events: EventWriter<TerrainMeshRebuilt>
) {
    query
        .iter()
        .for_each(|(entity, heights, mesh_handle, terrain_coordinate)| {
            let neighbors = [
                tile_to_terrain
                    .0
                    .get(&(terrain_coordinate.0 - IVec2::X))
                    .and_then(|entries| entries.first())
                    .and_then(|entity| heights_query.get(*entity).ok())
                    .map(|heights| heights.0.as_ref()),
                tile_to_terrain
                    .0
                    .get(&(terrain_coordinate.0 + IVec2::X))
                    .and_then(|entries| entries.first())
                    .and_then(|entity| heights_query.get(*entity).ok())
                    .map(|heights| heights.0.as_ref()),
                tile_to_terrain
                    .0
                    .get(&(terrain_coordinate.0 - IVec2::Y))
                    .and_then(|entries| entries.first())
                    .and_then(|entity| heights_query.get(*entity).ok())
                    .map(|heights| heights.0.as_ref()),
                tile_to_terrain
                    .0
                    .get(&(terrain_coordinate.0 + IVec2::Y))
                    .and_then(|entries| entries.first())
                    .and_then(|entity| heights_query.get(*entity).ok())
                    .map(|heights| heights.0.as_ref()),
            ];

            let mesh = create_terrain_mesh(
                terrain_settings.tile_size(),
                terrain_settings.edge_points,
                &heights.0,
                &neighbors,
            );

            if let Some(existing_mesh) = mesh_handle.and_then(|handle| meshes.get_mut(handle)) {
                *existing_mesh = mesh;
            } else {
                let new_handle = meshes.add(mesh);

                commands.entity(entity).insert((new_handle,));
            }

            repaint_texture_events.send(TerrainMeshRebuilt(terrain_coordinate.0));

            commands.entity(entity).insert(Collider::heightfield(
                heights.0.to_vec(),
                terrain_settings.edge_points as usize,
                terrain_settings.edge_points as usize,
                Vec3::new(
                    terrain_settings.tile_size(),
                    1.0,
                    terrain_settings.tile_size(),
                ),
            ));
        });
}

fn face_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a).normalize()
}

fn create_terrain_mesh(
    size: f32,
    edge_length: u16,
    heights: &[f32],
    neighbours: &[Option<&[f32]>; 4],
) -> Mesh {
    let vertex_count = edge_length;

    assert_eq!(vertex_count * vertex_count, heights.len() as u16);

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

    info!("Last UV: {:?} {:?}", uvs.first(), uvs.last());

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

    indices.chunks_exact(3).for_each(|face| {
        let [a, b, c] = [face[0], face[1], face[2]];
        let normal = face_normal(
            positions[a as usize],
            positions[b as usize],
            positions[c as usize],
        );

        [a, b, c].iter().for_each(|pos| {
            normals[*pos as usize] += normal;
            adjacency_counts[*pos as usize] += 1;
        });
    });

    // Add neighbors.

    let vertex_edge = (vertex_count - 1) as f32;
    let step = (1.0 / vertex_edge) * size;

    // -X direction.
    if let Some(neighbors) = neighbours[0] {
        // Ignoring corners.
        for x in (0..(num_vertices - edge_length as usize))
            .skip(edge_length.into())
            .step_by(edge_length.into())
        {
            let s = positions[x];
            // 3 bottom triangles.

            let a_i = x + edge_length as usize;
            let a = Vec3::new(s.x, heights[a_i], s.z + step);
            let b_i = x + (edge_length as usize * 2) - 2;
            let b = Vec3::new(s.x - step, neighbors[b_i], s.z + step);
            let c_i = x + edge_length as usize - 2;
            let c = Vec3::new(s.x - step, neighbors[c_i], s.z);
            let d_i = x - edge_length as usize;
            let d = Vec3::new(s.x, heights[d_i], s.z - step);

            let face_a = face_normal(b, a, s);
            let face_b = face_normal(c, b, s);
            let face_c = face_normal(d, c, s);

            adjacency_counts[x] += 3;

            normals[x] += face_a + face_b + face_c;
        }
    }

    // +X direction.
    if let Some(neighbors) = neighbours[1] {
        // Ignoring corners.
        for x in (0..(num_vertices - edge_length as usize))
            .skip(edge_length as usize + edge_length as usize - 1)
            .step_by(edge_length.into())
        {
            let s = positions[x];
            // 3 bottom triangles.

            let a_i = x - edge_length as usize;
            let a = Vec3::new(s.x, heights[a_i], s.z - step);
            let b_i = x + 2 - (edge_length * 2) as usize;
            let b = Vec3::new(s.x + step, neighbors[b_i], s.z - step);
            let c_i = x - edge_length as usize + 2;
            let c = Vec3::new(s.x + step, neighbors[c_i], s.z);
            let d_i = x + edge_length as usize;
            let d = Vec3::new(s.x, heights[d_i], s.z + step);

            let face_a = face_normal(b, a, s);
            let face_b = face_normal(c, b, s);
            let face_c = face_normal(d, c, s);

            adjacency_counts[x] += 3;

            normals[x] += face_a + face_b + face_c;
        }
    }

    // -Y
    if let Some(neighbors) = neighbours[2] {
        let neighbor_row = &neighbors[edge_length as usize * (edge_length as usize - 2)..];

        // Ignoring corners.
        for x in 1..(edge_length - 1) as usize {
            let s = positions[x];
            // 3 bottom triangles.
            let a = Vec3::new(s.x - step, heights[x - 1], s.z);
            let b = Vec3::new(s.x, neighbor_row[x], s.z - step);
            let c = Vec3::new(s.x + step, neighbor_row[x + 1], s.z - step);
            let d = Vec3::new(s.x + step, heights[x + 1], s.z);

            let face_a = face_normal(b, a, s);
            let face_b = face_normal(c, b, s);
            let face_c = face_normal(d, c, s);

            adjacency_counts[x] += 3;

            normals[x] += face_a + face_b + face_c;
        }
    }
    // +Y
    if let Some(neighbors) = neighbours[3] {
        let neighbor_row = &neighbors[edge_length as usize..(edge_length as usize * 2)];

        // Ignoring corners.
        for x in 1..(edge_length - 1) as usize {
            let s_x = (edge_length as usize * (edge_length as usize - 1)) + x;

            let s: Vec3 = positions[s_x];
            // 3 top triangles.
            let a = Vec3::new(s.x + step, heights[s_x + 1], s.z);
            let b = Vec3::new(s.x, neighbor_row[x], s.z + step);
            let c = Vec3::new(s.x - step, neighbor_row[x - 1], s.z + step);
            let d = Vec3::new(s.x - step, heights[s_x - 1], s.z);

            let face_a = face_normal(b, a, s);
            let face_b = face_normal(c, b, s);
            let face_c = face_normal(d, c, s);

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
