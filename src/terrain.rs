use bevy::{math::{Vec2, Vec3}, render::{mesh::{Indices, Mesh, PrimitiveTopology}, render_asset::RenderAssetUsages}};

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