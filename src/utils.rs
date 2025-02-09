use bevy_math::{UVec4, Vec2, Vec3};

use crate::{Heights, TerrainSettings};

pub fn distance_squared_to_line_segment(v: Vec2, w: Vec2, p: Vec2) -> (f32, f32) {
    let vw = w - v;
    let pv = p - v;

    // Compute squared length of the segment (w - v)
    let l2 = vw.length_squared();

    // Handle degenerate case where v == w
    if l2 == 0.0 {
        return (pv.length_squared(), 1.0);
    }

    // Calculate the projection factor t
    let t: f32 = pv.dot(vw) / l2;
    let t_clamped = t.clamp(0.0, 1.0);

    // Compute the projection point on the segment
    let projection = v + vw * t_clamped;

    // Return the squared distance and the projection factor t
    (p.distance_squared(projection), t)
}

/// Returns the (interpolated) height at a position in a terrain tile.
///
/// If `relative_location` is outside the tile it will be clamped inside of it.
pub fn get_height_at_position_in_tile(
    relative_location: Vec2,
    heights: &Heights,
    terrain_settings: &TerrainSettings,
) -> f32 {
    let normalized_position = (relative_location / terrain_settings.tile_size())
        .clamp(Vec2::ZERO, Vec2::splat(1.0 - f32::EPSILON));

    // Convert to point out vertex.
    let vertex_space_position = normalized_position * (terrain_settings.edge_points - 1) as f32;

    let vertex_a = (vertex_space_position.y as usize * terrain_settings.edge_points as usize)
        + vertex_space_position.x as usize;
    let vertex_b = vertex_a + 1;
    let vertex_c = vertex_a + terrain_settings.edge_points as usize;
    let vertex_d = vertex_a + terrain_settings.edge_points as usize + 1;

    let quad_normalized_pos = vertex_space_position - vertex_space_position.floor();

    // Skip the bounds checks.
    // SAFETY: These can never fail because of us clamping the normalized position.
    unsafe {
        get_height_at_position_in_quad(
            *heights.0.get_unchecked(vertex_a),
            *heights.0.get_unchecked(vertex_b),
            *heights.0.get_unchecked(vertex_c),
            *heights.0.get_unchecked(vertex_d),
            quad_normalized_pos.x,
            quad_normalized_pos.y,
        )
    }
}

/// Returns the flat normal at a position in a terrain tile.
///
/// If `relative_location` is outside the tile it will be clamped inside of it.
pub fn get_flat_normal_at_position_in_tile(
    relative_location: Vec2,
    heights: &Heights,
    terrain_settings: &TerrainSettings,
) -> Vec3 {
    let normalized_position = (relative_location / terrain_settings.tile_size())
        .clamp(Vec2::ZERO, Vec2::splat(1.0 - f32::EPSILON));

    // Convert to point out vertex.
    let vertex_space_position = normalized_position * (terrain_settings.edge_points - 1) as f32;

    let vertex_a = (vertex_space_position.y as usize * terrain_settings.edge_points as usize)
        + vertex_space_position.x as usize;
    let vertex_b = vertex_a + 1;
    let vertex_c = vertex_a + terrain_settings.edge_points as usize;
    let vertex_d = vertex_a + terrain_settings.edge_points as usize + 1;

    let quad_normalized_pos = vertex_space_position - vertex_space_position.floor();

    // Skip the bounds checks.
    // SAFETY: These can never fail because of us clamping the normalized position.
    unsafe {
        let a = *heights.0.get_unchecked(vertex_a);
        let b = *heights.0.get_unchecked(vertex_b);
        let c = *heights.0.get_unchecked(vertex_c);
        let d = *heights.0.get_unchecked(vertex_d);

        let a = Vec3::new(0.0, a, 0.0);
        let b = Vec3::new(1.0, b, 0.0);
        let c = Vec3::new(0.0, c, 1.0);
        let d = Vec3::new(1.0, d, 1.0);

        if quad_normalized_pos.x + quad_normalized_pos.y > 0.5 {
            face_normal(a, b, c)
        } else {
            face_normal(b, c, d)
        }
    }
}

/// Returns the (interpolated) height at a position in a terrain quad made up of A, B, C, D points.
///
/// X & Y should each be normalized within the quad ([0.0, 1.0]).
#[inline]
pub fn get_height_at_position_in_quad(a: f32, b: f32, c: f32, d: f32, x: f32, y: f32) -> f32 {
    // Determine which triangle the point (x, y) lies in
    if x + y <= 1.0 {
        // Point is in triangle ABC
        closest_height_in_triangle(a, b, c, x, y)
    } else {
        // Point is in triangle BCD
        let (u, v, w) = barycentric_coordinates(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), x, y);

        b * u + c * v + d * w
    }
}

#[inline]
fn closest_height_in_triangle(a: f32, b: f32, c: f32, x: f32, y: f32) -> f32 {
    // Calculate barycentric coordinates for the point (x, y) within the triangle
    let u = 1.0 - x - y;
    let v = x;
    let w = y;

    // Return the interpolated height based on barycentric coordinates
    a * u + b * v + c * w
}

/// Returns the (interpolated) normal at a position in a terrain quad made up of A, B, C, D points.
///
/// X & Y should each be normalized within the quad ([0.0, 1.0]).
#[inline]
pub fn get_normal_at_position_in_quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3, x: f32, y: f32) -> Vec3 {
    // Determine which triangle the point (x, y) lies in0
    if x + y < 1.0 {
        // Point is in triangle ABC
        closest_normal_in_triangle(a, b, c, x, y)
    } else {
        // Point is in triangle BCD
        let (u, v, w) = barycentric_coordinates(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), x, y);
        (b * u + c * v + d * w).normalize_or_zero()
    }
}

/// Calculate the closest normal in a grid-spaced triangle.
/// Only works for triangles with vertices at (0, 0), (1, 0), and (0, 1).
#[inline]
fn closest_normal_in_triangle(a: Vec3, b: Vec3, c: Vec3, x: f32, y: f32) -> Vec3 {
    // Calculate barycentric coordinates for the point (x, y) within the triangle
    let u = 1.0 - x - y; // Weight for vertex a
    let v = x;           // Weight for vertex b
    let w = y;           // Weight for vertex c

    // Return the interpolated height based on barycentric coordinates
    (a * u + b * v + c * w).normalize_or_zero()
}

/// Calculate the barycentric coordinates for a point (px, py) within a triangle
pub fn barycentric_coordinates(a: Vec2, b: Vec2, c: Vec2, px: f32, py: f32) -> (f32, f32, f32) {
    let p = Vec2::new(px, py);

    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    (u, v, w)
}

#[inline]
pub fn index_to_x_z(index: usize, edge_points: usize) -> (usize, usize) {
    let x = index % edge_points;
    let z = index / edge_points;

    (x, z)
}
#[inline]
pub fn index_to_x_z_simd(indices: UVec4, edge_points: u32) -> (UVec4, UVec4) {
    let edge_points = UVec4::splat(edge_points);
    let x = indices % edge_points;
    let z = indices / edge_points;

    (x, z)
}

#[inline]
pub fn face_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a).normalize()
}

/// Proportionate normals grabbed from https://github.com/bevyengine/bevy/pull/16050
#[inline]
pub fn face_area_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a)
}


#[test]
fn test_height_in_tile() {
    let terrain_settings = TerrainSettings {
        tile_size_power: std::num::NonZeroU8::MIN,
        edge_points: 2,
        max_spline_simplification_distance_squared: 1.0,
        max_tile_updates_per_frame: std::num::NonZeroU8::MIN,
    };

    let heights = Heights([1.0, 1.0, 1.0, 2.0].into());

    let height = get_height_at_position_in_tile(Vec2::splat(2.0), &heights, &terrain_settings);

    assert!((height - 2.0).abs() <= f32::EPSILON);
}

#[test]
fn test_normal_in_quad() {
    let a = Vec3::new(1.0, 0.0, 0.0).normalize();
    let b = Vec3::new(0.3, 0.0, 0.5).normalize();
    let c = Vec3::new(0.0, 1.0, 0.0).normalize();
    let d = Vec3::new(0.0, 0.0, 1.0).normalize();

    let normal_a = get_normal_at_position_in_quad(a, b, c, d, 0.0, 0.0);
    let normal_b = get_normal_at_position_in_quad(a, b, c, d, 1.0, 0.0);
    let normal_c = get_normal_at_position_in_quad(a, b, c, d, 0.0, 1.0);
    let normal_d = get_normal_at_position_in_quad(a, b, c, d, 1.0, 1.0);

    assert!((normal_a - a).length() <= f32::EPSILON, "Checking A, expected: {a:?}, RECIEVED: A: {normal_a:?}, B: {normal_b:?}, C: {normal_c:?}, D: {normal_d:?}");
    assert!((normal_b - b).length() <= f32::EPSILON, "Checking B, expected: {b:?}, RECIEVED: A: {normal_a:?}, B: {normal_b:?}, C: {normal_c:?}, D: {normal_d:?}");
    assert!((normal_c - c).length() <= f32::EPSILON, "Checking C, expected: {c:?}, RECIEVED: A: {normal_a:?}, B: {normal_b:?}, C: {normal_c:?}, D: {normal_d:?}");
    assert!((normal_d - d).length() <= f32::EPSILON, "Checking D, expected: {d:?}, RECIEVED: A: {normal_a:?}, B: {normal_b:?}, C: {normal_c:?}, D: {normal_d:?}");
}