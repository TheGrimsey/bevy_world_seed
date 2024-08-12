#import bevy_pbr::forward_io::VertexOutput

@group(2) @binding(0) var texture_map: texture_2d<f32>;
@group(2) @binding(1) var texture_map_sampler: sampler;

@group(2) @binding(2) var texture_a: texture_2d<f32>;
@group(2) @binding(3) var texture_a_sampler: sampler;

@group(2) @binding(4) var texture_b: texture_2d<f32>;
@group(2) @binding(5) var texture_b_sampler: sampler;

@group(2) @binding(6) var texture_c: texture_2d<f32>;
@group(2) @binding(7) var texture_c_sampler: sampler;

@group(2) @binding(8) var texture_d: texture_2d<f32>;
@group(2) @binding(9) var texture_d_sampler: sampler;

@fragment
fn fragment(
    mesh: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    var texture_weights = textureSample(texture_map, texture_map_sampler, mesh.uv);

    var diffuse_a = textureSample(texture_a, texture_a_sampler, mesh.uv) * texture_weights.x;
    var diffuse_b = textureSample(texture_d, texture_b_sampler, mesh.uv) * texture_weights.y;
    var diffuse_c = textureSample(texture_c, texture_c_sampler, mesh.uv) * texture_weights.z;
    var diffuse_d = textureSample(texture_d, texture_d_sampler, mesh.uv) * texture_weights.w;

    var color = diffuse_a + diffuse_b + diffuse_c + diffuse_d;

    return color;
}