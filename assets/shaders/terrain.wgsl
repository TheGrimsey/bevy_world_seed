#import bevy_pbr::forward_io::VertexOutput

@group(2) @binding(0) var texture_map: texture_2d<f32>;
@group(2) @binding(1) var texture_map_sampler: sampler;

@group(2) @binding(0) var texture_a: texture_2d<f32>;
@group(2) @binding(1) var texture_a_sampler: sampler;

@group(2) @binding(0) var texture_b: texture_2d<f32>;
@group(2) @binding(1) var texture_b_sampler: sampler;

@group(2) @binding(0) var texture_c: texture_2d<f32>;
@group(2) @binding(1) var texture_c_sampler: sampler;

@group(2) @binding(0) var texture_d: texture_2d<f32>;
@group(2) @binding(1) var texture_d_sampler: sampler;

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    return textureSample(texture_map, texture_map_sampler, mesh.uv);
}