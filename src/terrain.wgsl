#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{FragmentOutput, VertexOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{FragmentOutput, VertexOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

#import bevy_pbr::{
    mesh_bindings::mesh,
    mesh_functions,
    skinning,
    pbr_fragment::pbr_input_from_standard_material,
    view_transformations::position_world_to_clip,
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,

    @location(0) height: f32,
    @location(1) normal: vec3<f32>,
    @builtin(vertex_index) index: u32,
}

@group(2) @binding(34) var<uniform> tile_size: f32;
@group(2) @binding(35) var<uniform> edge_points: u32;

@group(2) @binding(20) var texture_map: texture_2d<f32>;
@group(2) @binding(21) var texture_map_sampler: sampler;


@group(2) @binding(22) var texture_a: texture_2d<f32>;
@group(2) @binding(23) var texture_a_sampler: sampler;

@group(2) @binding(24) var<uniform> texture_a_scale: f32;

@group(2) @binding(25) var texture_b: texture_2d<f32>;
@group(2) @binding(26) var texture_b_sampler: sampler;

@group(2) @binding(27) var<uniform> texture_b_scale: f32;

@group(2) @binding(28) var texture_c: texture_2d<f32>;
@group(2) @binding(29) var texture_c_sampler: sampler;

@group(2) @binding(30) var<uniform> texture_c_scale: f32;

@group(2) @binding(31) var texture_d: texture_2d<f32>;
@group(2) @binding(32) var texture_d_sampler: sampler;

@group(2) @binding(33) var<uniform> texture_d_scale: f32;

@vertex
fn vertex(vertex_no_morph: Vertex) -> VertexOutput {
    var out: VertexOutput;

    var vertex = vertex_no_morph;

    // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
    // See https://github.com/gfx-rs/naga/issues/2416 .
    var world_from_local = mesh_functions::get_world_from_local(vertex_no_morph.instance_index);

    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
        // See https://github.com/gfx-rs/naga/issues/2416
        vertex_no_morph.instance_index
    );

    var x = f32(vertex.index % (edge_points + 1)) / f32(edge_points);
    var z = f32(vertex.index / (edge_points + 1)) / f32(edge_points);

    var position = vec3<f32>(x * tile_size, vertex.height, z * tile_size);

    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);

#ifdef VERTEX_UVS_A
    out.uv = vec2<f32>(x, z);
#endif

    return out;
}

@fragment
fn fragment(
    mesh: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    // generate a PbrInput struct from the StandardMaterial bindings
    var pbr_input = pbr_input_from_standard_material(mesh, is_front);

    // Yoinking some code from Bevy pbr shader.
    // https://github.com/bevyengine/bevy/blob/bd8faa7ae17dcd8b4df2beba28876759fb4fdef5/crates/bevy_pbr/src/render/pbr.wgsl
#ifdef PREPASS_PIPELINE
    // write the gbuffer, lighting pass id, and optionally normal and motion_vector textures
    let out = deferred_output(in, pbr_input);
#else

#ifdef VERTEX_UVS_A
    var texture_weights = textureSample(texture_map, texture_map_sampler, mesh.uv);

    var diffuse_a = textureSample(texture_a, texture_a_sampler, fract(mesh.uv * texture_a_scale)) * texture_weights.x;
    var diffuse_b = textureSample(texture_b, texture_b_sampler, fract(mesh.uv * texture_b_scale)) * texture_weights.y;
    var diffuse_c = textureSample(texture_c, texture_c_sampler, fract(mesh.uv * texture_c_scale)) * texture_weights.z;
    var diffuse_d = textureSample(texture_d, texture_d_sampler, fract(mesh.uv * texture_d_scale)) * texture_weights.w;

    var color = diffuse_a + diffuse_b + diffuse_c + diffuse_d;
    
    // Replace color with out mixed color.
    pbr_input.material.base_color =  color;
#endif
    // apply lighting
    var out = apply_pbr_lighting(pbr_input);
    // apply post processsing. 
    out = main_pass_post_lighting_processing(pbr_input, out);
#endif

    return out;
}