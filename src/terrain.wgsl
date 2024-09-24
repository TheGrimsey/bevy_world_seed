#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing, calculate_tbn_mikktspace, apply_normal_mapping},
    pbr_bindings
}
#endif

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
}

@group(2) @binding(20) var texture_map: texture_2d<f32>;
@group(2) @binding(21) var texture_map_sampler: sampler;

@group(2) @binding(22) var texture_a: texture_2d<f32>;
@group(2) @binding(23) var texture_a_sampler: sampler;

@group(2) @binding(24) var texture_a_normal: texture_2d<f32>;
@group(2) @binding(25) var texture_a_normal_sampler: sampler;

@group(2) @binding(26) var<uniform> texture_a_scale: f32;

@group(2) @binding(27) var texture_b: texture_2d<f32>;
@group(2) @binding(28) var texture_b_sampler: sampler;

@group(2) @binding(29) var texture_b_normal: texture_2d<f32>;
@group(2) @binding(30) var texture_b_normal_sampler: sampler;

@group(2) @binding(31) var<uniform> texture_b_scale: f32;

@group(2) @binding(32) var texture_c: texture_2d<f32>;
@group(2) @binding(33) var texture_c_sampler: sampler;

@group(2) @binding(34) var texture_c_normal: texture_2d<f32>;
@group(2) @binding(35) var texture_c_normal_sampler: sampler;

@group(2) @binding(36) var<uniform> texture_c_scale: f32;

@group(2) @binding(37) var texture_d: texture_2d<f32>;
@group(2) @binding(38) var texture_d_sampler: sampler;

@group(2) @binding(39) var texture_d_normal: texture_2d<f32>;
@group(2) @binding(40) var texture_d_normal_sampler: sampler;

@group(2) @binding(41) var<uniform> texture_d_scale: f32;

@fragment
fn fragment(
    mesh: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    var texture_weights = textureSample(texture_map, texture_map_sampler, mesh.uv);

    var texture_a_uv = fract(mesh.uv * texture_a_scale);
    var texture_b_uv = fract(mesh.uv * texture_b_scale);
    var texture_c_uv = fract(mesh.uv * texture_c_scale);
    var texture_d_uv = fract(mesh.uv * texture_d_scale);
    
    var diffuse_a = textureSample(texture_a, texture_a_sampler, texture_a_uv) * texture_weights.x;
    var diffuse_b = textureSample(texture_b, texture_b_sampler, texture_b_uv) * texture_weights.y;
    var diffuse_c = textureSample(texture_c, texture_c_sampler, texture_c_uv) * texture_weights.z;
    var diffuse_d = textureSample(texture_d, texture_d_sampler, texture_d_uv) * texture_weights.w;

    var color = diffuse_a + diffuse_b + diffuse_c + diffuse_d;
    
    var normal_a = textureSample(texture_a_normal, texture_a_normal_sampler, texture_a_uv) * texture_weights.x;
    var normal_b = textureSample(texture_b_normal, texture_b_normal_sampler, texture_b_uv) * texture_weights.y;
    var normal_c = textureSample(texture_c_normal, texture_c_normal_sampler, texture_c_uv) * texture_weights.z;
    var normal_d = textureSample(texture_d_normal, texture_d_normal_sampler, texture_d_uv) * texture_weights.w;

    var normal = normal_a + normal_b + normal_c + normal_d;

    // generate a PbrInput struct from the StandardMaterial bindings
    var pbr_input = pbr_input_from_standard_material(mesh, is_front);
    
    // Replace color with out mixed color.
    pbr_input.material.base_color =  color;

    let TBN = calculate_tbn_mikktspace(pbr_input.world_normal, mesh.world_tangent);

    pbr_input.N = apply_normal_mapping(
        pbr_bindings::material.flags,
        TBN,
        false,
        is_front,
        normal.rgb,
    );

// Yoinking some code from Bevy pbr shader.
// https://github.com/bevyengine/bevy/blob/bd8faa7ae17dcd8b4df2beba28876759fb4fdef5/crates/bevy_pbr/src/render/pbr.wgsl
#ifdef PREPASS_PIPELINE
    // write the gbuffer, lighting pass id, and optionally normal and motion_vector textures
    let out = deferred_output(in, pbr_input);
#else
    // apply lighting
    var out = apply_pbr_lighting(pbr_input);
    // apply post processsing. 
    out = main_pass_post_lighting_processing(pbr_input, out);
#endif

    return out;
}