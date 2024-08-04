use bevy::{asset::{Asset, Handle}, pbr::Material, prelude::{AlphaMode, Image}, reflect::TypePath, render::render_resource::{AsBindGroup, ShaderRef}};


// This struct defines the data that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub(super) struct TerrainMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub(super) texture_spread: Option<Handle<Image>>,
    pub(super) texture_a: Option<Handle<Image>>,
    pub(super) texture_b: Option<Handle<Image>>,
    pub(super) texture_c: Option<Handle<Image>>,
    pub(super) texture_d: Option<Handle<Image>>,
}

/// The Material trait is very configurable, but comes with sensible defaults for all methods.
/// You only need to implement functions for features that need non-default behavior. See the Material api docs for details!
impl Material for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/custom_material.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}
