use std::num::NonZeroU8;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{load_internal_asset, Asset, AssetApp, Assets, Handle};
use bevy_log::{info, info_span};
use bevy_math::{IVec2, UVec4, Vec2, Vec3, Vec3Swizzles, Vec4};
use bevy_pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy_ecs::prelude::{Commands, Component, Entity, EventReader, Query, Res, ResMut, Resource, With, Without, IntoSystemConfigs, ReflectComponent, ReflectResource, Local, DetectChanges};
use bevy_render::{
    primitives::Aabb,
    render_asset::RenderAssetUsages,
    render_resource::{AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat},
    texture::TextureFormatPixelInfo,
    prelude::{Mesh, Image, Shader}
};
use bevy_transform::prelude::GlobalTransform;
use bevy_reflect::{Reflect, prelude::ReflectDefault};

use crate::{
    distance_squared_to_line_segment, easing::EasingFunction, meshing::TerrainMeshRebuilt, modifiers::{
        ModifierFalloffProperty, ShapeModifier, TerrainSplineCached, TerrainSplineProperties,
        TileToModifierMapping,
    }, noise::{NoiseCache, NoiseIndexCache, StrengthCombinator, TerrainNoiseSettings, TileBiomes}, terrain::{Terrain, TileToTerrain}, utils::{get_height_at_position_in_quad, get_normal_at_position_in_quad, index_to_x_z, index_to_x_z_simd}, Heights, TerrainSets, TerrainSettings
};

pub const TERRAIN_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(138167552981664683109966343978676199666);

pub struct TerrainTexturingPlugin(pub TerrainTexturingSettings);
impl Plugin for TerrainTexturingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TerrainMaterialExtended>::default());

        app.insert_resource(self.0.clone());
        app.insert_resource(GlobalTexturingRules { rules: vec![] });

        app.register_asset_reflect::<TerrainMaterialExtended>()
            .register_type::<TextureModifierOperation>()
            .register_type::<TextureModifierFalloffProperty>()
            .register_type::<GlobalTexturingRules>();

        load_internal_asset!(
            app,
            TERRAIN_SHADER_HANDLE,
            "terrain.wgsl",
            Shader::from_wgsl
        );

        app.add_systems(
            PostUpdate,
            (
                insert_texture_map.in_set(TerrainSets::Init),
                update_terrain_texture_maps
                    .after(TerrainSets::Meshing)
                    .in_set(TerrainSets::Material),
            )
                .chain(),
        );

        app.init_resource::<TerrainTextureRebuildQueue>();
    }
}

#[derive(Resource, Clone)]
pub struct TerrainTexturingSettings {
    /// Determines the resolution of the texture map.
    ///
    /// The resulting resolution is `1 << texture_resolution_power`.
    ///
    /// Each step this is increased results in a 4x increase in RAM & VRAM usage.
    ///
    /// Setting this equal to [`TerrainSettings::tile_size_power`] would mean textures can change every 1m.
    pub texture_resolution_power: NonZeroU8,

    /// The maximum amount of tiles to update in a single frame.
    pub max_tile_updates_per_frame: NonZeroU8,
}
impl TerrainTexturingSettings {
    pub fn resolution(&self) -> u32 {
        1 << self.texture_resolution_power.get() as u32
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct TextureModifierOperation {
    pub texture: Handle<Image>,
    pub normal_texture: Option<Handle<Image>>,
    /// Maximum strength to apply the texture of this modifier with.
    ///
    /// Range between `0.0..=1.0`
    pub max_strength: f32,
    /// Represents the size of the texture in world units.
    ///
    /// `1.0` means the texture will repeat every world unit.
    pub units_per_texture: f32,
}

/// Determines the falloff distance for texture operations.
///
/// Overrides [`ModifierFalloffProperty`] if present.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct TextureModifierFalloffProperty {
    pub falloff: f32,
    pub easing_function: EasingFunction,
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Debug, Clone, PartialEq)]
pub enum TexturingRuleEvaluator {
    /// Applies texture to everything above `height` with a falloff distance of `falloff`
    Above {
        /// Y above which the rule will apply at full strength in world units.
        height: f32,
        /// Falloff for strength below `height` in world units.
        ///
        /// Texture strength will linearly reduce for this distance.
        falloff: f32,
    },
    /// Applies texture to everything below `height` with a falloff distance of `falloff`
    Below {
        /// Y below which the rule will apply at full strength in world units.
        height: f32,
        /// Falloff for strength above `height` in world units.
        ///
        /// Texture strength will linearly reduce for this distance.
        falloff: f32,
    },
    /// Applies texture to everything between `max_height` & `min_height` with a falloff distance of `falloff`
    Between {
        /// Y below which the rule will apply at full strength in world units.
        max_height: f32,
        /// Y above which the rule will apply at full strength in world units.
        min_height: f32,
        /// Falloff for strength above `max_height` & below `min_height` in world units.
        ///
        /// Texture strength will linearly reduce for this distance.
        falloff: f32,
    },
    /// Applies texture to everything with a normal angle greater than `angle_radians`
    AngleGreaterThan {
        /// Angle in radians above which the rule will apply at full strength.
        angle_radians: f32,
        /// Falloff for strength below `angle_radians` in radians.
        ///
        /// Texture strength will linearly reduce for this many radians.
        falloff_radians: f32,
    },
    /// Applies texture to everything with a normal angle less than `angle_radians`
    AngleLessThan {
        /// Angle in radians below which the rule will apply at full strength.
        angle_radians: f32,
        /// Falloff for strength above `angle_radians` in radians.
        ///
        /// Texture strength will linearly reduce for this many radians.
        falloff_radians: f32,
    },
    /// Applies texture to everything with a normal angle greater than `min_angle_radians` and less than `max_angle_radians` 
    AngleBetween {
        min_angle_radians: f32,
        max_angle_radians: f32,
        /// Falloff for strength outside of range in radians.
        ///
        /// Texture strength will linearly reduce for this many radians.
        falloff_radians: f32,
    },
    /// Applies a texture when in the biome.
    /// 
    /// Using this rule will cause noise sampling when generating. This may slow down updating textures.
    InBiome {
        biome: u32
    }
}
impl TexturingRuleEvaluator {
    pub fn needs_noise(&self) -> bool {
        match self {
            TexturingRuleEvaluator::Above { .. }
            | TexturingRuleEvaluator::Below { .. }
            | TexturingRuleEvaluator::Between { .. }
            | TexturingRuleEvaluator::AngleGreaterThan { .. }
            | TexturingRuleEvaluator::AngleLessThan { .. }
            | TexturingRuleEvaluator::AngleBetween { .. } => false,
            TexturingRuleEvaluator::InBiome { .. } => true,
        }
    }

    pub fn can_apply_to_tile(&self, min: f32, max: f32, tile_biomes: &TileBiomes) -> bool {
        match self {
            TexturingRuleEvaluator::Above { height, falloff } => {
                max >= (*height - *falloff)
            },
            TexturingRuleEvaluator::Below { height, falloff } => {
                min <= (*height + *falloff)
            },
            TexturingRuleEvaluator::Between { max_height, min_height, falloff } => {
                max >= (*min_height - *falloff) && min <= (*max_height + falloff)
            },
            // TODO: Early filter by angles. We can estimate the worst case angle by taking the heights & max distance between two connected vertices.
            TexturingRuleEvaluator::AngleGreaterThan { .. } => true,
            TexturingRuleEvaluator::AngleLessThan { .. } => true,
            TexturingRuleEvaluator::AngleBetween { .. } => true,
            TexturingRuleEvaluator::InBiome { biome } => {
                // Anything below 1 / 255 means it's rounded to 0.
                tile_biomes.0.get(*biome as usize).is_some_and(|val| *val >= (1.0 / 255.0))
            },
        }
    }

    pub fn eval_simd(&self, height_at_position: Vec4, angle_at_position: Vec4, biomes: &[Vec4]) -> Vec4 {
        match self {
            TexturingRuleEvaluator::Above { height, falloff } => {
                Vec4::ONE - ((Vec4::splat(*height) - height_at_position).max(Vec4::ZERO) / falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE)
            }
            TexturingRuleEvaluator::Below { height, falloff } => {
                Vec4::ONE - ((height_at_position - Vec4::splat(*height)).max(Vec4::ZERO) / falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE)
            }
            TexturingRuleEvaluator::Between {
                max_height,
                min_height,
                falloff,
            } => {
                let strength_below = Vec4::ONE
                    - ((Vec4::splat(*min_height) - height_at_position).max(Vec4::ZERO) / falloff.max(f32::EPSILON))
                        .clamp(Vec4::ZERO, Vec4::ONE);
                let strength_above = Vec4::ONE
                    - ((height_at_position - Vec4::splat(*max_height)).max(Vec4::ZERO) / falloff.max(f32::EPSILON))
                        .clamp(Vec4::ZERO, Vec4::ONE);

                strength_below.min(strength_above)
            }
            TexturingRuleEvaluator::AngleGreaterThan {
                angle_radians,
                falloff_radians,
            } => {
                Vec4::ONE - ((Vec4::splat(*angle_radians) - angle_at_position).max(Vec4::ZERO)
                    / falloff_radians.max(f32::EPSILON))
                .clamp(Vec4::ZERO, Vec4::ONE)
            }
            TexturingRuleEvaluator::AngleLessThan {
                angle_radians,
                falloff_radians,
            } => {
                Vec4::ONE - ((angle_at_position - Vec4::splat(*angle_radians)).max(Vec4::ZERO)
                    / falloff_radians.max(f32::EPSILON))
                .clamp(Vec4::ZERO, Vec4::ONE)
            },
            TexturingRuleEvaluator::AngleBetween { min_angle_radians, max_angle_radians, falloff_radians } => {
                let strength_above = 
                Vec4::ONE - ((Vec4::splat(*max_angle_radians) - angle_at_position).max(Vec4::ZERO)
                        / falloff_radians.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE);
                
                let strength_below = 
                Vec4::ONE - ((angle_at_position - Vec4::splat(*min_angle_radians)).max(Vec4::ZERO)
                        / falloff_radians.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE);

                strength_below.min(strength_above)
            }
            TexturingRuleEvaluator::InBiome { biome } => {
                biomes.get(*biome as usize).cloned().unwrap_or(Vec4::ZERO)
            },
        }
    }
}

#[derive(Reflect)]
pub struct TexturingRule {
    pub evaluators: Vec<TexturingRuleEvaluator>,
    pub evaulator_combinator: StrengthCombinator,

    /// The texture to apply with this rule.
    pub texture: Handle<Image>,
    pub normal_texture: Option<Handle<Image>>,
    /// Represents the size of the texture in world units.
    ///
    /// `1.0` means the texture will repeat every world unit.
    pub units_per_texture: f32,
}

/// Defines the rules for procedural texturing.
#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct GlobalTexturingRules {
    pub rules: Vec<TexturingRule>,
}

#[derive(Asset, AsBindGroup, Default, Debug, Clone, Reflect)]
#[reflect(Default, Debug)]
pub(super) struct TerrainMaterial {
    #[texture(20)]
    #[sampler(21)]
    texture_map: Handle<Image>,

    #[uniform(22)]
    texture_a_scale: f32,

    #[texture(23)]
    #[sampler(24)]
    texture_b: Option<Handle<Image>>,
    #[texture(25)]
    #[sampler(26)]
    texture_b_normal: Option<Handle<Image>>,

    #[uniform(27)]
    texture_b_scale: f32,

    #[texture(28)]
    #[sampler(29)]
    texture_c: Option<Handle<Image>>,
    #[texture(30)]
    #[sampler(31)]
    texture_c_normal: Option<Handle<Image>>,

    #[uniform(32)]
    texture_c_scale: f32,

    #[texture(33)]
    #[sampler(34)]
    texture_d: Option<Handle<Image>>,
    #[texture(35)]
    #[sampler(36)]
    texture_d_normal: Option<Handle<Image>>,

    #[uniform(37)]
    texture_d_scale: f32,
}

type TerrainMaterialExtended = ExtendedMaterial<StandardMaterial, TerrainMaterial>;
trait TerrainMaterialExtendedMethods {
    fn clear_textures(&mut self);
    fn clear_slot(&mut self, index: usize);
    fn get_texture_slot(
        &mut self,
        image: &Handle<Image>,
        normal: &Option<Handle<Image>>,
        units_per_texture: f32,
        tile_size: f32,
    ) -> Option<(usize, bool)>;
}

impl TerrainMaterialExtendedMethods for TerrainMaterialExtended {
    fn clear_textures(&mut self) {
        self.base.base_color_texture = None;
        self.extension.texture_b = None;
        self.extension.texture_c = None;
        self.extension.texture_d = None;

        self.base.normal_map_texture = None;
        self.extension.texture_b_normal = None;
        self.extension.texture_c_normal = None;
        self.extension.texture_d_normal = None;
    }

    fn clear_slot(&mut self, index: usize) {
        match index {
            0 => {
                self.base.base_color_texture = None;
                self.base.normal_map_texture = None;
            },
            1 => {
                self.extension.texture_b = None;
                self.extension.texture_b_normal = None;
            },
            2 => {
                self.extension.texture_c = None;
                self.extension.texture_c_normal = None;
            },
            3 => {
                self.extension.texture_d = None;
                self.extension.texture_d_normal = None;
            },
            _ => {}
        }
    }

    fn get_texture_slot(
        &mut self,
        image: &Handle<Image>,
        normal: &Option<Handle<Image>>,
        units_per_texture: f32,
        tile_size: f32,
    ) -> Option<(usize, bool)> {
        let scale = 1.0 / (units_per_texture / tile_size);
        // Find the first matching or empty texture slot (& assign it to the input texture if applicable).
        if self.base.base_color_texture.as_ref().is_some_and(|entry| {
            entry == image && self.extension.texture_a_scale == scale && self.base.normal_map_texture == *normal
        }) {
            Some((0, false))
        } else if self.extension.texture_b.as_ref().is_some_and(|entry| {
            entry == image && self.extension.texture_b_scale == scale && self.extension.texture_b_normal == *normal
        }) {
            Some((1, false))
        } else if self.extension.texture_c.as_ref().is_some_and(|entry| {
            entry == image && self.extension.texture_c_scale == scale && self.extension.texture_c_normal == *normal
        }) {
            Some((2, false))
        } else if self.extension.texture_d.as_ref().is_some_and(|entry| {
            entry == image && self.extension.texture_d_scale == scale && self.extension.texture_d_normal == *normal
        }) {
            Some((3, false))
        } else if self.base.base_color_texture.is_none() {
            self.base.base_color_texture = Some(image.clone());
            self.base.normal_map_texture.clone_from(normal);
            self.extension.texture_a_scale = scale;

            Some((0, true))
        } else if self.extension.texture_b.is_none() {
            self.extension.texture_b = Some(image.clone());
            self.extension.texture_b_normal.clone_from(normal);
            self.extension.texture_b_scale = scale;

            Some((1, true))
        } else if self.extension.texture_c.is_none() {
            self.extension.texture_c = Some(image.clone());
            self.extension.texture_c_normal.clone_from(normal);
            self.extension.texture_c_scale = scale;

            Some((2, true))
        } else if self.extension.texture_d.is_none() {
            self.extension.texture_d = Some(image.clone());
            self.extension.texture_d_normal.clone_from(normal);
            self.extension.texture_d_scale = scale;

            Some((3, true))
        } else {
            None
        }
    }
}

impl MaterialExtension for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        TERRAIN_SHADER_HANDLE.into()
    }
}

fn insert_texture_map(
    texture_settings: Res<TerrainTexturingSettings>,
    mut commands: Commands,
    mut materials: ResMut<Assets<TerrainMaterialExtended>>,
    mut images: ResMut<Assets<Image>>,
    query: Query<Entity, (With<Terrain>, Without<Handle<TerrainMaterialExtended>>)>,
) {
    let resolution = texture_settings.resolution();
    let texture_format = TextureFormat::Rgba8Unorm;
    let size = texture_format.pixel_size() * resolution as usize * resolution as usize;

    query.iter().for_each(|entity| {
        let image = Image::new(
            Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![0; size],
            texture_format,
            RenderAssetUsages::all(),
        );

        let image_handle = images.add(image);

        let material = TerrainMaterial {
            texture_map: image_handle,
            ..Default::default()
        };

        let material_handle = materials.add(TerrainMaterialExtended {
            base: StandardMaterial {
                perceptual_roughness: 1.0,
                reflectance: 0.0,
                ..Default::default()
            },
            extension: material,
        });

        commands
            .entity(entity)
            .insert((material_handle, Aabb::default()));
    });
}

/// Queue of terrain tiles which textures are to be rebuilt.
#[derive(Resource, Default)]
pub struct TerrainTextureRebuildQueue(Vec<IVec2>);
impl TerrainTextureRebuildQueue {
    pub fn get(&self) -> &[IVec2] {
        &self.0
    }
    pub fn count(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

fn update_terrain_texture_maps(
    shape_modifier_query: Query<(
        &TextureModifierOperation,
        &ShapeModifier,
        Option<&ModifierFalloffProperty>,
        Option<&TextureModifierFalloffProperty>,
        &GlobalTransform,
    )>,
    spline_query: Query<(
        &TextureModifierOperation,
        &TerrainSplineCached,
        &TerrainSplineProperties,
        Option<&ModifierFalloffProperty>,
        Option<&TextureModifierFalloffProperty>,
    )>,
    tiles_query: Query<(
        &Heights,
        &Handle<TerrainMaterialExtended>,
        &Handle<Mesh>,
        &Terrain,
        &Aabb,
        &TileBiomes
    )>,
    texturing_settings: (
        Res<TerrainTexturingSettings>,
        Res<GlobalTexturingRules>,
    ),
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    tile_to_terrain: Res<TileToTerrain>,
    mut event_reader: EventReader<TerrainMeshRebuilt>,
    mut tile_generate_queue: ResMut<TerrainTextureRebuildQueue>,
    mut materials: ResMut<Assets<TerrainMaterialExtended>>,
    mut images: ResMut<Assets<Image>>,
    meshes: Res<Assets<Mesh>>,
    noise_resources: (
        Res<TerrainNoiseSettings>,
        Res<NoiseCache>,
        Res<NoiseIndexCache>
    ), 
    mut needs_noise: Local<bool>,
    mut data_samples: Local<Vec<Vec4>>,
    mut biome_samples: Local<Vec<Vec4>>
) {
    if texturing_settings.1.is_changed() {
        *needs_noise = texturing_settings.1.rules.iter().any(|rule| rule.evaluators.iter().any(|evaulator| evaulator.needs_noise()));
    }

    for TerrainMeshRebuilt(tile) in event_reader.read() {
        if !tile_generate_queue.0.contains(tile) {
            tile_generate_queue.0.push(*tile);
        }
    }

    if tile_generate_queue.is_empty() {
        return;
    }

    let tile_size = terrain_settings.tile_size();
    let resolution = texturing_settings.0.resolution();
    let scale = tile_size / resolution as f32;
    let vertex_scale = (terrain_settings.edge_points - 1) as f32 / resolution as f32;
    let inv_tile_size_scale = scale * (7.0 / tile_size);

    let tiles_to_generate = tile_generate_queue
        .count()
        .min(texturing_settings.0.max_tile_updates_per_frame.get() as usize);

    tiles_query
        .iter_many(
            tile_generate_queue
                .0
                .drain(..tiles_to_generate)
                .filter_map(|tile| tile_to_terrain.0.get(&tile))
                .flatten(),
        )
        .for_each(|(heights, material, mesh, terrain_coordinate, aabb, tile_biomes)| {
            let Some(material) = materials.get_mut(material) else {
                return;
            };
            let Some(mesh) = meshes.get(mesh) else {
                return;
            };
            let Some(texture) = images.get_mut(material.extension.texture_map.id()) else {
                return;
            };

            texture.data.fill(0);
            material.clear_textures();

            let terrain_translation =
                (terrain_coordinate.0 << terrain_settings.tile_size_power.get()).as_vec2();

            if !texturing_settings.1.rules.is_empty() {
                let _span = info_span!("Apply global texturing rules.").entered();

                let normals = mesh
                    .attribute(Mesh::ATTRIBUTE_NORMAL)
                    .unwrap()
                    .as_float3()
                    .unwrap();

                let min = aabb.center.y - aabb.half_extents.y;
                let max = aabb.center.y + aabb.half_extents.y;

                for rule in texturing_settings.1.rules.iter() {
                    if (
                        matches!(rule.evaulator_combinator, StrengthCombinator::Min | StrengthCombinator::Multiply) && !rule.evaluators.iter().all(|evaluator| evaluator.can_apply_to_tile(min, max, tile_biomes))
                        || !rule.evaluators.iter().any(|evaluator| evaluator.can_apply_to_tile(min, max, tile_biomes))
                    ) {
                        continue;
                    }

                    let Some((texture_channel, is_new)) = material.get_texture_slot(
                        &rule.texture,
                        &rule.normal_texture,
                        rule.units_per_texture,
                        tile_size,
                    ) else {
                        info!("Hit max texture channels.");
                        continue;
                    };
                    
                    let mut applied = false;
                    let needs_noise = *needs_noise && rule.evaluators.iter().any(|evaluator| evaluator.needs_noise()); 

                    for (i, val) in texture.data.chunks_exact_mut(16).enumerate() {
                        let true_i = (i * 4) as u32;
                        let (x, z) = index_to_x_z_simd(UVec4::new(true_i, true_i + 1, true_i + 2, true_i + 3), resolution);

                        let x_f = x.as_vec4() * vertex_scale;
                        let z_f = z.as_vec4() * vertex_scale;

                        let vertex_x = x_f.as_uvec4();
                        let vertex_z = z_f.as_uvec4();

                        let vertex_a = (vertex_z * terrain_settings.edge_points as u32) + vertex_x;
                        let vertex_b = vertex_a + 1;
                        let vertex_c = vertex_a + terrain_settings.edge_points as u32;
                        let vertex_d = vertex_a + terrain_settings.edge_points as u32 + 1;

                        let local_x = x_f.fract();
                        let local_z = z_f.fract();

                        if needs_noise {
                            let world_x = x_f + Vec4::splat(terrain_translation.x);
                            let world_z = z_f + Vec4::splat(terrain_translation.y);

                            data_samples.clear();
                            noise_resources.0.sample_data_simd(&noise_resources.1, &noise_resources.2, world_x, world_z, &mut data_samples);

                            biome_samples.clear();
                            noise_resources.0.sample_biomes_simd(&noise_resources.1, &noise_resources.2, world_x, world_z, &data_samples, &mut biome_samples);
                        }

                        // Skip the bounds checks.
                        let height_at_position = unsafe {
                            Vec4::new(
                                get_height_at_position_in_quad(
                                    *heights.0.get_unchecked(vertex_a.x as usize),
                                    *heights.0.get_unchecked(vertex_b.x as usize),
                                    *heights.0.get_unchecked(vertex_c.x as usize),
                                    *heights.0.get_unchecked(vertex_d.x as usize),
                                    local_x.x,
                                    local_z.x,
                                ),
                                get_height_at_position_in_quad(
                                    *heights.0.get_unchecked(vertex_a.y as usize),
                                    *heights.0.get_unchecked(vertex_b.y as usize),
                                    *heights.0.get_unchecked(vertex_c.y as usize),
                                    *heights.0.get_unchecked(vertex_d.y as usize),
                                    local_x.y,
                                    local_z.y,
                                ),
                                get_height_at_position_in_quad(
                                    *heights.0.get_unchecked(vertex_a.z as usize),
                                    *heights.0.get_unchecked(vertex_b.z as usize),
                                    *heights.0.get_unchecked(vertex_c.z as usize),
                                    *heights.0.get_unchecked(vertex_d.z as usize),
                                    local_x.z,
                                    local_z.z,
                                ),
                                get_height_at_position_in_quad(
                                    *heights.0.get_unchecked(vertex_a.w as usize),
                                    *heights.0.get_unchecked(vertex_b.w as usize),
                                    *heights.0.get_unchecked(vertex_c.w as usize),
                                    *heights.0.get_unchecked(vertex_d.w as usize),
                                    local_x.w,
                                    local_z.w,
                                )
                            )
                        };

                        // Skip the bounds checks.
                        let normal_angle = unsafe {
                            Vec4::new(
                                get_normal_at_position_in_quad(
                                    (*normals.get_unchecked(vertex_a.x as usize)).into(),
                                    (*normals.get_unchecked(vertex_b.x as usize)).into(),
                                    (*normals.get_unchecked(vertex_c.x as usize)).into(),
                                    (*normals.get_unchecked(vertex_d.x as usize)).into(),
                                    local_x.x,
                                    local_z.x,
                                ).dot(Vec3::Y).acos(),
                                get_normal_at_position_in_quad(
                                    (*normals.get_unchecked(vertex_a.y as usize)).into(),
                                    (*normals.get_unchecked(vertex_b.y as usize)).into(),
                                    (*normals.get_unchecked(vertex_c.y as usize)).into(),
                                    (*normals.get_unchecked(vertex_d.y as usize)).into(),
                                    local_x.y,
                                    local_z.y,
                                ).dot(Vec3::Y).acos(),
                                get_normal_at_position_in_quad(
                                    (*normals.get_unchecked(vertex_a.z as usize)).into(),
                                    (*normals.get_unchecked(vertex_b.z as usize)).into(),
                                    (*normals.get_unchecked(vertex_c.z as usize)).into(),
                                    (*normals.get_unchecked(vertex_d.z as usize)).into(),
                                    local_x.z,
                                    local_z.z,
                                ).dot(Vec3::Y).acos(),
                                get_normal_at_position_in_quad(
                                    (*normals.get_unchecked(vertex_a.w as usize)).into(),
                                    (*normals.get_unchecked(vertex_b.w as usize)).into(),
                                    (*normals.get_unchecked(vertex_c.w as usize)).into(),
                                    (*normals.get_unchecked(vertex_d.w as usize)).into(),
                                    local_x.w,
                                    local_z.w,
                                ).dot(Vec3::Y).acos(),
                            )
                        };

                        let strength = if let Some(evaluator) = rule.evaluators.first() {
                            let initial_strength = evaluator.eval_simd(height_at_position, normal_angle, &biome_samples);

                            rule.evaluators.iter().skip(1).fold(initial_strength, |acc, filter| {
                                let eval_sample = filter.eval_simd(height_at_position, normal_angle, &biome_samples);
                            
                                rule.evaulator_combinator.combine_simd(acc, eval_sample)
                            })
                        } else {
                            Vec4::ONE
                        };

                        // Apply texture.
                        let strength = strength * Vec4::splat(255.0);
                        
                        if strength.cmpge(Vec4::splat(f32::EPSILON)).any() {
                            apply_texture(&mut val[0..4], texture_channel, strength.x as u8);
                            apply_texture(&mut val[4..8], texture_channel, strength.y as u8);
                            apply_texture(&mut val[8..12], texture_channel, strength.z as u8);
                            apply_texture(&mut val[12..16], texture_channel, strength.w as u8);

                            applied = true;
                        }
                    }

                    if is_new && !applied {
                        material.clear_slot(texture_channel);
                    }
                }
            }

            // Secondly, set by shape-modifiers.
            if let Some(shapes) = tile_to_modifier.shape.get(&terrain_coordinate.0) {
                let _span = info_span!("Apply shape modifiers").entered();

                for entry in shapes.iter() {
                    if let Ok((
                        texture_modifier,
                        shape_modifier,
                        modifier_falloff,
                        texture_modifier_falloff,
                        global_transform,
                    )) = shape_modifier_query.get(entry.entity)
                    {
                        let Some((texture_channel, is_new)) = material.get_texture_slot(
                            &texture_modifier.texture,
                            &texture_modifier.normal_texture,
                            texture_modifier.units_per_texture,
                            tile_size,
                        ) else {
                            info!("Hit max texture channels.");
                            return;
                        };

                        let mut applied = false;
                        let shape_translation = global_transform.translation().xz();
                        let (falloff, easing_function) = texture_modifier_falloff
                            .map(|falloff| (falloff.falloff, falloff.easing_function))
                            .or(modifier_falloff
                                .map(|falloff| (falloff.falloff, falloff.easing_function)))
                            .unwrap_or((f32::EPSILON, EasingFunction::Linear));

                        match shape_modifier {
                            ShapeModifier::Circle { radius } => {
                                for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                                    let (x, z) = index_to_x_z(i, resolution as usize);

                                    let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                                    let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                                    let overlap_index = overlap_y * 8 + overlaps_x;
                                    if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                        continue;
                                    }

                                    let pixel_position = terrain_translation
                                        + Vec2::new(x as f32 * scale, z as f32 * scale);

                                    let strength = (1.0
                                        - ((pixel_position.distance(shape_translation) - radius)
                                            / falloff))
                                        .clamp(0.0, texture_modifier.max_strength);

                                    let eased_strength = easing_function.ease(strength);

                                    // Apply texture.
                                    let scaled_strength = (eased_strength * 255.0) as u8;
                                    if scaled_strength > 0 {
                                        apply_texture(val, texture_channel, scaled_strength);
                                        applied = true;
                                    }
                                }
                            }
                            ShapeModifier::Rectangle { x, z } => {
                                let rect_min = Vec2::new(-x, -z);
                                let rect_max = Vec2::new(*x, *z);

                                for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                                    let (x, z) = index_to_x_z(i, resolution as usize);

                                    let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                                    let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                                    let overlap_index = overlap_y * 8 + overlaps_x;
                                    if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                        continue;
                                    }

                                    let pixel_position = terrain_translation
                                        + Vec2::new(x as f32 * scale, z as f32 * scale);
                                    let pixel_local = global_transform
                                        .affine()
                                        .inverse()
                                        .transform_point3(Vec3::new(
                                            pixel_position.x,
                                            0.0,
                                            pixel_position.y,
                                        ))
                                        .xz();

                                    let d_x = (rect_min.x - pixel_local.x)
                                        .max(pixel_local.x - rect_max.x)
                                        .max(0.0);
                                    let d_y = (rect_min.y - pixel_local.y)
                                        .max(pixel_local.y - rect_max.y)
                                        .max(0.0);
                                    let d_d = (d_x * d_x + d_y * d_y).sqrt();

                                    let strength = (1.0 - (d_d / falloff))
                                        .clamp(0.0, texture_modifier.max_strength);
                                    let eased_strength = easing_function.ease(strength);

                                    // Apply texture.
                                    let scaled_strength = (eased_strength * 255.0) as u8;
                                    if scaled_strength > 0 {
                                        apply_texture(val, texture_channel, scaled_strength);
                                        applied = true;
                                    }
                                }
                            }
                        }

                        if is_new && !applied {
                            material.clear_slot(texture_channel);
                        }
                    }
                }
            }

            // Finally, set by splines.
            if let Some(splines) = tile_to_modifier.splines.get(&terrain_coordinate.0) {
                let _span = info_span!("Apply splines").entered();

                for entry in splines.iter() {
                    if let Ok((
                        texture_modifier,
                        spline,
                        spline_properties,
                        modifier_falloff,
                        texture_modifier_falloff,
                    )) = spline_query.get(entry.entity)
                    {
                        let Some((texture_channel, is_new)) = material.get_texture_slot(
                            &texture_modifier.texture,
                            &texture_modifier.normal_texture,
                            texture_modifier.units_per_texture,
                            tile_size,
                        ) else {
                            info!("Hit max texture channels.");
                            continue;
                        };
                        let (falloff, easing_function) = texture_modifier_falloff
                            .map(|falloff| (falloff.falloff, falloff.easing_function))
                            .or(modifier_falloff
                                .map(|falloff| (falloff.falloff, falloff.easing_function)))
                            .unwrap_or((f32::EPSILON, EasingFunction::Linear));

                        let mut applied = false;

                        for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                            let (x, z) = index_to_x_z(i, resolution as usize);

                            let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                            let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                            let overlap_index = overlap_y * 8 + overlaps_x;
                            if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                continue;
                            }

                            let vertex_position =
                                terrain_translation + Vec2::new(x as f32, z as f32) * scale;
                            let mut distance = f32::INFINITY;

                            for points in spline.points.windows(2) {
                                let a_2d = points[0].xz();
                                let b_2d = points[1].xz();

                                let (new_distance, _) =
                                    distance_squared_to_line_segment(a_2d, b_2d, vertex_position);

                                if new_distance < distance {
                                    distance = new_distance;
                                }
                            }

                            let strength = (1.0
                                - ((distance.sqrt() - spline_properties.half_width) / falloff))
                                .clamp(0.0, texture_modifier.max_strength);
                            let eased_strength = easing_function.ease(strength);

                            // Apply texture.
                            // Apply texture.
                            let scaled_strength = (eased_strength * 255.0) as u8;
                            if scaled_strength > 0 {
                                apply_texture(val, texture_channel, scaled_strength);
                                applied = true;
                            }
                        }
                        
                        if is_new && !applied {
                            material.clear_slot(texture_channel);
                        }
                    }
                }
            }
        });
}

pub fn apply_texture(channels: &mut [u8], target_channel: usize, strength: u8) {
    // Problem: Must be fast. Must be understandable.

    // Idea: Try to apply the full strength. Removing from the highest if there is not enough.

    // Don't decrease the strength of our channel.
    if channels[target_channel] < strength {
        if strength == 255 {
            channels.fill(0);
        } else {
            channels[target_channel] = 0;
            let total: i16 =
                channels[0] as i16 + channels[1] as i16 + channels[2] as i16 + channels[3] as i16;

            let overflow = total + strength as i16 - 255;

            if overflow > 0 {
                let mut to_remove = overflow as u8;

                while to_remove > 0 {
                    let mut min_channel = 0;
                    let mut min = u16::MAX;
                    for (i, val) in channels.iter().enumerate() {
                        if i != target_channel && (*val as u16) < min && *val > 0 {
                            min_channel = i;
                            min = *val as u16;
                        }
                    }

                    let subtract = channels[min_channel].min(to_remove);
                    channels[min_channel] -= subtract;

                    to_remove -= subtract;
                }
            }
        }
        channels[target_channel] = strength;
    }
}
