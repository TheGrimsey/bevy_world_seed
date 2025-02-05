use std::{num::NonZeroU8, sync::Arc};

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{load_internal_asset, Asset, AssetApp, Assets, Handle};
use bevy_log::{info, info_span};
use bevy_math::{IVec2, UVec4, Vec2, Vec3, Vec3Swizzles, Vec4};
use bevy_pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, MeshMaterial3d, StandardMaterial};
use bevy_ecs::{prelude::{Commands, Component, DetectChanges, Entity, EventReader, IntoSystemConfigs, Local, Query, ReflectComponent, ReflectResource, Res, ResMut, Resource, With, Without}, schedule::common_conditions::any_with_component};
use bevy_render::{
    primitives::Aabb,
    render_asset::RenderAssetUsages,
    render_resource::{AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat},
    prelude::{Mesh, Shader, Mesh3d}
};
use bevy_transform::prelude::GlobalTransform;
use bevy_reflect::{Reflect, prelude::ReflectDefault};
use bevy_image::Image;
use bevy_tasks::{futures_lite, AsyncComputeTaskPool, Task};

use crate::{
    calc_shape_modifier_strength, distance_squared_to_line_segment, easing::EasingFunction, meshing::TerrainMeshRebuilt, modifiers::{
        ModifierFalloffNoiseProperty, ModifierFalloffProperty, ModifierStrengthLimitProperty, ShapeModifier, TerrainSplineCached, TerrainSplineProperties, TileToModifierMapping
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
                insert_material.in_set(TerrainSets::Init),
                (
                    apply_texture_maps.run_if(any_with_component::<TextureGenerationTask>),
                    update_terrain_texture_maps
                )
                .chain()
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

    /// The maximum amount of texture generation tasks to run at once.
    pub max_texture_generation_tasks: NonZeroU8,
}
impl TerrainTexturingSettings {
    pub fn resolution(&self) -> u32 {
        1 << self.texture_resolution_power.get() as u32
    }
}

#[derive(Component, Reflect, Clone)]
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
#[derive(Component, Reflect, Clone)]
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

#[derive(Reflect, Clone)]
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
#[derive(Resource, Reflect, Clone)]
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
}

impl MaterialExtension for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        TERRAIN_SHADER_HANDLE.into()
    }
}

fn insert_material(
    mut commands: Commands,
    mut materials: ResMut<Assets<TerrainMaterialExtended>>,
    query: Query<Entity, (With<Terrain>, Without<MeshMaterial3d<TerrainMaterialExtended>>)>,
) {
    query.iter().for_each(|entity| {
        let material_handle = materials.add(TerrainMaterialExtended {
            base: StandardMaterial {
                perceptual_roughness: 1.0,
                reflectance: 0.0,
                ..Default::default()
            },
            extension: TerrainMaterial::default(),
        });

        commands
            .entity(entity)
            .insert((MeshMaterial3d(material_handle), Aabb::default()));
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

fn apply_texture_maps(
    mut query: Query<(Entity, &mut TextureGenerationTask, &MeshMaterial3d<TerrainMaterialExtended>)>,
    mut commands: Commands,
    mut materials: ResMut<Assets<TerrainMaterialExtended>>,
    mut textures: ResMut<Assets<Image>>,
) {
    query.iter_mut().for_each(|(entity, mut texture_generation_task, material)| {
        if let Some(splat_map) = futures_lite::future::block_on(futures_lite::future::poll_once(&mut texture_generation_task.task)) {
            if let Some(material) = materials.get_mut(&material.0) {
                material.clear_textures();

                material.base.base_color_texture = splat_map.textures[0].clone();
                material.extension.texture_a_scale = splat_map.units_per_texture[0];
                material.base.normal_map_texture = splat_map.normal_textures[0].clone();

                material.extension.texture_b = splat_map.textures[1].clone();
                material.extension.texture_b_scale = splat_map.units_per_texture[1];
                material.extension.texture_b_normal = splat_map.normal_textures[1].clone();

                material.extension.texture_c = splat_map.textures[2].clone();
                material.extension.texture_c_scale = splat_map.units_per_texture[2];
                material.extension.texture_c_normal = splat_map.normal_textures[2].clone();

                material.extension.texture_d = splat_map.textures[3].clone();
                material.extension.texture_d_scale = splat_map.units_per_texture[3];
                material.extension.texture_d_normal = splat_map.normal_textures[3].clone();

                let texture = textures.add(Image::new(
                    Extent3d {
                        width: splat_map.resolution,
                        height: splat_map.resolution,
                        depth_or_array_layers: 1,
                    },
                    TextureDimension::D2,
                    splat_map.data,
                    TextureFormat::Rgba8Unorm,
                    RenderAssetUsages::RENDER_WORLD,
                ));
                material.extension.texture_map = texture;
            }

            commands.entity(entity).remove::<TextureGenerationTask>();
        }
    });
}

fn update_terrain_texture_maps(
    shape_modifier_query: Query<(
        &TextureModifierOperation,
        &ShapeModifier,
        (
            Option<&ModifierFalloffProperty>,
            Option<&TextureModifierFalloffProperty>,
            Option<&ModifierFalloffNoiseProperty>,
        ),
        Option<&ModifierStrengthLimitProperty>,
        &GlobalTransform,
    )>,
    spline_query: Query<(
        &TextureModifierOperation,
        &TerrainSplineCached,
        &TerrainSplineProperties,
        Option<&ModifierFalloffProperty>,
        Option<&TextureModifierFalloffProperty>,
    )>,
    mut tiles_query: Query<(
        Entity,
        &Heights,
        &Mesh3d,
        &Terrain,
        &Aabb,
        &TileBiomes,
        Option<&mut TextureGenerationTask>,
    )>,
    textures_generating: Query<(), With<TextureGenerationTask>>,
    texturing_settings: (
        Res<TerrainTexturingSettings>,
        Res<GlobalTexturingRules>,
    ),
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    tile_to_terrain: Res<TileToTerrain>,
    mut event_reader: EventReader<TerrainMeshRebuilt>,
    mut tile_generate_queue: ResMut<TerrainTextureRebuildQueue>,
    meshes: Res<Assets<Mesh>>,
    noise_resources: (
        Res<TerrainNoiseSettings>,
        Res<NoiseCache>,
        Res<NoiseIndexCache>
    ),
    mut cached_terrain_details: Local<Option<Arc<CachedTerrainDetails>>>,
    mut commands: Commands
) {
    let tile_size = terrain_settings.tile_size();
    let resolution = texturing_settings.0.resolution();
    let vertex_scale = (terrain_settings.edge_points - 1) as f32 / resolution as f32;

    if texturing_settings.0.is_changed() || texturing_settings.1.is_changed() || terrain_settings.is_changed() || cached_terrain_details.is_none() {
        let needs_noise = texturing_settings.1.rules.iter().any(|rule| rule.evaluators.iter().any(|evaulator| evaulator.needs_noise()));
        
        *cached_terrain_details = Some(Arc::new(CachedTerrainDetails {
            vertex_scale,
            tile_size,
            texturing_rules: if texturing_settings.1.rules.is_empty() {
                None
            } else {
                Some(texturing_settings.1.clone())
            },
            terrain_noise_settings: noise_resources.0.clone(),
            terrain_settings: terrain_settings.clone(),
            needs_noise,
        }));
    }
    let cached_terrain_details = cached_terrain_details.as_ref().unwrap();

    for TerrainMeshRebuilt(tile) in event_reader.read() {
        if !tile_generate_queue.0.contains(tile) {
            tile_generate_queue.0.push(*tile);
        }
    }

    if tile_generate_queue.is_empty() {
        return;
    }
    let thread_pool = AsyncComputeTaskPool::get();

    let tiles_to_generate = tile_generate_queue
        .count()
        .min(texturing_settings.0.max_texture_generation_tasks.get() as usize).saturating_sub(textures_generating.iter().len());

    let mut iter = tiles_query
        .iter_many_mut(
            tile_generate_queue
                .0
                .drain(..tiles_to_generate)
                .filter_map(|tile| tile_to_terrain.0.get(&tile))
                .flatten(),
        );

    while let Some((entity, heights, mesh, terrain_coordinate, aabb, tile_biomes, texture_generation_task)) = iter.fetch_next(){
        let Some(mesh) = meshes.get(mesh) else {
            return;
        };
        
        let terrain_translation =
            (terrain_coordinate.0 << terrain_settings.tile_size_power.get()).as_vec2();

        let min_height = aabb.center.y - aabb.half_extents.y;
        let max_height = aabb.center.y + aabb.half_extents.y;

        let normals = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .unwrap()
            .as_float3()
            .unwrap()
            .to_vec()
            .into_boxed_slice();

        let shapes = if let Some(shapes) = tile_to_modifier.shape.get(&terrain_coordinate.0) {
            shapes.iter().filter_map(|shape| {
                shape_modifier_query.get(shape.entity).ok().map(|(texture_modifier, shape_modifier, (
                    modifier_falloff,
                    texture_modifier_falloff,
                    modifier_falloff_noise,
                ), strength_limit, transform)| {
                    ShapeDefinition {
                        overlap_bits: shape.overlap_bits,
                        texture_modifier: texture_modifier.clone(),
                        shape_modifier: shape_modifier.clone(),
                        modifier_falloff: modifier_falloff.cloned(),
                        texture_modifier_falloff: texture_modifier_falloff.cloned(),
                        modifier_falloff_noise: modifier_falloff_noise.cloned(),
                        modifier_strength_limit: strength_limit.cloned(),
                        global_transform: *transform,
                    }
                })
            }).collect()
        } else {
            vec![]
        };
        let splines = if let Some(splines) = tile_to_modifier.splines.get(&terrain_coordinate.0) {
            splines.iter().filter_map(|entry| {
                spline_query.get(entry.entity).ok().map(|(texture_modifier, spline, properties, modifier_falloff, texture_modifier_falloff)| {
                    SplineDefinition {
                        overlap_bits: entry.overlap_bits,
                        texture_modifier: texture_modifier.clone(),
                        spline: spline.clone(),
                        spline_properties: properties.clone(),
                        modifier_falloff: modifier_falloff.cloned(),
                        texture_modifier_falloff: texture_modifier_falloff.cloned(),
                    }
                })
            }).collect()
        } else {
            vec![]
        };

        let task = thread_pool.spawn(
            generate_splat_map(
                resolution,
                cached_terrain_details.clone(),
                terrain_translation,
                normals,
                heights.clone(),
                min_height,
                max_height,
                tile_biomes.clone(),
                shapes,
                splines
            )
        );

        if let Some(mut texture_generation_task) = texture_generation_task {
            texture_generation_task.task = task;
        } else {
            commands.entity(entity).insert(TextureGenerationTask { task });
        }
    }
}

#[derive(Component)]
struct TextureGenerationTask {
    task: Task<SplatMap>
}

struct SplatMap {
    resolution: u32,
    data: Vec<u8>,

    textures: [Option<Handle<Image>>; 4],
    normal_textures: [Option<Handle<Image>>; 4],
    units_per_texture: [f32; 4],
}
impl SplatMap {
    fn get_texture_slot(
        &mut self,
        image: &Handle<Image>,
        normal: &Option<Handle<Image>>,
        units_per_texture: f32,
        tile_size: f32,
    ) -> Option<(usize, bool)> {
        let scale = 1.0 / (units_per_texture / tile_size);
        
        // Find the first matching or empty texture slot (& assign it to the input texture if applicable).
        let existing = self.textures.iter().enumerate().position(|(i, entry)| entry.as_ref().is_some_and(|entry| {
            entry == image && self.units_per_texture[i] == scale && self.normal_textures[i] == *normal
        }));
        if let Some(existing) = existing {
            return Some((existing, false));
        }

        for i in 0..4 {
            if self.textures[i].is_none() {
                self.textures[i] = Some(image.clone());
                self.normal_textures[i] = normal.clone();
                self.units_per_texture[i] = scale;

                return Some((i, true));
            }
        }

        None
    }
}

struct CachedTerrainDetails {
    vertex_scale: f32,
    tile_size: f32,
    texturing_rules: Option<GlobalTexturingRules>,
    terrain_noise_settings: TerrainNoiseSettings,
    terrain_settings: TerrainSettings,
    needs_noise: bool,
}

struct ShapeDefinition {
    overlap_bits: u64,
    texture_modifier: TextureModifierOperation,
    shape_modifier: ShapeModifier,
    modifier_falloff: Option<ModifierFalloffProperty>,
    texture_modifier_falloff: Option<TextureModifierFalloffProperty>,
    modifier_falloff_noise: Option<ModifierFalloffNoiseProperty>,
    modifier_strength_limit: Option<ModifierStrengthLimitProperty>,
    global_transform: GlobalTransform,
}

struct SplineDefinition {
    overlap_bits: u64,
    texture_modifier: TextureModifierOperation,
    spline: TerrainSplineCached,
    spline_properties: TerrainSplineProperties,
    modifier_falloff: Option<ModifierFalloffProperty>,
    texture_modifier_falloff: Option<TextureModifierFalloffProperty>,
}

async fn generate_splat_map(
    resolution: u32,
    cached_terrain_details: Arc<CachedTerrainDetails>,
    terrain_translation: Vec2,
    normals: Box<[[f32; 3]]>,
    heights: Heights,
    min_height: f32,
    max_height: f32,
    tile_biomes: TileBiomes,
    shapes: Vec<ShapeDefinition>,
    splines: Vec<SplineDefinition>
) -> SplatMap {
    let _span = info_span!("Generating Terrain Splat Map").entered();
    let mut noise_cache = NoiseCache::default();    
    let mut noise_index_cache = NoiseIndexCache::default();
    noise_index_cache.fill_cache(&cached_terrain_details.terrain_noise_settings, &mut noise_cache);

    let scale = cached_terrain_details.tile_size / resolution as f32;
    let inv_tile_size_scale = scale * (7.0 / cached_terrain_details.tile_size);
    let mut splat_map = SplatMap {
        resolution,
        data: vec![0; (resolution * resolution) as usize * 4],
        textures: [const { None }; 4],
        normal_textures: [const { None }; 4],
        units_per_texture: [0.0; 4],
    };

    if let Some(texturing_rules) = &cached_terrain_details.texturing_rules {
        let _span = info_span!("Apply global texturing rules.").entered();
        let mut data_samples = Vec::with_capacity(cached_terrain_details.terrain_noise_settings.data.len());
        let mut biome_samples = Vec::with_capacity(cached_terrain_details.terrain_noise_settings.biome.len());

        for rule in texturing_rules.rules.iter() {
            if (
                matches!(rule.evaulator_combinator, StrengthCombinator::Min | StrengthCombinator::Multiply) && !rule.evaluators.iter().all(|evaluator| evaluator.can_apply_to_tile(min_height, max_height, &tile_biomes))
                || !rule.evaluators.iter().any(|evaluator| evaluator.can_apply_to_tile(min_height, max_height, &tile_biomes))
            ) {
                continue;
            }

            let Some((texture_channel, is_new)) = splat_map.get_texture_slot(
                &rule.texture,
                &rule.normal_texture,
                rule.units_per_texture,
                cached_terrain_details.tile_size,
            ) else {
                info!("Hit max texture channels.");
                continue;
            };
            
            let mut applied = false;
            let needs_noise = cached_terrain_details.needs_noise && rule.evaluators.iter().any(|evaluator| evaluator.needs_noise()); 

            for (i, val) in splat_map.data.chunks_exact_mut(16).enumerate() {
                let true_i = (i * 4) as u32;
                let (x, z) = index_to_x_z_simd(UVec4::new(true_i, true_i + 1, true_i + 2, true_i + 3), resolution);

                let x_f = x.as_vec4() * cached_terrain_details.vertex_scale;
                let z_f = z.as_vec4() * cached_terrain_details.vertex_scale;

                let vertex_x = x_f.as_uvec4();
                let vertex_z = z_f.as_uvec4();

                let vertex_a = (vertex_z * cached_terrain_details.terrain_settings.edge_points as u32) + vertex_x;
                let vertex_b = vertex_a + 1;
                let vertex_c = vertex_a + cached_terrain_details.terrain_settings.edge_points as u32;
                let vertex_d = vertex_a + cached_terrain_details.terrain_settings.edge_points as u32 + 1;

                let local_x = x_f.fract();
                let local_z = z_f.fract();

                if needs_noise {
                    let world_x = x_f + Vec4::splat(terrain_translation.x);
                    let world_z = z_f + Vec4::splat(terrain_translation.y);

                    data_samples.clear();
                    cached_terrain_details.terrain_noise_settings.sample_data_simd(&noise_cache, &noise_index_cache, world_x, world_z, &mut data_samples);

                    biome_samples.clear();
                    cached_terrain_details.terrain_noise_settings.sample_biomes_simd(&noise_cache, &noise_index_cache, world_x, world_z, &data_samples, &mut biome_samples);
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
                splat_map.textures[texture_channel] = None;
                splat_map.normal_textures[texture_channel] = None;
            }
        }
    }
    
    // Secondly, set by shape-modifiers.
    {
        let _span = info_span!("Apply shape modifiers").entered();

        for entry in shapes.iter() {
            let Some((texture_channel, is_new)) = splat_map.get_texture_slot(
                &entry.texture_modifier.texture,
                &entry.texture_modifier.normal_texture,
                entry.texture_modifier.units_per_texture,
                cached_terrain_details.tile_size,
            ) else {
                info!("Hit max texture channels.");
                continue;
            };

            let mut applied = false;
            let (falloff, easing_function) = entry.texture_modifier_falloff.as_ref()
                .map(|falloff| (falloff.falloff, falloff.easing_function))
                .or(entry.modifier_falloff.as_ref()
                    .map(|falloff| (falloff.falloff, falloff.easing_function)))
                .unwrap_or((f32::EPSILON, EasingFunction::Linear));

            // Cache the noise index.
            let modifier_falloff_noise = entry.modifier_falloff_noise.as_ref().map(|falloff_noise| {
                (
                    falloff_noise,
                    noise_cache.get_simplex_index(falloff_noise.noise.seed),
                )
            });

            for (i, val) in splat_map.data.chunks_exact_mut(4).enumerate() {
                let (x, z) = index_to_x_z(i, resolution as usize);

                let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                let overlap_index = overlap_y * 8 + overlaps_x;
                if (entry.overlap_bits & 1 << overlap_index) == 0 {
                    continue;
                }

                let pixel_position = terrain_translation
                    + Vec2::new(x as f32 * scale, z as f32 * scale);

                let strength = calc_shape_modifier_strength(&noise_cache, falloff, modifier_falloff_noise, pixel_position, &entry.shape_modifier, entry.modifier_strength_limit.as_ref(), &entry.global_transform);
                let eased_strength = easing_function.ease(strength);

                // Apply texture.
                let scaled_strength = (eased_strength * 255.0) as u8;
                if scaled_strength > 0 {
                    apply_texture(val, texture_channel, scaled_strength);
                    applied = true;
                }
            }

            if is_new && !applied {
                splat_map.textures[texture_channel] = None;
                splat_map.normal_textures[texture_channel] = None;
            }
        }
    }

    {
        let _span = info_span!("Apply splines").entered();

        for entry in splines.iter() {
            let Some((texture_channel, is_new)) = splat_map.get_texture_slot(
                &entry.texture_modifier.texture,
                &entry.texture_modifier.normal_texture,
                entry.texture_modifier.units_per_texture,
                cached_terrain_details.tile_size,
            ) else {
                info!("Hit max texture channels.");
                continue;
            };
            let (falloff, easing_function) = entry.texture_modifier_falloff.as_ref()
                .map(|falloff| (falloff.falloff, falloff.easing_function))
                .or(entry.modifier_falloff.as_ref()
                    .map(|falloff| (falloff.falloff, falloff.easing_function)))
                .unwrap_or((f32::EPSILON, EasingFunction::Linear));

            let mut applied = false;

            for (i, val) in splat_map.data.chunks_exact_mut(4).enumerate() {
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

                for points in entry.spline.points.windows(2) {
                    let a_2d = points[0].xz();
                    let b_2d = points[1].xz();

                    let (new_distance, _) =
                        distance_squared_to_line_segment(a_2d, b_2d, vertex_position);

                    if new_distance < distance {
                        distance = new_distance;
                    }
                }

                let strength = (1.0
                    - ((distance.sqrt() - entry.spline_properties.half_width) / falloff))
                    .clamp(0.0, entry.texture_modifier.max_strength);
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
                splat_map.textures[texture_channel] = None;
                splat_map.normal_textures[texture_channel] = None;
            }
        }
    }

    splat_map
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
