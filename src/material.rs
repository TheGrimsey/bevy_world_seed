use std::num::NonZeroU8;

use bevy::{
    app::{App, Plugin, PostUpdate},
    asset::{load_internal_asset, Asset, AssetApp, Assets, Handle},
    log::{info, info_span},
    math::{IVec2, Vec2, Vec3, Vec3Swizzles},
    pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial},
    prelude::{
        default, Commands, Component, Entity, EventReader, GlobalTransform, Image,
        IntoSystemConfigs, Mesh, Query, ReflectComponent, ReflectDefault, ReflectResource, Res,
        ResMut, Resource, Shader, With, Without,
    },
    reflect::Reflect,
    render::{
        primitives::Aabb,
        render_asset::RenderAssetUsages,
        render_resource::{AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat},
        texture::TextureFormatPixelInfo,
    },
};

use crate::{
    distance_squared_to_line_segment,
    easing::EasingFunction,
    meshing::TerrainMeshRebuilt,
    modifiers::{
        ModifierFalloffProperty, ShapeModifier, TerrainSplineCached, TerrainSplineProperties,
        TileToModifierMapping,
    },
    terrain::{Terrain, TileToTerrain},
    utils::{get_height_at_position_in_quad, get_normal_at_position_in_quad, index_to_x_z},
    Heights, TerrainSets, TerrainSettings,
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
#[derive(Reflect, Debug)]
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
}
impl TexturingRuleEvaluator {
    pub fn eval(&self, height_at_position: f32, angle_at_position: f32) -> f32 {
        match self {
            TexturingRuleEvaluator::Above { height, falloff } => {
                1.0 - ((height - height_at_position).max(0.0) / falloff.max(f32::EPSILON))
                    .clamp(0.0, 1.0)
            }
            TexturingRuleEvaluator::Below { height, falloff } => {
                1.0 - ((height_at_position - height).max(0.0) / falloff.max(f32::EPSILON))
                    .clamp(0.0, 1.0)
            }
            TexturingRuleEvaluator::Between {
                max_height,
                min_height,
                falloff,
            } => {
                let strength_below = 1.0
                    - ((min_height - height_at_position).max(0.0) / falloff.max(f32::EPSILON))
                        .clamp(0.0, 1.0);
                let strength_above = 1.0
                    - ((height_at_position - max_height).max(0.0) / falloff.max(f32::EPSILON))
                        .clamp(0.0, 1.0);

                strength_below.min(strength_above)
            }
            TexturingRuleEvaluator::AngleGreaterThan {
                angle_radians,
                falloff_radians,
            } => {
                1.0 - ((angle_radians - angle_at_position).max(0.0)
                    / falloff_radians.max(f32::EPSILON))
                .clamp(0.0, 1.0)
            }
            TexturingRuleEvaluator::AngleLessThan {
                angle_radians,
                falloff_radians,
            } => {
                1.0 - ((angle_at_position - angle_radians).max(0.0)
                    / falloff_radians.max(f32::EPSILON))
                .clamp(0.0, 1.0)
            },
            TexturingRuleEvaluator::AngleBetween { min_angle_radians, max_angle_radians, falloff_radians } => {
                let strength_above = 
                    1.0 - ((min_angle_radians - angle_at_position).max(0.0)
                        / falloff_radians.max(f32::EPSILON))
                    .clamp(0.0, 1.0);
                
                let strength_below = 
                    1.0 - ((max_angle_radians - angle_at_position).max(0.0)
                        / falloff_radians.max(f32::EPSILON))
                    .clamp(0.0, 1.0);

                strength_below.min(strength_above)
            }
        }
    }
}

#[derive(Reflect)]
pub struct TexturingRule {
    pub evaluator: TexturingRuleEvaluator,
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

// This struct defines the data that will be passed to your shader
#[derive(Asset, AsBindGroup, Default, Debug, Clone, Reflect)]
#[reflect(Default, Debug)]
pub(super) struct TerrainMaterial {
    #[texture(20)]
    #[sampler(21)]
    texture_map: Handle<Image>,

    #[texture(22)]
    #[sampler(23)]
    texture_a: Option<Handle<Image>>,
    #[texture(24)]
    #[sampler(25)]
    texture_a_normal: Option<Handle<Image>>,

    #[uniform(26)]
    texture_a_scale: f32,

    #[texture(27)]
    #[sampler(28)]
    texture_b: Option<Handle<Image>>,
    #[texture(29)]
    #[sampler(30)]
    texture_b_normal: Option<Handle<Image>>,

    #[uniform(31)]
    texture_b_scale: f32,

    #[texture(32)]
    #[sampler(33)]
    texture_c: Option<Handle<Image>>,
    #[texture(34)]
    #[sampler(35)]
    texture_c_normal: Option<Handle<Image>>,

    #[uniform(36)]
    texture_c_scale: f32,

    #[texture(37)]
    #[sampler(38)]
    texture_d: Option<Handle<Image>>,
    #[texture(39)]
    #[sampler(40)]
    texture_d_normal: Option<Handle<Image>>,

    #[uniform(41)]
    texture_d_scale: f32,
}
impl TerrainMaterial {
    fn clear_textures(&mut self) {
        self.texture_a = None;
        self.texture_b = None;
        self.texture_c = None;
        self.texture_d = None;

        self.texture_a_normal = None;
        self.texture_b_normal = None;
        self.texture_c_normal = None;
        self.texture_d_normal = None;
    }

    fn get_texture_slot(
        &mut self,
        image: &Handle<Image>,
        normal: &Option<Handle<Image>>,
        units_per_texture: f32,
        tile_size: f32,
    ) -> Option<usize> {
        let scale = 1.0 / (units_per_texture / tile_size);
        // Find the first matching or empty texture slot (& assign it to the input texture if applicable).
        if self.texture_a.as_ref().is_some_and(|entry| {
            entry == image && self.texture_a_scale == scale && self.texture_a_normal == *normal
        }) {
            Some(0)
        } else if self.texture_b.as_ref().is_some_and(|entry| {
            entry == image && self.texture_b_scale == scale && self.texture_b_normal == *normal
        }) {
            Some(1)
        } else if self.texture_c.as_ref().is_some_and(|entry| {
            entry == image && self.texture_c_scale == scale && self.texture_c_normal == *normal
        }) {
            Some(2)
        } else if self.texture_d.as_ref().is_some_and(|entry| {
            entry == image && self.texture_d_scale == scale && self.texture_d_normal == *normal
        }) {
            Some(3)
        } else if self.texture_a.is_none() {
            self.texture_a = Some(image.clone());
            self.texture_a_normal.clone_from(normal);
            self.texture_a_scale = scale;

            Some(0)
        } else if self.texture_b.is_none() {
            self.texture_b = Some(image.clone());
            self.texture_b_normal.clone_from(normal);
            self.texture_b_scale = scale;

            Some(1)
        } else if self.texture_c.is_none() {
            self.texture_c = Some(image.clone());
            self.texture_c_normal.clone_from(normal);
            self.texture_c_scale = scale;

            Some(2)
        } else if self.texture_d.is_none() {
            self.texture_d = Some(image.clone());
            self.texture_d_normal.clone_from(normal);
            self.texture_d_scale = scale;

            Some(3)
        } else {
            None
        }
    }
}

type TerrainMaterialExtended = ExtendedMaterial<StandardMaterial, TerrainMaterial>;

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

    query.iter().for_each(|entity| {
        let image = Image::new(
            Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![0; texture_format.pixel_size() * resolution as usize * resolution as usize],
            texture_format,
            RenderAssetUsages::all(),
        );

        let image_handle = images.add(image);

        let material = TerrainMaterial {
            texture_map: image_handle,
            ..default()
        };

        let material_handle = materials.add(TerrainMaterialExtended {
            base: StandardMaterial {
                perceptual_roughness: 1.0,
                reflectance: 0.0,
                ..default()
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
    )>,
    texture_settings: Res<TerrainTexturingSettings>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    tile_to_terrain: Res<TileToTerrain>,
    mut event_reader: EventReader<TerrainMeshRebuilt>,
    mut tile_generate_queue: ResMut<TerrainTextureRebuildQueue>,
    mut materials: ResMut<Assets<TerrainMaterialExtended>>,
    mut images: ResMut<Assets<Image>>,
    meshes: Res<Assets<Mesh>>,
    texturing_rules: Res<GlobalTexturingRules>,
) {
    for TerrainMeshRebuilt(tile) in event_reader.read() {
        if !tile_generate_queue.0.contains(tile) {
            tile_generate_queue.0.push(*tile);
        }
    }

    if tile_generate_queue.is_empty() {
        return;
    }

    let tile_size = terrain_settings.tile_size();
    let resolution = texture_settings.resolution();
    let scale = tile_size / resolution as f32;
    let vertex_scale = (terrain_settings.edge_points - 1) as f32 / resolution as f32;
    let inv_tile_size_scale = scale * (7.0 / tile_size);

    let tiles_to_generate = tile_generate_queue
        .count()
        .min(texture_settings.max_tile_updates_per_frame.get() as usize);

    tiles_query
        .iter_many(
            tile_generate_queue
                .0
                .drain(..tiles_to_generate)
                .filter_map(|tile| tile_to_terrain.0.get(&tile))
                .flatten(),
        )
        .for_each(|(heights, material, mesh, terrain_coordinate)| {
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
            material.extension.clear_textures();

            let terrain_translation =
                (terrain_coordinate.0 << terrain_settings.tile_size_power.get()).as_vec2();

            if !texturing_rules.rules.is_empty() {
                let _span = info_span!("Apply global texturing rules.").entered();

                for rule in texturing_rules.rules.iter() {
                    let Some(texture_channel) = material.extension.get_texture_slot(
                        &rule.texture,
                        &rule.normal_texture,
                        rule.units_per_texture,
                        tile_size,
                    ) else {
                        info!("Hit max texture channels.");
                        return;
                    };
                    let normals = mesh
                        .attribute(Mesh::ATTRIBUTE_NORMAL)
                        .unwrap()
                        .as_float3()
                        .unwrap();

                    for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                        let (x, z) = index_to_x_z(i, resolution as usize);

                        let x_f = x as f32 * vertex_scale;
                        let z_f = z as f32 * vertex_scale;

                        let vertex_x = x_f as usize;
                        let vertex_z = z_f as usize;

                        let vertex_a =
                            (vertex_z * terrain_settings.edge_points as usize) + vertex_x;
                        let vertex_b = vertex_a + 1;
                        let vertex_c = vertex_a + terrain_settings.edge_points as usize;
                        let vertex_d = vertex_a + terrain_settings.edge_points as usize + 1;

                        let local_x = x_f - x_f.floor();
                        let local_z = z_f - z_f.floor();

                        // TODO: We are doing this redundantly for each rule, where a single rule can only use one of these.

                        // Skip the bounds checks.
                        let height_at_position = unsafe {
                            get_height_at_position_in_quad(
                                *heights.0.get_unchecked(vertex_a),
                                *heights.0.get_unchecked(vertex_b),
                                *heights.0.get_unchecked(vertex_c),
                                *heights.0.get_unchecked(vertex_d),
                                local_x,
                                local_z,
                            )
                        };

                        // Skip the bounds checks.
                        let normal_at_position = unsafe {
                            get_normal_at_position_in_quad(
                                (*normals.get_unchecked(vertex_a)).into(),
                                (*normals.get_unchecked(vertex_b)).into(),
                                (*normals.get_unchecked(vertex_c)).into(),
                                (*normals.get_unchecked(vertex_d)).into(),
                                local_x,
                                local_z,
                            )
                        };
                        let normal_angle = normal_at_position.normalize().dot(Vec3::Y).acos();

                        let strength = rule.evaluator.eval(height_at_position, normal_angle);

                        // Apply texture.
                        apply_texture(val, texture_channel, strength);
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
                        let Some(texture_channel) = material.extension.get_texture_slot(
                            &texture_modifier.texture,
                            &texture_modifier.normal_texture,
                            texture_modifier.units_per_texture,
                            tile_size,
                        ) else {
                            info!("Hit max texture channels.");
                            return;
                        };
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
                                    apply_texture(val, texture_channel, eased_strength);
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
                                    apply_texture(val, texture_channel, eased_strength);
                                }
                            }
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
                        let Some(texture_channel) = material.extension.get_texture_slot(
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
                            apply_texture(val, texture_channel, eased_strength);
                        }
                    }
                }
            }
        });
}

pub fn apply_texture(channels: &mut [u8], target_channel: usize, target_strength: f32) {
    // Problem: Must be fast. Must be understandable.

    // Idea: Try to apply the full strength. Removing from the highest if there is not enough.
    let strength = (target_strength * 255.0) as u8;

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
