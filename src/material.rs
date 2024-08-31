use std::num::NonZeroU32;

use bevy::{
    app::{App, Plugin, PostUpdate},
    asset::{load_internal_asset, Asset, AssetApp, Assets, Handle},
    log::{info, info_span},
    math::{IVec2, Vec2, Vec3, Vec3Swizzles},
    pbr::{ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline, MaterialPlugin, StandardMaterial, PBR_PREPASS_SHADER_HANDLE},
    prelude::{
        default, Commands, Component, Entity, EventReader, GlobalTransform, Image, IntoSystemConfigs, Local, Mesh, Query, ReflectComponent, ReflectDefault, ReflectResource, Res, ResMut, Resource, Shader, With, Without
    },
    reflect::Reflect,
    render::{
        mesh::MeshVertexAttribute, primitives::Aabb, render_asset::RenderAssetUsages, render_resource::{AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat, VertexFormat}, texture::TextureFormatPixelInfo
    },
};

use crate::{
    distance_to_line_segment, meshing::TerrainMeshRebuilt, modifiers::{
        Shape, ShapeModifier, TerrainSpline, TerrainSplineCached, TileToModifierMapping
    }, terrain::{TerrainCoordinate, TileToTerrain}, utils::{get_height_at_position, get_normal_at_position}, Heights, TerrainSets, TerrainSettings
};

pub const TERRAIN_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(138167552981664683109966343978676199666);
pub const TERRAIN_VERTEX_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(138167552981664683109966343978676199676);

pub struct TerrainTexturingPlugin(pub TerrainTexturingSettings);
impl Plugin for TerrainTexturingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TerrainMaterialExtended>::default());

        app.insert_resource(self.0.clone());
        app.insert_resource(GlobalTexturingRules { rules: vec![] });

        app.register_asset_reflect::<TerrainMaterial>()
            .register_type::<TextureModifier>()
            .register_type::<GlobalTexturingRules>();

        
            load_internal_asset!(app, TERRAIN_SHADER_HANDLE, "terrain.wgsl", Shader::from_wgsl);
            load_internal_asset!(app, TERRAIN_VERTEX_SHADER_HANDLE, "terrain.v.wgsl", Shader::from_wgsl);

        app.add_systems(
            PostUpdate,
            (
                insert_rendering_components,
                update_terrain_texture_maps.after(TerrainSets::Modifiers),
            )
                .chain(),
        );
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
    pub texture_resolution_power: u8,

    pub max_tile_updates_per_frame: NonZeroU32,
}
impl TerrainTexturingSettings {
    pub fn resolution(&self) -> u32 {
        1 << self.texture_resolution_power as u32
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct TextureModifier {
    pub texture: Handle<Image>,
    pub max_strength: f32,
    /// Represents how much the texture will tile in a single terrain tile.
    /// 
    /// `1.0` means the texture will cover the entire tile. `2.0` means the texture will repeat twice in each direction. 
    pub tiling_factor: f32
}

#[derive(Reflect, Debug)]
pub enum TexturingRuleEvaluator {
    Above {
        height: f32,
        falloff: f32,
    },
    Below {
        height: f32,
        falloff: f32,
    },
    Between {
        max_height: f32,
        min_height: f32,
        falloff: f32,
    },
    AngleGreaterThan {
        angle_radians: f32,
    },
    AngleLessThan {
        angle_radians: f32,
    }
}
impl TexturingRuleEvaluator {
    pub fn eval(&self, height_at_position: f32, angle: f32) -> f32 {
        match self {
            TexturingRuleEvaluator::Above { height, falloff } => {
                1.0 - ((height - height_at_position).max(0.0) / falloff).clamp(0.0, 1.0)
            }
            TexturingRuleEvaluator::Below { height, falloff } => {
                1.0 - ((height - height_at_position).max(0.0) / falloff).clamp(0.0, 1.0)
            },
            TexturingRuleEvaluator::Between {
                max_height,
                min_height,
                falloff,
            } => {
                let strength_below = 1.0 - ((min_height - height_at_position).max(0.0) / falloff).clamp(0.0, 1.0);
                let strength_above = 1.0 - ((height_at_position - max_height).max(0.0) / falloff).clamp(0.0, 1.0);
                
                strength_below.min(strength_above)
            },
            TexturingRuleEvaluator::AngleGreaterThan { angle_radians } => {
                if angle >= *angle_radians {
                    1.0
                } else {
                    0.0
                }
            },
            TexturingRuleEvaluator::AngleLessThan { angle_radians } => {
                if angle < *angle_radians {
                    1.0
                } else {
                    0.0
                }
            },
        }
    }
}

#[derive(Reflect)]
pub struct TexturingRule {
    pub evaluator: TexturingRuleEvaluator,
    pub texture: Handle<Image>,
    /// Represents how much the texture will tile in a single terrain tile.
    /// 
    /// `1.0` means the texture will cover the entire tile. `2.0` means the texture will repeat twice in each direction. 
    pub tiling_factor: f32
}

#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct GlobalTexturingRules {
    pub rules: Vec<TexturingRule>,
}

// This struct defines the data that will be passed to your shader
#[derive(Asset, AsBindGroup, Default, Debug, Clone, Reflect)]
#[reflect(Default, Debug)]
pub(super) struct TerrainMaterial {
    #[uniform(34)]
    tile_size: f32,
    #[uniform(35)]
    edge_points: u32,

    #[texture(20)]
    #[sampler(21)]
    texture_map: Handle<Image>,
    #[texture(22)]
    #[sampler(23)]
    texture_a: Option<Handle<Image>>,

    #[uniform(24)]
    texture_a_scale: f32,

    #[texture(25)]
    #[sampler(26)]
    texture_b: Option<Handle<Image>>,
    
    #[uniform(27)]
    texture_b_scale: f32,

    #[texture(28)]
    #[sampler(29)]
    texture_c: Option<Handle<Image>>,
    
    #[uniform(30)]
    texture_c_scale: f32,

    #[texture(31)]
    #[sampler(32)]
    texture_d: Option<Handle<Image>>,
    
    #[uniform(33)]
    texture_d_scale: f32,
}
impl TerrainMaterial {
    pub fn clear_textures(&mut self) {
        self.texture_a = None;
        self.texture_b = None;
        self.texture_c = None;
        self.texture_d = None;
    }

    pub fn get_texture_slot(&mut self, image: &Handle<Image>, scale: f32) -> Option<usize> {
        /*
         *   There has to be a better way to do this, right?
         */

        if self.texture_a.as_ref().is_some_and(|entry| entry == image && self.texture_a_scale == scale) {
            Some(0)
        } else if self.texture_b.as_ref().is_some_and(|entry| entry == image && self.texture_b_scale == scale) {
            Some(1)
        } else if self.texture_c.as_ref().is_some_and(|entry| entry == image && self.texture_c_scale == scale) {
            Some(2)
        } else if self.texture_d.as_ref().is_some_and(|entry| entry == image && self.texture_d_scale == scale) {
            Some(3)
        } else if self.texture_a.is_none() {
            self.texture_a = Some(image.clone());
            self.texture_a_scale = scale;

            Some(0)
        } else if self.texture_b.is_none() {
            self.texture_b = Some(image.clone());
            self.texture_b_scale = scale;

            Some(1)
        } else if self.texture_c.is_none() {
            self.texture_c = Some(image.clone());
            self.texture_c_scale = scale;

            Some(2)
        } else if self.texture_d.is_none() {
            self.texture_d = Some(image.clone());
            self.texture_d_scale = scale;

            Some(3)
        } else {
            None
        }
    }
}

/*
*   This is likely overkill.
*   We should figure out a better way to get shadows and lightning than this.
*/
pub const ATTRIBUTE_HEIGHTS: MeshVertexAttribute =
    MeshVertexAttribute::new("Height", 724683397550405073, VertexFormat::Float32);

impl MaterialExtension for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        TERRAIN_SHADER_HANDLE.into()
    }

    fn vertex_shader() -> ShaderRef {
        TERRAIN_VERTEX_SHADER_HANDLE.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        TERRAIN_VERTEX_SHADER_HANDLE.into()
    }

    fn deferred_vertex_shader() -> ShaderRef {
        TERRAIN_VERTEX_SHADER_HANDLE.into()
    }

    fn specialize(
            _pipeline: &MaterialExtensionPipeline,
            descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
            layout: &bevy::render::mesh::MeshVertexBufferLayoutRef,
            _key: MaterialExtensionKey<Self>,
        ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            ATTRIBUTE_HEIGHTS.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        
        descriptor.vertex.shader_defs.push("VERTEX_NORMALS".into());
        descriptor.vertex.shader_defs.push("VERTEX_UVS_A".into());

        if let Some(fragment) = descriptor.fragment.as_mut() {
            let shader_defs = &mut fragment.shader_defs;
            shader_defs.push("VERTEX_UVS_A".into());
            shader_defs.push("VERTEX_NORMALS".into());
        }

        Ok(())
    }
}

type TerrainMaterialExtended = ExtendedMaterial<StandardMaterial, TerrainMaterial>;

fn insert_rendering_components(
    terrain_settings: Res<TerrainSettings>,
    texture_settings: Res<TerrainTexturingSettings>,
    mut commands: Commands,
    mut materials: ResMut<Assets<TerrainMaterialExtended>>,
    mut images: ResMut<Assets<Image>>,
    query: Query<Entity, (With<Heights>, Without<Handle<TerrainMaterialExtended>>)>,
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
            edge_points: terrain_settings.edge_points as u32 - 1,
            tile_size: terrain_settings.tile_size(),
            ..default()
        };
        let standard_material = StandardMaterial {
            perceptual_roughness: 1.0,
            ..default()
        };
        let material_handle = materials.add(TerrainMaterialExtended {
            base: standard_material,
            extension: material,
        });

        commands.entity(entity).insert((
            material_handle,
            Aabb::default()
        ));
    });
}

fn update_terrain_texture_maps(
    shape_modifier_query: Query<(&TextureModifier, &ShapeModifier, &GlobalTransform)>,
    spline_query: Query<(
        &TextureModifier,
        &TerrainSplineCached,
        &TerrainSpline,
    )>,
    tiles_query: Query<(&Heights, &Handle<TerrainMaterialExtended>, &Handle<Mesh>, &TerrainCoordinate)>,
    texture_settings: Res<TerrainTexturingSettings>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    tile_to_terrain: Res<TileToTerrain>,
    mut event_reader: EventReader<TerrainMeshRebuilt>,
    mut tile_generate_queue: Local<Vec<IVec2>>,
    mut materials: ResMut<Assets<TerrainMaterialExtended>>,
    mut images: ResMut<Assets<Image>>,
    meshes: Res<Assets<Mesh>>,
    texturing_rules: Res<GlobalTexturingRules>,
) {
    for TerrainMeshRebuilt(tile) in event_reader.read() {
        if !tile_generate_queue.contains(tile) {
            tile_generate_queue.push(*tile);
        }
    }

    if tile_generate_queue.is_empty() {
        return;
    }

    let tile_size = terrain_settings.tile_size();
    let resolution = texture_settings.resolution();
    let scale = tile_size / resolution as f32;
    let vertex_scale = (terrain_settings.edge_points - 1) as f32 / resolution as f32;
    let inv_tile_size_scale =  scale * (7.0 / tile_size);

    let tiles_to_generate = tile_generate_queue
        .len()
        .min(texture_settings.max_tile_updates_per_frame.get() as usize);

    tiles_query
        .iter_many(
            tile_generate_queue
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
                    let Some(texture_channel) = material.extension.get_texture_slot(&rule.texture, rule.tiling_factor) else {
                        info!("Hit max texture channels.");
                        return;
                    };

                    for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                        let x = i % resolution as usize;
                        let z = i / resolution as usize;

                        let x_f = x as f32 * vertex_scale;
                        let z_f = z as f32 * vertex_scale;

                        let vertex_x = x_f as usize;
                        let vertex_z = z_f as usize;

                        let vertex_a =
                            (vertex_z * terrain_settings.edge_points as usize) + vertex_x;
                        let vertex_b = vertex_a + 1;
                        let vertex_c = vertex_a + terrain_settings.edge_points as usize;
                        let vertex_d = vertex_a + terrain_settings.edge_points as usize + 1;

                        // TODO: We are doing this redundantly for each rule, where a single rule can only use one of these.

                        let local_x = x_f - x_f.round();
                        let local_z = z_f - z_f.round();

                        let height_at_position = unsafe { get_height_at_position(
                            *heights.0.get_unchecked(vertex_a),
                            *heights.0.get_unchecked(vertex_b),
                            *heights.0.get_unchecked(vertex_c),
                            *heights.0.get_unchecked(vertex_d),
                            local_x,
                            z_f - z_f.round(),
                        )};

                        let normals = mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap().as_float3().unwrap();
                        let normal_at_position = unsafe { get_normal_at_position(
                            (*normals.get_unchecked(vertex_a)).into(),
                            (*normals.get_unchecked(vertex_b)).into(),
                            (*normals.get_unchecked(vertex_c)).into(),
                            (*normals.get_unchecked(vertex_d)).into(),
                            local_x,
                            local_z,
                        )};
                        let normal_angle = normal_at_position.dot(Vec3::Y).acos();

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
                    if let Ok((texture_modifier, modifier, global_transform)) =
                        shape_modifier_query.get(entry.entity)
                    {
                        let Some(texture_channel) =
                            material.extension.get_texture_slot(&texture_modifier.texture, texture_modifier.tiling_factor)
                        else {
                            info!("Hit max texture channels.");
                            return;
                        };
                        let shape_translation = global_transform.translation().xz();

                        match modifier.shape {
                            Shape::Circle { radius } => {
                                for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                                    let x = i % resolution as usize;
                                    let z = i / resolution as usize;
                                    
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
                                            / modifier.falloff)
                                            .clamp(0.0, 1.0))
                                    .min(texture_modifier.max_strength);

                                    // Apply texture.
                                    apply_texture(val, texture_channel, strength);
                                }
                            }
                            Shape::Rectangle { x, z } => {
                                let rect_min = Vec2::new(-x, -z);
                                let rect_max = Vec2::new(x, z);

                                for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                                    let x = i % resolution as usize;
                                    let z = i / resolution as usize;
                                    
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

                                    let strength = (1.0 - (d_d / modifier.falloff).clamp(0.0, 1.0))
                                        .min(texture_modifier.max_strength);

                                    // Apply texture.
                                    apply_texture(val, texture_channel, strength);
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
                    if let Ok((texture_modifier, spline, spline_properties)) =
                        spline_query.get(entry.entity)
                    {
                        let Some(texture_channel) =
                            material.extension.get_texture_slot(&texture_modifier.texture, texture_modifier.tiling_factor)
                        else {
                            info!("Hit max texture channels.");
                            continue;
                        };

                        for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                            let x = i % resolution as usize;
                            let z = i / resolution as usize;
                            
                            let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                            let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                            let overlap_index = overlap_y * 8 + overlaps_x;
                            if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                continue;
                            }
                        
                            let vertex_position = terrain_translation + Vec2::new(x as f32, z as f32) * scale;
                            let mut distance = f32::INFINITY;

                            for points in spline.points.windows(2) {
                                let a_2d = points[0].xz();
                                let b_2d = points[1].xz();
            
                                let (new_distance, _) = distance_to_line_segment(a_2d, b_2d, vertex_position);
            
                                if new_distance < distance {
                                    distance = new_distance;
                                }
                            }

                            let strength = (1.0
                                - ((distance.sqrt() - spline_properties.width)
                                    / spline_properties.falloff)
                                    .clamp(0.0, 1.0))
                            .min(texture_modifier.max_strength);

                            // Apply texture.
                            apply_texture(val, texture_channel, strength);
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
            let total: i16 = channels[0] as i16 + channels[1] as i16 + channels[2] as i16 + channels[3] as i16;

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
