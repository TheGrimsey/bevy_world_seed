use std::num::NonZeroU32;

use bevy::{
    app::{App, Plugin, PostUpdate, Startup},
    asset::{Asset, AssetApp, AssetServer, Assets, Handle},
    log::{info, info_span},
    math::{IVec2, Vec2, Vec3, Vec3Swizzles},
    pbr::{Material, MaterialPlugin},
    prelude::{
        default, AlphaMode, Commands, Component, Entity, EventReader, GlobalTransform, Image,
        IntoSystemConfigs, Local, Query, ReflectComponent, ReflectDefault, ReflectResource, Res,
        ResMut, Resource, With, Without,
    },
    reflect::Reflect,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat},
        texture::TextureFormatPixelInfo,
    },
};

use crate::{
    minimum_distance,
    modifiers::{
        Shape, ShapeModifier, TerrainSplineCached, TerrainSplineProperties, TileToModifierMapping,
    },
    terrain::{TerrainCoordinate, TileToTerrain},
    Heights, RebuildTile, TerrainSets, TerrainSettings,
};

pub struct TerrainTexturingPlugin(pub TerrainTexturingSettings);
impl Plugin for TerrainTexturingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TerrainMaterial>::default());

        app.insert_resource(self.0.clone());
        app.insert_resource(GlobalTexturingRules { rules: vec![] });

        app.register_asset_reflect::<TerrainMaterial>()
            .register_type::<TextureModifier>()
            .register_type::<GlobalTexturingRules>();

        app.add_systems(
            PostUpdate,
            (
                insert_texture_map,
                update_terrain_texture_maps.after(TerrainSets::Modifiers),
            )
                .chain(),
        );

        app.add_systems(Startup, insert_rules);
    }
}

fn insert_rules(mut texturing_rules: ResMut<GlobalTexturingRules>, asset_server: Res<AssetServer>) {
    texturing_rules.rules.push(TexturingRule {
        evaluator: TexturingRuleEvaluator::Above {
            height: 1.0,
            falloff: 2.0,
        },
        texture: asset_server.load("textures/brown_mud_leaves.jpg"),
    });
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
    pub max_texture_strength: f32,
}

#[derive(Reflect)]
enum TexturingRuleEvaluator {
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
}

#[derive(Reflect)]
struct TexturingRule {
    evaluator: TexturingRuleEvaluator,
    texture: Handle<Image>,
}

#[derive(Resource, Reflect)]
#[reflect(Resource)]
struct GlobalTexturingRules {
    rules: Vec<TexturingRule>,
}

// This struct defines the data that will be passed to your shader
#[derive(Asset, AsBindGroup, Default, Debug, Clone, Reflect)]
#[reflect(Default, Debug)]
pub(super) struct TerrainMaterial {
    #[texture(0)]
    #[sampler(1)]
    texture_map: Handle<Image>,
    #[texture(2)]
    #[sampler(3)]
    texture_a: Option<Handle<Image>>,
    #[texture(4)]
    #[sampler(5)]
    texture_b: Option<Handle<Image>>,
    #[texture(6)]
    #[sampler(7)]
    texture_c: Option<Handle<Image>>,
    #[texture(8)]
    #[sampler(9)]
    texture_d: Option<Handle<Image>>,
}
impl TerrainMaterial {
    pub fn clear_textures(&mut self) {
        self.texture_a = None;
        self.texture_b = None;
        self.texture_c = None;
        self.texture_d = None;
    }

    pub fn get_texture_slot(&mut self, image: &Handle<Image>) -> Option<usize> {
        /*
         *   There has to be a better way to do this, right?
         */

        if self.texture_a.as_ref().is_some_and(|entry| entry == image) {
            Some(0)
        } else if self.texture_b.as_ref().is_some_and(|entry| entry == image) {
            Some(1)
        } else if self.texture_c.as_ref().is_some_and(|entry| entry == image) {
            Some(2)
        } else if self.texture_d.as_ref().is_some_and(|entry| entry == image) {
            Some(3)
        } else if self.texture_a.is_none() {
            self.texture_a = Some(image.clone());

            Some(0)
        } else if self.texture_b.is_none() {
            self.texture_b = Some(image.clone());

            Some(1)
        } else if self.texture_c.is_none() {
            self.texture_c = Some(image.clone());

            Some(2)
        } else if self.texture_d.is_none() {
            self.texture_d = Some(image.clone());

            Some(3)
        } else {
            None
        }
    }
}

impl Material for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

fn insert_texture_map(
    texture_settings: Res<TerrainTexturingSettings>,
    mut commands: Commands,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>,
    query: Query<Entity, (With<Heights>, Without<Handle<TerrainMaterial>>)>,
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

        let material_handle = materials.add(material);

        commands.entity(entity).insert(material_handle);
    });
}

fn update_terrain_texture_maps(
    shape_modifier_query: Query<(&TextureModifier, &ShapeModifier, &GlobalTransform)>,
    spline_query: Query<(
        &TextureModifier,
        &TerrainSplineCached,
        &TerrainSplineProperties,
    )>,
    tiles_query: Query<(&Heights, &Handle<TerrainMaterial>, &TerrainCoordinate)>,
    texture_settings: Res<TerrainTexturingSettings>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    tile_to_terrain: Res<TileToTerrain>,
    mut event_reader: EventReader<RebuildTile>,
    mut tile_generate_queue: Local<Vec<IVec2>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>,
    texturing_rules: Res<GlobalTexturingRules>,
) {
    for RebuildTile(tile) in event_reader.read() {
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
    let vertex_scale = (terrain_settings.edge_length - 1) as f32 / resolution as f32;

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
        .for_each(|(heights, material, terrain_coordinate)| {
            let Some(material) = materials.get_mut(material) else {
                return;
            };
            let Some(texture) = images.get_mut(material.texture_map.id()) else {
                return;
            };

            texture.data.fill(0);
            material.clear_textures();

            let terrain_translation =
                (terrain_coordinate.0 << terrain_settings.tile_size_power).as_vec2();

            {
                let _span = info_span!("Apply global texturing rules.").entered();

                for rule in texturing_rules.rules.iter() {
                    let Some(texture_channel) = material.get_texture_slot(&rule.texture) else {
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
                            (vertex_z * terrain_settings.edge_length as usize) + vertex_x;
                        let vertex_b = vertex_a + 1;
                        let vertex_c = vertex_a + terrain_settings.edge_length as usize;
                        let vertex_d = vertex_a + terrain_settings.edge_length as usize + 1;

                        if vertex_a >= heights.0.len()
                            || vertex_b >= heights.0.len()
                            || vertex_c >= heights.0.len()
                            || vertex_d >= heights.0.len()
                        {
                            info!("HI?");
                        }

                        let height_at_position = get_height_at_position(
                            heights.0[vertex_a],
                            heights.0[vertex_b],
                            heights.0[vertex_c],
                            heights.0[vertex_d],
                            x_f - x_f.round(),
                            z_f - z_f.round(),
                        );

                        let strength = match rule.evaluator {
                            TexturingRuleEvaluator::Above { height, falloff } => {
                                if height_at_position >= height {
                                    1.0
                                } else {
                                    1.0 - ((height_at_position - height).abs() / falloff)
                                        .clamp(0.0, 1.0)
                                }
                            }
                            TexturingRuleEvaluator::Below { height, falloff } => 1.0,
                            TexturingRuleEvaluator::Between {
                                max_height,
                                min_height,
                                falloff,
                            } => 1.0,
                        };

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
                            material.get_texture_slot(&texture_modifier.texture)
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

                                    let pixel_position = terrain_translation
                                        + Vec2::new(x as f32 * scale, z as f32 * scale);

                                    let strength = (1.0
                                        - ((pixel_position.distance(shape_translation) - radius)
                                            / modifier.falloff)
                                            .clamp(0.0, 1.0))
                                    .min(texture_modifier.max_texture_strength);

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
                                        .min(texture_modifier.max_texture_strength);

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
                            material.get_texture_slot(&texture_modifier.texture)
                        else {
                            info!("Hit max texture channels.");
                            continue;
                        };

                        for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                            let x = i % resolution as usize;
                            let z = i / resolution as usize;

                            let local_vertex_position =
                                Vec2::new(x as f32 * scale, z as f32 * scale);
                            let overlaps = (local_vertex_position / tile_size * 7.0).as_ivec2();
                            let overlap_index = overlaps.y * 8 + overlaps.x;
                            if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                continue;
                            }

                            let vertex_position = terrain_translation + local_vertex_position;
                            let mut distance = f32::INFINITY;

                            for (a, b) in spline.points.iter().zip(spline.points.iter().skip(1)) {
                                let a_2d = a.xz();
                                let b_2d = b.xz();

                                let (new_distance, _) =
                                    minimum_distance(a_2d, b_2d, vertex_position);

                                if new_distance < distance {
                                    distance = new_distance;
                                }
                            }

                            let strength = (1.0
                                - ((distance.sqrt() - spline_properties.width)
                                    / spline_properties.falloff)
                                    .clamp(0.0, 1.0))
                            .min(texture_modifier.max_texture_strength);

                            // Apply texture.
                            apply_texture(val, texture_channel, strength);
                        }
                    }
                }
            }
        });
}

fn apply_texture(channels: &mut [u8], target_channel: usize, target_strength: f32) {
    // Problem: Must be fast. Must be understandable.

    // Idea: Try to apply the full strength. Removing from the highest if there is not enough.
    let strength = (target_strength * 255.0) as u8;

    // Don't decrease the strength of our channel.
    if channels[target_channel] < strength {
        if strength == 255 {
            channels.fill(0);
            channels[target_channel] = 255;
        } else {
            let total: u8 = channels.iter().sum();

            let remainder = 255 - total as i16 + channels[target_channel] as i16;

            if remainder >= 0 {
                channels[target_channel] = strength;
            } else {
                let mut to_remove = remainder.unsigned_abs() as u8;

                while to_remove > 0 {
                    let mut min_channel = 0;
                    let mut min = u8::MAX;
                    for (i, val) in channels.iter().enumerate() {
                        if i != target_channel && *val < min && *val > 0 {
                            min_channel = i;
                            min = *val;
                        }
                    }

                    let subtract = channels[min_channel].min(to_remove);
                    channels[min_channel] -= subtract;

                    to_remove -= subtract;
                }
            }
        }
    }
}

/*
*   A -- B
*   I    I
*   I    I
*   C -- D
*/

// TODO: Make into util function.
fn get_height_at_position(a: f32, b: f32, c: f32, d: f32, x: f32, y: f32) -> f32 {
    let a = Vec3::new(0.0, a, 0.0);
    let b = Vec3::new(1.0, b, 0.0);
    let c = Vec3::new(0.0, c, 1.0);
    let d = Vec3::new(1.0, d, 1.0);

    if x <= 0.5 && y <= 0.5 {
        closest_height_in_triangle(a, b, c, Vec3::new(x, 0.0, y))
    } else {
        closest_height_in_triangle(b, c, d, Vec3::new(x, 0.0, y))
    }
}

fn closest_height_in_triangle(a: Vec3, b: Vec3, c: Vec3, position: Vec3) -> f32 {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = position - a;

    let mut denom = v0.x * v1.z - v0.z * v1.x;

    let mut u = v1.z * v2.x - v1.x * v2.z;
    let mut v = v0.x * v2.z - v0.z * v2.x;

    if denom < 0.0 {
        denom = -denom;
        u = -u;
        v = -v;
    }

    a.y + (v0.y * u + v1.y * v) / denom
}
