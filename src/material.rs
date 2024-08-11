use std::num::NonZeroU32;

use bevy::{app::{App, Plugin, PostUpdate}, asset::{Asset, AssetApp, Assets, Handle}, log::info_span, math::{IVec2, Vec2, Vec3, Vec3Swizzles}, pbr::{Material, MaterialPlugin}, prelude::{default, AlphaMode, Commands, Component, Entity, EventReader, GlobalTransform, Image, Local, Query, ReflectDefault, Res, ResMut, Resource, With, Without}, reflect::Reflect, render::{render_asset::RenderAssetUsages, render_resource::{AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat}, texture::TextureFormatPixelInfo}};

use crate::{minimum_distance, modifiers::{Shape, ShapeModifier, TerrainSplineCached, TerrainSplineProperties, TileToModifierMapping}, terrain::{TerrainCoordinate, TileToTerrain}, Heights, RebuildTile, TerrainSettings};

pub struct TerrainTexturingPlugin;
impl Plugin for TerrainTexturingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            MaterialPlugin::<TerrainMaterial>::default()
        );

        app.insert_resource(TerrainTexturingSettings {
            texture_resolution_power: 6,
            max_tile_updates_per_frame: NonZeroU32::new(2).unwrap(),
        });

        app.register_asset_reflect::<TerrainMaterial>();

        app.add_systems(PostUpdate, (
            insert_texture_map,
            update_terrain_texture_maps
        ));
    }
}

#[derive(Resource)]
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

#[derive(Component)]
pub struct TextureModifier {
    pub texture: Handle<Image>,
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

/// The Material trait is very configurable, but comes with sensible defaults for all methods.
/// You only need to implement functions for features that need non-default behavior. See the Material api docs for details!
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
    query: Query<Entity, (With<Heights>, Without<Handle<TerrainMaterial>>)>
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
            RenderAssetUsages::all()
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
    spline_query: Query<(&TextureModifier, &TerrainSplineCached, &TerrainSplineProperties)>,
    tiles_query: Query<(&Handle<TerrainMaterial>, &TerrainCoordinate)>,
    texture_settings: Res<TerrainTexturingSettings>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    tile_to_terrain: Res<TileToTerrain>,
    mut event_reader: EventReader<RebuildTile>,
    mut tile_generate_queue: Local<Vec<IVec2>>,
    materials: Res<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>
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

    let tiles_to_generate = tile_generate_queue.len().min(texture_settings.max_tile_updates_per_frame.get() as usize);


    tiles_query.iter_many(tile_generate_queue.drain(..tiles_to_generate).filter_map(|tile| tile_to_terrain.0.get(&tile)).flatten()).for_each(|(material, terrain_coordinate)| {
        let Some(material) = materials.get(material) else {
            return;
        };
        let Some(texture) = images.get_mut(&material.texture_map) else {
            return;
        };

        let terrain_translation = (terrain_coordinate.0 << terrain_settings.tile_size_power).as_vec2();

        // Secondly, set by shape-modifiers.
        if let Some(shapes) = tile_to_modifier.shape.get(&terrain_coordinate.0) {
            let _span = info_span!("Apply shape modifiers").entered();
            
            shape_modifier_query.iter_many(shapes.iter()).for_each(|(texture_modifier, modifier, global_transform)| {
                let shape_translation = global_transform.translation().xz();
    
                match modifier.shape {
                    Shape::Circle { radius } => {
                        for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                            let x = i % resolution as usize;
                            let z = i / resolution as usize;
                        
                            let pixel_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
    
                            let strength = 1.0 - ((pixel_position.distance(shape_translation) - radius) / modifier.falloff).clamp(0.0, 1.0);
    
                            // Apply texture.
                        }
                    },
                    Shape::Rectangle { x, z } => {
                        let rect_min = Vec2::new(-x, -z) / 2.0;
                        let rect_max = Vec2::new(x, z) / 2.0;
    
                        for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                            let x = i % terrain_settings.edge_length as usize;
                            let z = i / terrain_settings.edge_length as usize;
                        
                            let pixel_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
                            let pixel_local = global_transform.affine().inverse().transform_point3(Vec3::new(pixel_position.x, 0.0, pixel_position.y)).xz();
    
                            let d_x = (rect_min.x - pixel_local.x).max(pixel_local.x - rect_max.x).max(0.0);
                            let d_y = (rect_min.y - pixel_local.y).max(pixel_local.y - rect_max.y).max(0.0);
                            let d_d = (d_x*d_x + d_y*d_y).sqrt();
    
                            let strength = 1.0 - (d_d / modifier.falloff).clamp(0.0, 1.0);
    
                            // Apply texture.
                        }
                    },
                }
            });
        }

        // Finally, set by splines.
        if let Some(splines) = tile_to_modifier.splines.get(&terrain_coordinate.0) {
            let _span = info_span!("Apply splines").entered();

            for entry in splines.iter() {
                if let Ok((texture_modifier, spline, spline_properties)) = spline_query.get(entry.entity) {
                    for (i, val) in texture.data.chunks_exact_mut(4).enumerate() {
                        let x = i % terrain_settings.edge_length as usize;
                        let z = i / terrain_settings.edge_length as usize;
        
                        let local_vertex_position = Vec2::new(x as f32 * scale, z as f32 * scale);
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
        
                            let (new_distance, _) = minimum_distance(a_2d, b_2d, vertex_position);
        
                            if new_distance < distance {
                                distance = new_distance;
                            }
                        }
        
                        let strength = 1.0 - ((distance.sqrt() - spline_properties.width) / spline_properties.falloff).clamp(0.0, 1.0);
                        
                        // Apply texture.
                    }
                }
            }
        }
    });
}