#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::num::NonZeroU32;

use bevy::{
    app::{App, Plugin, PostUpdate}, log::info_span, math::{FloatExt, IVec2, Vec2, Vec3, Vec3Swizzles}, prelude::{resource_changed, Component, Event, EventReader, IntoSystemConfigs, Local, Query, ReflectDefault, ReflectResource, Res, Resource, SystemSet, TransformSystem}, reflect::Reflect, transform::components::GlobalTransform
};
use debug_draw::TerrainDebugDrawPlugin;
#[cfg(feature = "rendering")]
use material::{TerrainTexturingPlugin, TerrainTexturingSettings};
#[cfg(feature = "rendering")]
use meshing::TerrainMeshingPlugin;
use noise::{NoiseFn, Simplex};
use modifiers::{update_shape_modifier_aabb, update_terrain_spline_aabb, update_terrain_spline_cache, update_tile_modifier_priorities, ModifierOperation, ModifierPriority, Shape, ShapeModifier, TerrainSpline, TerrainSplineCached, TerrainSplineProperties, TerrainTileAabb, TileToModifierMapping};
use terrain::{update_tiling, TerrainCoordinate, TileToTerrain};

pub mod modifiers;
pub mod terrain;

#[cfg(feature = "rendering")]
mod debug_draw;
#[cfg(feature = "rendering")]
mod meshing;
#[cfg(feature = "rendering")]
pub mod material;

/// System sets containing the crate's systems.
#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
pub enum TerrainSets {
    Modifiers,
    Heights,
}


pub struct TerrainPlugin {
    pub noise_settings: Option<TerrainNoiseLayers>,
    pub terrain_settings: TerrainSettings,
    #[cfg(feature = "rendering")]
    pub texturing_settings: TerrainTexturingSettings,
    #[cfg(feature = "rendering")]
    pub debug_draw: bool
}
impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "rendering")]
        {
            app.add_plugins((
                TerrainMeshingPlugin,
                TerrainTexturingPlugin(self.texturing_settings.clone()),
            ));

            if self.debug_draw {
                app.add_plugins(TerrainDebugDrawPlugin);
            }
        }

        app.add_systems(PostUpdate, (
            update_tiling.before(update_terrain_heights),
            (
                (
                    (
                        (
                            update_terrain_spline_cache,
                            update_terrain_spline_aabb
                        ).chain(),
                        update_shape_modifier_aabb,
                    ),
                    update_tile_modifier_priorities.run_if(resource_changed::<TileToModifierMapping>),
                ).chain().in_set(TerrainSets::Modifiers),
                update_terrain_heights.in_set(TerrainSets::Heights)
            ).chain()
        ).after(TransformSystem::TransformPropagate));

        app.insert_resource(self.noise_settings.clone().unwrap_or_default());
        app.insert_resource(self.terrain_settings.clone());

        app
            .init_resource::<TileToModifierMapping>()
            .init_resource::<TileToTerrain>()

            .register_type::<TerrainSpline>()
            .register_type::<TerrainSplineCached>()
            .register_type::<TerrainNoiseLayer>()
            .register_type::<TerrainNoiseLayers>()
            .register_type::<TerrainTileAabb>()
            .register_type::<TerrainSplineProperties>()
            .register_type::<TerrainCoordinate>()
            .register_type::<ShapeModifier>()
            .register_type::<ModifierOperation>()
            .register_type::<ModifierPriority>()

            .add_event::<RebuildTile>();
    }
}

/// Cache of Simplex noise instances & which seeds they map to.
#[derive(Default)]
struct NoiseCache {
    /// Seeds of the simplex noises.
    /// 
    /// These are separated from the noises for cache coherency. Simplex is a big struct.
    seeds: Vec<u32>,
    noises: Vec<Simplex>
}
impl NoiseCache {
    fn get(&mut self, seed: u32) -> &Simplex {
        if let Some(index) = self.seeds.iter().position(|existing_seed| *existing_seed == seed) {
            &self.noises[index]
        } else {
            self.seeds.push(seed);
            self.noises.push(Simplex::new(seed));
    
            self.noises.last().unwrap()
        }
    }
}

#[derive(Reflect, Clone)]
#[reflect(Default)]
pub struct TerrainNoiseLayer {
    pub height_scale: f32,
    pub planar_scale: f32,
    pub seed: u32,
}
impl TerrainNoiseLayer {
    pub fn get(&self, x: f32, z: f32, noise: &Simplex) -> f32 {
        noise.get([(x * self.planar_scale) as f64, (z  * self.planar_scale) as f64]) as f32 * self.height_scale
    }
}
impl Default for TerrainNoiseLayer {
    fn default() -> Self {
        Self { height_scale: 1.0, planar_scale: 1.0, seed: 1 }
    }
}

#[derive(Resource, Reflect, Clone, Default)]
#[reflect(Resource)]
pub struct TerrainNoiseLayers {
    pub layers: Vec<TerrainNoiseLayer>
}

#[derive(Resource, Reflect, Clone)]
#[reflect(Resource)]
pub struct TerrainSettings {
    pub tile_size_power: u32,
    pub edge_length: u32,
    pub max_tile_updates_per_frame: NonZeroU32,
    /// Points closer than this square distance are removed. 
    pub max_spline_simplification_distance: f32,
}
impl TerrainSettings {
    pub fn tile_size(&self) -> f32 {
        (1 << self.tile_size_power) as f32
    }
}

#[derive(Event)]
struct RebuildTile(IVec2);

#[derive(Component)]
pub struct Heights(pub Box<[f32]>);

fn update_terrain_heights(
    terrain_noise_layers: Res<TerrainNoiseLayers>,
    shape_modifier_query: Query<(&ShapeModifier, &ModifierOperation, &GlobalTransform)>,
    spline_query: Query<(&TerrainSplineCached, &TerrainSplineProperties)>,
    mut heights: Query<(&mut Heights, &TerrainCoordinate)>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    tile_to_terrain: Res<TileToTerrain>,
    mut tile_generate_queue: Local<Vec<IVec2>>,
    mut noise_cache: Local<NoiseCache>,
    mut event_reader: EventReader<RebuildTile>,
) {
    for RebuildTile(tile) in event_reader.read() {
        if !tile_generate_queue.contains(tile) {
            tile_generate_queue.push(*tile);
        }
    }

    if tile_generate_queue.is_empty() {
        return;
    }

    let scale = terrain_settings.tile_size() / (terrain_settings.edge_length - 1) as f32;

    let tile_size = terrain_settings.tile_size();
    let inv_tile_size_scale =  scale * (7.0 / tile_size);

    let tiles_to_generate = tile_generate_queue.len().min(terrain_settings.max_tile_updates_per_frame.get() as usize);

    let mut iter = heights.iter_many_mut(tile_generate_queue.drain(..tiles_to_generate).filter_map(|tile| tile_to_terrain.0.get(&tile)).flatten());

    while let Some((mut heights, terrain_coordinate)) = iter.fetch_next() {
        let terrain_translation = (terrain_coordinate.0 << terrain_settings.tile_size_power).as_vec2();

        // Clear heights.
        heights.0.fill(0.0);

        // First, set by noise.
        {
            let _span = info_span!("Apply noise").entered();
            for (i, val) in heights.0.iter_mut().enumerate() {
                let x = i % terrain_settings.edge_length as usize;
                let z = i / terrain_settings.edge_length as usize;
                
                let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
    
                for noise_layer in terrain_noise_layers.layers.iter() {
                    *val += noise_layer.get(vertex_position.x, vertex_position.y, noise_cache.get(noise_layer.seed));
                }

            }
        }

        // Secondly, set by shape-modifiers.
        if let Some(shapes) = tile_to_modifier.shape.get(&terrain_coordinate.0) {
            let _span = info_span!("Apply shape modifiers").entered();
            
            for entry in shapes.iter() {
                if let Ok((modifier, operation, global_transform)) = shape_modifier_query.get(entry.entity) {
                    let shape_translation = global_transform.translation().xz();
                    
                    match modifier.shape {
                        Shape::Circle { radius } => {
                            for (i, val) in heights.0.iter_mut().enumerate() {
                                let x = i % terrain_settings.edge_length as usize;
                                let z = i / terrain_settings.edge_length as usize;

                                let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                                let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                                let overlap_index = overlap_y * 8 + overlaps_x;
                                if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                    continue;
                                }
                            
                                let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
                            
                                let strength = 1.0 - ((vertex_position.distance(shape_translation) - radius) / modifier.falloff).clamp(0.0, 1.0);
                            
                                *val = apply_modifier(modifier, operation, vertex_position, shape_translation, *val, global_transform, strength, &mut noise_cache, false);
                            }
                        },
                        Shape::Rectangle { x, z } => {
                            let rect_min = Vec2::new(-x, -z);
                            let rect_max = Vec2::new(x, z);
                        
                            for (i, val) in heights.0.iter_mut().enumerate() {
                                let x = i % terrain_settings.edge_length as usize;
                                let z = i / terrain_settings.edge_length as usize;
                                
                                let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                                let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                                let overlap_index = overlap_y * 8 + overlaps_x;
                                if (entry.overlap_bits & 1 << overlap_index) == 0 {
                                    continue;
                                }
                            
                                let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
                                let vertex_local = global_transform.affine().inverse().transform_point3(Vec3::new(vertex_position.x, 0.0, vertex_position.y)).xz();
                            
                                let d_x = (rect_min.x - vertex_local.x).max(vertex_local.x - rect_max.x).max(0.0);
                                let d_y = (rect_min.y - vertex_local.y).max(vertex_local.y - rect_max.y).max(0.0);
                                let d_d = (d_x*d_x + d_y*d_y).sqrt();
                            
                                let strength = 1.0 - (d_d / modifier.falloff).clamp(0.0, 1.0);
                            
                                *val = apply_modifier(modifier, operation, vertex_position, shape_translation, *val, global_transform, strength, &mut noise_cache, false);
                            }
                        },
                    }
                }
            }
        }

        // Finally, set by splines.
        if let Some(splines) = tile_to_modifier.splines.get(&terrain_coordinate.0) {
            let _span = info_span!("Apply splines").entered();

            for entry in splines.iter() {
                if let Ok((spline, spline_properties)) = spline_query.get(entry.entity) {
                    for (i, val) in heights.0.iter_mut().enumerate() {
                        let x = i % terrain_settings.edge_length as usize;
                        let z = i / terrain_settings.edge_length as usize;
        
                        let overlaps_x = (x as f32 * inv_tile_size_scale) as u32;
                        let overlap_y = (z as f32 * inv_tile_size_scale) as u32;
                        let overlap_index = overlap_y * 8 + overlaps_x;
                        if (entry.overlap_bits & 1 << overlap_index) == 0 {
                            continue;
                        }
    
                        let vertex_position = terrain_translation + Vec2::new(x as f32, z as f32) * scale;
                        let mut distance = f32::INFINITY;
                        let mut height = None;
    
                        for points in spline.points.windows(2) {
                            let a_2d = points[0].xz();
                            let b_2d = points[1].xz();
        
                            let (new_distance, t) = minimum_distance(a_2d, b_2d, vertex_position);
        
                            if new_distance < distance {
                                distance = new_distance;
                                height = Some(points[0].lerp(points[1], t).y);
                            }
                        }
        
                        if let Some(height) = height {
                            let strength = 1.0 - ((distance.sqrt() - spline_properties.width) / spline_properties.falloff).clamp(0.0, 1.0);
                            *val = val.lerp(height, strength);
                        }
                    }
                }
            }
        }
    }
}

fn apply_modifier(modifier: &ShapeModifier, operation: &ModifierOperation, vertex_position: Vec2, shape_translation: Vec2, val: f32, global_transform: &GlobalTransform, strength: f32, noise_cache: &mut Local<NoiseCache>, set_with_position_y: bool) -> f32 {
    let mut new_val = match operation {
        ModifierOperation::Set => {
            // Relative position so we can apply the rotation from the shape modifier. This gets us tilted circles.
            let position = vertex_position - shape_translation;

            let height = if set_with_position_y {
                position.y 
            } else {
                global_transform.transform_point(Vec3::new(position.x, 0.0, position.y)).y
            };

            val.lerp(height, strength)
        },
        ModifierOperation::Change(change) => {
            val + *change * strength
        },
        ModifierOperation::Step { step, smoothing } => {
            val.lerp((((val / *step) * smoothing).round() / smoothing) * *step , strength)
        },
        ModifierOperation::Noise { noise } => {
            val.lerp(noise.get(vertex_position.x, vertex_position.y, noise_cache.get(noise.seed)), strength)
        },
    };

    if !modifier.allow_raising {
        new_val = new_val.min(val);
    }
    if !modifier.allow_lowering {
        new_val = new_val.max(val);
    }

    new_val

}

pub fn minimum_distance(v: Vec2, w: Vec2, p: Vec2) -> (f32, f32) {
    let vw = w - v;
    let pv = p - v;
    
    // Compute squared length of the segment (w - v)
    let l2 = vw.length_squared();

    // Handle degenerate case where v == w
    if l2 == 0.0 {
        return (pv.length_squared(), 1.0);
    }

    // Calculate the projection factor t
    let t = (pv.dot(vw) / l2).clamp(0.0, 1.0);
    
    // Compute the projection point on the segment
    let projection = v + vw * t;

    // Return the squared distance and the projection factor t
    (p.distance_squared(projection), t)
}
