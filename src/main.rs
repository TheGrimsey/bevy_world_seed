#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::num::NonZeroU32;

use bevy::{
    app::{App, PostUpdate, Startup}, asset::{AssetServer, Assets}, color::{palettes::css::RED, Color}, core::Name, log::info_span, math::{FloatExt, IVec2, Vec2, Vec3, Vec3Swizzles}, pbr::{DirectionalLight, DirectionalLightBundle, PbrBundle, StandardMaterial}, prelude::{default, resource_changed, BuildChildren, Camera3dBundle, Capsule3d, Commands, Component, CubicCardinalSpline, CubicCurve, CubicGenerator, Event, EventReader, IntoSystemConfigs, Local, PerspectiveProjection, Projection, Query, ReflectDefault, ReflectResource, Res, ResMut, Resource, SystemSet, TransformSystem}, reflect::Reflect, render::{mesh::Mesh, view::VisibilityBundle}, transform::{bundles::TransformBundle, components::{GlobalTransform, Transform}}, DefaultPlugins
};
use bevy_editor_pls::EditorPlugin;
use bevy_rapier3d::prelude::Collider;
use debug_draw::TerrainDebugDrawPlugin;
#[cfg(feature = "rendering")]
use material::TerrainTexturingPlugin;
use material::TextureModifier;
#[cfg(feature = "rendering")]
use meshing::TerrainMeshingPlugin;
use noise::{NoiseFn, Simplex};
use modifiers::{update_shape_modifier_aabb, update_terrain_spline_aabb, update_terrain_spline_cache, update_tile_modifier_priorities, ModifierOperation, ModifierPriority, Shape, ShapeModifier, ShapeModifierBundle, TerrainSpline, TerrainSplineBundle, TerrainSplineCached, TerrainSplineProperties, TerrainTileAabb, TileToModifierMapping};
use terrain::{update_tiling, TerrainCoordinate, TileToTerrain};

mod debug_draw;
mod modifiers;
mod terrain;

#[cfg(feature = "rendering")]
mod meshing;
#[cfg(feature = "rendering")]
mod material;

/// System sets containing the crate's systems.
#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
pub enum TerrainSets {
    Modifiers,
    Heights,
}


// Our Bevy app's entry point
fn main() {
    // Bevy apps are created using the builder pattern. We use the builder to add systems,
    // resources, and plugins to our app
    let mut app = App::new();


    app
        .add_plugins(DefaultPlugins);

    #[cfg(feature = "rendering")]
    {
        app.add_plugins((
            TerrainMeshingPlugin,
            TerrainTexturingPlugin
        ));
    }

    app
        .add_plugins(EditorPlugin::new())
        .add_plugins(TerrainDebugDrawPlugin)
        // This call to run() starts the app we just built!
        .add_systems(Startup, spawn_terrain)
        .add_systems(PostUpdate, (
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
        ).after(TransformSystem::TransformPropagate))
        .insert_resource(TerrainNoiseLayers {
            layers: vec![
                TerrainNoiseLayer { height_scale: 3.0, planar_scale: 1.0 / 20.0, seed: 1 }
            ],
        })
        .insert_resource(TerrainSettings {
            tile_size_power: 5,
            edge_length: 65,
            max_tile_updates_per_frame: NonZeroU32::new(2).unwrap(),
            max_spline_simplification_distance: 3.0
        })
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

        .add_event::<RebuildTile>()

        .run();
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

#[derive(Reflect)]
#[reflect(Default)]
struct TerrainNoiseLayer {
    height_scale: f32,
    planar_scale: f32,
    seed: u32,
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

#[derive(Resource, Reflect)]
#[reflect(Resource)]
struct TerrainNoiseLayers {
    layers: Vec<TerrainNoiseLayer>
}

#[derive(Resource, Reflect)]
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
pub struct RebuildTile(IVec2);

#[derive(Component)]
struct Heights(Box<[f32]>);

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
            
            shape_modifier_query.iter_many(shapes.iter()).for_each(|(modifier, operation, global_transform)| {
                let shape_translation = global_transform.translation().xz();
    
                match modifier.shape {
                    Shape::Circle { radius } => {
                        for (i, val) in heights.0.iter_mut().enumerate() {
                            let x = i % terrain_settings.edge_length as usize;
                            let z = i / terrain_settings.edge_length as usize;
                        
                            let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
    
                            let strength = 1.0 - ((vertex_position.distance(shape_translation) - radius) / modifier.falloff).clamp(0.0, 1.0);
    
                            *val = apply_modifier(modifier, operation, vertex_position, shape_translation, *val, global_transform, strength, &mut noise_cache);
                        }
                    },
                    Shape::Rectangle { x, z } => {
                        let rect_min = Vec2::new(-x, -z);
                        let rect_max = Vec2::new(x, z);
    
                        for (i, val) in heights.0.iter_mut().enumerate() {
                            let x = i % terrain_settings.edge_length as usize;
                            let z = i / terrain_settings.edge_length as usize;
                        
                            let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
                            let vertex_local = global_transform.affine().inverse().transform_point3(Vec3::new(vertex_position.x, 0.0, vertex_position.y)).xz();
    
                            let d_x = (rect_min.x - vertex_local.x).max(vertex_local.x - rect_max.x).max(0.0);
                            let d_y = (rect_min.y - vertex_local.y).max(vertex_local.y - rect_max.y).max(0.0);
                            let d_d = (d_x*d_x + d_y*d_y).sqrt();
    
                            let strength = 1.0 - (d_d / modifier.falloff).clamp(0.0, 1.0);
    
                            *val = apply_modifier(modifier, operation, vertex_position, shape_translation, *val, global_transform, strength, &mut noise_cache);
                        }
                    },
                }
            });
        }

        // Finally, set by splines.
        if let Some(splines) = tile_to_modifier.splines.get(&terrain_coordinate.0) {
            let _span = info_span!("Apply splines").entered();

            for entry in splines.iter() {
                if let Ok((spline, spline_properties)) = spline_query.get(entry.entity) {
                    for (i, val) in heights.0.iter_mut().enumerate() {
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
                        let mut height = None;
    
                        for (a, b) in spline.points.iter().zip(spline.points.iter().skip(1)) {
                            let a_2d = a.xz();
                            let b_2d = b.xz();
        
                            let (new_distance, t) = minimum_distance(a_2d, b_2d, vertex_position);
        
                            if new_distance < distance {
                                distance = new_distance;
                                height = Some(a.lerp(*b, t).y);
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

fn apply_modifier(modifier: &ShapeModifier, operation: &ModifierOperation, vertex_position: Vec2, shape_translation: Vec2, val: f32, global_transform: &GlobalTransform, strength: f32, noise_cache: &mut Local<NoiseCache>) -> f32 {
    let mut new_val = match operation {
        ModifierOperation::Set => {
            // Relative position so we can apply the rotation from the shape modifier. This gets us tilted circles.
            let position = vertex_position - shape_translation;

            val.lerp(global_transform.transform_point(Vec3::new(position.x, 0.0, position.y)).y, strength)
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

fn minimum_distance(v: Vec2, w: Vec2, p: Vec2) -> (f32, f32) {
    // Return minimum distance between line segment vw and point p
    let l2 = v.distance_squared(w);  // i.e. |w-v|^2 -  avoid a sqrt
    if l2 == 0.0 {
        return (p.distance_squared(v), 1.0);   // v == w case
    }
    // Consider the line extending the segment, parameterized as v + t (w - v).
    // We find projection of point p onto the line. 
    // It falls where t = [(p-v) . (w-v)] / |w-v|^2
    // We clamp t from [0,1] to handle points outside the segment vw.
    let t = ((p - v).dot(w - v) / l2).clamp(0.0, 1.0);
    
    let projection = v + t * (w - v);  // Projection falls on the segment
    
    (p.distance_squared(projection), t)
  }

fn spawn_terrain(
    mut commands: Commands,
    mut mesh: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    terrain_settings: Res<TerrainSettings>,
) {
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            illuminance: 1000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_translation(Vec3::new(32.0, 25.0, 16.0)).looking_at(Vec3::ZERO, Vec3::Y).with_translation(Vec3::ZERO),
        ..default()
    });

    let spline: CubicCurve<Vec3> = CubicCardinalSpline::new_catmull_rom(vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(8.0, 0.0, 8.0),
        Vec3::new(16.0, 0.0, 16.0),
        Vec3::new(32.0, 1.0, 32.0),
    ]).to_curve();

    commands.spawn((
        TerrainSplineBundle {
            tile_aabb: TerrainTileAabb::default(),
            spline: TerrainSpline {
                curve: spline
            },
            properties: TerrainSplineProperties {
                width: 8.0,
                falloff: 4.0
            },
            spline_cached: TerrainSplineCached::default(),
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::default()
        },
        TextureModifier {
            texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
            max_texture_strength: 0.95
        },
        Name::new("Spline")
    ));

    commands.spawn((
        ShapeModifierBundle {
            aabb: TerrainTileAabb::default(),
            modifier: ShapeModifier {
                shape: Shape::Circle {
                    radius: 4.0
                },
                falloff: 4.0,
                allow_lowering: true,
                allow_raising: true
            },
            operation: ModifierOperation::Set,
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(Vec3::new(10.0, 5.0, 48.0))),
        },
        TextureModifier {
            texture: asset_server.load("textures/cracked_concrete_diff_1k.jpg"),
            max_texture_strength: 0.95
        },
        Name::new("Modifier (Circle)")
    ));

    commands.spawn((
        ShapeModifierBundle {
            aabb: TerrainTileAabb::default(),
            modifier: ShapeModifier {
                shape: Shape::Rectangle {
                    x: 2.5,
                    z: 5.0
                },
                falloff: 12.0,
                allow_raising: true,
                allow_lowering: true
            },
            operation: ModifierOperation::Set,
            priority: ModifierPriority(2),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(Vec3::new(32.0, 5.0, 50.0))),
        },
        TextureModifier {
            texture: asset_server.load("textures/brown_mud_leaves.jpg"),
            max_texture_strength: 0.95
        },
        Name::new("Modifier (Rectangle)")
    ));

    let size = terrain_settings.edge_length as usize * terrain_settings.edge_length as usize;
    let flat_heights = vec![0.0; size].into_boxed_slice();

    commands.spawn((
        TerrainCoordinate(IVec2::new(0, 0)),
        Heights(flat_heights.clone()),
        TransformBundle::default(),
        VisibilityBundle::default(),
        Name::new("Terrain")
    ));

    commands.spawn((
        TerrainCoordinate(IVec2::new(0, 0)),
        Heights(flat_heights.clone()),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(terrain_settings.tile_size(), 0.0, 0.0))),
        VisibilityBundle::default(),
        Name::new("Terrain 2")
    ));
    
    commands.spawn((
        TerrainCoordinate(IVec2::new(0, 0)),
        Heights(flat_heights.clone()),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(0.0, 0.0, terrain_settings.tile_size()))),
        VisibilityBundle::default(),
        Name::new("Terrain 3")
    ));
    
    commands.spawn((
        TerrainCoordinate(IVec2::new(0, 0)),
        Heights(flat_heights),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(terrain_settings.tile_size(), 0.0, terrain_settings.tile_size()))),
        VisibilityBundle::default(),
        Name::new("Terrain 4")
    ));

    commands.spawn((
        PbrBundle {
            mesh: mesh.add(Capsule3d::new(0.25, 1.8 / 2.0)),
            material: materials.add(StandardMaterial::from_color(RED)),
            transform: Transform::from_translation(Vec3::new(0.0, 1.8 / 2.0, 0.0)),
            ..default()
        },
        Collider::capsule_y(1.8 / 2.0, 0.25),
        Name::new("Character")
    )).with_children(|child_builder| {
        child_builder.spawn(Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.0, 2.0)),
            projection: Projection::Perspective(PerspectiveProjection {
                fov: 80.0_f32.to_radians(),
                ..default()
            }),
            ..default()
        });
    });
}
