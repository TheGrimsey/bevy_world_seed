use std::time::Duration;

use bevy::{
    app::{App, Startup, Update}, asset::{Assets, Handle}, color::{palettes::css::{BLUE, DARK_CYAN, LIGHT_CYAN, RED, SILVER}, Color}, core::Name, log::info_span, math::{IVec2, Quat, Vec2, Vec3, Vec3Swizzles}, pbr::{DirectionalLight, DirectionalLightBundle, PbrBundle, StandardMaterial}, prelude::{default, resource_changed, BuildChildren, Camera3dBundle, Capsule3d, Changed, Commands, Component, CubicCardinalSpline, CubicCurve, CubicGenerator, Entity, Gizmos, IntoSystem, IntoSystemConfigs, Local, PerspectiveProjection, Projection, Query, ReflectComponent, ReflectResource, Res, ResMut, Resource}, reflect::Reflect, render::{mesh::Mesh, view::VisibilityBundle}, time::common_conditions::on_timer, transform::{bundles::TransformBundle, components::{GlobalTransform, Transform}}, utils::HashSet, DefaultPlugins
};
use bevy_editor_pls::EditorPlugin;
use bevy_rapier3d::prelude::Collider;
use noise::{NoiseFn, Simplex};
use modifiers::{update_shape_modifier_aabb, update_terrain_spline_aabb, update_terrain_spline_cache, update_tile_modifier_priorities, ModifierPriority, Shape, ShapeModifier, ShapeModifierBundle, TerrainAabb, TerrainSpline, TerrainSplineBundle, TerrainSplineCached, TerrainSplineProperties, TerrainTileAabb, TileToModifierMapping};
use terrain::create_terrain_mesh;

mod modifiers;
mod terrain;

// Our Bevy app's entry point
fn main() {
    // Bevy apps are created using the builder pattern. We use the builder to add systems,
    // resources, and plugins to our app
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EditorPlugin::new())
        // This call to run() starts the app we just built!
        .add_systems(Startup, spawn_terrain)
        .add_systems(Update, (
            debug_draw_terrain_modifiers.run_if(|draw_debug: Res<DrawDebug>| draw_debug.0),
            (
                (
                    (
                        update_terrain_spline_cache,
                        update_terrain_spline_aabb
                    ).chain(),
                    update_shape_modifier_aabb,
                ),
                update_tile_modifier_priorities.run_if(resource_changed::<DirtyTiles>),
                update_terrain_heights,
                update_mesh_from_heights,
            ).chain().run_if(on_timer(Duration::from_millis(500)))
        ))

        .insert_resource(DrawDebug(true))
        .insert_resource(TerrainNoiseLayers {
            layers: vec![
                TerrainNoiseLayer { height_scale: 3.0, planar_scale: 1.0 / 20.0, noise: Simplex::new(1) }
            ],
        })
        .insert_resource(MaximumSplineSimplificationDistance(3.0))
        .insert_resource(TerrainSettings {
            tile_size_power: 5,
            edge_length: 65
        })
        .init_resource::<DirtyTiles>()
        .init_resource::<TileToModifierMapping>()

        .register_type::<DrawDebug>()
        .register_type::<TerrainSpline>()
        .register_type::<TerrainSplineCached>()
        .register_type::<TerrainNoiseLayers>()
        .register_type::<TerrainAabb>()
        .register_type::<TerrainTileAabb>()
        .register_type::<TerrainSplineProperties>()
        .register_type::<TerrainCoordinate>()
        .register_type::<MaximumSplineSimplificationDistance>()
        .register_type::<ShapeModifier>()

        .run();
}

#[derive(Resource, Reflect)]
#[reflect(Resource)]
struct DrawDebug(bool);

#[derive(Reflect)]
struct TerrainNoiseLayer {
    height_scale: f32,
    planar_scale: f32,
    #[reflect(ignore)]
    noise: Simplex
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
    pub edge_length: u32
}
impl TerrainSettings {
    pub fn tile_size(&self) -> f32 {
        (1 << self.tile_size_power) as f32
    }
}

#[derive(Resource, Default)]
struct DirtyTiles(HashSet<IVec2>);

#[derive(Resource, Reflect)]
#[reflect(Resource)]
/// Points closer than this square distance are removed. 
struct MaximumSplineSimplificationDistance(f32);

#[derive(Component)]
struct Heights(Box<[f32]>);

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct TerrainCoordinate(IVec2); 

fn update_mesh_from_heights(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<(Entity, &Heights, Option<&Handle<Mesh>>), Changed<Heights>>,
    mut terrain_material: Local<Option<Handle<StandardMaterial>>>,
    terrain_settings: Res<TerrainSettings>
) {
    let material = terrain_material.get_or_insert_with(|| materials.add(StandardMaterial::from_color(Color::from(SILVER))));

    query.iter().for_each(|(entity, heights, mesh_handle)| {
        let mesh = create_terrain_mesh(Vec2::splat(terrain_settings.tile_size()), terrain_settings.edge_length.try_into().unwrap(), &heights.0);

        if let Some(existing_mesh) = mesh_handle.and_then(|handle| meshes.get_mut(handle)) {
            *existing_mesh = mesh;
        } else {
            let new_handle = meshes.add(mesh);

            commands.entity(entity).insert((
                new_handle,
                material.clone()
            ));
        }

        commands.entity(entity).insert(Collider::heightfield(heights.0.to_vec(), terrain_settings.edge_length as usize, terrain_settings.edge_length as usize, Vec3::new(terrain_settings.tile_size(), 1.0, terrain_settings.tile_size())));
    });
}


fn update_terrain_heights(
    terrain_noise_layers: Res<TerrainNoiseLayers>,
    shape_modifier_query: Query<(&ShapeModifier, &GlobalTransform, &ModifierPriority)>,
    spline_query: Query<(&TerrainSplineCached, &TerrainSplineProperties, &ModifierPriority)>,
    mut heights: Query<(&mut Heights, &TerrainCoordinate)>,
    mut noise: Local<Option<Simplex>>,
    terrain_settings: Res<TerrainSettings>,
    tile_to_modifier: Res<TileToModifierMapping>,
    mut dirty_tiles: ResMut<DirtyTiles>
) {
    let scale = terrain_settings.tile_size() / (terrain_settings.edge_length - 1) as f32;

    let tile_size = terrain_settings.tile_size();

    heights.iter_many(dirty_tiles.0.drain().take(n)).for_each(|(mut heights, terrain_coordinate)| {
        let terrain_translation = (terrain_coordinate.0 << terrain_settings.tile_size_power).as_vec2(); 

        // First, set by noise.
        {
            let _span = info_span!("Apply noise").entered();
            for (i, val) in heights.0.iter_mut().enumerate() {
                let mut new_val = 0.0;
    
                let x = i % terrain_settings.edge_length as usize;
                let z = i / terrain_settings.edge_length as usize;
                
                let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
    
                for noise_layer in terrain_noise_layers.layers.iter() {
                    let noise_x = vertex_position.x * noise_layer.planar_scale;
                    let noise_z = vertex_position.y * noise_layer.planar_scale;
        
                    new_val += noise_layer.noise.get([noise_x as f64, noise_z as f64]) as f32 * noise_layer.height_scale;
                }
    
                *val = new_val;
            }
        }

        // Secondly, set by shape-modifiers.
        if let Some(shapes) = tile_to_modifier.shape.get(&terrain_coordinate.0) {
            let _span = info_span!("Apply shape modifiers").entered();
            
            shape_modifier_query.iter_many(shapes.iter()).for_each(|(modifier, global_transform, _)| {
                let shape_translation = global_transform.translation().xz();
    
                match modifier.shape {
                    Shape::Circle { radius } => {
                        for (i, val) in heights.0.iter_mut().enumerate() {
                            let x = i % terrain_settings.edge_length as usize;
                            let z = i / terrain_settings.edge_length as usize;
                        
                            let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
    
                            let strength = 1.0 - ((vertex_position.distance(shape_translation) - radius) / modifier.falloff).clamp(0.0, 1.0);
    
                            // Relative position so we can apply the rotation from the shape modifier. This gets us tilted circles.
                            let position = vertex_position - shape_translation;
    
                            *val = val.lerp(global_transform.transform_point(Vec3::new(position.x, 0.0, position.y)).y, strength);
                        }
                    },
                    Shape::Rectangle { x, z } => {
                        let rect_min = Vec2::new(-x, -z) / 2.0;
                        let rect_max = Vec2::new(x, z) / 2.0;
    
                        for (i, val) in heights.0.iter_mut().enumerate() {
                            let x = i % terrain_settings.edge_length as usize;
                            let z = i / terrain_settings.edge_length as usize;
                        
                            let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
                            let vertex_local = global_transform.affine().inverse().transform_point3(Vec3::new(vertex_position.x, 0.0, vertex_position.y)).xz();
    
                            let d_x = (rect_min.x - vertex_local.x).max(vertex_local.x - rect_max.x).max(0.0);
                            let d_y = (rect_min.y - vertex_local.y).max(vertex_local.y - rect_max.y).max(0.0);
                            let d_d = (d_x*d_x + d_y*d_y).sqrt();
    
                            let strength = 1.0 - (d_d / modifier.falloff).clamp(0.0, 1.0);
    
                            // Relative position so we can apply the rotation from the shape modifier. This gets us tilted shapes.
                            let position = vertex_position - shape_translation;
    
                            *val = val.lerp(global_transform.transform_point(Vec3::new(position.x, 0.0, position.y)).y, strength);
                        }
                    },
                }
            });
        }

        // Finally, set by splines.
        if let Some(splines) = tile_to_modifier.splines.get(&terrain_coordinate.0) {
            let _span = info_span!("Apply splines").entered();

            for entry in splines.iter() {
                if let Ok((spline, spline_properties, _)) = spline_query.get(entry.entity) {
                    for (i, val) in heights.0.iter_mut().enumerate() {
                        let x = i % terrain_settings.edge_length as usize;
                        let z = i / terrain_settings.edge_length as usize;
        
                        let overlaps = (Vec2::new(x as f32 * scale, z as f32 * scale) / tile_size * 8.0).as_ivec2();
                        let overlap_index = overlaps.y * 8 + overlaps.x;
                        if (entry.overlap_bits & 1 << overlap_index) == 0 {
                            continue;
                        }
    
                        let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);
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
    });
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
            aabb: TerrainAabb::default(),
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
            },
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(Vec3::new(10.0, 5.0, 48.0))),
        },
        Name::new("Modifier (Circle)")
    ));

    commands.spawn((
        ShapeModifierBundle {
            aabb: TerrainTileAabb::default(),
            modifier: ShapeModifier {
                shape: Shape::Rectangle {
                    x: 5.0,
                    z: 10.0
                },
                falloff: 12.0,
            },
            priority: ModifierPriority(1),
            transform_bundle: TransformBundle::from_transform(Transform::from_translation(Vec3::new(48.0, 5.0, 10.0))),
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
        TerrainCoordinate(IVec2::new(1, 0)),
        Heights(flat_heights.clone()),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(terrain_settings.tile_size(), 0.0, 0.0))),
        VisibilityBundle::default(),
        Name::new("Terrain 2")
    ));
    
    commands.spawn((
        TerrainCoordinate(IVec2::new(0, 1)),
        Heights(flat_heights.clone()),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(0.0, 0.0, terrain_settings.tile_size()))),
        VisibilityBundle::default(),
        Name::new("Terrain 3")
    ));
    
    commands.spawn((
        TerrainCoordinate(IVec2::new(1, 1)),
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

fn debug_draw_terrain_modifiers(
    mut gizmos: Gizmos,
    spline_query: Query<(&TerrainSplineCached, &TerrainSplineProperties, &TerrainTileAabb)>,
    shape_query: Query<(&ShapeModifier, &GlobalTransform)>,
    terrain_settings: Res<TerrainSettings>
) {
    spline_query.iter().for_each(|(spline, spline_properties, terrain_abb)| {
        for (a, b) in spline.points.iter().zip(spline.points.iter().skip(1)) {
            gizmos.line(*a, *a + Vec3::Y, Color::from(BLUE));

            gizmos.line(*a, *b, Color::from(BLUE));

            let distance = a.distance(*b);

            gizmos.rect(a.lerp(*b, 0.5), Quat::from_axis_angle(Vec3::X, 90.0_f32.to_radians()) * Quat::from_axis_angle(Vec3::Z, 45.0_f32.to_radians()), Vec2::new(distance, spline_properties.width*2.0), Color::from(BLUE));
        
            // Debug draw AABB.
            for x in terrain_abb.min.x..=terrain_abb.max.x {
                for y in terrain_abb.min.y..=terrain_abb.max.y {
                    gizmos.rect(Vec3::new((x << terrain_settings.tile_size_power) as f32 + terrain_settings.tile_size() / 2.0, 0.0, (y << terrain_settings.tile_size_power) as f32 + terrain_settings.tile_size() / 2.0), Quat::from_axis_angle(Vec3::X, 90.0_f32.to_radians()), Vec2::splat(terrain_settings.tile_size()), Color::from(BLUE));        
                }
            }
        }
    });

    shape_query.iter().for_each(|(shape, global_transform)| {
        match shape.shape {
            Shape::Circle { radius } => {
                gizmos.circle(global_transform.translation(), global_transform.up(), radius, Color::from(LIGHT_CYAN));

                // Falloff.
                gizmos.circle(global_transform.translation(), global_transform.up(), shape.falloff + radius, Color::from(DARK_CYAN));
            },
            Shape::Rectangle { x, z } => {
                let (_, rot, translation) = global_transform.to_scale_rotation_translation();
                gizmos.rect(translation, rot * Quat::from_axis_angle(Vec3::X, 90.0_f32.to_radians()), Vec2::new(x, z), Color::from(LIGHT_CYAN));
            },
        }
    });
}