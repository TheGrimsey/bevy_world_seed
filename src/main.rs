use std::time::Duration;

use bevy::{
    app::{App, Startup, Update}, asset::{Assets, Handle}, color::{palettes::css::{BLUE, SILVER}, Color}, log::info, math::{FloatExt, Quat, Vec2, Vec3, Vec3Swizzles}, pbr::{DirectionalLight, DirectionalLightBundle, StandardMaterial}, prelude::{default, Changed, Commands, Component, CubicCardinalSpline, CubicCurve, CubicGenerator, Entity, Gizmos, IntoSystemConfigs, Local, Or, Query, ReflectComponent, ReflectResource, Res, ResMut, Resource}, reflect::Reflect, render::{mesh::Mesh, view::VisibilityBundle}, time::common_conditions::on_timer, transform::{bundles::TransformBundle, components::{GlobalTransform, Transform}}, DefaultPlugins
};
use bevy_editor_pls::EditorPlugin;
use noise::{NoiseFn, Simplex};
use terrain::create_terrain_mesh;

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
            debug_draw_terrain_spline.run_if(|draw_debug: Res<DrawDebug>| draw_debug.0),
            (
                update_terrain_spline_cache,
                update_terrain_heights,
                update_mesh_from_heights,
            ).chain().run_if(on_timer(Duration::from_millis(500)))
        ))

        .insert_resource(DrawDebug(true))
        .insert_resource(TerrainNoiseLayers {
            layers: vec![
                TerrainNoiseLayer { height_scale: 2.0, planar_scale: 1.0 / 40.0 }
            ],
        })
        .insert_resource(MaximumSplineSimplificationDistance(3.0))

        .register_type::<DrawDebug>()
        .register_type::<TerrainSpline>()
        .register_type::<TerrainSplineCached>()
        .register_type::<TerrainNoiseLayers>()
        .register_type::<MaximumSplineSimplificationDistance>()

        .run();
}

#[derive(Resource, Reflect)]
#[reflect(Resource)]
struct DrawDebug(bool);

#[derive(Reflect)]
struct TerrainNoiseLayer {
    height_scale: f32,
    planar_scale: f32
}

#[derive(Resource, Reflect)]
#[reflect(Resource)]
struct TerrainNoiseLayers {
    layers: Vec<TerrainNoiseLayer>
}

#[derive(Resource, Reflect)]
#[reflect(Resource)]
/// Points closer than this square distance are removed. 
struct MaximumSplineSimplificationDistance(f32);

#[derive(Component)]
struct Heights(Box<[f32]>);

#[derive(Component, Reflect)]
#[reflect(Component)]
struct TerrainSpline {
    curve: CubicCurve<Vec3>,
    width: f32,
    falloff: f32
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
struct TerrainSplineCached {
    points: Vec<Vec3>,
    width: f32,
    falloff: f32
}

const EDGE_LENGTH: usize = 64;

fn update_mesh_from_heights(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<(Entity, &Heights, Option<&Handle<Mesh>>), Changed<Heights>>,
    mut terrain_material: Local<Option<Handle<StandardMaterial>>>
) {
    let material = terrain_material.get_or_insert_with(|| materials.add(StandardMaterial::from_color(Color::from(SILVER))));

    query.iter().for_each(|(entity, heights, mesh_handle)| {
        let mesh = create_terrain_mesh(Vec2::splat(32.0), EDGE_LENGTH.try_into().unwrap(), &heights.0);

        if let Some(existing_mesh) = mesh_handle.and_then(|handle| meshes.get_mut(handle)) {
            *existing_mesh = mesh;
        } else {
            let new_handle = meshes.add(mesh);

            commands.entity(entity).insert((
                new_handle,
                material.clone()
            ));
        }
    });
}

fn update_terrain_spline_cache(
    mut query: Query<(&mut TerrainSplineCached, &TerrainSpline, &GlobalTransform), Or<(Changed<TerrainSpline>, Changed<GlobalTransform>)>>,
    spline_simplification_distance: Res<MaximumSplineSimplificationDistance>
) {
    query.par_iter_mut().for_each(|(mut spline_cached, spline, global_transform)| {
        spline_cached.points.clear();

        spline_cached.points.extend(spline.curve.iter_positions(80).map(|point| global_transform.transform_point(point)));

        // Filter points that are very close together.
        let dedup_distance = (spline.width * spline.width).min(spline_simplification_distance.0);

        let pre_dedup = spline_cached.points.len();
        spline_cached.points.dedup_by(|a, b| a.distance_squared(*b) < dedup_distance);
        let post = spline_cached.points.len();

        info!("Pre: {pre_dedup} Post: {post}, DIFF: {} (Dedup distance: {dedup_distance})", pre_dedup - post);

        spline_cached.width = spline.width;
        spline_cached.falloff = spline.falloff;
    });
}

fn update_terrain_heights(
    terrain_noise_layers: Res<TerrainNoiseLayers>,
    spline_query: Query<&TerrainSplineCached>,
    mut heights: Query<(&mut Heights, &GlobalTransform)>,
    mut noise: Local<Option<Simplex>>
) {
    let noise = noise.get_or_insert_with(|| Simplex::new(1));

    heights.par_iter_mut().for_each(|(mut heights, terrain_transform)| {
        // First, set by noise.
        for (i, val) in heights.0.iter_mut().enumerate() {
            let mut new_val = 0.0;

            for noise_layer in terrain_noise_layers.layers.iter() {
                let x = (i % EDGE_LENGTH) as f64 * noise_layer.planar_scale as f64;
                let y = (i / EDGE_LENGTH) as f64 * noise_layer.planar_scale as f64;
    
                new_val += noise.get([x, y]) as f32 * noise_layer.height_scale;
            }

            *val = new_val;
        }

        // Secondly, set by splines.
        spline_query.iter().for_each(|spline| {
            for (i, val) in heights.0.iter_mut().enumerate() {
                let x = i % EDGE_LENGTH;
                let z = i / EDGE_LENGTH;

                let vertex_position = terrain_transform.transform_point(Vec3::new(x as f32 * 0.5, 0.0, z as f32 * 0.5)).xz();
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
                    let strength = 1.0 - ((distance.sqrt() - spline.width) / spline.falloff).clamp(0.0, 1.0);
                    *val = val.lerp(height, strength);
                }
            }
        });
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
) {
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            illuminance: 1000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_translation(Vec3::new(32.0, 25.0, 16.0)).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    let spline: CubicCurve<Vec3> = CubicCardinalSpline::new_catmull_rom(vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(8.0, 0.0, 8.0),
        Vec3::new(16.0, 0.0, 16.0),
        Vec3::new(32.0, 1.0, 32.0),
    ]).to_curve();

    commands.spawn((
        TerrainSpline {
            curve: spline,
            width: 2.0,
            falloff: 1.5
        },
        TerrainSplineCached::default(),
        TransformBundle::default()
    ));

    commands.spawn((
        Heights([0.0; EDGE_LENGTH*EDGE_LENGTH].into()),
        TransformBundle::default(),
        VisibilityBundle::default()
    ));
}

fn debug_draw_terrain_spline(
    mut gizmos: Gizmos,
    query: Query<&TerrainSplineCached>
) {
    query.iter().for_each(|spline: &TerrainSplineCached| {
        for (a, b) in spline.points.iter().zip(spline.points.iter().skip(1)) {
            gizmos.line(*a, *a + Vec3::Y, Color::from(BLUE));

            gizmos.line(*a, *b, Color::from(BLUE));

            let distance = a.distance(*b);

            gizmos.rect(a.lerp(*b, 0.5), Quat::from_axis_angle(Vec3::X, 90.0_f32.to_radians()) * Quat::from_axis_angle(Vec3::Z, 45.0_f32.to_radians()), Vec2::new(distance, spline.width*2.0), Color::from(BLUE));
            //gizmos.circle(position, Dir3::Y, spline.width, Color::from(BLUE));
            //gizmos.circle(position, Dir3::Y, spline.width + spline.falloff, Color::from(RED));
        }
    });
}