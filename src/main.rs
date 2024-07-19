use std::time::Duration;

use bevy::{
    app::{App, Startup, Update}, asset::{Assets, Handle}, color::{palettes::css::{BLUE, RED, SILVER}, Color}, ecs::reflect, log::info, math::{Dir3, FloatExt, Quat, Vec2, Vec3}, pbr::{DirectionalLight, DirectionalLightBundle, StandardMaterial}, prelude::{default, Changed, Commands, Component, CubicCardinalSpline, CubicCurve, CubicGenerator, Entity, Gizmos, IntoSystemConfigs, Local, Query, ReflectComponent, ReflectResource, Res, ResMut, Resource}, reflect::Reflect, render::{mesh::Mesh, view::VisibilityBundle}, time::common_conditions::on_timer, transform::{bundles::TransformBundle, components::{GlobalTransform, Transform}}, DefaultPlugins
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
            update_mesh_from_heights,
            debug_draw_terrain_spline.run_if(|draw_debug: Res<DrawDebug>| draw_debug.0),
            update_terrain_heights.run_if(on_timer(Duration::from_millis(500)))
        ))

        .insert_resource(DrawDebug(true))
        .insert_resource(TerrainNoiseLayers {
            layers: vec![
                TerrainNoiseLayer { height_scale: 2.0, planar_scale: 1.0 / 40.0 }
            ],
        })

        .register_type::<DrawDebug>()
        .register_type::<TerrainSpline>()
        .register_type::<TerrainNoiseLayers>()

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


#[derive(Component)]
struct Heights(Box<[f32]>);

#[derive(Component, Reflect)]
#[reflect(Component)]
struct TerrainSpline {
    curve: CubicCurve<Vec3>,
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

fn update_terrain_heights(
    terrain_noise_layers: Res<TerrainNoiseLayers>,
    splines: Query<(&TerrainSpline, &GlobalTransform)>,
    mut heights: Query<(&mut Heights, &GlobalTransform)>
) {
    let noise = Simplex::new(1);

    heights.par_iter_mut().for_each(|(mut heights, global_transform)| {
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
        splines.iter().for_each(|(spline, global_transform)| {
            let box_width = spline.width + spline.falloff;

            for position in spline.curve.iter_positions(80) {
                let min_x = (((position.x - box_width) / 0.5) as usize).max(0);
                let min_z = (((position.z - box_width) / 0.5) as usize).max(0);
        
                let max_x = (((position.x + box_width) / 0.5) as usize).min(63);
                let max_z = (((position.z + box_width) / 0.5) as usize).min(63);
        
                for z in min_z..max_z {
                    let row = z * EDGE_LENGTH;
                    for x in min_x..max_x {
                        let vertex_position = Vec3::new(x as f32 * 0.5, position.y, z as f32 * 0.5);
        
                        let distance = position.distance(vertex_position);
        
                        let strength = 1.0 - ((distance - spline.width) / spline.falloff).clamp(0.0, 1.0);
        
                        let i = row + x;
        
                        heights.0[i] = heights.0[i].lerp(position.y, strength);
                    }
                }
            }
        });
    });
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
    query: Query<&TerrainSpline>
) {
    query.iter().for_each(|spline| {
        for (a, b) in spline.curve.iter_positions(40).zip(spline.curve.iter_positions(40).skip(1)) {
            gizmos.line(a, a + Vec3::Y, Color::from(BLUE));

            gizmos.line(a, b, Color::from(BLUE));

            let distance = a.distance(b);

            gizmos.rect(a.lerp(b, 0.5), Quat::from_axis_angle(Vec3::X, 90.0_f32.to_radians()) * Quat::from_axis_angle(Vec3::Z, 45.0_f32.to_radians()), Vec2::new(distance, spline.width*2.0), Color::from(BLUE));
            //gizmos.circle(position, Dir3::Y, spline.width, Color::from(BLUE));
            //gizmos.circle(position, Dir3::Y, spline.width + spline.falloff, Color::from(RED));
        }
    });
}