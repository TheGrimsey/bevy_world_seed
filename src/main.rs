use bevy::{
    app::{App, Startup, Update}, asset::{Assets, Handle}, color::{palettes::css::{BLUE, RED, SILVER}, Color}, ecs::reflect, log::info, math::{Dir3, FloatExt, Quat, Vec2, Vec3}, pbr::{DirectionalLight, DirectionalLightBundle, StandardMaterial}, prelude::{default, Changed, Commands, Component, CubicCardinalSpline, CubicCurve, CubicGenerator, Entity, Gizmos, IntoSystemConfigs, Local, Query, ReflectComponent, ReflectResource, Res, ResMut, Resource}, reflect::Reflect, render::{mesh::Mesh, view::VisibilityBundle}, transform::{bundles::TransformBundle, components::Transform}, DefaultPlugins
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
            debug_draw_terrain_spline.run_if(|draw_debug: Res<DrawDebug>| draw_debug.0)
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

    let noise = Simplex::new(1);
    
    let mut heights = [0.0; EDGE_LENGTH*EDGE_LENGTH];
    for (i, val) in heights.iter_mut().enumerate() {
        let x = (i % EDGE_LENGTH) as f64 / 40.0;
        let y = (i / EDGE_LENGTH) as f64 / 40.0;

        *val = noise.get([x, y]) as f32 * 2.5;
    }

    let spline = CubicCardinalSpline::new_catmull_rom(vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(8.0, 1.0, 8.0),
        Vec3::new(16.0, 1.0, 16.0),
        Vec3::new(32.0, 1.0, 32.0),
    ]).to_curve();

    let width = 2.0;
    let falloff = 1.75;

    let box_width = width + falloff;

    for position in spline.iter_positions(80) {
        let min_x = (((position.x - box_width) / 0.5) as usize).max(0);
        let min_z = (((position.z - box_width) / 0.5) as usize).max(0);

        let max_x = (((position.x + box_width) / 0.5) as usize).min(63);
        let max_z = (((position.z + box_width) / 0.5) as usize).min(63);

        for z in min_z..max_z {
            let row = z * EDGE_LENGTH;
            for x in min_x..max_x {
                let vertex_position = Vec3::new(x as f32 * 0.5, position.y, z as f32 * 0.5);

                let distance = position.distance(vertex_position);

                let strength = 1.0 - ((distance - width) / falloff).clamp(0.0, 1.0);

                let i = row + x;

                heights[i] = heights[i].lerp(0.25, strength);
            }
        }
    }

    commands.spawn(TerrainSpline {
        curve: spline,
        width,
        falloff
    });

    commands.spawn((
        Heights(heights.into()),
        TransformBundle::default(),
        VisibilityBundle::default()
    ));
}

fn debug_draw_terrain_spline(
    mut gizmos: Gizmos,
    query: Query<&TerrainSpline>
) {
    query.iter().for_each(|spline| {
        for position in spline.curve.iter_positions(40) {
            gizmos.line(position, position + Vec3::Y, Color::from(BLUE));

            gizmos.circle(position, Dir3::Y, spline.width, Color::from(BLUE));
            gizmos.circle(position, Dir3::Y, spline.width + spline.falloff, Color::from(RED));
        }
    });
}