use bevy::{app::{App, Plugin, Update}, color::{palettes::css::{BLUE, DARK_CYAN, LIGHT_CYAN}, Color}, math::{Quat, Vec2, Vec3}, prelude::{Gizmos, GlobalTransform, IntoSystemConfigs, Query, ReflectResource, Res, Resource}, reflect::Reflect};

use crate::{modifiers::{Shape, ShapeModifier, TerrainSplineCached, TerrainSplineProperties, TerrainTileAabb}, TerrainSettings};

pub struct TerrainDebugDrawPlugin;
impl Plugin for TerrainDebugDrawPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Update, debug_draw_terrain_modifiers.run_if(|draw_debug: Res<TerrainDebugDraw>| draw_debug.0))
            
            .insert_resource(TerrainDebugDraw(true))
            .register_type::<TerrainDebugDraw>()    
        ;
    }
}

#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct TerrainDebugDraw(pub bool);

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