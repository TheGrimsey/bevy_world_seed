use bevy_app::{App, Plugin, Update};
use bevy_color::{
    palettes::css::{BLUE, DARK_CYAN, LIGHT_CYAN},
    Color,
};
use bevy_math::{Quat, Vec2, Vec3, Vec3Swizzles};
use bevy_ecs::prelude::{Query, Resource, Res, IntoSystemConfigs, ReflectResource};
use bevy_transform::prelude::GlobalTransform;
use bevy_gizmos::prelude::Gizmos;
use bevy_reflect::Reflect;

use crate::modifiers::{
    ModifierFalloffProperty, ShapeModifier, TerrainSplineCached, TerrainSplineProperties,
};

pub struct TerrainDebugDrawPlugin;
impl Plugin for TerrainDebugDrawPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            debug_draw_terrain_modifiers.run_if(|draw_debug: Res<TerrainDebugDraw>| draw_debug.0),
        )
        .insert_resource(TerrainDebugDraw(false))
        .register_type::<TerrainDebugDraw>();
    }
}

#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct TerrainDebugDraw(pub bool);

fn debug_draw_terrain_modifiers(
    mut gizmos: Gizmos,
    spline_query: Query<(&TerrainSplineCached, &TerrainSplineProperties)>,
    shape_query: Query<(
        &ShapeModifier,
        Option<&ModifierFalloffProperty>,
        &GlobalTransform,
    )>,
) {
    spline_query.iter().for_each(|(spline, spline_properties)| {
        for (a, b) in spline.points.iter().zip(spline.points.iter().skip(1)) {
            gizmos.line(*a, *a + Vec3::Y, Color::from(BLUE));

            gizmos.line(*a, *b, Color::from(BLUE));

            let distance = a.distance(*b);

            let forward = *b - *a;
            let rotation = (-forward.x).atan2(-forward.z);

            gizmos.rect(
                a.lerp(*b, 0.5),
                Quat::from_axis_angle(Vec3::X, 90.0_f32.to_radians())
                    * Quat::from_rotation_z(rotation),
                Vec2::new(distance, spline_properties.half_width * 2.0),
                Color::from(BLUE),
            );
        }
    });

    shape_query
        .iter()
        .for_each(|(shape, modifier_falloff, global_transform)| {
            let falloff = modifier_falloff.map_or(f32::EPSILON, |falloff| falloff.falloff);

            let (scale, rot, translation) = global_transform.to_scale_rotation_translation();
            let rotation = rot * Quat::from_axis_angle(Vec3::X, 90.0_f32.to_radians());

            match shape {
                ShapeModifier::Circle { radius } => {
                    gizmos.ellipse(
                        translation,
                        rotation,
                        scale.xz() * *radius,
                        Color::from(LIGHT_CYAN),
                    );

                    // Falloff.
                    gizmos.ellipse(
                        translation,
                        rotation,
                        scale.xz() * (falloff + radius),
                        Color::from(DARK_CYAN),
                    );
                }
                ShapeModifier::Rectangle { x, z } => {
                    gizmos.rect(
                        translation,
                        rotation,
                        scale.xz() * Vec2::new(*x, *z) * 2.0,
                        Color::from(LIGHT_CYAN),
                    );

                    gizmos.rect(
                        translation,
                        rotation,
                        scale.xz() * (Vec2::new(*x, *z) * 2.0 + falloff),
                        Color::from(DARK_CYAN),
                    );
                }
            }
        });
}
