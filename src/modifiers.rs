use bevy::{ecs::reflect, math::{Vec2, Vec3, Vec3Swizzles}, prelude::{Bundle, Changed, Component, CubicCurve, GlobalTransform, Or, Query, ReflectComponent, Res, TransformBundle}, reflect::Reflect};

use crate::MaximumSplineSimplificationDistance;

#[derive(Bundle)]
pub struct ShapeModifierBundle {
    pub aabb: TerrainAabb,
    pub modifier: ShapeModifier,
    pub priority: ModifierPriority,
    pub transform_bundle: TransformBundle
}

#[derive(Reflect)]
pub enum Shape {
    Circle {
        radius: f32,
    },
    Rectangle {
        x: f32,
        z: f32, 
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct ShapeModifier {
    pub shape: Shape,
    pub falloff: f32
}

#[derive(Bundle)]
pub struct TerrainSplineBundle {
    pub aabb: TerrainAabb,
    pub spline: TerrainSpline,
    pub properties: TerrainSplineProperties,
    pub spline_cached: TerrainSplineCached,
    pub priority: ModifierPriority,
    pub transform_bundle: TransformBundle
}
#[derive(Component, Reflect, Default, PartialEq, Eq, PartialOrd, Ord)]
#[reflect(Component)]
pub struct ModifierPriority(pub u32);

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct TerrainAabb {
    min: Vec2,
    max: Vec2
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct TerrainSpline {
    pub curve: CubicCurve<Vec3>,
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct TerrainSplineProperties {
    pub width: f32,
    pub falloff: f32
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub(super) struct TerrainSplineCached {
    pub(super) points: Vec<Vec3>,
}

pub(super) fn update_terrain_spline_cache(
    mut query: Query<(&mut TerrainSplineCached, &mut TerrainAabb, &TerrainSpline, &TerrainSplineProperties, &GlobalTransform), Or<(Changed<TerrainSpline>, Changed<GlobalTransform>)>>,
    spline_simplification_distance: Res<MaximumSplineSimplificationDistance>
) {
    query.par_iter_mut().for_each(|(mut spline_cached, mut terrain_aabb, spline, spline_properties, global_transform)| {
        spline_cached.points.clear();

        spline_cached.points.extend(spline.curve.iter_positions(80).map(|point| global_transform.transform_point(point)));

        // Filter points that are very close together.
        let dedup_distance = (spline_properties.width * spline_properties.width).min(spline_simplification_distance.0);

        spline_cached.points.dedup_by(|a, b| a.distance_squared(*b) < dedup_distance);

        if spline_cached.points.is_empty() {
            terrain_aabb.min = Vec2::ZERO;
            terrain_aabb.max = Vec2::ZERO;
        } else {
            let (min, max) = spline_cached.points.iter().fold((spline_cached.points[0].xz(), spline_cached.points[0].xz()), |(min, max), point| (
                min.min(point.xz()),
                max.max(point.xz())
            ));

            terrain_aabb.min = min;
            terrain_aabb.max = max;
        }
    });
}