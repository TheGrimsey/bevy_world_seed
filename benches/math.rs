use bevy::math::Vec2;
use bevy_terrain_test::{material::{apply_texture, get_height_at_position, TexturingRuleEvaluator}, minimum_distance};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Minimum distance", |b| {
        b.iter(black_box(|| minimum_distance(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0,  0.57))))
    });
    
    c.bench_function("Height at position", |b| {
        b.iter(black_box(|| get_height_at_position(1.0, 0.0, 0.5, 1.0, 0.5, 0.45)))
    });

    c.bench_function("Apply texture", |b| {
        b.iter(black_box(|| apply_texture(&mut [0, 230, 20, 5], 2, 1.0)))
    });
    
    c.bench_function("Evalutate rule (Above)", |b| {
        b.iter(black_box(|| TexturingRuleEvaluator::Above { height: 10.0, falloff: 1.0 }.eval(9.5, 0.0)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);