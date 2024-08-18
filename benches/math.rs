use bevy::math::Vec2;
use bevy_terrain_test::minimum_distance;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Minimum distance", |b| {
        b.iter(black_box(|| minimum_distance(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0,  0.57))))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);