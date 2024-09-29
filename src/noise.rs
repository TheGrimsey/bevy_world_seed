use bevy_lookup_curve::LookupCurve;
use ::noise::{NoiseFn, Simplex};
use bevy::{
    asset::{Assets, Handle}, math::Vec2, prelude::{ReflectDefault, ReflectResource, Resource}, reflect::Reflect
};

/// Cache of Simplex noise instances & which seeds they map to.
#[derive(Default, Resource)]
pub struct NoiseCache {
    /// Seeds of the simplex noises.
    ///
    /// These are separated from the noises for cache coherency. Simplex is a big struct.
    seeds: Vec<u32>,
    noises: Vec<Simplex>,
}
impl NoiseCache {
    pub fn get(&mut self, seed: u32) -> &Simplex {
        if let Some(index) = self
            .seeds
            .iter()
            .position(|existing_seed| *existing_seed == seed)
        {
            &self.noises[index]
        } else {
            self.seeds.push(seed);
            self.noises.push(Simplex::new(seed));

            self.noises.last().unwrap()
        }
    }
}

#[derive(Clone, Reflect)]
pub struct TerrainNoiseSplineLayer {
    pub amplitude_curve: Handle<LookupCurve>,
    /// Scale of the noise on the XZ-plane.
    ///
    /// Increasing this causes the value to change quicker. Lower frequency means smoother changes in height.
    pub frequency: f32,
    /// Seed for the noise function.
    pub seed: u32,
}
impl TerrainNoiseSplineLayer {
    /// Sample noise at the x & z coordinates.
    ///
    /// `noise` is expected to be a Simplex noise initialized with this `TerrainNoiseBaseLayer`'s `seed`.
    /// It is not contained within the noise layer to keep the size of a layer smaller.
    pub fn sample(&self, x: f32, z: f32, noise: &Simplex, lookup_curves: &Assets<LookupCurve>) -> f32 {
        let noise_value =
            noise.get([(x * self.frequency) as f64, (z * self.frequency) as f64]) as f32;

        lookup_curves.get(&self.amplitude_curve).map_or(0.0, |curve| curve.lookup((noise_value / 2.0) + 0.5))
    }
}

#[derive(Reflect, Clone)]
#[reflect(Default)]
pub struct TerrainNoiseDetailLayer {
    /// Amplitude of the noise.
    ///
    /// Increasing this increases the variance of terrain heights from this layer.
    pub amplitude: f32,
    /// Scale of the noise on the XZ-plane.
    ///
    /// Increasing this causes the value to change quicker. Lower frequency means smoother changes in height.
    pub frequency: f32,
    /// Seed for the noise function.
    pub seed: u32,
}
impl TerrainNoiseDetailLayer {
    /// Sample noise at the x & z coordinates.
    ///
    /// `noise` is expected to be a Simplex noise initialized with this `TerrainNoiseLayer`'s `seed`.
    /// It is not contained within the noise layer to keep the size of a layer smaller.
    pub fn sample(&self, x: f32, z: f32, noise: &Simplex) -> f32 {
        noise.get([(x * self.frequency) as f64, (z * self.frequency) as f64]) as f32
            * self.amplitude
    }
}
impl Default for TerrainNoiseDetailLayer {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            frequency: 1.0,
            seed: 1,
        }
    }
}

/// Noise layers to be applied to Terrain tiles.
#[derive(Resource, Reflect, Clone, Default)]
#[reflect(Resource)]
pub struct TerrainNoiseSettings {
    pub splines: Vec<TerrainNoiseSplineLayer>,

    pub layers: Vec<TerrainNoiseDetailLayer>,
}
impl TerrainNoiseSettings {
    /// Samples noise height at the position.
    ///
    /// Returns 0.0 if there are no noise layers.
    pub fn sample_position(&self, noise_cache: &mut NoiseCache, pos: Vec2, lookup_curves: &Assets<LookupCurve>) -> f32 {
        let spline_height = self.splines.iter().fold(0.0, |acc, layer| {
            acc + layer.sample(pos.x, pos.y, noise_cache.get(layer.seed), lookup_curves)
        });

        self.layers
            .iter()
            .fold(spline_height, |acc, layer: &TerrainNoiseDetailLayer| {
                acc + layer.sample(pos.x, pos.y, noise_cache.get(layer.seed))
            })
    }
}
