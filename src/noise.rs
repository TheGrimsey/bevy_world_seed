use bevy_lookup_curve::LookupCurve;
use ::noise::{NoiseFn, Simplex};
use bevy::{
    asset::{Assets, Handle}, math::{Vec2, Vec4}, prelude::{ReflectDefault, ReflectResource, Resource}, reflect::Reflect
};

use crate::{utils::index_to_x_z, TerrainSettings};

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
        let index = self.get_simplex_index(seed);

        &self.noises[index]
    }

    /// SAFETY: This is fine as long as the noise has already been initialized (using for example [`NoiseCache::get_simplex_index`]) 
    #[inline]
    pub(super) unsafe fn get_by_index(&self, index: usize) -> &Simplex {
        self.noises.get_unchecked(index)
    }

    #[inline]
    pub fn get_simplex_index(&mut self, seed: u32) -> usize {
        if let Some(index) = self
            .seeds
            .iter()
            .position(|existing_seed| *existing_seed == seed)
        {
            index
        } else {
            self.seeds.push(seed);
            self.noises.push(Simplex::new(seed));

            self.noises.len() - 1
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
    #[inline]
    pub fn sample(&self, x: f32, z: f32, noise: &Simplex, lookup_curves: &Assets<LookupCurve>) -> f32 {
        let noise_value =
            noise.get([(x * self.frequency) as f64, (z * self.frequency) as f64]) as f32;

        lookup_curves.get(&self.amplitude_curve).map_or(0.0, |curve| curve.lookup((noise_value / 2.0) + 0.5))
    }
    
    #[inline]
    pub fn sample_simd(
        &self,
        x: Vec4,
        z: Vec4,
        noise: &Simplex,
        lookup_curves: &Assets<LookupCurve>,
    ) -> Vec4 {
        // Step 1: Get the noise values for all 4 positions (x, z)
        let noise_values = Vec4::new(
            noise.get([(x.x * self.frequency) as f64, (z.x * self.frequency) as f64]) as f32,
            noise.get([(x.y * self.frequency) as f64, (z.y * self.frequency) as f64]) as f32,
            noise.get([(x.z * self.frequency) as f64, (z.z * self.frequency) as f64]) as f32,
            noise.get([(x.w * self.frequency) as f64, (z.w * self.frequency) as f64]) as f32,
        );

        // Step 2: Normalize noise values from [-1, 1] to [0, 1]
        let normalized_noise = (noise_values / 2.0) + Vec4::splat(0.5);

        // Step 3: Fetch the lookup curve and apply it to all 4 noise values
        if let Some(curve) = lookup_curves.get(&self.amplitude_curve) {
            Vec4::new(
                curve.lookup(normalized_noise.x),
                curve.lookup(normalized_noise.y),
                curve.lookup(normalized_noise.z),
                curve.lookup(normalized_noise.w),
            )
        } else {
            Vec4::ZERO  // Default to 0 if the curve isn't found
        }
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
    #[inline]
    pub fn sample(&self, x: f32, z: f32, noise: &Simplex) -> f32 {
        noise.get([(x * self.frequency) as f64, (z * self.frequency) as f64]) as f32
            * self.amplitude
    }
    
    #[inline]
    pub fn sample_simd(&self, x: Vec4, z: Vec4, noise: &Simplex) -> Vec4 {
        // Step 1: Get the noise values for all 4 positions (x, z)
        let noise_values = Vec4::new(
            noise.get([(x.x * self.frequency) as f64, (z.x * self.frequency) as f64]) as f32,
            noise.get([(x.y * self.frequency) as f64, (z.y * self.frequency) as f64]) as f32,
            noise.get([(x.z * self.frequency) as f64, (z.z * self.frequency) as f64]) as f32,
            noise.get([(x.w * self.frequency) as f64, (z.w * self.frequency) as f64]) as f32,
        );

        // Step 2: Multiply by the amplitude
        noise_values * Vec4::splat(self.amplitude)
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
        let spline_height = self.splines
            .iter()
            .fold(0.0, |acc, layer| {
                acc + layer.sample(pos.x, pos.y, noise_cache.get(layer.seed), lookup_curves)
        });

        let layer_height = self.layers
            .iter()
            .fold(0.0, |acc, layer: &TerrainNoiseDetailLayer| {
                acc + layer.sample(pos.x, pos.y, noise_cache.get(layer.seed))
            });
            
        spline_height + layer_height
    }
}

pub(super) fn apply_noise_simd(heights: &mut [f32], terrain_settings: &TerrainSettings, terrain_translation: Vec2, scale: f32, noise_cache: &NoiseCache, noise_spline_index_cache: &[u32], noise_detail_index_cache: &[u32], lookup_curves: &Assets<LookupCurve>, terrain_noise_layers: &TerrainNoiseSettings) {
    let edge_points = terrain_settings.edge_points as usize;
    let length = heights.len();
    let simd_len = length / 4 * 4;  // Length rounded down to the nearest multiple of 4

    // Process in chunks of 4
    for i in (0..simd_len).step_by(4) {
        // Unpack four (x, z) pairs in parallel
        let (x1, z1) = index_to_x_z(i, edge_points);
        let (x2, z2) = index_to_x_z(i + 1, edge_points);
        let (x3, z3) = index_to_x_z(i + 2, edge_points);
        let (x4, z4) = index_to_x_z(i + 3, edge_points);

        // Create SIMD vectors for x and z positions
        let x_positions = Vec4::new(x1 as f32 * scale, x2 as f32 * scale, x3 as f32 * scale, x4 as f32 * scale);
        let z_positions = Vec4::new(z1 as f32 * scale, z2 as f32 * scale, z3 as f32 * scale, z4 as f32 * scale);

        // Add terrain translation to the positions
        let x_translated = x_positions + Vec4::splat(terrain_translation.x);
        let z_translated = z_positions + Vec4::splat(terrain_translation.y);

        // Accumulate spline and layer heights for all 4 points in parallel
        let mut spline_heights = Vec4::ZERO;
        let mut layer_heights = Vec4::ZERO;

        unsafe {
            // Process all spline layers
            for (j, layer) in terrain_noise_layers.splines.iter().enumerate() {
                let noise = noise_cache.get_by_index(noise_spline_index_cache[j] as usize);
                let spline_values = layer.sample_simd(x_translated, z_translated, noise, lookup_curves);
                spline_heights += spline_values;
            }

            // Process all detail layers
            for (j, layer) in terrain_noise_layers.layers.iter().enumerate() {
                let noise = noise_cache.get_by_index(noise_detail_index_cache[j] as usize);
                let layer_values = layer.sample_simd(x_translated, z_translated, noise);
                layer_heights += layer_values;
            }

            // Store the results back into the heights array
            heights[i] = spline_heights.x + layer_heights.x;
            heights[i + 1] = spline_heights.y + layer_heights.y;
            heights[i + 2] = spline_heights.z + layer_heights.z;
            heights[i + 3] = spline_heights.w + layer_heights.w;
        }
    }

    // Process any remaining heights that aren't divisible by 4
    for i in simd_len..length {
        let (x, z) = index_to_x_z(i, edge_points);
        let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);

        unsafe {
            let mut spline_height = 0.0;
            for (j, layer) in terrain_noise_layers.splines.iter().enumerate() {
                spline_height += layer.sample(vertex_position.x, vertex_position.y, noise_cache.get_by_index(noise_spline_index_cache[j] as usize), lookup_curves);
            }

            let mut layer_height = 0.0;
            for (j, layer) in terrain_noise_layers.layers.iter().enumerate() {
                layer_height += layer.sample(vertex_position.x, vertex_position.y, noise_cache.get_by_index(noise_detail_index_cache[j] as usize));
            }

            heights[i] = spline_height + layer_height;
        }
    }
}