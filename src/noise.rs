use ::noise::{NoiseFn, Simplex};
use bevy::{
    asset::{Assets, Handle}, math::{UVec4, Vec2, Vec4}, prelude::{ReflectDefault, ReflectResource, Resource}, reflect::Reflect
};
use bevy_lookup_curve::LookupCurve;

use crate::{
    easing::EasingFunction,
    utils::{index_to_x_z, index_to_x_z_simd},
    TerrainSettings,
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
        let index = self.get_simplex_index(seed);

        &self.noises[index]
    }

    /// SAFETY: This is fine as long as the noise has already been initialized (using for example [`NoiseCache::get_simplex_index`])
    #[inline]
    pub(super) unsafe fn get_by_index(&self, index: usize) -> &Simplex {
        self.noises.get_unchecked(index)
    }

    /// Returns the index containing the noise with the supplied seed.
    /// Inserts it if it doesn't exist.
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
    /// Applies domain warping to the noise layer.
    pub domain_warp: Vec<DomainWarping>
}
impl TerrainNoiseSplineLayer {
    /// Sample noise at the x & z coordinates WITHOUT amplitude curve.
    ///
    /// `noise` is expected to be a Simplex noise initialized with this `TerrainNoiseBaseLayer`'s `seed`.
    /// It is not contained within the noise layer to keep the size of a layer smaller.
    ///
    /// The result is normalized.
    #[inline]
    pub fn sample_raw(&self, x: f32, z: f32, noise: &Simplex) -> f32 {
        let (x, z) = self.domain_warp.iter().fold((x, z), |(x, z), warp| warp.warp(x, z, noise));

        (noise.get([(x * self.frequency) as f64, (z * self.frequency) as f64]) / 2.0 + 0.5) as f32
    }

    /// Sample noise at the x & z coordinates.
    ///
    /// `noise` is expected to be a Simplex noise initialized with this `TerrainNoiseBaseLayer`'s `seed`.
    /// It is not contained within the noise layer to keep the size of a layer smaller.
    #[inline]
    pub fn sample(
        &self,
        x: f32,
        z: f32,
        noise: &Simplex,
        lookup_curves: &Assets<LookupCurve>,
    ) -> f32 {
        lookup_curves
            .get(&self.amplitude_curve)
            .map_or(0.0, |curve| curve.lookup(self.sample_raw(x, z, noise)))
    }

    /// Sample the spline layer excluding the curve.
    ///
    /// The result is normalized.
    #[inline]
    fn sample_simd_raw(&self, noise: &Simplex, x: Vec4, z: Vec4) -> Vec4 {
        let (x, z) = self.domain_warp.iter().fold((x, z), |(x, z), warp| warp.warp_simd(x, z, noise));

        // Step 1: Get the noise values for all 4 positions (x, z)
        let noise_values = Vec4::new(
            noise.get([(x.x * self.frequency) as f64, (z.x * self.frequency) as f64]) as f32,
            noise.get([(x.y * self.frequency) as f64, (z.y * self.frequency) as f64]) as f32,
            noise.get([(x.z * self.frequency) as f64, (z.z * self.frequency) as f64]) as f32,
            noise.get([(x.w * self.frequency) as f64, (z.w * self.frequency) as f64]) as f32,
        );

        // Step 2: Normalize noise values from [-1, 1] to [0, 1]
        (noise_values / 2.0) + Vec4::splat(0.5)
    }

    #[inline]
    pub fn sample_simd(
        &self,
        x: Vec4,
        z: Vec4,
        noise: &Simplex,
        lookup_curves: &Assets<LookupCurve>,
    ) -> Vec4 {
        // Fetch the lookup curve and apply it to all 4 noise values
        if let Some(curve) = lookup_curves.get(&self.amplitude_curve) {
            let normalized_noise = self.sample_simd_raw(noise, x, z);

            Vec4::new(
                curve.lookup(normalized_noise.x),
                curve.lookup(normalized_noise.y),
                curve.lookup(normalized_noise.z),
                curve.lookup(normalized_noise.w),
            )
        } else {
            Vec4::ZERO // Default to 0 if the curve isn't found
        }
    }
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq, Default)]
#[reflect(Default)]
pub struct DomainWarping {
    pub amplitude: f32,
    pub frequency: f32,
    pub z_offset: f32
}
impl DomainWarping {
    pub fn warp(&self, x: f32, z: f32, noise: &Simplex) -> (f32, f32) {
        (
            x + self.amplitude * noise.get([(x * self.frequency) as f64, (z * self.frequency) as f64]) as f32,
            z + self.amplitude * noise.get([(x * self.frequency + self.z_offset) as f64, (z * self.frequency + self.z_offset) as f64]) as f32,
        )
    }

    pub fn warp_simd(&self, x: Vec4, z: Vec4, noise: &Simplex) -> (Vec4, Vec4) {
        let noise_x = x * self.frequency;
        let noise_z = z * self.frequency;
        
        let offset_x = Vec4::new(
            noise.get([noise_x.x as f64, noise_z.x as f64]) as f32,
            noise.get([noise_x.y as f64, noise_z.y as f64]) as f32,
            noise.get([noise_x.z as f64, noise_z.z as f64]) as f32,
            noise.get([noise_x.w as f64, noise_z.w as f64]) as f32
        );

        let noise_x = noise_x + self.z_offset;
        let noise_z = noise_z + self.z_offset;

        let offset_z = Vec4::new(
            noise.get([noise_x.x as f64, noise_z.x as f64]) as f32,
            noise.get([noise_x.y as f64, noise_z.y as f64]) as f32,
            noise.get([noise_x.z as f64, noise_z.z as f64]) as f32,
            noise.get([noise_x.w as f64, noise_z.w as f64]) as f32
        );

        (
            x + self.amplitude * offset_x,
            z + self.amplitude * offset_z,
        )
    }
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq, Default)]
#[reflect(Default)]
pub enum NoiseScaling {
    /// Noise is scaled to range -1.0..=1.0
    #[default]
    Normalized,
    /// Noise is scaled to range 0.0..=1.0.
    /// 
    /// This makes the noise only additive.
    Unitized,
    /// Noise is normalized and the absolute value is returned.
    /// 
    /// Useful to make wavy hills.
    Billow,
    /// Noise is normalized and the complement to the absolute value is returned.
    /// 
    /// Useful to make shapr alpine like ridges.
    Ridged
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq)]
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
    /// Applies domain warping to the noise layer.
    pub domain_warp: Vec<DomainWarping>,
    /// Scaling of the noise.
    pub scaling: NoiseScaling
}
impl TerrainNoiseDetailLayer {
    /// Sample noise at the x & z coordinates WITHOUT amplitude.
    ///
    /// `noise` is expected to be a Simplex noise initialized with this `TerrainNoiseLayer`'s `seed`.
    /// It is not contained within the noise layer to keep the size of a layer smaller.
    ///
    /// The result is scaled according to `scaling`.
    #[inline]
    pub fn sample_scaled_raw(&self, x: f32, z: f32, noise: &Simplex) -> f32 {
        let (x, z) = self.domain_warp.iter().fold((x, z), |(x, z), warp| warp.warp(x, z, noise));

        let noise = noise.get([(x * self.frequency) as f64, (z * self.frequency) as f64]) as f32;

        match &self.scaling {
            NoiseScaling::Normalized => noise,
            NoiseScaling::Unitized => noise / 2.0 + 0.5,
            NoiseScaling::Billow => noise.abs(),
            NoiseScaling::Ridged => 1.0 - noise.abs(),
        }
    }

    /// Sample noise at the x & z coordinates.
    ///
    /// `noise` is expected to be a Simplex noise initialized with this `TerrainNoiseLayer`'s `seed`.
    /// It is not contained within the noise layer to keep the size of a layer smaller.
    #[inline]
    pub fn sample(&self, x: f32, z: f32, noise: &Simplex) -> f32 {
        self.sample_scaled_raw(x, z, noise) * self.amplitude
    }

    /// The result is normalized.
    #[inline]
    fn sample_simd_scaled_raw(&self, x: Vec4, z: Vec4, noise: &Simplex) -> Vec4 {
        let (x, z) = self.domain_warp.iter().fold((x, z), |(x, z), warp| warp.warp_simd(x, z, noise));

        let x = x * self.frequency;
        let z = z * self.frequency;

        let noise =Vec4::new(
            noise.get([x.x as f64, z.x as f64]) as f32,
            noise.get([x.y as f64, z.y as f64]) as f32,
            noise.get([x.z as f64, z.z as f64]) as f32,
            noise.get([x.w as f64, z.w as f64]) as f32,
        );

        match &self.scaling {
            NoiseScaling::Normalized => noise,
            NoiseScaling::Unitized => noise / 2.0 + Vec4::splat(0.5),
            NoiseScaling::Billow => noise.abs(),
            NoiseScaling::Ridged => Vec4::ONE - noise.abs(),
        }
    }

    #[inline]
    pub fn sample_simd(&self, x: Vec4, z: Vec4, noise: &Simplex) -> Vec4 {
        // Step 1: Get the noise values for all 4 positions (x, z)
        let noise_values = self.sample_simd_scaled_raw(x, z, noise);

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
            domain_warp: vec![],
            scaling: NoiseScaling::Normalized
        }
    }
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Default, Clone, PartialEq)]
#[reflect(Default)]
pub struct FilteredTerrainNoiseDetailLayer {
    pub layer: TerrainNoiseDetailLayer,

    /// Filter this detail layer to only apply during certain conditions.
    pub filter: Vec<NoiseFilter>,
    pub filter_combinator: FilterCombinator
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Default, Clone, PartialEq)]
#[reflect(Default)]
pub enum FilterComparingTo {
    #[default]
    ToSelf,
    /// Sample from a data noise.
    Data {
        index: u32
    },
    Spline {
        index: u32,
    },
    Detail {
        index: u32,
    },
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Default, Clone, PartialEq)]
#[reflect(Default)]
pub struct NoiseFilter {
    pub condition: NoiseFilterCondition,
    pub falloff: f32,
    pub falloff_easing_function: EasingFunction,
    pub compare_to: FilterComparingTo,
}
impl NoiseFilter {
    fn get_filter(
        &self,
        noise_value: f32,
    ) -> f32 {
        let strength = match &self.condition {
            NoiseFilterCondition::Above(threshold) => {
                1.0 - ((threshold - noise_value) / self.falloff.max(f32::EPSILON)).clamp(0.0, 1.0)
            }
            NoiseFilterCondition::Below(threshold) => {
                1.0 - ((noise_value - threshold) / self.falloff.max(f32::EPSILON)).clamp(0.0, 1.0)
            }
            NoiseFilterCondition::Between { min, max } => {
                let strength_below = 1.0
                    - ((min - noise_value) / self.falloff.max(f32::EPSILON))
                        .clamp(0.0, 1.0);
                let strength_above = 1.0
                    - ((noise_value - max) / self.falloff.max(f32::EPSILON))
                        .clamp(0.0, 1.0);

                strength_below.min(strength_above)
            }
        };

        self.falloff_easing_function.ease(strength)
    }
    fn get_filter_simd(
        &self,
        noise_value: Vec4,
    ) -> Vec4 {
        let strength = match &self.condition {
            NoiseFilterCondition::Above(threshold) => {
                Vec4::ONE
                    - ((Vec4::splat(*threshold) - noise_value)
                        / self.falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE)
            }
            NoiseFilterCondition::Below(threshold) => {
                Vec4::ONE
                    - ((noise_value - Vec4::splat(*threshold))
                        / self.falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE)
            }
            NoiseFilterCondition::Between { min, max } => {
                let strength_below = 1.0
                    - ((Vec4::splat(*min) - noise_value)
                        / self.falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE);
                let strength_above = 1.0
                    - ((noise_value - Vec4::splat(*max))
                        / self.falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE);

                strength_below.min(strength_above)
            }
        };

        self.falloff_easing_function.ease_simd(strength)
    }
}


#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Default, Clone, PartialEq)]
#[reflect(Default)]
pub enum FilterCombinator {
    #[default]
    Max,
    Min,
    Multiply,
    Sum,
    SumUncapped
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq)]
#[reflect(Default)]
pub enum NoiseFilterCondition {
    Above(f32),
    Below(f32),
    Between { min: f32, max: f32 },
}
impl Default for NoiseFilterCondition {
    fn default() -> Self {
        Self::Above(0.5)
    }
}

/// Noise layers to be applied to Terrain tiles.
#[derive(Resource, Reflect, Clone, Default)]
#[reflect(Resource)]
pub struct TerrainNoiseSettings {
    /// Data noise.
    /// 
    /// Not applied to the world but can be used to for filters.
    pub data: Vec<TerrainNoiseDetailLayer>,

    pub splines: Vec<TerrainNoiseSplineLayer>,

    pub layers: Vec<FilteredTerrainNoiseDetailLayer>,
}
impl TerrainNoiseSettings {
    /// Samples noise height at the position.
    ///
    /// Returns 0.0 if there are no noise layers.
    pub fn sample_position(
        &self,
        noise_cache: &mut NoiseCache,
        pos: Vec2,
        lookup_curves: &Assets<LookupCurve>,
    ) -> f32 {
        let spline_height = self.splines.iter().fold(0.0, |acc, layer| {
            acc + layer.sample(pos.x, pos.y, noise_cache.get(layer.seed), lookup_curves)
        });

        let layer_height = self.layers.iter().fold(0.0, |acc, layer| {
            acc + layer
                .layer
                .sample(pos.x, pos.y, noise_cache.get(layer.layer.seed))
        });

        spline_height + layer_height
    }
}

pub(super) fn apply_noise_simd(
    heights: &mut [f32],
    terrain_settings: &TerrainSettings,
    terrain_translation: Vec2,
    scale: f32,
    noise_cache: &NoiseCache,
    noise_data_index_cache: &[u32],
    noise_spline_index_cache: &[u32],
    noise_detail_index_cache: &[u32],
    lookup_curves: &Assets<LookupCurve>,
    terrain_noise_layers: &TerrainNoiseSettings,
) {
    let edge_points = terrain_settings.edge_points as usize;
    let length = heights.len();
    let simd_len = length / 4 * 4; // Length rounded down to the nearest multiple of 4

    // Process in chunks of 4
    for i in (0..simd_len).step_by(4) {
        // Unpack four (x, z) pairs in parallel
        let (x, z) = index_to_x_z_simd(
            UVec4::new(i as u32, i as u32 + 1, i as u32 + 2, i as u32 + 3),
            edge_points as u32,
        );

        // Create SIMD vectors for x and z positions
        let x_positions = x.as_vec4() * scale;
        let z_positions = z.as_vec4() * scale;

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
                let spline_values =
                    layer.sample_simd(x_translated, z_translated, noise, lookup_curves);
                spline_heights += spline_values;
            }

            // Process all detail layers
            for (j, layer) in terrain_noise_layers.layers.iter().enumerate() {
                let noise = noise_cache.get_by_index(noise_detail_index_cache[j] as usize);
                let mut layer_values = layer.layer.sample_simd(x_translated, z_translated, noise);

                if let Some(initial_filter) = layer.filter.first() {
                    let sample_filter = |filter: &NoiseFilter| match &filter.compare_to {
                        FilterComparingTo::ToSelf => layer_values / layer.layer.amplitude,
                        FilterComparingTo::Data { index } => terrain_noise_layers
                            .data
                            .get(*index as usize)
                            .map(|layer| {
                                layer.sample_simd_scaled_raw(
                                    x_translated,
                                    z_translated,
                                    noise_cache.get_by_index(
                                        noise_data_index_cache[*index as usize] as usize,
                                    ),
                                )
                            })
                            .unwrap_or(Vec4::ZERO),
                        FilterComparingTo::Spline { index } => terrain_noise_layers
                            .splines
                            .get(*index as usize)
                            .map(|spline| {
                                spline.sample_simd_raw(
                                    noise_cache.get_by_index(
                                        noise_spline_index_cache[*index as usize] as usize,
                                    ),
                                    x_translated,
                                    z_translated,
                                )
                            })
                            .unwrap_or(Vec4::ZERO),
                        FilterComparingTo::Detail { index } => terrain_noise_layers
                            .layers
                            .get(*index as usize)
                            .map(|layer| {
                                layer.layer.sample_simd_scaled_raw(
                                    x_translated,
                                    z_translated,
                                    noise_cache.get_by_index(
                                        noise_detail_index_cache[*index as usize] as usize,
                                    ),
                                )
                            })
                            .unwrap_or(Vec4::ZERO),
                    };
                    let initial_filter_strength = initial_filter.get_filter_simd(sample_filter(initial_filter));

                    let calculated_filter_strength = layer.filter.iter().skip(1).fold(initial_filter_strength, |acc, filter| {
                        let filter_sample = filter.get_filter_simd(sample_filter(filter));
                        
                        match layer.filter_combinator {
                            FilterCombinator::Max => acc.max(filter_sample),
                            FilterCombinator::Min => acc.min(filter_sample),
                            FilterCombinator::Multiply => acc * filter_sample,
                            FilterCombinator::Sum => (acc + filter_sample).min(Vec4::splat(1.0)),
                            FilterCombinator::SumUncapped => acc + filter_sample,
                        }
                    });

                    layer_values *= calculated_filter_strength;
                }

                layer_heights += layer_values;
            }

            let final_heights = spline_heights + layer_heights;

            // Store the results back into the heights array
            heights[i] = final_heights.x;
            heights[i + 1] = final_heights.y;
            heights[i + 2] = final_heights.z;
            heights[i + 3] = final_heights.w;
        }
    }

    // Process any remaining heights that aren't divisible by 4
    for (i, height) in heights.iter_mut().enumerate().skip(simd_len) {
        let (x, z) = index_to_x_z(i, edge_points);
        let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);

        unsafe {
            let mut spline_height = 0.0;
            for (j, layer) in terrain_noise_layers.splines.iter().enumerate() {
                spline_height += layer.sample(
                    vertex_position.x,
                    vertex_position.y,
                    noise_cache.get_by_index(noise_spline_index_cache[j] as usize),
                    lookup_curves,
                );
            }

            let mut layer_height = 0.0;
            for (j, layer) in terrain_noise_layers.layers.iter().enumerate() {
                let mut layer_value = layer.layer.sample(
                    vertex_position.x,
                    vertex_position.y,
                    noise_cache.get_by_index(noise_detail_index_cache[j] as usize),
                );

                if let Some(initial_filter) = layer.filter.first() {
                    let sample_filter = |filter: &NoiseFilter| match &filter.compare_to {
                        FilterComparingTo::ToSelf => layer_value / layer.layer.amplitude,
                        FilterComparingTo::Data { index } => terrain_noise_layers
                            .data
                            .get(*index as usize)
                            .map(|layer| {
                                layer.sample_scaled_raw(
                                    vertex_position.x,
                                    vertex_position.y,
                                    noise_cache.get_by_index(
                                        noise_data_index_cache[*index as usize] as usize,
                                    ),
                                )
                            })
                            .unwrap_or(0.0),
                        FilterComparingTo::Spline { index } => terrain_noise_layers
                            .splines
                            .get(*index as usize)
                            .map(|spline| {
                                spline.sample_raw(
                                    vertex_position.x,
                                    vertex_position.y,
                                    noise_cache.get_by_index(
                                        noise_spline_index_cache[*index as usize] as usize,
                                    ),
                                )
                            })
                            .unwrap_or(0.0),
                        FilterComparingTo::Detail { index } => terrain_noise_layers
                            .layers
                            .get(*index as usize)
                            .map(|layer| {
                                layer.layer.sample_scaled_raw(
                                    vertex_position.x,
                                    vertex_position.y,
                                    noise_cache.get_by_index(
                                        noise_detail_index_cache[*index as usize] as usize,
                                    ),
                                )
                            })
                            .unwrap_or(0.0),
                    };
                    let initial_filter_strength = initial_filter.get_filter(sample_filter(initial_filter));

                    let calculated_filter_strength = layer.filter.iter().skip(1).fold(initial_filter_strength, |acc, filter| {
                        let filter_sample = filter.get_filter(sample_filter(filter));
                        
                        match layer.filter_combinator {
                            FilterCombinator::Max => acc.max(filter_sample),
                            FilterCombinator::Min => acc.min(filter_sample),
                            FilterCombinator::Multiply => acc * filter_sample,
                            FilterCombinator::Sum => (acc + filter_sample).min(1.0),
                            FilterCombinator::SumUncapped => acc + filter_sample,
                        }
                    });

                    layer_value *= calculated_filter_strength;
                }

                layer_height += layer_value;
            }

            *height = spline_height + layer_height;
        }
    }
}
