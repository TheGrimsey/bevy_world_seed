
#[cfg(feature = "count_samples")]
use std::cell::Cell;

#[cfg(feature = "count_samples")]
use bevy_log::info;
use ::noise::{NoiseFn, Simplex};
use bevy_asset::{Assets, Handle};
use bevy_math::{UVec4, Vec2, Vec4};
use bevy_ecs::prelude::{ReflectResource, Resource, Component, ReflectComponent};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_lookup_curve::LookupCurve;

use crate::{
    easing::EasingFunction,
    utils::{index_to_x_z, index_to_x_z_simd},
    TerrainSettings,
};

/// This is a cache of the index mapping between noise settings & the NoiseCache.
/// 
/// Yeah.
#[derive(Default, Resource, Debug)]
pub struct NoiseIndexCache {
    pub data_index_cache: Vec<u32>,
    pub spline_index_cache: Vec<u32>,

    // Offset for each group's noise cache.
    pub group_offset_cache: Vec<u32>,
    // All noise groups offsets.
    pub group_index_cache: Vec<u32>,
}
impl NoiseIndexCache {
    pub fn fill_cache(
        &mut self,
        terrain_noise_settings: &TerrainNoiseSettings,
        noise_cache: &mut NoiseCache
    ) {
        self.data_index_cache.clear();
        self.data_index_cache.extend(
            terrain_noise_settings
                .data
                .iter()
                .map(|layer| noise_cache.get_simplex_index(layer.seed) as u32),
        );
        
        self.spline_index_cache.clear();
        self.spline_index_cache.extend(
            terrain_noise_settings
                .splines
                .iter()
                .map(|spline| noise_cache.get_simplex_index(spline.seed) as u32),
        );

        self.group_offset_cache.clear();
        self.group_index_cache.clear();

        for group in terrain_noise_settings.noise_groups.iter() {
            let offset = self.group_index_cache.len();
            self.group_offset_cache.push(offset as u32);

            self.group_index_cache.extend(
                group.layers.iter().map(|layer| match &layer.operation {
                    LayerOperation::Noise { noise } => noise_cache.get_simplex_index(noise.seed) as u32,
                    LayerOperation::Step { .. } => 0,
                })
            );
        }
    }
}

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

    /// # Safety
    ///  This is fine as long as the noise has already been initialized (using for example [`NoiseCache::get_simplex_index`])
    #[inline]
    pub unsafe fn get_by_index(&self, index: usize) -> &Simplex {
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
    pub domain_warp: Vec<DomainWarping>,

    pub filters: Vec<NoiseFilter>,
    pub filter_combinator: StrengthCombinator
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

        #[cfg(feature = "count_samples")]
        SAMPLES.set(SAMPLES.get() + 1);

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
        noise_settings: &TerrainNoiseSettings,
        noise_cache: &NoiseCache,
        data_noise_values: &[f32],
        biome_values: &[f32],
        spline_noise_cache: &[u32],
        noise: &Simplex,
        lookup_curves: &Assets<LookupCurve>,
    ) -> f32 {
        if let Some(curve) = lookup_curves.get(&self.amplitude_curve) {
            let strength = calc_filter_strength(Vec2::new(x, z), &self.filters, self.filter_combinator, noise_settings, noise_cache, data_noise_values, biome_values, spline_noise_cache);
            
            curve.lookup(self.sample_raw(x, z, noise)) * strength
        } else {
            0.0
        }
    }

    /// Sample the spline layer excluding the curve.
    ///
    /// The result is normalized.
    #[inline]
    fn sample_simd_raw(&self, x: Vec4, z: Vec4, noise: &Simplex) -> Vec4 {
        let (x, z) = self.domain_warp.iter().fold((x, z), |(x, z), warp| warp.warp_simd(x, z, noise));

        #[cfg(feature = "count_samples")]
        SAMPLES.set(SAMPLES.get() + 4);

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
        noise_settings: &TerrainNoiseSettings,
        noise_cache: &NoiseCache,
        data_noise_values: &[Vec4],
        biome_values: &[Vec4],
        spline_noise_cache: &[u32],
        noise: &Simplex,
        lookup_curves: &Assets<LookupCurve>,
    ) -> Vec4 {
        // Fetch the lookup curve and apply it to all 4 noise values
        if let Some(curve) = lookup_curves.get(&self.amplitude_curve) {
            let strength = calc_filter_strength_simd(x, z, &self.filters, self.filter_combinator, noise_settings, noise_cache, data_noise_values, biome_values, spline_noise_cache);
            
            let normalized_noise = self.sample_simd_raw(x, z, noise);

            Vec4::new(
                curve.lookup(normalized_noise.x),
                curve.lookup(normalized_noise.y),
                curve.lookup(normalized_noise.z),
                curve.lookup(normalized_noise.w),
            ) * strength
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
pub struct LayerNoiseSettings {
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
impl LayerNoiseSettings {
    /// Sample noise at the x & z coordinates WITHOUT amplitude.
    ///
    /// `noise` is expected to be a Simplex noise initialized with this `TerrainNoiseLayer`'s `seed`.
    /// It is not contained within the noise layer to keep the size of a layer smaller.
    ///
    /// The result is scaled according to `scaling`.
    #[inline]
    pub fn sample_scaled_raw(&self, x: f32, z: f32, noise: &Simplex) -> f32 {
        let (x, z) = self.domain_warp.iter().fold((x, z), |(x, z), warp| warp.warp(x, z, noise));

        #[cfg(feature = "count_samples")]
        SAMPLES.set(SAMPLES.get() + 1);

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

        #[cfg(feature = "count_samples")]
        SAMPLES.set(SAMPLES.get() + 4);
        
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
impl Default for LayerNoiseSettings {
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
#[derive(Reflect, Clone, PartialEq)]
#[reflect(Default)]
pub enum LayerOperation {
    Noise {
        noise: LayerNoiseSettings
    },
    /// Creates terracing.
    Step {
        step: f32,
    },
}
impl Default for LayerOperation {
    fn default() -> Self {
        LayerOperation::Noise { noise: LayerNoiseSettings::default() }
    }
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Default, Clone, PartialEq)]
#[reflect(Default)]
pub struct NoiseLayer {
    pub operation: LayerOperation,

    pub filters: Vec<NoiseFilter>,
    pub filter_combinator: StrengthCombinator
}

#[derive(Component, Reflect, Default, Clone, PartialEq)]
#[reflect(Component)]
pub struct TileBiomes(pub Vec<f32>);

pub fn calc_filter_strength(
    pos: Vec2,
    filters: &[NoiseFilter],
    combinator: StrengthCombinator,
    noise_settings: &TerrainNoiseSettings,
    noise_cache: &NoiseCache,
    data_noise_values: &[f32],
    biome_values: &[f32],
    spline_noise_cache: &[u32]
) -> f32 {
    if let Some(initial_filter) = filters.first() {
        unsafe {
            let sample_filter = |filter: &NoiseFilter| match &filter.compare_to {
                FilterComparingTo::Data { index } => data_noise_values
                    .get(*index as usize)
                    .cloned()
                    .unwrap_or(0.0),
                FilterComparingTo::Spline { index } => noise_settings
                    .splines
                    .get(*index as usize)
                    .map_or(0.0, |spline| {
                        spline.sample_raw(
                            pos.x,
                            pos.y,
                            noise_cache.get_by_index(spline_noise_cache[*index as usize] as usize),
                        )
                    }),
                FilterComparingTo::Biome { index } => {
                    biome_values.get(*index as usize).cloned().unwrap_or_else(||
                        noise_settings
                            .biome
                            .get(*index as usize)
                            .map_or(0.0, |biome| {
                                calc_filter_strength(pos, &biome.filters, biome.filter_combinator, noise_settings, noise_cache, data_noise_values, biome_values, spline_noise_cache)
                            })
                    )
                }
            };
            let initial_filter_strength = initial_filter.get_filter(sample_filter(initial_filter));
    
            let calculated_filter_strength = filters.iter().skip(1).fold(initial_filter_strength, |acc, filter| {
                let filter_sample = filter.get_filter(sample_filter(filter));
                
                combinator.combine(acc, filter_sample)
            });
    
            calculated_filter_strength
        }
    } else {
        1.0
    }
}

fn calc_filter_strength_simd(
    x: Vec4,
    z: Vec4,
    filters: &[NoiseFilter],
    combinator: StrengthCombinator,
    noise_settings: &TerrainNoiseSettings,
    noise_cache: &NoiseCache,
    data_noise_values: &[Vec4],
    biome_values: &[Vec4],
    spline_noise_cache: &[u32]
) -> Vec4 {
    if let Some(initial_filter) = filters.first() {
        unsafe {
            let sample_filter = |filter: &NoiseFilter| match &filter.compare_to {
                FilterComparingTo::Data { index } => data_noise_values
                    .get(*index as usize)
                    .cloned()
                    .unwrap_or(Vec4::ZERO),
                FilterComparingTo::Spline { index } => noise_settings
                    .splines
                    .get(*index as usize)
                    .map_or(Vec4::ZERO, |spline| {
                        spline.sample_simd_raw(
                            x,
                            z,
                            noise_cache.get_by_index(spline_noise_cache[*index as usize] as usize),
                        )
                    }),   
                FilterComparingTo::Biome { index } => biome_values.get(*index as usize).cloned().unwrap_or_else(||
                    noise_settings
                        .biome
                        .get(*index as usize)
                        .map_or(Vec4::ZERO, |biome| {
                            calc_filter_strength_simd(x, z, &biome.filters, biome.filter_combinator, noise_settings, noise_cache, data_noise_values, biome_values, spline_noise_cache)
                        })
                )
            };
            let initial_filter_strength = initial_filter.get_filter_simd(sample_filter(initial_filter));
    
            let calculated_filter_strength = filters.iter().skip(1).fold(initial_filter_strength, |acc, filter| {
                let filter_sample = filter.get_filter_simd(sample_filter(filter));
                
                combinator.combine_simd(acc, filter_sample)
            });
    
            calculated_filter_strength
        }
    } else {
        Vec4::ONE
    }
}

/**
 * A grouping of noise layers which can be filtered upon.
 * 
 * Useable as a holder for Biome specific noise.
*/
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Default, Clone, PartialEq)]
#[reflect(Default)]
pub struct NoiseGroup {
    pub layers: Vec<NoiseLayer>,

    /// Filter this detail layer to only apply during certain conditions.
    pub filters: Vec<NoiseFilter>,
    pub filter_combinator: StrengthCombinator
}
impl NoiseGroup {
    pub fn sample(
        &self,
        noise_settings: &TerrainNoiseSettings,
        noise_cache: &NoiseCache,
        data_noise_values: &[f32],
        biome_values: &[f32],
        spline_noise_cache: &[u32],
        layer_noise_cache: &[u32],
        pos: Vec2,
    ) -> f32 {
        let group_strength = calc_filter_strength(pos, &self.filters, self.filter_combinator, noise_settings, noise_cache, data_noise_values, biome_values, spline_noise_cache);
        if group_strength <= f32::EPSILON {
            return 0.0;
        }

        unsafe {
            let group_height = self.layers.iter().enumerate().fold(0.0, |acc, (i, layer)| {
                let layer_strength = calc_filter_strength(pos, &layer.filters, layer.filter_combinator, noise_settings, noise_cache, data_noise_values, biome_values, spline_noise_cache);
    
                if layer_strength > f32::EPSILON {
                    let layer_value = match &layer.operation {
                        LayerOperation::Noise { noise } => {
                            noise.sample(
                                pos.x,
                                pos.y,
                                noise_cache.get_by_index(layer_noise_cache[i] as usize),
                            )
                        },
                        LayerOperation::Step { step } => {
                            (acc / *step).floor() * *step
                        },
                    };
        
                    acc + layer_value * layer_strength
                } else {
                    acc
                }
            });
         
            group_height * group_strength
        }
    }
    
    pub fn sample_simd(
        &self,
        noise_settings: &TerrainNoiseSettings,
        noise_cache: &NoiseCache,
        data_noise_values: &[Vec4],
        biome_values: &[Vec4],
        spline_noise_cache: &[u32],
        layer_noise_cache: &[u32],
        x: Vec4,
        z: Vec4
    ) -> Vec4 {
        let group_strength = calc_filter_strength_simd(x, z, &self.filters, self.filter_combinator, noise_settings, noise_cache, data_noise_values, biome_values, spline_noise_cache);
        if group_strength.cmple(Vec4::splat(f32::EPSILON)).all() {
            return Vec4::ZERO;
        }

        unsafe {
            let group_height = self.layers.iter().enumerate().fold(Vec4::ZERO, |acc, (i, layer)| {
                let layer_strength = calc_filter_strength_simd(x, z, &layer.filters, layer.filter_combinator, noise_settings, noise_cache, data_noise_values, biome_values, spline_noise_cache);
    
                if layer_strength.cmpge(Vec4::splat(f32::EPSILON)).any() {
                    let layer_value = match &layer.operation {
                        LayerOperation::Noise { noise } => {
                            noise.sample_simd(
                                x,
                                z,
                                noise_cache.get_by_index(layer_noise_cache[i] as usize),
                            )
                        },
                        LayerOperation::Step { step } => {
                            (acc / Vec4::splat(*step)).floor() * Vec4::splat(*step)
                        },
                    };
        
                    acc + layer_value * layer_strength
                } else {
                    acc
                }
            });

            group_height * group_strength
        }
    }
}


#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default)]
pub enum FilterComparingTo {
    /// Sample from a data noise.
    Data {
        index: u32
    },
    Spline {
        index: u32,
    },
    Biome {
        index: u32
    }
}
impl Default for FilterComparingTo {
    fn default() -> Self {
        FilterComparingTo::Data { index: 0 }
    }
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Default, Clone, PartialEq, Debug)]
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
        let strength = self.condition.evaluate_condition(noise_value, self.falloff);

        self.falloff_easing_function.ease(strength)
    }
    fn get_filter_simd(
        &self,
        noise_value: Vec4,
    ) -> Vec4 {
        let strength = self.condition.evaluate_condition_simd(noise_value, self.falloff);

        self.falloff_easing_function.ease_simd(strength)
    }
}


#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Default, Clone, Copy, PartialEq, Debug)]
#[reflect(Default)]
pub enum StrengthCombinator {
    #[default]
    Max,
    Min,
    Multiply,
    Sum,
    SumUncapped
}
impl StrengthCombinator {
    #[inline]
    pub fn combine(&self, a: f32, b: f32) -> f32 {
        match self {
            StrengthCombinator::Max => a.max(b),
            StrengthCombinator::Min => a.min(b),
            StrengthCombinator::Multiply => a * b,
            StrengthCombinator::Sum => (a + b).min(1.0),
            StrengthCombinator::SumUncapped => a + b,
        }
    }
    
    #[inline]
    pub fn combine_simd(&self, a: Vec4, b: Vec4) -> Vec4 {
        match self {
            StrengthCombinator::Max => a.max(b),
            StrengthCombinator::Min => a.min(b),
            StrengthCombinator::Multiply => a * b,
            StrengthCombinator::Sum => (a + b).min(Vec4::ONE),
            StrengthCombinator::SumUncapped => a + b,
        }
    }
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Reflect, Clone, PartialEq, Debug)]
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
impl NoiseFilterCondition {
    pub fn evaluate_condition(&self, value: f32, falloff: f32) -> f32 {
        match &self {
            NoiseFilterCondition::Above(threshold) => {
                1.0 - ((threshold - value) / falloff.max(f32::EPSILON)).clamp(0.0, 1.0)
            }
            NoiseFilterCondition::Below(threshold) => {
                1.0 - ((value - threshold) / falloff.max(f32::EPSILON)).clamp(0.0, 1.0)
            }
            NoiseFilterCondition::Between { min, max } => {
                let strength_below = 1.0
                    - ((min - value) / falloff.max(f32::EPSILON))
                        .clamp(0.0, 1.0);
                let strength_above = 1.0
                    - ((value - max) / falloff.max(f32::EPSILON))
                        .clamp(0.0, 1.0);

                strength_below.min(strength_above)
            }
        }
    }

    pub fn evaluate_condition_simd(&self, value: Vec4, falloff: f32) -> Vec4 {
        match &self {
            NoiseFilterCondition::Above(threshold) => {
                Vec4::ONE
                    - ((Vec4::splat(*threshold) - value)
                        / falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE)
            }
            NoiseFilterCondition::Below(threshold) => {
                Vec4::ONE
                    - ((value - Vec4::splat(*threshold))
                        / falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE)
            }
            NoiseFilterCondition::Between { min, max } => {
                let strength_below = 1.0
                    - ((Vec4::splat(*min) - value)
                        / falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE);
                let strength_above = 1.0
                    - ((value - Vec4::splat(*max))
                        / falloff.max(f32::EPSILON))
                    .clamp(Vec4::ZERO, Vec4::ONE);

                strength_below.min(strength_above)
            }
        }
    }
}

/**
 * A biome.
 * 
 * A collection of filters determining where a biome is placed.
 */
#[derive(Resource, Reflect, Clone, Default, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BiomeSettings {
    pub filters: Vec<NoiseFilter>,
    pub filter_combinator: StrengthCombinator

    // Biome flags?
}

/// Noise layers to be applied to Terrain tiles.
#[derive(Resource, Reflect, Clone, Default)]
#[reflect(Resource)]
pub struct TerrainNoiseSettings {
    pub biome: Vec<BiomeSettings>,

    /// Data noise.
    /// 
    /// Not applied to the world but can be used to for filters.
    pub data: Vec<LayerNoiseSettings>,

    pub splines: Vec<TerrainNoiseSplineLayer>,

    pub noise_groups: Vec<NoiseGroup>,
}
impl TerrainNoiseSettings {
    pub fn sample_data(
        &self,
        noise_cache: &NoiseCache,
        noise_index_cache: &NoiseIndexCache,
        pos: Vec2,
        data: &mut Vec<f32>
    ) {
        unsafe {
            data.extend(
                self.data.iter().zip(noise_index_cache.data_index_cache.iter())
                .map(|(data_layer, noise_index)| data_layer.sample_scaled_raw(pos.x, pos.y, noise_cache.get_by_index(*noise_index as usize)))
            );
        }
    }
    
    pub fn sample_data_simd(
        &self,
        noise_cache: &NoiseCache,
        noise_index_cache: &NoiseIndexCache,
        x: Vec4,
        z: Vec4,
        data: &mut Vec<Vec4>
    ) {
        unsafe {
            data.extend(
                self.data.iter().zip(noise_index_cache.data_index_cache.iter())
                .map(|(data_layer, noise_index)| data_layer.sample_simd_scaled_raw(x, z, noise_cache.get_by_index(*noise_index as usize)))
            );
        }
    }

    pub fn sample_biomes(
        &self,
        noise_cache: &NoiseCache,
        noise_index_cache: &NoiseIndexCache,
        pos: Vec2,
        data_noise_values: &[f32],
        biomes: &mut Vec<f32>
    ) {
        biomes.extend(
            self.biome.iter()
            .map(|biome| calc_filter_strength(pos, &biome.filters, biome.filter_combinator, self, noise_cache, data_noise_values, &[], &noise_index_cache.spline_index_cache))
        );
    }
    pub fn sample_biomes_simd(
        &self,
        noise_cache: &NoiseCache,
        noise_index_cache: &NoiseIndexCache,
        x: Vec4,
        z: Vec4,
        data_noise_values: &[Vec4],
        biomes: &mut Vec<Vec4>
    ) {
        biomes.extend(
            self.biome.iter()
            .map(|biome| calc_filter_strength_simd(x, z, &biome.filters, biome.filter_combinator, self, noise_cache, data_noise_values, &[], &noise_index_cache.spline_index_cache))
        );
    }

    /// Samples noise height at the position.
    ///
    /// Returns 0.0 if there are no noise layers.
    pub fn sample_position(
        &self,
        noise_cache: &NoiseCache,
        noise_index_cache: &NoiseIndexCache,
        pos: Vec2,
        lookup_curves: &Assets<LookupCurve>,
        data_noise_values: &[f32],
        biome_values: &[f32]
    ) -> f32 {
        unsafe {
            let spline_height = self.splines.iter().enumerate().fold(0.0, |acc, (i, layer)| {
                acc + layer.sample(
                    pos.x,
                    pos.y,
                    self,
                    noise_cache,
                    data_noise_values,
                    biome_values,
                    &noise_index_cache.spline_index_cache,
                    noise_cache.get_by_index(noise_index_cache.spline_index_cache[i] as usize),
                    lookup_curves,
                )
            });

            let layer_height = self.noise_groups.iter().enumerate().fold(0.0, |acc, (i, group)| {
                acc + group.sample(self, noise_cache, data_noise_values, biome_values, &noise_index_cache.spline_index_cache, &noise_index_cache.group_index_cache[noise_index_cache.group_offset_cache[i] as usize..], pos)
            });

            spline_height + layer_height
        }
    }

    pub fn sample_position_simd(
        &self,
        noise_cache: &NoiseCache,
        noise_index_cache: &NoiseIndexCache,
        lookup_curves: &Assets<LookupCurve>,
        x: Vec4,
        z: Vec4,
        data_cached: &[Vec4],
        biome_values: &[Vec4],
    ) -> Vec4 {
        unsafe {
            let spline_height = self.splines.iter().enumerate().fold(Vec4::ZERO, |acc, (i, layer)| {
                acc + layer.sample_simd(
                    x,
                    z,
                    self,
                    noise_cache,
                    data_cached,
                    biome_values,
                    &noise_index_cache.spline_index_cache,
                    noise_cache.get_by_index(noise_index_cache.spline_index_cache[i] as usize),
                    lookup_curves,
                )
            });

            let layer_height = self.noise_groups.iter().enumerate().fold(Vec4::ZERO, |acc, (i, group)| {
                acc + group.sample_simd(self, noise_cache, data_cached, biome_values, &noise_index_cache.spline_index_cache, &noise_index_cache.group_index_cache[noise_index_cache.group_offset_cache[i] as usize..], x, z)
            });

            spline_height + layer_height
        }
    }
}

#[cfg(feature = "count_samples")]
thread_local! {
    pub static SAMPLES: Cell<usize> = const { Cell::new(0) }; 
}

pub(super) fn apply_noise_simd(
    heights: &mut [f32],
    terrain_settings: &TerrainSettings,
    terrain_translation: Vec2,
    scale: f32,
    noise_cache: &NoiseCache,
    noise_index_cache: &NoiseIndexCache,
    lookup_curves: &Assets<LookupCurve>,
    terrain_noise_layers: &TerrainNoiseSettings,
) -> TileBiomes {

    let edge_points = terrain_settings.edge_points as usize;
    let length = heights.len();
    let simd_len = length / 4 * 4; // Length rounded down to the nearest multiple of 4

    #[cfg(feature = "count_samples")]
    SAMPLES.set(0);

    let mut tile_biomes = TileBiomes(vec![0.0; terrain_noise_layers.biome.len()]);

    let mut data_simd = Vec::with_capacity(terrain_noise_layers.data.len());
    let mut biomes_simd = Vec::with_capacity(terrain_noise_layers.biome.len());

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

        data_simd.clear();
        terrain_noise_layers.sample_data_simd(noise_cache, noise_index_cache, x_translated, z_translated, &mut data_simd);

        biomes_simd.clear();
        terrain_noise_layers.sample_biomes_simd(noise_cache, noise_index_cache, x_translated, z_translated, &data_simd, &mut biomes_simd);
        for (i, biome) in biomes_simd.iter().enumerate() {
            let max_biome_strength = &mut tile_biomes.0[i];

            *max_biome_strength = max_biome_strength.max(biome.max_element());
        }


        let final_heights = terrain_noise_layers.sample_position_simd(noise_cache, noise_index_cache, lookup_curves, x_translated, z_translated, &data_simd, &biomes_simd);

        // Store the results back into the heights array
        heights[i] = final_heights.x;
        heights[i + 1] = final_heights.y;
        heights[i + 2] = final_heights.z;
        heights[i + 3] = final_heights.w;
    }


    let mut data = Vec::with_capacity(terrain_noise_layers.data.len());
    let mut biomes = Vec::with_capacity(terrain_noise_layers.biome.len());

    // Process any remaining heights that aren't divisible by 4
    for (i, height) in heights.iter_mut().enumerate().skip(simd_len) {
        let (x, z) = index_to_x_z(i, edge_points);
        let vertex_position = terrain_translation + Vec2::new(x as f32 * scale, z as f32 * scale);

        data.clear();
        terrain_noise_layers.sample_data(noise_cache, noise_index_cache, vertex_position, &mut data);

        biomes.clear();
        terrain_noise_layers.sample_biomes(noise_cache, noise_index_cache, vertex_position, &data, &mut biomes);
        for (i, biome) in biomes.iter().enumerate() {
            let max_biome_strength = &mut tile_biomes.0[i];

            *max_biome_strength = max_biome_strength.max(*biome);
        }

        *height = terrain_noise_layers.sample_position(noise_cache, noise_index_cache, vertex_position, lookup_curves, &data, &biomes);
    }
    
    #[cfg(feature = "count_samples")]
    info!("Average samples: {}", SAMPLES.get() / length);

    tile_biomes
}
