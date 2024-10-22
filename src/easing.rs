use bevy::math::Vec4;
use bevy::prelude::ReflectDefault;
use bevy::reflect::Reflect;
use core::f32;

#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, Reflect, Default, PartialEq, Eq)]
#[reflect(Default)]
pub enum EasingFunction {
    #[default]
    Linear,

    // Sine functions.
    SineIn,
    SineOut,
    SineInOut,

    // Cubic functions
    CubicIn,
    CubicOut,
    CubicInOut,

    // Back functions
    BackIn,
    BackOut,
    BackInOut,
}

// Back easing constant.
const BACK_CONSTANT: f32 = 1.70158;

impl EasingFunction {
    #[inline]
    pub fn ease(&self, x: f32) -> f32 {
        match self {
            EasingFunction::Linear => x,
            EasingFunction::SineIn => 1.0 - ((x * f32::consts::PI) / 2.0).cos(),
            EasingFunction::SineOut => ((x * f32::consts::PI) / 2.0).sin(),
            EasingFunction::SineInOut => -((x * f32::consts::PI).cos() - 1.0) / 2.0,
            EasingFunction::CubicIn => x * x * x,
            EasingFunction::CubicOut => 1.0 - (1.0 - x).powf(3.0),
            EasingFunction::CubicInOut => {
                if x < 0.5 {
                    4.0 * x * x * x
                } else {
                    1.0 - (-2.0 * x + 2.0).powf(3.0) / 2.0
                }
            }
            EasingFunction::BackIn => {
                let c3 = BACK_CONSTANT + 1.0;

                c3 * x * x * x - BACK_CONSTANT * x * x
            }
            EasingFunction::BackOut => {
                let c3 = BACK_CONSTANT + 1.0;

                1.0 + c3 * (x - 1.0).powf(3.0) + BACK_CONSTANT * (x - 1.0).powf(2.0)
            }
            EasingFunction::BackInOut => {
                let c2 = BACK_CONSTANT * 1.525;

                if x < 0.5 {
                    ((2.0 * x).powf(2.0) * ((c2 + 1.0) * 2.0 * x - c2)) / 2.0
                } else {
                    ((2.0 * x - 2.0).powf(2.0) * ((c2 + 1.0) * (2.0 * x - 2.0) + c2) + 2.0) / 2.0
                }
            }
        }
    }

    #[inline]
    pub fn ease_simd(&self, x: Vec4) -> Vec4 {
        match self {
            EasingFunction::Linear => x,
            EasingFunction::SineIn => {
                let half_pi = (x * f32::consts::PI) / 2.0;

                Vec4::ONE
                    - Vec4::new(
                        half_pi.x.cos(),
                        half_pi.y.cos(),
                        half_pi.z.cos(),
                        half_pi.w.cos(),
                    )
            }
            EasingFunction::SineOut => {
                let half_pi = (x * f32::consts::PI) / 2.0;

                Vec4::new(
                    half_pi.x.sin(),
                    half_pi.y.sin(),
                    half_pi.z.sin(),
                    half_pi.w.sin(),
                )
            }
            EasingFunction::SineInOut => {
                let mul_pi = x * f32::consts::PI;

                -(Vec4::new(
                    mul_pi.x.cos(),
                    mul_pi.y.cos(),
                    mul_pi.z.cos(),
                    mul_pi.w.cos(),
                ) - Vec4::ONE)
                    / 2.0
            }
            EasingFunction::CubicIn => x * x * x,
            EasingFunction::CubicOut => Vec4::ONE - (Vec4::ONE - x).powf(3.0),
            EasingFunction::CubicInOut => {
                let two = Vec4::splat(2.0);

                let is_less = x.cmplt(Vec4::splat(0.5));

                let cubic_in = 4.0 * x * x * x;
                let cubic_out = Vec4::ONE - (-two * x + two).powf(3.0) / two;

                Vec4::select(is_less, cubic_in, cubic_out)
            }
            EasingFunction::BackIn => {
                let c3 = BACK_CONSTANT + 1.0;

                c3 * x * x * x - BACK_CONSTANT * x * x
            }
            EasingFunction::BackOut => {
                let c3 = BACK_CONSTANT + 1.0;

                1.0 + c3 * (x - 1.0).powf(3.0) + BACK_CONSTANT * (x - 1.0).powf(2.0)
            }
            EasingFunction::BackInOut => {
                let c2 = BACK_CONSTANT * 1.525;

                let is_less = x.cmplt(Vec4::splat(0.5));

                let back_in = ((2.0 * x).powf(2.0) * ((c2 + 1.0) * 2.0 * x - c2)) / 2.0;
                let back_out =
                    ((2.0 * x - 2.0).powf(2.0) * ((c2 + 1.0) * (2.0 * x - 2.0) + c2) + 2.0) / 2.0;

                Vec4::select(is_less, back_in, back_out)
            }
        }
    }
}
