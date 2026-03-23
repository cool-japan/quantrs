//! # NoiseSpectrum - Trait Implementations
//!
//! This module contains trait implementations for `NoiseSpectrum`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

use super::types::{NoiseSpectrum, NoiseType};

impl Default for NoiseSpectrum {
    fn default() -> Self {
        Self {
            frequencies: Array1::linspace(1e3, 1e9, 1000),
            power_spectral_density: Array1::from_elem(1000, 1e-15),
            dominant_noise_type: NoiseType::OneOverF,
            bandwidth: 1e9,
        }
    }
}
