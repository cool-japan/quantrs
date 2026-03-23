//! # SystemNoiseModel - Trait Implementations
//!
//! This module contains trait implementations for `SystemNoiseModel`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

use super::types::{GateType, NoiseSpectrum, SystemNoiseModel};

impl Default for SystemNoiseModel {
    fn default() -> Self {
        Self {
            t1_coherence_time: Array1::from_elem(4, 100.0),
            t2_dephasing_time: Array1::from_elem(4, 50.0),
            gate_error_rates: {
                let mut rates = HashMap::new();
                rates.insert(GateType::SingleQubit, 0.001);
                rates.insert(GateType::TwoQubit, 0.01);
                rates.insert(GateType::Measurement, 0.02);
                rates.insert(GateType::Preparation, 0.001);
                rates
            },
            measurement_error_rate: 0.02,
            thermal_temperature: 15.0,
            noise_spectrum: NoiseSpectrum::default(),
            crosstalk_matrix: Array2::eye(4),
        }
    }
}
