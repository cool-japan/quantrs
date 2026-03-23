//! # TemporalCorrelationModel - Trait Implementations
//!
//! This module contains trait implementations for `TemporalCorrelationModel`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};

use super::types::{OneOverFParameters, TemporalCorrelationModel};

impl Default for TemporalCorrelationModel {
    fn default() -> Self {
        Self {
            autocorrelation: Array1::zeros(1),
            power_spectrum: Array1::zeros(1),
            timescales: Vec::new(),
            one_over_f_params: OneOverFParameters::default(),
        }
    }
}
