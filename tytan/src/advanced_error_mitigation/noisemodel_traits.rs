//! # NoiseModel - Trait Implementations
//!
//! This module contains trait implementations for `NoiseModel`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::{EnvironmentalNoiseModel, NoiseModel, TemporalCorrelationModel};

impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            single_qubit_errors: HashMap::new(),
            two_qubit_errors: HashMap::new(),
            crosstalk_matrix: Array2::zeros((1, 1)),
            temporal_correlations: TemporalCorrelationModel::default(),
            environmental_noise: EnvironmentalNoiseModel::default(),
            validation_score: 0.0,
        }
    }
}
