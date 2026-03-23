//! # TensorNetworkMetrics - Trait Implementations
//!
//! This module contains trait implementations for `TensorNetworkMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, ArrayD};
use scirs2_core::random::prelude::*;

use super::types::{EntanglementMeasures, TensorNetworkMetrics};

impl Default for TensorNetworkMetrics {
    fn default() -> Self {
        Self {
            compression_efficiency: 1.0,
            convergence_rate: 1.0,
            memory_efficiency: 1.0,
            computational_speed: 1.0,
            approximation_accuracy: 1.0,
            entanglement_measures: EntanglementMeasures {
                entanglement_entropy: Array1::zeros(1),
                mutual_information: Array2::zeros((1, 1)),
                entanglement_spectrum: vec![Array1::zeros(1)],
                topological_entropy: 0.0,
            },
            overall_performance: 1.0,
        }
    }
}
