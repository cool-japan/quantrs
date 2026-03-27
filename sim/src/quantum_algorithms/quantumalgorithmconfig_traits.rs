//! # QuantumAlgorithmConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumAlgorithmConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{OptimizationLevel, QuantumAlgorithmConfig};

impl Default for QuantumAlgorithmConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Maximum,
            use_classical_preprocessing: true,
            enable_error_mitigation: true,
            max_circuit_depth: 1000,
            precision_tolerance: 1e-10,
            enable_parallel: true,
            resource_estimation_accuracy: 0.95,
        }
    }
}
