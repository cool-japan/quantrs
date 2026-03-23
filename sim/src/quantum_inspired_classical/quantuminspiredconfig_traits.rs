//! # QuantumInspiredConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumInspiredConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    AlgorithmCategory, AlgorithmConfig, BenchmarkingConfig, GraphConfig, LinalgConfig, MLConfig,
    OptimizationConfig, QuantumInspiredConfig, SamplingConfig,
};

impl Default for QuantumInspiredConfig {
    fn default() -> Self {
        Self {
            num_variables: 16,
            algorithm_category: AlgorithmCategory::Optimization,
            algorithm_config: AlgorithmConfig::default(),
            optimization_config: OptimizationConfig::default(),
            ml_config: Some(MLConfig::default()),
            sampling_config: SamplingConfig::default(),
            linalg_config: LinalgConfig::default(),
            graph_config: GraphConfig::default(),
            benchmarking_config: BenchmarkingConfig::default(),
            enable_quantum_heuristics: true,
            precision: 1e-8,
            random_seed: None,
        }
    }
}
