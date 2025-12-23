//! # Hybridoptimizationconfig - Trait Implementations
//!
//! This module contains trait implementations for `Hybridoptimizationconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::collections::HashMap;

impl Default for HybridOptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_tolerance: 1e-6,
            optimizer: HybridOptimizer::Adam,
            parameter_bounds: None,
            adaptive_learning_rate: true,
            multi_objective_weights: HashMap::new(),
            enable_parallel_exploration: true,
            enable_scirs2_optimization: true,
        }
    }
}
