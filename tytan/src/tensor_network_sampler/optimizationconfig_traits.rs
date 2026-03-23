//! # OptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `OptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    LineSearchConfig, LineSearchMethod, OptimizationAlgorithm, OptimizationConfig,
    RegularizationConfig,
};

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            algorithm: OptimizationAlgorithm::DMRG,
            max_iterations: 1000,
            tolerance: 1e-8,
            learning_rate: 0.01,
            regularization: RegularizationConfig {
                l1_strength: 0.0,
                l2_strength: 0.001,
                bond_dimension_penalty: 0.0,
                entropy_regularization: 0.0,
            },
            line_search: LineSearchConfig {
                method: LineSearchMethod::Backtracking,
                max_step_size: 1.0,
                backtracking_params: (0.5, 1e-4),
                wolfe_conditions: false,
            },
        }
    }
}
