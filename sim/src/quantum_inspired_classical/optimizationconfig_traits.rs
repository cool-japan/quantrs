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
    ConstraintMethod, ObjectiveFunction, OptimizationAlgorithm, OptimizationConfig,
};

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            algorithm_type: OptimizationAlgorithm::QuantumGeneticAlgorithm,
            objective_function: ObjectiveFunction::Quadratic,
            bounds: vec![(-10.0, 10.0); 16],
            constraint_method: ConstraintMethod::PenaltyFunction,
            multi_objective: false,
            parallel_evaluation: true,
        }
    }
}
