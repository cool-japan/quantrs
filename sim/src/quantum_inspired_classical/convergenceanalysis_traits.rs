//! # ConvergenceAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `ConvergenceAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::ConvergenceAnalysis;

impl Default for ConvergenceAnalysis {
    fn default() -> Self {
        Self {
            convergence_rate: 0.0,
            iterations_to_convergence: 0,
            final_gradient_norm: f64::INFINITY,
            converged: false,
            convergence_criterion: "tolerance".to_string(),
        }
    }
}
