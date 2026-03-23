//! # PerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::PerformanceMetrics;

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            solution_fidelity: 1.0,
            annealing_efficiency: 1.0,
            error_suppression_factor: 1.0,
            protocol_stability: 1.0,
            adaptation_effectiveness: 1.0,
        }
    }
}
