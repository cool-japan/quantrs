//! # ComparisonStats - Trait Implementations
//!
//! This module contains trait implementations for `ComparisonStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::ComparisonStats;

impl Default for ComparisonStats {
    fn default() -> Self {
        Self {
            quantum_inspired_performance: 0.0,
            classical_performance: 0.0,
            speedup_factor: 1.0,
            solution_quality_ratio: 1.0,
            convergence_speed_ratio: 1.0,
        }
    }
}
