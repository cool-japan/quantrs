//! # GravitySimulationStats - Trait Implementations
//!
//! This module contains trait implementations for `GravitySimulationStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::GravitySimulationStats;

impl Default for GravitySimulationStats {
    fn default() -> Self {
        Self {
            total_time: 0.0,
            memory_usage: 0,
            calculations_performed: 0,
            avg_time_per_step: 0.0,
            peak_memory_usage: 0,
        }
    }
}
