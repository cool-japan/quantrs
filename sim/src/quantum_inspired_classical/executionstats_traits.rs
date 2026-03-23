//! # ExecutionStats - Trait Implementations
//!
//! This module contains trait implementations for `ExecutionStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::ExecutionStats;

impl Default for ExecutionStats {
    fn default() -> Self {
        Self {
            total_runtime: 0.0,
            avg_runtime_per_iteration: 0.0,
            peak_memory_usage: 0,
            successful_runs: 0,
            failed_runs: 0,
        }
    }
}
