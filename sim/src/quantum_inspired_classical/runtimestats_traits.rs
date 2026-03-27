//! # RuntimeStats - Trait Implementations
//!
//! This module contains trait implementations for `RuntimeStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::RuntimeStats;

impl Default for RuntimeStats {
    fn default() -> Self {
        Self {
            function_evaluations: 0,
            gradient_evaluations: 0,
            cpu_time: 0.0,
            memory_usage: 0,
            quantum_operations: 0,
        }
    }
}
