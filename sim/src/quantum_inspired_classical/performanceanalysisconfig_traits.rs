//! # PerformanceAnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceAnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::PerformanceAnalysisConfig;

impl Default for PerformanceAnalysisConfig {
    fn default() -> Self {
        Self {
            analyze_convergence: true,
            analyze_scalability: true,
            analyze_quantum_advantage: true,
            record_memory_usage: true,
        }
    }
}
