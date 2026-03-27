//! # BenchmarkingConfig - Trait Implementations
//!
//! This module contains trait implementations for `BenchmarkingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{BenchmarkingConfig, PerformanceAnalysisConfig};

impl Default for BenchmarkingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_runs: 10,
            compare_classical: true,
            detailed_metrics: true,
            performance_analysis: PerformanceAnalysisConfig::default(),
        }
    }
}
