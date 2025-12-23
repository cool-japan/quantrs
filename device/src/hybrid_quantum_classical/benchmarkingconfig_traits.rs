//! # Benchmarkingconfig - Trait Implementations
//!
//! This module contains trait implementations for `Benchmarkingconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for BenchmarkingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            benchmark_suites: vec![BenchmarkSuite::StandardAlgorithms],
            comparison_targets: vec![ComparisonTarget::BaselineImplementation],
        }
    }
}
