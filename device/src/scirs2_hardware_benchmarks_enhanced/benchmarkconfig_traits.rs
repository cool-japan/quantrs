//! # BenchmarkConfig - Trait Implementations
//!
//! This module contains trait implementations for `BenchmarkConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use std::time::{Duration, Instant};

use super::types::BenchmarkConfig;

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_repetitions: 20,
            shots_per_circuit: 1000,
            max_circuit_depth: 100,
            timeout: Duration::from_secs(300),
            confidence_level: 0.95,
        }
    }
}
