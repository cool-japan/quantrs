//! # PerformanceThresholds - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::PerformanceThresholds;

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_gate_fidelity: 0.995,
            max_error_rate: 0.005,
            max_drift_rate: 0.001,
            min_readout_fidelity: 0.98,
            max_crosstalk: 0.01,
        }
    }
}
