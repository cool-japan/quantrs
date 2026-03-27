//! # QECMetrics - Trait Implementations
//!
//! This module contains trait implementations for `QECMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use scirs2_core::random::prelude::*;

use super::types::QECMetrics;

impl Default for QECMetrics {
    fn default() -> Self {
        Self {
            logical_error_rate: 0.0,
            syndrome_fidelity: 1.0,
            decoding_success_rate: 1.0,
            correction_latency: 0.0,
            resource_efficiency: 1.0,
            fault_tolerance_margin: 1.0,
            overall_performance: 1.0,
        }
    }
}
