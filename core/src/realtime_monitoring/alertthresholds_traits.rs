//! # AlertThresholds - Trait Implementations
//!
//! This module contains trait implementations for `AlertThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AlertThresholds;
use std::time::Duration;

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_gate_error_rate: 0.01,
            max_readout_error_rate: 0.05,
            min_coherence_time: Duration::from_micros(50),
            max_calibration_drift: 0.1,
            max_temperature: 300.0,
            max_queue_depth: 1000,
            max_execution_time: Duration::from_secs(300),
        }
    }
}
