//! # AlertThresholds - Trait Implementations
//!
//! This module contains trait implementations for `AlertThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant};

use super::types::AlertThresholds;

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            performance_threshold: 0.8,
            error_rate_threshold: 0.05,
            temperature_threshold: 300.0,
            vibration_threshold: 1e-6,
            coherence_threshold: Duration::from_micros(10),
        }
    }
}
