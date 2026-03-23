//! # AlertThresholds - Trait Implementations
//!
//! This module contains trait implementations for `AlertThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::AlertThresholds;

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            min_fidelity: 0.8,
            max_error_rate: 0.1,
            min_efficiency: 0.5,
            min_stability: 0.7,
        }
    }
}
