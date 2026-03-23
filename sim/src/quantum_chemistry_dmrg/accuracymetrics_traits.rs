//! # AccuracyMetrics - Trait Implementations
//!
//! This module contains trait implementations for `AccuracyMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::AccuracyMetrics;

impl Default for AccuracyMetrics {
    fn default() -> Self {
        Self {
            energy_accuracy: 0.0,
            dipole_accuracy: 0.0,
            bond_length_accuracy: 0.0,
            frequency_accuracy: 0.0,
        }
    }
}
