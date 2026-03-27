//! # OneOverFParameters - Trait Implementations
//!
//! This module contains trait implementations for `OneOverFParameters`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::OneOverFParameters;

impl Default for OneOverFParameters {
    fn default() -> Self {
        Self {
            amplitude: 0.0,
            exponent: 1.0,
            cutoff_frequency: 1.0,
            high_freq_rolloff: 0.0,
        }
    }
}
