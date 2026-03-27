//! # VqcConfig - Trait Implementations
//!
//! This module contains trait implementations for `VqcConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::VqcConfig;

impl Default for VqcConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            learning_rate: 0.01,
            tolerance: 1e-6,
            num_shots: 1024,
            regularization: 0.001,
            batch_size: 32,
            seed: None,
        }
    }
}
