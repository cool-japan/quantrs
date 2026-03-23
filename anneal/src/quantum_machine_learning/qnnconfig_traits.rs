//! # QnnConfig - Trait Implementations
//!
//! This module contains trait implementations for `QnnConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::QnnConfig;

impl Default for QnnConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_epochs: 100,
            batch_size: 32,
            tolerance: 1e-6,
            regularization: 0.001,
            seed: None,
        }
    }
}
