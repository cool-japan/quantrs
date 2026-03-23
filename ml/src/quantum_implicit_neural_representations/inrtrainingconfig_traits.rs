//! # INRTrainingConfig - Trait Implementations
//!
//! This module contains trait implementations for `INRTrainingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::INRTrainingConfig;

impl Default for INRTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            batch_size: 1024,
            learning_rate: 1e-4,
            log_interval: 100,
        }
    }
}
