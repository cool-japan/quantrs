//! # SSLTrainingConfig - Trait Implementations
//!
//! This module contains trait implementations for `SSLTrainingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::SSLTrainingConfig;

impl Default for SSLTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 256,
            learning_rate: 3e-4,
            weight_decay: 1e-4,
            log_interval: 10,
            save_interval: 50,
            early_stopping_patience: 15,
        }
    }
}
