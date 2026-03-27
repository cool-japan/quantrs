//! # TrainingConfig - Trait Implementations
//!
//! This module contains trait implementations for `TrainingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{LossFunction, Regularization, TrainingConfig};

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 100,
            batch_size: 32,
            validation_frequency: 10,
            loss_function: LossFunction::MeanSquaredError,
            regularization: Regularization {
                l1_weight: 0.0,
                l2_weight: 0.01,
                dropout_rate: 0.1,
                weight_decay: 0.01,
            },
        }
    }
}
