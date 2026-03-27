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

use super::types::{OptimizerType, TrainingConfig};

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 100,
            batch_size: 32,
            optimizer: OptimizerType::QuantumInspiredAdam,
            regularization: 0.001,
        }
    }
}
