//! # LearningRateScheduler - Trait Implementations
//!
//! This module contains trait implementations for `LearningRateScheduler`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{LRSchedulerType, LearningRateScheduler};

impl Default for LearningRateScheduler {
    fn default() -> Self {
        Self {
            scheduler_type: LRSchedulerType::Cosine,
            current_lr: 3e-4,
            warmup_epochs: 10,
            quantum_adaptive: false,
        }
    }
}
