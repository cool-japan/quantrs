//! # AdaptiveLearningConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveLearningConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{AdaptiveLearningConfig, LearningRateSchedule, PlasticityType};

impl Default for AdaptiveLearningConfig {
    fn default() -> Self {
        Self {
            enable_online: true,
            lr_schedule: LearningRateSchedule::Exponential,
            initial_lr: 0.01,
            min_lr: 1e-6,
            lr_decay: 0.95,
            adaptation_window: 100,
            plasticity_type: PlasticityType::Hebbian,
            enable_homeostasis: false,
            target_activity: 0.5,
            regulation_rate: 0.001,
            enable_meta_learning: false,
            meta_update_frequency: 1000,
        }
    }
}
