//! # NeuralSchedulerConfig - Trait Implementations
//!
//! This module contains trait implementations for `NeuralSchedulerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::NeuralSchedulerConfig;

impl Default for NeuralSchedulerConfig {
    fn default() -> Self {
        Self {
            enable_online_learning: true,
            schedule_network_layers: vec![256, 512, 256, 128],
            encoder_network_layers: vec![128, 256, 128],
            predictor_network_layers: vec![256, 128, 64],
            learning_rate: 0.001,
            batch_size: 32,
            max_schedule_points: 100,
            enable_transfer_learning: true,
            smoothness_weight: 0.1,
            enable_hardware_constraints: true,
        }
    }
}
