//! # AdvancedLearningConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdvancedLearningConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{ActivationFunction, AdvancedLearningConfig, LearningAlgorithm};

impl Default for AdvancedLearningConfig {
    fn default() -> Self {
        Self {
            algorithm: LearningAlgorithm::Ridge,
            regularization: 1e-6,
            l1_ratio: 0.5,
            forgetting_factor: 0.99,
            process_noise: 1e-4,
            measurement_noise: 1e-3,
            nn_architecture: vec![64, 32, 16],
            nn_activation: ActivationFunction::ReLU,
            epochs: 100,
            batch_size: 32,
            early_stopping_patience: 10,
            cv_folds: 5,
            enable_ensemble: false,
            ensemble_size: 5,
        }
    }
}
