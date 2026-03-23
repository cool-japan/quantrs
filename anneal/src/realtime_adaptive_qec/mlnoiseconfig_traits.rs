//! # MLNoiseConfig - Trait Implementations
//!
//! This module contains trait implementations for `MLNoiseConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{FeatureConfig, MLNoiseConfig, NeuralArchitecture};

impl Default for MLNoiseConfig {
    fn default() -> Self {
        Self {
            enable_neural_prediction: true,
            network_architecture: NeuralArchitecture::LSTM,
            training_window: 1000,
            prediction_horizon: Duration::from_secs(10),
            update_frequency: Duration::from_secs(60),
            feature_config: FeatureConfig::default(),
        }
    }
}
