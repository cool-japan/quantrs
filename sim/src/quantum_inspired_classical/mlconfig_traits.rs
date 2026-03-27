//! # MLConfig - Trait Implementations
//!
//! This module contains trait implementations for `MLConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    MLAlgorithm, MLConfig, NetworkArchitecture, TensorNetworkConfig, TrainingConfig,
};

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            algorithm_type: MLAlgorithm::QuantumInspiredNeuralNetwork,
            architecture: NetworkArchitecture::default(),
            training_config: TrainingConfig::default(),
            tensor_network_config: TensorNetworkConfig::default(),
        }
    }
}
