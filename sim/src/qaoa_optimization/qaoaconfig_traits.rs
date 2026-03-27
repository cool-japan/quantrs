//! # QAOAConfig - Trait Implementations
//!
//! This module contains trait implementations for `QAOAConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    QAOAConfig, QAOAInitializationStrategy, QAOAMixerType, QAOAOptimizationStrategy,
};

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            mixer_type: QAOAMixerType::Standard,
            initialization: QAOAInitializationStrategy::UniformSuperposition,
            optimization_strategy: QAOAOptimizationStrategy::Classical,
            max_iterations: 100,
            convergence_tolerance: 1e-6,
            learning_rate: 0.1,
            multi_angle: false,
            parameter_transfer: false,
            hardware_aware: true,
            shots: None,
            adaptive_layers: false,
            max_adaptive_layers: 10,
        }
    }
}
