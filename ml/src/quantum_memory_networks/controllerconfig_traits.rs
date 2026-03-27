//! # ControllerConfig - Trait Implementations
//!
//! This module contains trait implementations for `ControllerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ActivationFunction, ControllerArchitecture, ControllerConfig, QuantumEnhancementLevel,
};

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            architecture: ControllerArchitecture::QuantumLSTM,
            hidden_dims: vec![64, 32],
            activation: ActivationFunction::QuantumTanh,
            recurrent: true,
            quantum_enhancement: QuantumEnhancementLevel::Full,
        }
    }
}
