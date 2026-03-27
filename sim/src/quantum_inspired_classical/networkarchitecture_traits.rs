//! # NetworkArchitecture - Trait Implementations
//!
//! This module contains trait implementations for `NetworkArchitecture`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{ActivationFunction, NetworkArchitecture};

impl Default for NetworkArchitecture {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_layers: vec![32, 16],
            output_dim: 8,
            activation: ActivationFunction::QuantumInspiredTanh,
            quantum_connections: true,
        }
    }
}
