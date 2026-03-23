//! # QuantumParameters - Trait Implementations
//!
//! This module contains trait implementations for `QuantumParameters`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{QuantumParameters, QuantumWalkParams};

impl Default for QuantumParameters {
    fn default() -> Self {
        Self {
            superposition_strength: 0.5,
            entanglement_strength: 0.3,
            interference_strength: 0.2,
            tunneling_probability: 0.1,
            decoherence_rate: 0.01,
            measurement_probability: 0.1,
            quantum_walk_params: QuantumWalkParams::default(),
        }
    }
}
