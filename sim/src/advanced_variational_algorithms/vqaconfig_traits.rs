//! # VQAConfig - Trait Implementations
//!
//! This module contains trait implementations for `VQAConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{AdvancedOptimizerType, VQAConfig};

impl Default for VQAConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_tolerance: 1e-6,
            optimizer: AdvancedOptimizerType::QuantumAdam,
            learning_rate: 0.01,
            shots: None,
            gradient_clipping: Some(1.0),
            regularization: 0.0,
            parameter_bounds: None,
            warm_restart: None,
            hardware_aware: true,
            noise_aware: true,
        }
    }
}
