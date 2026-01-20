//! # Stateestimationconfig - Trait Implementations
//!
//! This module contains trait implementations for `Stateestimationconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for StateEstimationConfig {
    fn default() -> Self {
        Self {
            method: StateEstimationMethod::MaximumLikelihood,
            confidence_level: 0.95,
            update_frequency: 1.0,
            noise_modeling: NoiseModelingConfig::default(),
        }
    }
}
