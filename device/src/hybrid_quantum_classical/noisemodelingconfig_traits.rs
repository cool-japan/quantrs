//! # NoiseModelingConfig - Trait Implementations
//!
//! This module contains trait implementations for `NoiseModelingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for NoiseModelingConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_modeling: true,
            characterization_frequency: Duration::from_secs(300),
            mitigation_strategies: vec![NoiseMitigationStrategy::ZeroNoiseExtrapolation],
            adaptive_threshold: 0.01,
        }
    }
}
