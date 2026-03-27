//! # NoiseCharacterizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `NoiseCharacterizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::prelude::*;

use super::types::NoiseCharacterizationConfig;

impl Default for NoiseCharacterizationConfig {
    fn default() -> Self {
        Self {
            enable_advanced_statistics: true,
            enable_ml_predictions: true,
            enable_drift_monitoring: true,
            update_frequency_hours: 24.0,
            confidence_level: 0.95,
            protocol_repetitions: 100,
            enable_crosstalk_analysis: true,
            enable_temporal_analysis: true,
        }
    }
}
