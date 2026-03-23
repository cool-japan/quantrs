//! # AnalysisParameters - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisParameters`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AnalysisParameters;

impl Default for AnalysisParameters {
    fn default() -> Self {
        Self {
            temporal_window: 1000.0,
            frequency_resolution: 1e3,
            correlation_threshold: 0.1,
            ml_update_frequency: 100,
            prediction_horizon: 100.0,
        }
    }
}
