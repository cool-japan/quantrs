//! # AnalysisOptions - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::AnalysisOptions;

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            bootstrap_samples: 1000,
            confidence_level: 0.95,
            outlier_threshold: 3.0,
            trend_analysis: true,
            spectral_analysis: true,
        }
    }
}
