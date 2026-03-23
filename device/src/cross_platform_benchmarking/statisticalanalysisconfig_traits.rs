//! # StatisticalAnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalAnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::prelude::*;

use super::types::StatisticalAnalysisConfig;

impl Default for StatisticalAnalysisConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            enable_anova: true,
            enable_pairwise_comparisons: true,
            enable_outlier_detection: true,
            enable_distribution_fitting: true,
            enable_correlation_analysis: true,
            min_sample_size: 5,
        }
    }
}
