//! # AdaptiveFusionConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveFusionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "advanced_math")]
use quantrs2_circuit::prelude::*;

use super::types::{AdaptiveFusionConfig, FusionStrategy};

impl Default for AdaptiveFusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::Adaptive,
            max_fusion_size: 8,
            min_benefit_threshold: 1.1,
            enable_cross_qubit_fusion: true,
            enable_temporal_fusion: true,
            max_analysis_depth: 100,
            enable_ml_predictions: true,
            fusion_cache_size: 10_000,
            parallel_analysis: true,
        }
    }
}
