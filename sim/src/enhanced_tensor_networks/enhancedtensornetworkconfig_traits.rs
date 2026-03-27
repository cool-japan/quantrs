//! # EnhancedTensorNetworkConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedTensorNetworkConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{ContractionStrategy, EnhancedTensorNetworkConfig};

impl Default for EnhancedTensorNetworkConfig {
    fn default() -> Self {
        Self {
            max_bond_dimension: 1024,
            contraction_strategy: ContractionStrategy::Adaptive,
            memory_limit: 16_000_000_000,
            enable_approximations: true,
            svd_threshold: 1e-12,
            max_optimization_time_ms: 5000,
            parallel_contractions: true,
            use_scirs2_acceleration: true,
            enable_slicing: true,
            max_slices: 64,
        }
    }
}
