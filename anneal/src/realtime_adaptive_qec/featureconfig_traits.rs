//! # FeatureConfig - Trait Implementations
//!
//! This module contains trait implementations for `FeatureConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{FeatureConfig, FeatureNormalization, FeatureSelection};

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            enable_temporal: true,
            enable_spectral: true,
            enable_correlation: true,
            normalization: FeatureNormalization::ZScore,
            selection_method: FeatureSelection::Automatic,
        }
    }
}
