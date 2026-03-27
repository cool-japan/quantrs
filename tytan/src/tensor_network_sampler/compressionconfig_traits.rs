//! # CompressionConfig - Trait Implementations
//!
//! This module contains trait implementations for `CompressionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    CompressionConfig, CompressionMethod, QualityControlConfig, QualityMetric, RecoveryStrategy,
};

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            target_compression_ratio: 0.5,
            max_error: 1e-6,
            method: CompressionMethod::SVD,
            adaptive_compression: true,
            quality_control: QualityControlConfig {
                error_tolerance: 1e-8,
                quality_metrics: vec![
                    QualityMetric::RelativeError,
                    QualityMetric::FrobeniusNormError,
                ],
                validation_frequency: 10,
                recovery_strategies: vec![RecoveryStrategy::IncreaseBondDimension],
            },
        }
    }
}
