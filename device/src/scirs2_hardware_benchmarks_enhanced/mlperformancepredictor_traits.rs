//! # MLPerformancePredictor - Trait Implementations
//!
//! This module contains trait implementations for `MLPerformancePredictor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex};

use super::types::{BenchmarkFeatureExtractor, MLPerformancePredictor, PerformanceModel};

impl Default for MLPerformancePredictor {
    fn default() -> Self {
        Self {
            model: Arc::new(Mutex::new(PerformanceModel::new())),
            feature_extractor: Arc::new(BenchmarkFeatureExtractor::new()),
        }
    }
}
