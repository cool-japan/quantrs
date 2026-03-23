//! # ErrorStatistics - Trait Implementations
//!
//! This module contains trait implementations for `ErrorStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::ErrorStatistics;

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            total_errors_detected: 0,
            error_rates_by_type: HashMap::new(),
            temporal_error_distribution: Array1::zeros(1),
            spatial_error_distribution: Array2::zeros((1, 1)),
            mitigation_effectiveness: HashMap::new(),
            prediction_accuracy: 0.0,
        }
    }
}
