//! # AutoParallelEngine - calculate_variance_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Calculate variance of a vector of values
    pub(super) fn calculate_variance(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance
    }
}
