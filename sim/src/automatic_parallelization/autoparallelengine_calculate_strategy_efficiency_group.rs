//! # AutoParallelEngine - calculate_strategy_efficiency_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ParallelTask;

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Calculate strategy efficiency
    pub(super) fn calculate_strategy_efficiency(tasks: &[ParallelTask]) -> f64 {
        if tasks.is_empty() {
            return 0.0;
        }
        let total_cost: f64 = tasks.iter().map(|t| t.cost).sum();
        let max_cost = tasks.iter().map(|t| t.cost).fold(0.0, f64::max);
        if max_cost > 0.0 {
            total_cost / (max_cost * tasks.len() as f64)
        } else {
            0.0
        }
    }
}
