//! # QAOAOptimizer - should_add_layer_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    pub(super) fn should_add_layer(&self, cost_history: &[f64]) -> Result<bool> {
        if cost_history.len() < 10 {
            return Ok(false);
        }
        let recent_improvement =
            cost_history[cost_history.len() - 1] - cost_history[cost_history.len() - 10];
        Ok(recent_improvement.abs() < self.config.convergence_tolerance * 10.0)
    }
}
