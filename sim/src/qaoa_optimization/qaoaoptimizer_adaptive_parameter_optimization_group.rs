//! # QAOAOptimizer - adaptive_parameter_optimization_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    /// Adaptive parameter optimization
    pub(super) fn adaptive_parameter_optimization(&mut self, cost_history: &[f64]) -> Result<()> {
        if cost_history.len() > 5 {
            let recent_improvement =
                cost_history[cost_history.len() - 1] - cost_history[cost_history.len() - 5];
            if recent_improvement > 0.0 {
                self.config.learning_rate *= 1.1;
            } else {
                self.config.learning_rate *= 0.9;
            }
        }
        self.classical_parameter_optimization()
    }
}
