//! # EnhancedResourceEstimator - calculate_error_budget_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::parallel_ops_stubs::*;

use super::types::ErrorBudget;

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Calculate error budget
    pub(super) fn calculate_error_budget(&self) -> Result<ErrorBudget, QuantRS2Error> {
        let total = self.config.base_config.target_logical_error_rate;
        Ok(ErrorBudget {
            total_budget: total,
            gate_errors: total * 0.4,
            measurement_errors: total * 0.2,
            idle_errors: total * 0.2,
            crosstalk_errors: total * 0.1,
            readout_errors: total * 0.1,
        })
    }
}
