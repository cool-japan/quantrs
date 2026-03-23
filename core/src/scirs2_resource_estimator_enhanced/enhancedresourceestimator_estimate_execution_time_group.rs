//! # EnhancedResourceEstimator - estimate_execution_time_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::parallel_ops_stubs::*;

use super::types::GateStatistics;

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Estimate execution time
    pub(super) fn estimate_execution_time(
        &self,
        gate_stats: &GateStatistics,
    ) -> Result<f64, QuantRS2Error> {
        let mut total_time = 0.0;
        let gate_times = self.get_gate_times()?;
        for (gate_type, count) in &gate_stats.gate_counts {
            let time = gate_times.get(gate_type).copied().unwrap_or(1e-6);
            total_time += time * (*count as f64);
        }
        total_time *= 1.5;
        Ok(total_time)
    }
}
