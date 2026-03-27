//! # EnhancedResourceEstimator - estimate_magic_states_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::parallel_ops_stubs::*;
use crate::resource_estimator::{
    ErrorCorrectionCode, EstimationMode, HardwarePlatform, QuantumGate, ResourceEstimationConfig,
};

use super::types::GateStatistics;

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Estimate magic states
    pub(super) const fn estimate_magic_states(
        &self,
        gate_stats: &GateStatistics,
    ) -> Result<usize, QuantRS2Error> {
        let t_gates = gate_stats.non_clifford_count;
        let overhead = match self.config.base_config.estimation_mode {
            EstimationMode::Conservative => 15,
            EstimationMode::Optimistic => 10,
            EstimationMode::Realistic => 12,
        };
        Ok(t_gates * overhead)
    }
}
