//! # EnhancedResourceEstimator - estimate_memory_requirements_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::parallel_ops_stubs::*;

use super::types::{GateStatistics, MemoryRequirements};

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Estimate memory requirements
    pub(super) fn estimate_memory_requirements(
        &self,
        num_qubits: usize,
        gate_stats: &GateStatistics,
    ) -> Result<MemoryRequirements, QuantRS2Error> {
        let state_vector_size = (1 << num_qubits) * 16;
        let gate_memory = gate_stats.total_gates * 64;
        let workspace = state_vector_size / 2;
        Ok(MemoryRequirements {
            state_vector_memory: state_vector_size,
            gate_storage_memory: gate_memory,
            workspace_memory: workspace,
            total_memory: state_vector_size + gate_memory + workspace,
            memory_bandwidth: self.estimate_memory_bandwidth(gate_stats)?,
        })
    }
    /// Estimate memory bandwidth requirements
    pub(super) fn estimate_memory_bandwidth(
        &self,
        gate_stats: &GateStatistics,
    ) -> Result<f64, QuantRS2Error> {
        let ops_per_second = 1e9;
        let bytes_per_op = 32.0;
        Ok(ops_per_second * bytes_per_op / 1e9)
    }
}
