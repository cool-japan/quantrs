//! # EnhancedResourceEstimator - count_multi_qubit_gates_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::parallel_ops_stubs::*;
use crate::resource_estimator::{
    ErrorCorrectionCode, EstimationMode, HardwarePlatform, QuantumGate, ResourceEstimationConfig,
};

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Count multi-qubit gates (3+ qubits)
    pub(super) fn count_multi_qubit_gates(circuit: &[QuantumGate]) -> usize {
        circuit
            .iter()
            .filter(|gate| gate.target_qubits().len() > 2)
            .count()
    }
}
