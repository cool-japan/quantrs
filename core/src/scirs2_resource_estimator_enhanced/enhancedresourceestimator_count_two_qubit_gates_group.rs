//! # EnhancedResourceEstimator - count_two_qubit_gates_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::parallel_ops_stubs::*;
use crate::resource_estimator::{
    ErrorCorrectionCode, EstimationMode, HardwarePlatform, QuantumGate, ResourceEstimationConfig,
};

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Count two-qubit gates
    pub(super) fn count_two_qubit_gates(circuit: &[QuantumGate]) -> usize {
        circuit
            .iter()
            .filter(|gate| gate.target_qubits().len() == 2)
            .count()
    }
    /// Classify algorithmic complexity
    pub(super) fn classify_algorithmic_complexity(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<String, QuantRS2Error> {
        let depth = circuit.len();
        let two_qubit_ratio =
            Self::count_two_qubit_gates(circuit) as f64 / circuit.len().max(1) as f64;
        if depth < 100 && two_qubit_ratio < 0.2 {
            Ok("Low (BQP-easy)".to_string())
        } else if depth < 1000 && two_qubit_ratio < 0.5 {
            Ok("Medium (BQP-intermediate)".to_string())
        } else {
            Ok("High (BQP-hard)".to_string())
        }
    }
}
