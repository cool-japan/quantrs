//! # EnhancedResourceEstimator - count_non_clifford_gates_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::gate_translation::GateType;
use crate::parallel_ops_stubs::*;
use crate::resource_estimator::{
    ErrorCorrectionCode, EstimationMode, HardwarePlatform, QuantumGate, ResourceEstimationConfig,
};

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Count non-Clifford gates
    pub(super) fn count_non_clifford_gates(circuit: &[QuantumGate]) -> usize {
        circuit
            .iter()
            .filter(|gate| {
                matches!(
                    gate.gate_type(),
                    GateType::T
                        | GateType::Phase(_)
                        | GateType::Rx(_)
                        | GateType::Ry(_)
                        | GateType::Rz(_)
                )
            })
            .count()
    }
}
