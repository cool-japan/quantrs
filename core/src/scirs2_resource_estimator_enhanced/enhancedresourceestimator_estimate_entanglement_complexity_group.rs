//! # EnhancedResourceEstimator - estimate_entanglement_complexity_group Methods
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
    /// Estimate entanglement complexity
    pub(super) fn estimate_entanglement_complexity(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<f64, QuantRS2Error> {
        let entangling_gates = circuit
            .iter()
            .filter(|g| g.target_qubits().len() >= 2)
            .count();
        let total_gates = circuit.len().max(1);
        Ok(entangling_gates as f64 / total_gates as f64)
    }
}
