//! # EnhancedResourceEstimator - estimate_physical_qubits_group Methods
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
    /// Estimate physical qubits
    pub(super) const fn estimate_physical_qubits(
        &self,
        logical_qubits: usize,
        code_distance: usize,
    ) -> Result<usize, QuantRS2Error> {
        let qubits_per_logical = match self.config.base_config.error_correction_code {
            ErrorCorrectionCode::SurfaceCode => 2 * code_distance * code_distance,
            ErrorCorrectionCode::ColorCode => 3 * code_distance * code_distance,
            _ => code_distance * code_distance,
        };
        Ok(logical_qubits * qubits_per_logical)
    }
}
