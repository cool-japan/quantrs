//! # QuantumContinuousFlow - validation Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::FlowTrainingMetrics;

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Validate epoch
    pub(super) fn validate_epoch(
        &self,
        validation_data: &Array2<f64>,
    ) -> Result<FlowTrainingMetrics> {
        let mut val_nll = 0.0;
        let mut quantum_fidelity_sum = 0.0;
        let mut entanglement_sum = 0.0;
        let mut num_samples = 0;
        for sample_idx in 0..validation_data.nrows() {
            let x = validation_data.row(sample_idx).to_owned();
            let forward_output = self.forward(&x)?;
            val_nll += -forward_output.quantum_log_probability;
            quantum_fidelity_sum += forward_output.quantum_enhancement.fidelity_contribution;
            entanglement_sum += forward_output.quantum_enhancement.entanglement_contribution;
            num_samples += 1;
        }
        Ok(FlowTrainingMetrics {
            epoch: 0,
            negative_log_likelihood: val_nll / num_samples as f64,
            bits_per_dimension: (val_nll / num_samples as f64)
                / (validation_data.ncols() as f64 * (2.0_f64).ln()),
            quantum_likelihood: val_nll / num_samples as f64,
            entanglement_measure: entanglement_sum / num_samples as f64,
            invertibility_score: 1.0,
            jacobian_determinant_mean: 0.0,
            jacobian_determinant_std: 0.0,
            quantum_fidelity: quantum_fidelity_sum / num_samples as f64,
            coherence_time: 1.0,
            quantum_advantage_ratio: 1.0 + entanglement_sum / num_samples as f64,
        })
    }
}
