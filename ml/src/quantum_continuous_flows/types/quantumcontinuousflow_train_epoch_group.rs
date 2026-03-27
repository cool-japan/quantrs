//! # QuantumContinuousFlow - train_epoch_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{FlowTrainingConfig, FlowTrainingMetrics, QuantumFlowBatchMetrics};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Train single epoch
    pub(super) fn train_epoch(
        &mut self,
        data: &Array2<f64>,
        config: &FlowTrainingConfig,
        epoch: usize,
    ) -> Result<FlowTrainingMetrics> {
        let mut epoch_nll = 0.0;
        let mut quantum_fidelity_sum = 0.0;
        let mut entanglement_sum = 0.0;
        let mut jacobian_det_sum = 0.0;
        let mut num_batches = 0;
        let num_samples = data.nrows();
        for batch_start in (0..num_samples).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(num_samples);
            let batch_data = data.slice(scirs2_core::ndarray::s![batch_start..batch_end, ..]);
            let batch_metrics = self.train_batch(&batch_data, config)?;
            epoch_nll += batch_metrics.negative_log_likelihood;
            quantum_fidelity_sum += batch_metrics.quantum_fidelity;
            entanglement_sum += batch_metrics.entanglement_measure;
            jacobian_det_sum += batch_metrics.jacobian_determinant_mean;
            num_batches += 1;
        }
        let num_batches_f = num_batches as f64;
        Ok(FlowTrainingMetrics {
            epoch,
            negative_log_likelihood: epoch_nll / num_batches_f,
            bits_per_dimension: (epoch_nll / num_batches_f)
                / (data.ncols() as f64 * (2.0_f64).ln()),
            quantum_likelihood: epoch_nll / num_batches_f,
            entanglement_measure: entanglement_sum / num_batches_f,
            invertibility_score: 1.0,
            jacobian_determinant_mean: jacobian_det_sum / num_batches_f,
            jacobian_determinant_std: 1.0,
            quantum_fidelity: quantum_fidelity_sum / num_batches_f,
            coherence_time: 1.0,
            quantum_advantage_ratio: 1.0 + entanglement_sum / num_batches_f,
        })
    }
    /// Train single batch
    fn train_batch(
        &mut self,
        batch_data: &scirs2_core::ndarray::ArrayView2<f64>,
        config: &FlowTrainingConfig,
    ) -> Result<FlowTrainingMetrics> {
        let mut batch_nll = 0.0;
        let mut quantum_metrics_sum = QuantumFlowBatchMetrics::default();
        for sample_idx in 0..batch_data.nrows() {
            let x = batch_data.row(sample_idx).to_owned();
            let forward_output = self.forward(&x)?;
            let nll = -forward_output.quantum_log_probability;
            batch_nll += nll;
            quantum_metrics_sum.accumulate(&forward_output)?;
            self.update_flow_parameters(&forward_output, config)?;
        }
        let num_samples = batch_data.nrows() as f64;
        Ok(FlowTrainingMetrics {
            epoch: 0,
            negative_log_likelihood: batch_nll / num_samples,
            bits_per_dimension: (batch_nll / num_samples)
                / (batch_data.ncols() as f64 * (2.0_f64).ln()),
            quantum_likelihood: batch_nll / num_samples,
            entanglement_measure: quantum_metrics_sum.entanglement_measure / num_samples,
            invertibility_score: quantum_metrics_sum.invertibility_score / num_samples,
            jacobian_determinant_mean: quantum_metrics_sum.jacobian_determinant_mean / num_samples,
            jacobian_determinant_std: quantum_metrics_sum.jacobian_determinant_std / num_samples,
            quantum_fidelity: quantum_metrics_sum.quantum_fidelity / num_samples,
            coherence_time: quantum_metrics_sum.coherence_time / num_samples,
            quantum_advantage_ratio: quantum_metrics_sum.quantum_advantage_ratio / num_samples,
        })
    }
}
