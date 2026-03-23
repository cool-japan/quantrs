//! # QuantumReservoirComputer - train_output_weights_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Train output weights using ridge regression
    pub(super) fn train_output_weights(
        &mut self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<()> {
        if features.is_empty() || targets.is_empty() {
            return Ok(());
        }
        let n_samples = features.len().min(targets.len());
        let n_features = features[0].len();
        let n_outputs = targets[0].len().min(self.output_weights.nrows());
        let mut feature_matrix = Array2::zeros((n_samples, n_features));
        for (i, feature_vec) in features.iter().enumerate().take(n_samples) {
            for (j, &val) in feature_vec.iter().enumerate().take(n_features) {
                feature_matrix[[i, j]] = val;
            }
        }
        let mut target_matrix = Array2::zeros((n_samples, n_outputs));
        for (i, target_vec) in targets.iter().enumerate().take(n_samples) {
            for (j, &val) in target_vec.iter().enumerate().take(n_outputs) {
                target_matrix[[i, j]] = val;
            }
        }
        let lambda = 1e-6;
        let xtx = feature_matrix.t().dot(&feature_matrix);
        let mut xtx_reg = xtx;
        for i in 0..xtx_reg.nrows().min(xtx_reg.ncols()) {
            xtx_reg[[i, i]] += lambda;
        }
        let xty = feature_matrix.t().dot(&target_matrix);
        self.solve_linear_system(&xtx_reg, &xty)?;
        Ok(())
    }
    /// Solve linear system (simplified implementation)
    pub(super) fn solve_linear_system(&mut self, a: &Array2<f64>, b: &Array2<f64>) -> Result<()> {
        let min_dim = a.nrows().min(a.ncols()).min(b.nrows());
        for i in 0..min_dim.min(self.output_weights.nrows()) {
            for j in 0..b.ncols().min(self.output_weights.ncols()) {
                if a[[i, i]].abs() > 1e-15 {
                    self.output_weights[[i, j]] = b[[i, j]] / a[[i, i]];
                }
            }
        }
        Ok(())
    }
}
