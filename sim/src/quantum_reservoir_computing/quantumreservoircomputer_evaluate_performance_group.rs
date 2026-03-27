//! # QuantumReservoirComputer - evaluate_performance_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Evaluate performance on training data
    pub(super) fn evaluate_performance(
        &self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<(f64, f64)> {
        if features.is_empty() || targets.is_empty() {
            return Ok((0.0, 0.0));
        }
        let mut total_error = 0.0;
        let n_samples = features.len().min(targets.len());
        for i in 0..n_samples {
            let prediction = self.predict_output(&features[i])?;
            let error = self.calculate_prediction_error(&prediction, &targets[i]);
            total_error += error;
        }
        let training_error = total_error / n_samples as f64;
        let test_error = training_error;
        Ok((training_error, test_error))
    }
    /// Predict output for given features
    pub(super) fn predict_output(&self, features: &Array1<f64>) -> Result<Array1<f64>> {
        let feature_size = features.len().min(self.output_weights.ncols());
        let output_size = self.output_weights.nrows();
        let mut output = Array1::zeros(output_size);
        for i in 0..output_size {
            for j in 0..feature_size {
                output[i] += self.output_weights[[i, j]] * features[j];
            }
        }
        Ok(output)
    }
    /// Calculate prediction error
    pub(super) fn calculate_prediction_error(
        &self,
        prediction: &Array1<f64>,
        target: &Array1<f64>,
    ) -> f64 {
        let min_len = prediction.len().min(target.len());
        let mut error = 0.0;
        for i in 0..min_len {
            let diff = prediction[i] - target[i];
            error += diff * diff;
        }
        (error / min_len as f64).sqrt()
    }
}
