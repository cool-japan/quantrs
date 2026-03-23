//! # QuantumContinuousFlow - compute_base_log_probability_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use std::f64::consts::PI;

use super::types::QuantumDistributionType;

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Compute base distribution log probability
    pub(super) fn compute_base_log_probability(&self, z: &Array1<f64>) -> Result<f64> {
        match &self.base_distribution.distribution_type {
            QuantumDistributionType::QuantumGaussian {
                mean,
                covariance,
                quantum_enhancement,
            } => {
                let diff = z - mean;
                let mahalanobis_distance = diff
                    .iter()
                    .zip(covariance.diag().iter())
                    .map(|(d, cov)| d * d / cov.max(1e-8))
                    .sum::<f64>();
                let log_prob = -0.5
                    * (mahalanobis_distance
                        + z.len() as f64 * (2.0 * PI).ln()
                        + covariance.diag().iter().map(|x| x.ln()).sum::<f64>());
                let quantum_log_prob = log_prob * (1.0 + quantum_enhancement);
                Ok(quantum_log_prob)
            }
            _ => Ok(0.0),
        }
    }
}
