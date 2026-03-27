//! # QuantumContinuousFlow - analyze_flow_convergence_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::FlowConvergenceAnalysis;

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Analyze flow convergence
    pub(super) fn analyze_flow_convergence(
        &self,
        losses: &[f64],
    ) -> Result<FlowConvergenceAnalysis> {
        if losses.len() < 10 {
            return Ok(FlowConvergenceAnalysis::default());
        }
        let recent_losses = &losses[losses.len() - 10..];
        let early_losses = &losses[0..10];
        let recent_avg = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let early_avg = early_losses.iter().sum::<f64>() / early_losses.len() as f64;
        let convergence_rate = (early_avg - recent_avg) / early_avg;
        let variance = recent_losses
            .iter()
            .map(|&x| (x - recent_avg).powi(2))
            .sum::<f64>()
            / recent_losses.len() as f64;
        Ok(FlowConvergenceAnalysis {
            convergence_rate,
            final_loss: recent_avg,
            loss_variance: variance,
            is_converged: variance < 1e-6,
            invertibility_maintained: true,
        })
    }
}
