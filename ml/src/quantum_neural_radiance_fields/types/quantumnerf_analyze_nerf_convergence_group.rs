//! # QuantumNeRF - analyze_nerf_convergence_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::NeRFConvergenceAnalysis;

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Analyze NeRF convergence
    pub(super) fn analyze_nerf_convergence(
        &self,
        losses: &[f64],
    ) -> Result<NeRFConvergenceAnalysis> {
        if losses.len() < 10 {
            return Ok(NeRFConvergenceAnalysis::default());
        }
        let recent_losses = &losses[losses.len() - 10..];
        let early_losses = &losses[0..10];
        let recent_avg = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let early_avg = early_losses.iter().sum::<f64>() / early_losses.len() as f64;
        let convergence_rate = (early_avg - recent_avg) / early_avg;
        Ok(NeRFConvergenceAnalysis {
            convergence_rate,
            final_loss: recent_avg,
            rendering_quality_score: 1.0 / (1.0 + recent_avg),
            quantum_advantage_achieved: convergence_rate > 0.1,
        })
    }
}
