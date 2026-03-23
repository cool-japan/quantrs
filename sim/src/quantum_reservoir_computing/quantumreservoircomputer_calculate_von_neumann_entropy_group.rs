//! # QuantumReservoirComputer - calculate_von_neumann_entropy_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Calculate von Neumann entropy (simplified)
    pub(super) fn calculate_von_neumann_entropy(&self, _qubit: usize) -> Result<f64> {
        let state = &self.reservoir_state.state_vector;
        let mut entropy = 0.0;
        for amplitude in state {
            let prob = amplitude.norm_sqr();
            if prob > 1e-15 {
                entropy -= prob * prob.ln();
            }
        }
        Ok(entropy / (state.len() as f64).ln())
    }
}
