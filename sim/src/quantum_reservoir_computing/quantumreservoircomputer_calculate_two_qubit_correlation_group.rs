//! # QuantumReservoirComputer - calculate_two_qubit_correlation_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Calculate two-qubit correlation
    pub(super) fn calculate_two_qubit_correlation(
        &self,
        qubit1: usize,
        qubit2: usize,
    ) -> Result<f64> {
        let state = &self.reservoir_state.state_vector;
        let mut correlation = 0.0;
        for i in 0..state.len() {
            let bit1 = (i >> qubit1) & 1;
            let bit2 = (i >> qubit2) & 1;
            let sign = if bit1 == bit2 { 1.0 } else { -1.0 };
            correlation += sign * state[i].norm_sqr();
        }
        Ok(correlation)
    }
}
