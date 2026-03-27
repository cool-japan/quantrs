//! # QuantumReservoirComputer - calculate_single_qubit_expectation_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Calculate single qubit expectation value
    pub(super) fn calculate_single_qubit_expectation(
        &self,
        qubit: usize,
        pauli_matrix: &[Complex64; 4],
    ) -> Result<f64> {
        let state = &self.reservoir_state.state_vector;
        let mut expectation = 0.0;
        for i in 0..state.len() {
            for j in 0..state.len() {
                let i_bit = (i >> qubit) & 1;
                let j_bit = (j >> qubit) & 1;
                let matrix_element = pauli_matrix[i_bit * 2 + j_bit];
                expectation += (state[i].conj() * matrix_element * state[j]).re;
            }
        }
        Ok(expectation)
    }
}
