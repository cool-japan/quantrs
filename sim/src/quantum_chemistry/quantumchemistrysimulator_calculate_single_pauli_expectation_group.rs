//! # QuantumChemistrySimulator - calculate_single_pauli_expectation_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use crate::pauli::{PauliOperator, PauliOperatorSum, PauliString};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Calculate expectation value of single Pauli string
    pub(super) fn calculate_single_pauli_expectation(
        &self,
        state: &Array1<Complex64>,
        pauli_string: &PauliString,
    ) -> Result<Complex64> {
        let mut result_state = state.clone();
        for (qubit, pauli_op) in pauli_string.operators.iter().enumerate() {
            match pauli_op {
                PauliOperator::X => {
                    self.apply_pauli_x(&mut result_state, qubit)?;
                }
                PauliOperator::Y => {
                    self.apply_pauli_y(&mut result_state, qubit)?;
                }
                PauliOperator::Z => {
                    self.apply_pauli_z(&mut result_state, qubit)?;
                }
                PauliOperator::I => {}
            }
        }
        let expectation = state
            .iter()
            .zip(result_state.iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>();
        Ok(expectation * pauli_string.coefficient)
    }
    /// Apply Pauli-X operator to state
    pub(super) fn apply_pauli_x(&self, state: &mut Array1<Complex64>, qubit: usize) -> Result<()> {
        let n = state.len();
        let bit_mask = 1 << qubit;
        for i in 0..n {
            if (i & bit_mask) == 0 {
                let j = i | bit_mask;
                if j < n {
                    let temp = state[i];
                    state[i] = state[j];
                    state[j] = temp;
                }
            }
        }
        Ok(())
    }
    /// Apply Pauli-Y operator to state
    pub(super) fn apply_pauli_y(&self, state: &mut Array1<Complex64>, qubit: usize) -> Result<()> {
        let n = state.len();
        let bit_mask = 1 << qubit;
        for i in 0..n {
            if (i & bit_mask) == 0 {
                let j = i | bit_mask;
                if j < n {
                    let temp = state[i];
                    state[i] = -Complex64::i() * state[j];
                    state[j] = Complex64::i() * temp;
                }
            }
        }
        Ok(())
    }
    /// Apply Pauli-Z operator to state
    pub(super) fn apply_pauli_z(&self, state: &mut Array1<Complex64>, qubit: usize) -> Result<()> {
        let bit_mask = 1 << qubit;
        for i in 0..state.len() {
            if (i & bit_mask) != 0 {
                state[i] = -state[i];
            }
        }
        Ok(())
    }
}
