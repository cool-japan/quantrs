//! # QuantumChemistrySimulator - apply_pauli_z_public_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Apply Pauli Z (public version)
    pub fn apply_pauli_z_public(&self, state: &mut Array1<Complex64>, qubit: usize) -> Result<()> {
        self.apply_pauli_z(state, qubit)
    }
}
