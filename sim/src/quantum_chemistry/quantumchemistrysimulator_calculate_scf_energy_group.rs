//! # QuantumChemistrySimulator - calculate_scf_energy_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Calculate SCF energy
    pub(super) fn calculate_scf_energy(
        &self,
        density: &Array2<f64>,
        one_electron: &Array2<f64>,
        fock: &Array2<f64>,
    ) -> Result<f64> {
        let mut energy = 0.0;
        for i in 0..density.nrows() {
            for j in 0..density.ncols() {
                energy += density[[i, j]] * (one_electron[[i, j]] + fock[[i, j]]);
            }
        }
        Ok(0.5 * energy)
    }
}
