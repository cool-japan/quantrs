//! # QuantumChemistrySimulator - build_density_matrix_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Build density matrix from molecular orbitals
    pub(super) fn build_density_matrix(
        &self,
        orbitals: &Array2<f64>,
        num_electrons: usize,
    ) -> Result<Array2<f64>> {
        let num_orbitals = orbitals.nrows();
        let mut density = Array2::zeros((num_orbitals, num_orbitals));
        let occupied_orbitals = num_electrons / 2;
        for i in 0..num_orbitals {
            for j in 0..num_orbitals {
                for occ in 0..occupied_orbitals {
                    density[[i, j]] += 2.0 * orbitals[[i, occ]] * orbitals[[j, occ]];
                }
            }
        }
        Ok(density)
    }
    /// Build density matrix (public version)
    pub fn build_density_matrix_public(
        &self,
        orbitals: &Array2<f64>,
        num_electrons: usize,
    ) -> Result<Array2<f64>> {
        self.build_density_matrix(orbitals, num_electrons)
    }
}
