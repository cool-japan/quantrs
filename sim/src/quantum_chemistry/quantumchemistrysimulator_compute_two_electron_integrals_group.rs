//! # QuantumChemistrySimulator - compute_two_electron_integrals_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;

use super::types::Molecule;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Compute two-electron integrals
    pub(super) fn compute_two_electron_integrals(
        &self,
        _molecule: &Molecule,
        num_orbitals: usize,
    ) -> Result<Array4<f64>> {
        let mut integrals = Array4::zeros((num_orbitals, num_orbitals, num_orbitals, num_orbitals));
        for i in 0..num_orbitals {
            for j in 0..num_orbitals {
                for k in 0..num_orbitals {
                    for l in 0..num_orbitals {
                        if i == j && k == l && i == k {
                            integrals[[i, j, k, l]] = 0.625;
                        } else if (i == j && k == l) || (i == k && j == l) || (i == l && j == k) {
                            integrals[[i, j, k, l]] = 0.125;
                        }
                    }
                }
            }
        }
        Ok(integrals)
    }
    /// Compute two electron integrals (public version)
    pub fn compute_two_electron_integrals_public(
        &self,
        molecule: &Molecule,
        num_orbitals: usize,
    ) -> Result<Array4<f64>> {
        self.compute_two_electron_integrals(molecule, num_orbitals)
    }
}
