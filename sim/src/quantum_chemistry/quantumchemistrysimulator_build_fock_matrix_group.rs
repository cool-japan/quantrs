//! # QuantumChemistrySimulator - build_fock_matrix_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;

use super::types::MolecularHamiltonian;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Build Fock matrix from density matrix
    pub(super) fn build_fock_matrix(
        &self,
        fock: &mut Array2<f64>,
        density: &Array2<f64>,
        hamiltonian: &MolecularHamiltonian,
    ) -> Result<()> {
        let num_orbitals = hamiltonian.num_orbitals;
        fock.clone_from(&hamiltonian.one_electron_integrals);
        for i in 0..num_orbitals {
            for j in 0..num_orbitals {
                let mut two_electron_contribution = 0.0;
                for k in 0..num_orbitals {
                    for l in 0..num_orbitals {
                        two_electron_contribution +=
                            density[[k, l]] * hamiltonian.two_electron_integrals[[i, j, k, l]];
                        two_electron_contribution -= 0.5
                            * density[[k, l]]
                            * hamiltonian.two_electron_integrals[[i, k, j, l]];
                    }
                }
                fock[[i, j]] += two_electron_contribution;
            }
        }
        Ok(())
    }
}
