//! # QuantumChemistrySimulator - create_fermionic_hamiltonian_public_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use crate::fermionic_simulation::{FermionicHamiltonian, FermionicOperator, FermionicString};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Create fermionic Hamiltonian (public version)
    pub fn create_fermionic_hamiltonian_public(
        &self,
        one_electron: &Array2<f64>,
        two_electron: &Array4<f64>,
        num_orbitals: usize,
    ) -> Result<FermionicHamiltonian> {
        self.create_fermionic_hamiltonian(one_electron, two_electron, num_orbitals)
    }
}
