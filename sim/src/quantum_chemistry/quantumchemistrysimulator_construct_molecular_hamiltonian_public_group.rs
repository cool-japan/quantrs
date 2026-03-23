//! # QuantumChemistrySimulator - construct_molecular_hamiltonian_public_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::Molecule;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Construct molecular Hamiltonian (public version)
    pub fn construct_molecular_hamiltonian_public(&mut self, molecule: &Molecule) -> Result<()> {
        self.construct_molecular_hamiltonian(molecule)
    }
}
