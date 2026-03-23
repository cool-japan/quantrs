//! # QuantumChemistrySimulator - accessors Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::Molecule;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Set molecule for calculation
    pub fn set_molecule(&mut self, molecule: Molecule) -> Result<()> {
        self.molecule = Some(molecule);
        self.hamiltonian = None;
        self.hartree_fock = None;
        Ok(())
    }
}
