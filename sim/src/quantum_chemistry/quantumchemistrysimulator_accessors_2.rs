//! # QuantumChemistrySimulator - accessors Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::Molecule;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Get molecule reference
    #[must_use]
    pub const fn get_molecule(&self) -> Option<&Molecule> {
        self.molecule.as_ref()
    }
}
