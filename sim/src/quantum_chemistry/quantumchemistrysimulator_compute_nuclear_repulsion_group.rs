//! # QuantumChemistrySimulator - compute_nuclear_repulsion_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::Molecule;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Compute nuclear repulsion energy
    pub(super) fn compute_nuclear_repulsion(&self, molecule: &Molecule) -> Result<f64> {
        let mut nuclear_repulsion = 0.0;
        for i in 0..molecule.atomic_numbers.len() {
            for j in i + 1..molecule.atomic_numbers.len() {
                let distance = (molecule.positions[[i, 2]] - molecule.positions[[j, 2]])
                    .mul_add(
                        molecule.positions[[i, 2]] - molecule.positions[[j, 2]],
                        (molecule.positions[[i, 1]] - molecule.positions[[j, 1]]).mul_add(
                            molecule.positions[[i, 1]] - molecule.positions[[j, 1]],
                            (molecule.positions[[i, 0]] - molecule.positions[[j, 0]]).powi(2),
                        ),
                    )
                    .sqrt();
                if distance > 1e-10 {
                    nuclear_repulsion +=
                        f64::from(molecule.atomic_numbers[i] * molecule.atomic_numbers[j])
                            / distance;
                } else {
                    return Err(SimulatorError::NumericalError(
                        "Atoms are too close together (distance < 1e-10)".to_string(),
                    ));
                }
            }
        }
        Ok(nuclear_repulsion)
    }
    /// Compute nuclear repulsion (public version)
    pub fn compute_nuclear_repulsion_public(&self, molecule: &Molecule) -> Result<f64> {
        self.compute_nuclear_repulsion(molecule)
    }
}
