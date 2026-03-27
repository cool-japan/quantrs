//! # QuantumChemistrySimulator - compute_one_electron_integrals_group Methods
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
    /// Compute one-electron integrals
    pub(super) fn compute_one_electron_integrals(
        &self,
        molecule: &Molecule,
        num_orbitals: usize,
    ) -> Result<Array2<f64>> {
        let mut integrals = Array2::zeros((num_orbitals, num_orbitals));
        if molecule.atomic_numbers.len() == 2
            && molecule.atomic_numbers[0] == 1
            && molecule.atomic_numbers[1] == 1
        {
            let bond_length = (molecule.positions[[0, 2]] - molecule.positions[[1, 2]])
                .mul_add(
                    molecule.positions[[0, 2]] - molecule.positions[[1, 2]],
                    (molecule.positions[[0, 1]] - molecule.positions[[1, 1]]).mul_add(
                        molecule.positions[[0, 1]] - molecule.positions[[1, 1]],
                        (molecule.positions[[0, 0]] - molecule.positions[[1, 0]]).powi(2),
                    ),
                )
                .sqrt();
            let overlap = 0.6593 * (-0.1158 * bond_length * bond_length).exp();
            let kinetic = 1.2266f64.mul_add(-overlap, 0.7618);
            let nuclear_attraction = -1.2266;
            integrals[[0, 0]] = kinetic + nuclear_attraction;
            integrals[[1, 1]] = kinetic + nuclear_attraction;
            integrals[[0, 1]] = overlap * (kinetic + nuclear_attraction);
            integrals[[1, 0]] = integrals[[0, 1]];
        } else {
            for i in 0..num_orbitals {
                integrals[[i, i]] = -0.5
                    * f64::from(molecule.atomic_numbers[i.min(molecule.atomic_numbers.len() - 1)]);
                for j in i + 1..num_orbitals {
                    let distance = if i < molecule.positions.nrows()
                        && j < molecule.positions.nrows()
                    {
                        (molecule.positions[[i, 2]] - molecule.positions[[j, 2]])
                            .mul_add(
                                molecule.positions[[i, 2]] - molecule.positions[[j, 2]],
                                (molecule.positions[[i, 1]] - molecule.positions[[j, 1]]).mul_add(
                                    molecule.positions[[i, 1]] - molecule.positions[[j, 1]],
                                    (molecule.positions[[i, 0]] - molecule.positions[[j, 0]])
                                        .powi(2),
                                ),
                            )
                            .sqrt()
                    } else {
                        1.0
                    };
                    let coupling = -0.1 / (1.0 + distance);
                    integrals[[i, j]] = coupling;
                    integrals[[j, i]] = coupling;
                }
            }
        }
        Ok(integrals)
    }
    /// Compute one electron integrals (public version)
    pub fn compute_one_electron_integrals_public(
        &self,
        molecule: &Molecule,
        num_orbitals: usize,
    ) -> Result<Array2<f64>> {
        self.compute_one_electron_integrals(molecule, num_orbitals)
    }
}
