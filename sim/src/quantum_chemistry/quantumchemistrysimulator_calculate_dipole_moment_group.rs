//! # QuantumChemistrySimulator - calculate_dipole_moment_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Calculate molecular dipole moment from density matrix
    pub(super) fn calculate_dipole_moment(&self, density_matrix: &Array2<f64>) -> Array1<f64> {
        let mut dipole = Array1::zeros(3);
        if let Some(molecule) = &self.molecule {
            for (i, &atomic_number) in molecule.atomic_numbers.iter().enumerate() {
                if i < molecule.positions.nrows() {
                    dipole[0] += f64::from(atomic_number) * molecule.positions[[i, 0]];
                    dipole[1] += f64::from(atomic_number) * molecule.positions[[i, 1]];
                    dipole[2] += f64::from(atomic_number) * molecule.positions[[i, 2]];
                }
            }
            let num_orbitals = density_matrix.nrows();
            for i in 0..num_orbitals {
                for j in 0..num_orbitals {
                    let density_element = density_matrix[[i, j]];
                    if i == j {
                        let orbital_pos = i as f64 / num_orbitals as f64;
                        dipole[0] -= density_element * orbital_pos;
                        dipole[1] -= density_element * orbital_pos * 0.5;
                        dipole[2] -= density_element * orbital_pos * 0.3;
                    }
                }
            }
        }
        dipole
    }
}
