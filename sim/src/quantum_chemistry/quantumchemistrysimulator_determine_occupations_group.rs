//! # QuantumChemistrySimulator - determine_occupations_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Determine orbital occupations
    pub(super) fn determine_occupations(
        &self,
        energies: &Array1<f64>,
        num_electrons: usize,
    ) -> Array1<f64> {
        let mut occupations = Array1::zeros(energies.len());
        let mut remaining_electrons = num_electrons;
        let mut orbital_indices: Vec<usize> = (0..energies.len()).collect();
        orbital_indices.sort_by(|&a, &b| {
            energies[a]
                .partial_cmp(&energies[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &orbital in &orbital_indices {
            if remaining_electrons >= 2 {
                occupations[orbital] = 2.0;
                remaining_electrons -= 2;
            } else if remaining_electrons == 1 {
                occupations[orbital] = 1.0;
                remaining_electrons -= 1;
            } else {
                break;
            }
        }
        occupations
    }
}
