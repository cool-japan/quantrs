//! # QuantumGravitySimulator - extract_pauli_coefficients_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Extract Pauli matrix coefficients
    pub(super) fn extract_pauli_coefficients(&self, matrix: &Array2<Complex64>) -> [Complex64; 4] {
        let trace = matrix[[0, 0]] + matrix[[1, 1]];
        let a0 = trace / 2.0;
        let a1 = (matrix[[0, 1]] + matrix[[1, 0]]) / 2.0;
        let a2 = (matrix[[0, 1]] - matrix[[1, 0]]) / (2.0 * Complex64::i());
        let a3 = (matrix[[0, 0]] - matrix[[1, 1]]) / 2.0;
        [a0, a1, a2, a3]
    }
}
