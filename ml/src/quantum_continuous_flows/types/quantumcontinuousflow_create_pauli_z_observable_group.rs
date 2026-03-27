//! # QuantumContinuousFlow - create_pauli_z_observable_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};

use super::types::Observable;

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Create Pauli-Z observable
    pub(super) fn create_pauli_z_observable(qubit: usize) -> Observable {
        let pauli_z = scirs2_core::ndarray::array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ];
        Observable {
            name: format!("PauliZ_{}", qubit),
            matrix: pauli_z,
            qubits: vec![qubit],
        }
    }
}
