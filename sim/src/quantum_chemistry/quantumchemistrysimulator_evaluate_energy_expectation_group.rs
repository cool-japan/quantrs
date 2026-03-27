//! # QuantumChemistrySimulator - evaluate_energy_expectation_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};
use crate::pauli::{PauliOperator, PauliOperatorSum, PauliString};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::types::MolecularHamiltonian;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Evaluate energy expectation value
    pub(super) fn evaluate_energy_expectation(
        &self,
        circuit: &InterfaceCircuit,
        hamiltonian: &MolecularHamiltonian,
    ) -> Result<f64> {
        let final_state = self.get_circuit_final_state(circuit)?;
        if let Some(ref pauli_ham) = hamiltonian.pauli_hamiltonian {
            self.calculate_pauli_expectation(&final_state, pauli_ham)
        } else {
            Ok(hamiltonian.one_electron_integrals[[0, 0]])
        }
    }
    /// Calculate expectation value with Pauli Hamiltonian
    pub(super) fn calculate_pauli_expectation(
        &self,
        state: &Array1<Complex64>,
        pauli_ham: &PauliOperatorSum,
    ) -> Result<f64> {
        let mut expectation = 0.0;
        for pauli_term in &pauli_ham.terms {
            let pauli_expectation = self.calculate_single_pauli_expectation(state, pauli_term)?;
            expectation += pauli_expectation.re;
        }
        Ok(expectation)
    }
    /// Compute parameter gradient using finite differences
    pub(super) fn compute_parameter_gradient(
        &self,
        circuit: &InterfaceCircuit,
        hamiltonian: &MolecularHamiltonian,
    ) -> Result<Array1<f64>> {
        let num_params = self.vqe_optimizer.parameters.len();
        let mut gradient = Array1::zeros(num_params);
        let epsilon = 1e-4;
        for i in 0..num_params {
            let mut params_plus = self.vqe_optimizer.parameters.clone();
            params_plus[i] += epsilon;
            let circuit_plus = self.apply_ansatz_parameters(circuit, &params_plus)?;
            let energy_plus = self.evaluate_energy_expectation(&circuit_plus, hamiltonian)?;
            let mut params_minus = self.vqe_optimizer.parameters.clone();
            params_minus[i] -= epsilon;
            let circuit_minus = self.apply_ansatz_parameters(circuit, &params_minus)?;
            let energy_minus = self.evaluate_energy_expectation(&circuit_minus, hamiltonian)?;
            gradient[i] = (energy_plus - energy_minus) / (2.0 * epsilon);
        }
        Ok(gradient)
    }
}
