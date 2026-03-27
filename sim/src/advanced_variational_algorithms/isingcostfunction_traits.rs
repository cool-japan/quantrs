//! # IsingCostFunction - Trait Implementations
//!
//! This module contains trait implementations for `IsingCostFunction`.
//!
//! ## Implemented Traits
//!
//! - `CostFunction`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::functions::CostFunction;
use super::types::IsingCostFunction;

impl CostFunction for IsingCostFunction {
    fn evaluate(&self, _parameters: &[f64], circuit: &InterfaceCircuit) -> Result<f64> {
        let num_qubits = circuit.num_qubits;
        let mut cost = 0.0;
        for term in &self.problem_hamiltonian.terms {
            match term.pauli_string.as_str() {
                "ZZ" if term.qubits.len() == 2 => {
                    cost += term.coefficient.re;
                }
                "Z" if term.qubits.len() == 1 => {
                    cost += term.coefficient.re;
                }
                _ => {}
            }
        }
        Ok(cost)
    }
    fn get_observables(&self) -> Vec<String> {
        self.problem_hamiltonian
            .terms
            .iter()
            .map(|term| term.pauli_string.clone())
            .collect()
    }
    fn is_variational(&self) -> bool {
        true
    }
}
