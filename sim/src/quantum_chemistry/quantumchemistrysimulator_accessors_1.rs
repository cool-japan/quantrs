//! # QuantumChemistrySimulator - accessors Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use scirs2_core::random::prelude::*;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Get number of parameters in ansatz
    pub(super) fn get_ansatz_parameter_count(&self, circuit: &InterfaceCircuit) -> usize {
        let mut count = 0;
        for gate in &circuit.gates {
            match gate.gate_type {
                InterfaceGateType::RX(_) | InterfaceGateType::RY(_) | InterfaceGateType::RZ(_) => {
                    count += 1;
                }
                _ => {}
            }
        }
        count
    }
    /// Get ansatz parameter count (public version)
    #[must_use]
    pub fn get_ansatz_parameter_count_public(&self, circuit: &InterfaceCircuit) -> usize {
        self.get_ansatz_parameter_count(circuit)
    }
}
