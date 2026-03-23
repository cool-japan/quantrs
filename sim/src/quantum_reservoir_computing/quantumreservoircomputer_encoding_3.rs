//! # QuantumReservoirComputer - encoding Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Basis state encoding
    pub(super) fn encode_basis_state(&mut self, input: &Array1<f64>) -> Result<()> {
        let num_inputs = input.len().min(self.config.num_qubits);
        for i in 0..num_inputs {
            if input[i] > 0.5 {
                self.apply_single_qubit_gate(i, InterfaceGateType::X)?;
            }
        }
        Ok(())
    }
}
