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
    /// Amplitude encoding
    pub(super) fn encode_amplitude(&mut self, input: &Array1<f64>) -> Result<()> {
        let num_inputs = input.len().min(self.config.num_qubits);
        for i in 0..num_inputs {
            let angle = input[i] * std::f64::consts::PI;
            self.apply_single_qubit_rotation(i, InterfaceGateType::RY(angle))?;
        }
        Ok(())
    }
}
