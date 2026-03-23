//! # QuantumReservoirComputer - apply_gate_errors_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Apply gate errors
    pub(super) fn apply_gate_errors(&mut self) -> Result<()> {
        let error_rate = self.config.noise_level;
        for qubit in 0..self.config.num_qubits {
            if thread_rng().random::<f64>() < error_rate {
                let error_type = thread_rng().random_range(0..3);
                let gate_type = match error_type {
                    0 => InterfaceGateType::X,
                    1 => InterfaceGateType::PauliY,
                    _ => InterfaceGateType::PauliZ,
                };
                self.apply_single_qubit_gate(qubit, gate_type)?;
            }
        }
        Ok(())
    }
}
