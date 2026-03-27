//! # QuantumReservoirComputer - apply_measurement_errors_group Methods
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
    /// Apply measurement errors
    pub(super) fn apply_measurement_errors(&mut self) -> Result<()> {
        let error_rate = self.config.noise_level * 0.1;
        if thread_rng().random::<f64>() < error_rate {
            let qubit = thread_rng().random_range(0..self.config.num_qubits);
            self.apply_single_qubit_gate(qubit, InterfaceGateType::X)?;
        }
        Ok(())
    }
}
