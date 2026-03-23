//! # TensorNetworkSimulator - Trait Implementations
//!
//! This module contains trait implementations for `TensorNetworkSimulator`.
//!
//! ## Implemented Traits
//!
//! - `Simulator`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;

use super::functions::{
    cnot_matrix, cz_gate, pauli_h, pauli_x, pauli_y, pauli_z, rotation_x, rotation_y, rotation_z,
    s_gate, swap_gate, t_gate,
};
use super::types::TensorNetworkSimulator;

impl crate::simulator::Simulator for TensorNetworkSimulator {
    fn run<const N: usize>(
        &mut self,
        circuit: &quantrs2_circuit::prelude::Circuit<N>,
    ) -> crate::error::Result<crate::simulator::SimulatorResult<N>> {
        self.initialize_zero_state().map_err(|e| {
            crate::error::SimulatorError::ComputationError(format!(
                "Failed to initialize state: {e}"
            ))
        })?;
        let gates = circuit.gates();
        for gate in gates {
            self.apply_circuit_gate(gate.as_ref()).map_err(|e| {
                crate::error::SimulatorError::ComputationError(format!("Failed to apply gate: {e}"))
            })?;
        }
        let final_state = self.contract_to_state_vector::<N>().map_err(|e| {
            crate::error::SimulatorError::ComputationError(format!(
                "Failed to contract tensor network: {e}"
            ))
        })?;
        Ok(crate::simulator::SimulatorResult::new(final_state))
    }
}

impl Default for TensorNetworkSimulator {
    fn default() -> Self {
        Self::new(1)
    }
}
