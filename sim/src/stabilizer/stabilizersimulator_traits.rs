//! # StabilizerSimulator - Trait Implementations
//!
//! This module contains trait implementations for `StabilizerSimulator`.
//!
//! ## Implemented Traits
//!
//! - `Simulator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::simulator::{Simulator, SimulatorResult};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::random::prelude::*;

use super::functions::gate_to_stabilizer;
use super::types::StabilizerSimulator;

/// Implement the Simulator trait for `StabilizerSimulator`
impl Simulator for StabilizerSimulator {
    fn run<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> crate::error::Result<SimulatorResult<N>> {
        let mut sim = Self::new(N);
        for gate in circuit.gates() {
            match gate_to_stabilizer(gate) {
                Some(stab_gate) => sim.apply_gate(stab_gate)?,
                None => {
                    // The stabilizer formalism can only simulate Clifford gates.
                    // Silently dropping an unrecognised gate would fabricate an
                    // incorrect result, so surface it as an honest error instead.
                    return Err(crate::error::SimulatorError::UnsupportedOperation(format!(
                        "StabilizerSimulator cannot simulate non-Clifford gate '{}' on qubits {:?}",
                        gate.name(),
                        gate.qubits()
                    )));
                }
            }
        }
        let amplitudes = sim.get_statevector();
        Ok(SimulatorResult::new(amplitudes))
    }
}
