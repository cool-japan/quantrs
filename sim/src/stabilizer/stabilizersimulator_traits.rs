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
            if let Some(stab_gate) = gate_to_stabilizer(gate) {
                let _ = sim.apply_gate(stab_gate);
            }
        }
        let amplitudes = sim.get_statevector();
        Ok(SimulatorResult::new(amplitudes))
    }
}
