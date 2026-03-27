//! # TQParticleConservingLayer - Trait Implementations
//!
//! This module contains trait implementations for `TQParticleConservingLayer`.
//!
//! ## Implemented Traits
//!
//! - `TQModule`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::torchquantum::gates::{TQFSimGate, TQGivensRotation};
use crate::torchquantum::{TQDevice, TQModule, TQOperator, TQParameter};

use super::functions::{create_single_qubit_gate, create_two_qubit_gate};
use super::types::TQParticleConservingLayer;

impl TQModule for TQParticleConservingLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let pairs = self.get_wire_pairs();
        for (gate_idx, (w0, w1)) in pairs.iter().enumerate() {
            if gate_idx < self.gates.len() {
                self.gates[gate_idx].apply(qdev, &[*w0, *w1])?;
            }
        }
        Ok(())
    }
    fn parameters(&self) -> Vec<TQParameter> {
        self.gates.iter().flat_map(|g| g.parameters()).collect()
    }
    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }
    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
        let n_gates = Self::count_gates(n_wires, self.n_blocks, self.pattern);
        self.gates = (0..n_gates)
            .map(|_| TQGivensRotation::new(true, true))
            .collect();
    }
    fn is_static_mode(&self) -> bool {
        self.static_mode
    }
    fn static_on(&mut self) {
        self.static_mode = true;
        for gate in &mut self.gates {
            gate.static_on();
        }
    }
    fn static_off(&mut self) {
        self.static_mode = false;
        for gate in &mut self.gates {
            gate.static_off();
        }
    }
    fn name(&self) -> &str {
        "ParticleConservingLayer"
    }
    fn zero_grad(&mut self) {
        for gate in &mut self.gates {
            gate.zero_grad();
        }
    }
}
