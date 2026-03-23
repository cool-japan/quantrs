//! # TQExcitationPreservingLayer - Trait Implementations
//!
//! This module contains trait implementations for `TQExcitationPreservingLayer`.
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
use super::types::TQExcitationPreservingLayer;

impl TQModule for TQExcitationPreservingLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let n_pairs = if self.circular {
            self.n_wires
        } else {
            self.n_wires.saturating_sub(1)
        };
        let mut gate_idx = 0;
        for _ in 0..self.n_blocks {
            for pair in 0..n_pairs {
                let w0 = pair;
                let w1 = (pair + 1) % self.n_wires;
                if gate_idx < self.gates.len() {
                    self.gates[gate_idx].apply(qdev, &[w0, w1])?;
                    gate_idx += 1;
                }
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
        let n_pairs = if self.circular {
            n_wires
        } else {
            n_wires.saturating_sub(1)
        };
        let total_gates = n_pairs * self.n_blocks;
        self.gates = (0..total_gates)
            .map(|_| TQFSimGate::new(true, true))
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
        "ExcitationPreservingLayer"
    }
    fn zero_grad(&mut self) {
        for gate in &mut self.gates {
            gate.zero_grad();
        }
    }
}
