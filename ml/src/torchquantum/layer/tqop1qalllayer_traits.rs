//! # TQOp1QAllLayer - Trait Implementations
//!
//! This module contains trait implementations for `TQOp1QAllLayer`.
//!
//! ## Implemented Traits
//!
//! - `TQModule`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::torchquantum::{TQDevice, TQModule, TQOperator, TQParameter};

use super::functions::{create_single_qubit_gate, create_two_qubit_gate};
use super::types::TQOp1QAllLayer;

impl TQModule for TQOp1QAllLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for (wire, gate) in self.gates.iter_mut().enumerate() {
            gate.apply(qdev, &[wire])?;
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
        "Op1QAllLayer"
    }
    fn zero_grad(&mut self) {
        for gate in &mut self.gates {
            gate.zero_grad();
        }
    }
}
