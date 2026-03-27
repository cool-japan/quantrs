//! # TQRandomLayer - Trait Implementations
//!
//! This module contains trait implementations for `TQRandomLayer`.
//!
//! ## Implemented Traits
//!
//! - `TQModule`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::torchquantum::{TQDevice, TQModule, TQOperator, TQParameter};

use super::functions::{create_single_qubit_gate, create_two_qubit_gate};
use super::types::TQRandomLayer;

impl TQModule for TQRandomLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for (op_name, wires) in &self.gate_sequence {
            if wires.len() == 1 {
                let mut gate = create_single_qubit_gate(op_name, true, false);
                gate.apply(qdev, wires)?;
            } else {
                let mut gate = create_two_qubit_gate(op_name, false, false);
                gate.apply(qdev, wires)?;
            }
        }
        Ok(())
    }
    fn parameters(&self) -> Vec<TQParameter> {
        Vec::new()
    }
    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }
    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
        self.regenerate();
    }
    fn is_static_mode(&self) -> bool {
        self.static_mode
    }
    fn static_on(&mut self) {
        self.static_mode = true;
    }
    fn static_off(&mut self) {
        self.static_mode = false;
    }
    fn name(&self) -> &str {
        "RandomLayer"
    }
}
