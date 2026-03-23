//! # TQHardwareEfficient2Layer - Trait Implementations
//!
//! This module contains trait implementations for `TQHardwareEfficient2Layer`.
//!
//! ## Implemented Traits
//!
//! - `TQModule`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::torchquantum::{TQDevice, TQModule, TQOperator, TQParameter};

use super::functions::{create_single_qubit_gate, create_two_qubit_gate};
use super::types::TQHardwareEfficient2Layer;

impl TQModule for TQHardwareEfficient2Layer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.forward(qdev)?;
        }
        Ok(())
    }
    fn parameters(&self) -> Vec<TQParameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
    fn n_wires(&self) -> Option<usize> {
        Some(self.config.n_wires)
    }
    fn set_n_wires(&mut self, n_wires: usize) {
        self.config.n_wires = n_wires;
    }
    fn is_static_mode(&self) -> bool {
        self.static_mode
    }
    fn static_on(&mut self) {
        self.static_mode = true;
        for layer in &mut self.layers {
            layer.static_on();
        }
    }
    fn static_off(&mut self) {
        self.static_mode = false;
        for layer in &mut self.layers {
            layer.static_off();
        }
    }
    fn name(&self) -> &str {
        "HardwareEfficient2Layer"
    }
    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}
