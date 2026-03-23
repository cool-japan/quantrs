//! # TQRealAmplitudesLayer - Trait Implementations
//!
//! This module contains trait implementations for `TQRealAmplitudesLayer`.
//!
//! ## Implemented Traits
//!
//! - `TQModule`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::torchquantum::{TQDevice, TQModule, TQOperator, TQParameter};

use super::functions::{create_single_qubit_gate, create_two_qubit_gate};
use super::types::TQRealAmplitudesLayer;

impl TQModule for TQRealAmplitudesLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        self.inner.forward(qdev)
    }
    fn parameters(&self) -> Vec<TQParameter> {
        self.inner.parameters()
    }
    fn n_wires(&self) -> Option<usize> {
        self.inner.n_wires()
    }
    fn set_n_wires(&mut self, n_wires: usize) {
        self.inner.set_n_wires(n_wires);
    }
    fn is_static_mode(&self) -> bool {
        self.inner.is_static_mode()
    }
    fn static_on(&mut self) {
        self.inner.static_on();
    }
    fn static_off(&mut self) {
        self.inner.static_off();
    }
    fn name(&self) -> &str {
        "RealAmplitudesLayer"
    }
    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }
}
