//! # TQCXCXCXLayer - Trait Implementations
//!
//! This module contains trait implementations for `TQCXCXCXLayer`.
//!
//! ## Implemented Traits
//!
//! - `TQModule`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::torchquantum::{
    gates::{
        TQHadamard, TQPauliX, TQPauliY, TQPauliZ, TQRx, TQRy, TQRz, TQCNOT, TQCRX, TQCRY, TQCRZ,
        TQCZ, TQRXX, TQRYY, TQRZX, TQRZZ, TQS, TQSWAP, TQSX, TQT,
    },
    CType, TQDevice, TQModule, TQOperator, TQParameter,
};

use super::functions::{create_single_qubit_gate, create_two_qubit_gate};
use super::types::TQCXCXCXLayer;

impl TQModule for TQCXCXCXLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        for _ in 0..3 {
            for i in 0..(self.n_wires - 1) {
                let mut gate = TQCNOT::new();
                gate.apply(qdev, &[i, i + 1])?;
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
        "CXCXCXLayer"
    }
}
