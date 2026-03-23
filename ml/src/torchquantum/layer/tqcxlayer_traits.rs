//! # TQCXLayer - Trait Implementations
//!
//! This module contains trait implementations for `TQCXLayer`.
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
use super::types::TQCXLayer;

impl TQModule for TQCXLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let n_pairs = if self.circular {
            self.n_wires
        } else {
            self.n_wires.saturating_sub(1)
        };
        for i in 0..n_pairs {
            let wire0 = i;
            let wire1 = (i + 1) % self.n_wires;
            let mut gate = TQCNOT::new();
            gate.apply(qdev, &[wire0, wire1])?;
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
        "CXLayer"
    }
}
