//! # TQQFTLayer - Trait Implementations
//!
//! This module contains trait implementations for `TQQFTLayer`.
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
use super::types::TQQFTLayer;

impl TQModule for TQQFTLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        use crate::torchquantum::gates::{TQHadamard, TQCU1, TQSWAP};
        use std::f64::consts::PI;
        if self.inverse {
            if self.do_swaps {
                for wire in 0..(self.n_wires / 2) {
                    let mut swap_gate = TQSWAP::new();
                    swap_gate.apply(
                        qdev,
                        &[self.wires[wire], self.wires[self.n_wires - wire - 1]],
                    )?;
                }
            }
            for top_wire in (0..self.n_wires).rev() {
                for wire in ((top_wire + 1)..self.n_wires).rev() {
                    let lam = -PI / (1 << (wire - top_wire)) as f64;
                    let mut cu1_gate = TQCU1::new(true, false);
                    cu1_gate.apply_with_params(
                        qdev,
                        &[self.wires[wire], self.wires[top_wire]],
                        Some(&[lam]),
                    )?;
                }
                let mut h_gate = TQHadamard::new();
                h_gate.apply(qdev, &[self.wires[top_wire]])?;
            }
        } else {
            for top_wire in 0..self.n_wires {
                let mut h_gate = TQHadamard::new();
                h_gate.apply(qdev, &[self.wires[top_wire]])?;
                for wire in (top_wire + 1)..self.n_wires {
                    let lam = PI / (1 << (wire - top_wire)) as f64;
                    let mut cu1_gate = TQCU1::new(true, false);
                    cu1_gate.apply_with_params(
                        qdev,
                        &[self.wires[wire], self.wires[top_wire]],
                        Some(&[lam]),
                    )?;
                }
            }
            if self.do_swaps {
                for wire in 0..(self.n_wires / 2) {
                    let mut swap_gate = TQSWAP::new();
                    swap_gate.apply(
                        qdev,
                        &[self.wires[wire], self.wires[self.n_wires - wire - 1]],
                    )?;
                }
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
        self.wires = (0..n_wires).collect();
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
        if self.inverse {
            "InverseQFTLayer"
        } else {
            "QFTLayer"
        }
    }
}
