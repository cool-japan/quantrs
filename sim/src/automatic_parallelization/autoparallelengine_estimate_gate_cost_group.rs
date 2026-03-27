//! # AutoParallelEngine - estimate_gate_cost_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::autoparallelengine_type::AutoParallelEngine;
use quantrs2_core::gate::GateOp;

impl AutoParallelEngine {
    /// Estimate execution cost for a gate
    pub(super) fn estimate_gate_cost(gate: &dyn GateOp) -> f64 {
        match gate.num_qubits() {
            1 => 1.0,
            2 => 4.0,
            3 => 8.0,
            n => (2.0_f64).powi(n as i32),
        }
    }
}
