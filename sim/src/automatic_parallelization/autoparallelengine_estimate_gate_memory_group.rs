//! # AutoParallelEngine - estimate_gate_memory_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::gate::GateOp;
use scirs2_core::Complex64;

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Estimate memory requirement for a gate
    pub(super) fn estimate_gate_memory(gate: &dyn GateOp) -> usize {
        let num_qubits = gate.num_qubits();
        let state_size = 1 << num_qubits;
        state_size * std::mem::size_of::<Complex64>()
    }
}
