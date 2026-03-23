//! # AutoParallelEngine - predicates Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::autoparallelengine_type::AutoParallelEngine;
use quantrs2_core::gate::GateOp;

impl AutoParallelEngine {
    /// Check if a gate is a rotation gate
    pub(super) fn is_rotation_gate(gate: &dyn GateOp) -> bool {
        let gate_str = format!("{gate:?}");
        gate_str.contains("RX") || gate_str.contains("RY") || gate_str.contains("RZ")
    }
}
