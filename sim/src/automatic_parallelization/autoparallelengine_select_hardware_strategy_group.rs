//! # AutoParallelEngine - select_hardware_strategy_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use std::sync::{Arc, Barrier, Mutex, RwLock};

use super::types::{DependencyGraph, HardwareCharacteristics, HardwareStrategy};

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Select optimal hardware strategy based on circuit and hardware characteristics
    pub(super) fn select_hardware_strategy<const N: usize>(
        &self,
        hw_char: &HardwareCharacteristics,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<HardwareStrategy> {
        let gates = circuit.gates();
        let num_gates = gates.len();
        if hw_char.has_gpu && num_gates > 1000 {
            return Ok(HardwareStrategy::GPUOffload);
        }
        if hw_char.num_numa_nodes > 1 && N > 20 {
            return Ok(HardwareStrategy::NUMAAware);
        }
        let has_many_rotation_gates = self.count_rotation_gates(gates) > num_gates / 2;
        if has_many_rotation_gates && hw_char.simd_width >= 256 {
            return Ok(HardwareStrategy::SIMDOptimized);
        }
        if num_gates < 500 && N < 15 {
            return Ok(HardwareStrategy::CacheOptimized);
        }
        Ok(HardwareStrategy::Hybrid)
    }
    /// Count rotation gates in a gate list
    pub(super) fn count_rotation_gates(&self, gates: &[Arc<dyn GateOp + Send + Sync>]) -> usize {
        gates
            .iter()
            .filter(|g| Self::is_rotation_gate(g.as_ref()))
            .count()
    }
}
