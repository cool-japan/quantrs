//! # AutoParallelEngine - hybrid_hardware_parallelization_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use super::types::{DependencyGraph, HardwareCharacteristics, ParallelTask};

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Hybrid hardware-aware parallelization
    pub(super) fn hybrid_hardware_parallelization(
        &self,
        graph: &DependencyGraph,
        hw_char: &HardwareCharacteristics,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        let base_tasks = self.dependency_based_parallelization(graph)?;
        let cache_aware_tasks = Self::refine_for_cache(base_tasks, hw_char)?;
        if hw_char.num_numa_nodes > 1 {
            Self::refine_for_numa(cache_aware_tasks, hw_char)
        } else {
            Ok(cache_aware_tasks)
        }
    }
    /// Refine tasks for NUMA efficiency
    pub(super) const fn refine_for_numa(
        tasks: Vec<ParallelTask>,
        hw_char: &HardwareCharacteristics,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        Ok(tasks)
    }
}
