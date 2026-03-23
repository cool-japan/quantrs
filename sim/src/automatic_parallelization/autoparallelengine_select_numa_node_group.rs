//! # AutoParallelEngine - select_numa_node_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GateNode;

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Select NUMA node for a gate
    pub(super) fn select_numa_node(node: &GateNode, num_nodes: usize) -> usize {
        let qubit_sum: usize = node.qubits.iter().map(|q| q.0 as usize).sum();
        qubit_sum % num_nodes
    }
}
