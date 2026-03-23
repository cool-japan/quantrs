//! # AutoParallelEngine - gates_have_dependency_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DependencyGraph;

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Check if two gates have a dependency
    pub(super) fn gates_have_dependency(idx1: usize, idx2: usize, graph: &DependencyGraph) -> bool {
        if let Some(deps) = graph.reverse_edges.get(&idx2) {
            if deps.contains(&idx1) {
                return true;
            }
        }
        if let Some(deps) = graph.reverse_edges.get(&idx1) {
            if deps.contains(&idx2) {
                return true;
            }
        }
        false
    }
}
