//! # AutoParallelEngine - aggregate_distributed_results_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::parallel_ops::{current_num_threads, IndexedParallelIterator, ParallelIterator};
use scirs2_core::Complex64;

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Aggregate results from distributed execution
    pub(super) fn aggregate_distributed_results(
        node_results: Vec<Vec<Complex64>>,
    ) -> QuantRS2Result<Vec<Complex64>> {
        use scirs2_core::parallel_ops::{IndexedParallelIterator, ParallelIterator};
        let total_elements: usize = node_results.iter().map(std::vec::Vec::len).sum();
        let mut aggregated = Vec::with_capacity(total_elements);
        for node_result in node_results {
            aggregated.extend(node_result);
        }
        Ok(aggregated)
    }
}
