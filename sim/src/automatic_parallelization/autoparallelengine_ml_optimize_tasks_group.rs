//! # AutoParallelEngine - ml_optimize_tasks_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use super::types::{MLFeatures, ParallelTask};

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// ML-guided task optimization
    pub(super) fn ml_optimize_tasks(
        &self,
        tasks: Vec<ParallelTask>,
        features: &MLFeatures,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        let mut optimized = tasks;
        Self::balance_task_loads(&mut optimized)?;
        if features.num_gates < 50 {
            optimized = self.merge_small_tasks(optimized)?;
        }
        if features.parallelism_factor > 0.6 {
            optimized = Self::split_large_tasks(optimized)?;
        }
        Ok(optimized)
    }
    /// Balance task loads across tasks
    pub(super) fn balance_task_loads(tasks: &mut [ParallelTask]) -> QuantRS2Result<()> {
        tasks.sort_by(|a, b| {
            b.cost
                .partial_cmp(&a.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }
}
