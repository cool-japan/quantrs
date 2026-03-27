//! # DistributedGpuConfig - Trait Implementations
//!
//! This module contains trait implementations for `DistributedGpuConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;

use super::types::{DistributedGpuConfig, SyncStrategy};

impl Default for DistributedGpuConfig {
    fn default() -> Self {
        Self {
            num_gpus: 0,
            min_qubits_for_gpu: 15,
            max_state_size_per_gpu: 1 << 26,
            auto_load_balance: true,
            memory_overlap_ratio: 0.1,
            use_mixed_precision: false,
            sync_strategy: SyncStrategy::AllReduce,
        }
    }
}
