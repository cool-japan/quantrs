//! # AutoParallelEngine - detect_hardware_characteristics_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::HardwareCharacteristics;

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Detect hardware characteristics of the system
    pub(super) fn detect_hardware_characteristics() -> HardwareCharacteristics {
        use scirs2_core::parallel_ops::current_num_threads;
        let num_cores = current_num_threads();
        let l1_cache_size = 32 * 1024;
        let l2_cache_size = 256 * 1024;
        let l3_cache_size = 8 * 1024 * 1024;
        let memory_bandwidth = 50.0;
        let num_numa_nodes = if num_cores > 32 { 2 } else { 1 };
        let has_gpu = false;
        #[cfg(target_arch = "x86_64")]
        let simd_width = 256;
        #[cfg(not(target_arch = "x86_64"))]
        let simd_width = 128;
        HardwareCharacteristics {
            num_cores,
            l1_cache_size,
            l2_cache_size,
            l3_cache_size,
            memory_bandwidth,
            num_numa_nodes,
            has_gpu,
            simd_width,
        }
    }
}
