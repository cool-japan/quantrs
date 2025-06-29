//! Temporary GPU stubs to replace scirs2_core GPU types
//! TODO: Replace with scirs2_core when regex dependency issue is fixed

use serde::{Deserialize, Serialize};

/// SciRS2 GPU configuration stub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciRS2GpuConfig {
    pub device_id: usize,
    pub memory_pool_size: usize,
    pub enable_profiling: bool,
    pub enable_async: bool,
    pub enable_kernel_cache: bool,
    pub max_memory_mb: usize,
    pub simd_level: u8,
    pub compilation_flags: Vec<String>,
    pub enable_load_balancing: bool,
}

impl Default for SciRS2GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            enable_profiling: false,
            enable_async: true,
            enable_kernel_cache: true,
            max_memory_mb: 2048,
            simd_level: 2,
            compilation_flags: vec!["-O3".to_string(), "-fast-math".to_string()],
            enable_load_balancing: true,
        }
    }
}
