//! # Classicalresourceconfig - Trait Implementations
//!
//! This module contains trait implementations for `Classicalresourceconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ClassicalResourceConfig {
    fn default() -> Self {
        Self {
            cpu_cores: num_cpus::get(),
            memory_limit_mb: 8192.0,
            gpu_devices: vec![],
            thread_pool_size: num_cpus::get() * 2,
            priority_level: ProcessPriority::Normal,
        }
    }
}
