//! # ResourceConstraints - Trait Implementations
//!
//! This module contains trait implementations for `ResourceConstraints`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ResourceConstraints;

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_per_thread: 1024 * 1024 * 1024,
            max_cpu_utilization: 0.8,
            max_gates_per_thread: 1000,
            preferred_numa_node: None,
        }
    }
}
