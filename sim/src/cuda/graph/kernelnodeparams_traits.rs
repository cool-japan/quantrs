//! # KernelNodeParams - Trait Implementations
//!
//! This module contains trait implementations for `KernelNodeParams`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::KernelNodeParams;

impl Default for KernelNodeParams {
    fn default() -> Self {
        Self {
            function: 0,
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
            params: Vec::new(),
            extra: None,
        }
    }
}
