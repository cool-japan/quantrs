//! # SciRS2MemoryAllocator - Trait Implementations
//!
//! This module contains trait implementations for `SciRS2MemoryAllocator`.
//!
//! ## Implemented Traits
//!
//! - `Send`
//! - `Sync`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::SciRS2MemoryAllocator;

unsafe impl Send for SciRS2MemoryAllocator {}

unsafe impl Sync for SciRS2MemoryAllocator {}

impl Default for SciRS2MemoryAllocator {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            alignment: 64,
            allocation_count: 0,
        }
    }
}
