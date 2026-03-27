//! # MemoryPool - Trait Implementations
//!
//! This module contains trait implementations for `MemoryPool`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::MemoryPool;

#[cfg(not(feature = "advanced_math"))]
impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}
