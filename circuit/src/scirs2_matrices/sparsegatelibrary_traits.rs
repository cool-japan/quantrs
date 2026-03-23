//! # SparseGateLibrary - Trait Implementations
//!
//! This module contains trait implementations for `SparseGateLibrary`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::{
    parallel_ops::{IndexedParallelIterator, ParallelIterator},
    simd_ops::*,
};

use super::types::SparseGateLibrary;

impl Default for SparseGateLibrary {
    fn default() -> Self {
        Self::new()
    }
}
