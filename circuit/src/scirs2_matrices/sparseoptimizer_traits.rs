//! # SparseOptimizer - Trait Implementations
//!
//! This module contains trait implementations for `SparseOptimizer`.
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

use super::types::SparseOptimizer;

impl Default for SparseOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
