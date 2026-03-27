//! # SimdOperations - Trait Implementations
//!
//! This module contains trait implementations for `SimdOperations`.
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

use super::types::SimdOperations;

impl Default for SimdOperations {
    fn default() -> Self {
        Self::new()
    }
}
