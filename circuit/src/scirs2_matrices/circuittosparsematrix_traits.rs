//! # CircuitToSparseMatrix - Trait Implementations
//!
//! This module contains trait implementations for `CircuitToSparseMatrix`.
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

use super::types::CircuitToSparseMatrix;

impl Default for CircuitToSparseMatrix {
    fn default() -> Self {
        Self::new()
    }
}
