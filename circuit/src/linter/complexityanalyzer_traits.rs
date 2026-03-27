//! # ComplexityAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `ComplexityAnalyzer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ComplexityAnalyzer;

impl<const N: usize> Default for ComplexityAnalyzer<N> {
    fn default() -> Self {
        Self::new()
    }
}
