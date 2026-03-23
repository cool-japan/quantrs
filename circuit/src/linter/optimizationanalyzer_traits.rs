//! # OptimizationAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `OptimizationAnalyzer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::OptimizationAnalyzer;

impl<const N: usize> Default for OptimizationAnalyzer<N> {
    fn default() -> Self {
        Self::new()
    }
}
