//! # BestPracticesChecker - Trait Implementations
//!
//! This module contains trait implementations for `BestPracticesChecker`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BestPracticesChecker;

impl<const N: usize> Default for BestPracticesChecker<N> {
    fn default() -> Self {
        Self::new()
    }
}
