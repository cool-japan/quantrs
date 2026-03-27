//! # StyleChecker - Trait Implementations
//!
//! This module contains trait implementations for `StyleChecker`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StyleChecker;

impl<const N: usize> Default for StyleChecker<N> {
    fn default() -> Self {
        Self::new()
    }
}
