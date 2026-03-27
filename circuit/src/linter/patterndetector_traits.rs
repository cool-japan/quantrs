//! # PatternDetector - Trait Implementations
//!
//! This module contains trait implementations for `PatternDetector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PatternDetector;

impl<const N: usize> Default for PatternDetector<N> {
    fn default() -> Self {
        Self::new()
    }
}
