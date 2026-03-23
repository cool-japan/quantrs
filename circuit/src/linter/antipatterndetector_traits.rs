//! # AntiPatternDetector - Trait Implementations
//!
//! This module contains trait implementations for `AntiPatternDetector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AntiPatternDetector;

impl<const N: usize> Default for AntiPatternDetector<N> {
    fn default() -> Self {
        Self::new()
    }
}
