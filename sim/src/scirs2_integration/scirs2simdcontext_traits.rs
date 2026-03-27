//! # SciRS2SimdContext - Trait Implementations
//!
//! This module contains trait implementations for `SciRS2SimdContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::SciRS2SimdContext;

impl Default for SciRS2SimdContext {
    fn default() -> Self {
        Self::detect_capabilities()
    }
}
