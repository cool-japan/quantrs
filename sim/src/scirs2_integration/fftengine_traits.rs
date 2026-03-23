//! # FftEngine - Trait Implementations
//!
//! This module contains trait implementations for `FftEngine`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::FftEngine;

#[cfg(not(feature = "advanced_math"))]
impl Default for FftEngine {
    fn default() -> Self {
        Self::new()
    }
}
