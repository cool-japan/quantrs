//! # TensorCompression - Trait Implementations
//!
//! This module contains trait implementations for `TensorCompression`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::TensorCompression;

impl Default for TensorCompression {
    fn default() -> Self {
        Self::new()
    }
}
