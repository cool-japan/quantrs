//! # SciRS2Backend - Trait Implementations
//!
//! This module contains trait implementations for `SciRS2Backend`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::SciRS2Backend;

impl Default for SciRS2Backend {
    fn default() -> Self {
        Self::new()
    }
}
