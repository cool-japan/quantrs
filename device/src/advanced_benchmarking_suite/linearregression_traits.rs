//! # LinearRegression - Trait Implementations
//!
//! This module contains trait implementations for `LinearRegression`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::prelude::*;
use scirs2_core::random::prelude::*;

use super::types::LinearRegression;

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}
