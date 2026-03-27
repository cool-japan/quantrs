//! # CoherenceTimes - Trait Implementations
//!
//! This module contains trait implementations for `CoherenceTimes`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::CoherenceTimes;

impl Default for CoherenceTimes {
    fn default() -> Self {
        Self {
            t1: 50e-6,
            t2: 30e-6,
            t2_echo: 60e-6,
        }
    }
}
