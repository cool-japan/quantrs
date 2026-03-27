//! # SciRS2VectorizedFFT - Trait Implementations
//!
//! This module contains trait implementations for `SciRS2VectorizedFFT`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use std::collections::HashMap;

use super::types::{OptimizationLevel, SciRS2VectorizedFFT};

impl Default for SciRS2VectorizedFFT {
    fn default() -> Self {
        Self {
            plans: HashMap::new(),
            optimization_level: OptimizationLevel::Aggressive,
        }
    }
}
