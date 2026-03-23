//! # QuantumWalkParams - Trait Implementations
//!
//! This module contains trait implementations for `QuantumWalkParams`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::QuantumWalkParams;

impl Default for QuantumWalkParams {
    fn default() -> Self {
        Self {
            coin_bias: 0.5,
            step_size: 1.0,
            num_steps: 100,
            dimension: 1,
        }
    }
}
