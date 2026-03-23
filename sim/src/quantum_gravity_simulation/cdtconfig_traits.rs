//! # CDTConfig - Trait Implementations
//!
//! This module contains trait implementations for `CDTConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::CDTConfig;

impl Default for CDTConfig {
    fn default() -> Self {
        Self {
            num_simplices: 10_000,
            time_slicing: 0.1,
            spatial_volume: 1000.0,
            bare_coupling: 0.1,
            cosmological_coupling: 0.01,
            monte_carlo_moves: true,
            mc_sweeps: 1000,
            acceptance_threshold: 0.5,
        }
    }
}
