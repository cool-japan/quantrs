//! # AdSCFTConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdSCFTConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::AdSCFTConfig;

impl Default for AdSCFTConfig {
    fn default() -> Self {
        Self {
            ads_dimension: 5,
            cft_dimension: 4,
            ads_radius: 1.0,
            central_charge: 100.0,
            temperature: 0.0,
            black_hole_formation: false,
            holographic_entanglement: true,
            degrees_of_freedom: 1000,
        }
    }
}
