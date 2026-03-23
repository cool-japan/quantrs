//! # AsymptoticSafetyConfig - Trait Implementations
//!
//! This module contains trait implementations for `AsymptoticSafetyConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::AsymptoticSafetyConfig;

impl Default for AsymptoticSafetyConfig {
    fn default() -> Self {
        Self {
            uv_newton_constant: 0.1,
            uv_cosmological_constant: 0.01,
            truncation_order: 4,
            energy_scale: 1.0,
            critical_exponents: vec![-2.0, 0.5, 1.2],
            higher_derivatives: true,
            rg_flow_steps: 1000,
        }
    }
}
