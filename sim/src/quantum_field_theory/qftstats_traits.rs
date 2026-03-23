//! # QFTStats - Trait Implementations
//!
//! This module contains trait implementations for `QFTStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::QFTStats;

impl Default for QFTStats {
    fn default() -> Self {
        Self {
            simulation_time: 0.0,
            field_evaluations: 0,
            pi_samples: 0,
            correlation_calculations: 0,
            rg_steps: 0,
            avg_plaquette: None,
            topological_charge: None,
        }
    }
}
