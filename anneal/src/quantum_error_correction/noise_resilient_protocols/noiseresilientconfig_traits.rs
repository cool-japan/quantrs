//! # NoiseResilientConfig - Trait Implementations
//!
//! This module contains trait implementations for `NoiseResilientConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::NoiseResilientConfig;

impl Default for NoiseResilientConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_scheduling: true,
            error_threshold: 0.05,
            max_adaptation_steps: 5,
            min_annealing_time_factor: 0.5,
            max_annealing_time_factor: 3.0,
            enable_protocol_switching: true,
            enable_real_time_correction: true,
            enable_decoherence_compensation: true,
        }
    }
}
