//! # WaveFunctionConfig - Trait Implementations
//!
//! This module contains trait implementations for `WaveFunctionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{WaveFunctionConfig, WaveFunctionType};

impl Default for WaveFunctionConfig {
    fn default() -> Self {
        Self {
            wave_function_type: WaveFunctionType::SlaterJastrow,
            num_parameters: 32,
            jastrow_strength: 1.0,
            backflow_enabled: false,
        }
    }
}
