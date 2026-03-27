//! # SciRS2SimdConfig - Trait Implementations
//!
//! This module contains trait implementations for `SciRS2SimdConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::SciRS2SimdConfig;

impl Default for SciRS2SimdConfig {
    fn default() -> Self {
        Self {
            force_instruction_set: None,
            override_simd_lanes: None,
            enable_aggressive_simd: true,
            numa_aware_allocation: true,
        }
    }
}
