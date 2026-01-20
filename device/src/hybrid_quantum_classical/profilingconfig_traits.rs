//! # Profilingconfig - Trait Implementations
//!
//! This module contains trait implementations for `Profilingconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            level: ProfilingLevel::Basic,
            sampling_frequency: 1.0,
            output_format: ProfilingOutputFormat::JSON,
        }
    }
}
