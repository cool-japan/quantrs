//! # ReadoutProtocols - Trait Implementations
//!
//! This module contains trait implementations for `ReadoutProtocols`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::ReadoutProtocols;

impl Default for ReadoutProtocols {
    fn default() -> Self {
        Self {
            state_discrimination: true,
            readout_optimization: true,
            iq_calibration: true,
            threshold_optimization: true,
        }
    }
}
