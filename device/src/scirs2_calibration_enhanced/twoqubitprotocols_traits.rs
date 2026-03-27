//! # TwoQubitProtocols - Trait Implementations
//!
//! This module contains trait implementations for `TwoQubitProtocols`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::TwoQubitProtocols;

impl Default for TwoQubitProtocols {
    fn default() -> Self {
        Self {
            chevron_pattern: true,
            cphase_calibration: true,
            iswap_calibration: true,
            cnot_calibration: true,
            zz_interaction: true,
        }
    }
}
