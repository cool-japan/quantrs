//! # SingleQubitProtocols - Trait Implementations
//!
//! This module contains trait implementations for `SingleQubitProtocols`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::SingleQubitProtocols;

impl Default for SingleQubitProtocols {
    fn default() -> Self {
        Self {
            rabi_oscillations: true,
            ramsey_fringes: true,
            drag_calibration: true,
            amplitude_calibration: true,
            phase_calibration: true,
        }
    }
}
