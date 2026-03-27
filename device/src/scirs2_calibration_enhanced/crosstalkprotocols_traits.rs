//! # CrosstalkProtocols - Trait Implementations
//!
//! This module contains trait implementations for `CrosstalkProtocols`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::CrosstalkProtocols;

impl Default for CrosstalkProtocols {
    fn default() -> Self {
        Self {
            simultaneous_gates: true,
            spectator_qubits: true,
            drive_crosstalk: true,
            measurement_crosstalk: true,
        }
    }
}
