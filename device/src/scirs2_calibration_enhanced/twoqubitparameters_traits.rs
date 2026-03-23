//! # TwoQubitParameters - Trait Implementations
//!
//! This module contains trait implementations for `TwoQubitParameters`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::TwoQubitParameters;

impl Default for TwoQubitParameters {
    fn default() -> Self {
        Self {
            coupling_strength: 100e6,
            detuning: 0.0,
            cnot_angle: std::f64::consts::PI,
            cnot_duration: 200e-9,
            zz_strength: 1e6,
            gate_fidelity: 0.99,
        }
    }
}
