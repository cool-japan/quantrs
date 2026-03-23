//! # QubitParameters - Trait Implementations
//!
//! This module contains trait implementations for `QubitParameters`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::QubitParameters;

impl Default for QubitParameters {
    fn default() -> Self {
        Self {
            frequency: 5e9,
            anharmonicity: -300e6,
            t1: 50e-6,
            t2_star: 30e-6,
            t2_echo: 60e-6,
            pi_pulse_amplitude: 0.5,
            pi_pulse_duration: 40e-9,
            drag_coefficient: 0.1,
            gate_fidelity: 0.999,
        }
    }
}
