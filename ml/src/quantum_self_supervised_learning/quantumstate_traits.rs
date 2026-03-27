//! # QuantumState - Trait Implementations
//!
//! This module contains trait implementations for `QuantumState`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::types::QuantumState;

impl Default for QuantumState {
    fn default() -> Self {
        Self {
            amplitudes: Array1::ones(1).mapv(|_: f64| Complex64::new(1.0, 0.0)),
            phases: Array1::ones(1).mapv(|_: f64| Complex64::new(1.0, 0.0)),
            entanglement_measure: 0.0,
            coherence_time: 1.0,
            fidelity: 1.0,
        }
    }
}
