//! # StepMetrics - Trait Implementations
//!
//! This module contains trait implementations for `StepMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};
use std::f64::consts::PI;

use super::types::StepMetrics;

impl Default for StepMetrics {
    fn default() -> Self {
        Self {
            entanglement_preservation: 0.0,
            phase_coherence: 0.0,
            denoising_confidence: 0.0,
            quantum_advantage: 1.0,
        }
    }
}
