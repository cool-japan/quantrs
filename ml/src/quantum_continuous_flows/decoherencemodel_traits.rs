//! # DecoherenceModel - Trait Implementations
//!
//! This module contains trait implementations for `DecoherenceModel`.
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

use super::types::DecoherenceModel;

impl Default for DecoherenceModel {
    fn default() -> Self {
        Self {
            t1_time: 100.0,
            t2_time: 50.0,
            gate_error_rate: 0.001,
            measurement_error_rate: 0.01,
        }
    }
}
