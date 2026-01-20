//! # NeRFTrainingConfig - Trait Implementations
//!
//! This module contains trait implementations for `NeRFTrainingConfig`.
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

use super::types::NeRFTrainingConfig;

impl Default for NeRFTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            rays_per_batch: 1024,
            learning_rate: 5e-4,
            learning_rate_decay: 0.999,
            quantum_loss_weight: 0.1,
            log_interval: 100,
        }
    }
}
