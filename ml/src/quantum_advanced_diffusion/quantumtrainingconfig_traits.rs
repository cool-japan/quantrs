//! # QuantumTrainingConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumTrainingConfig`.
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

use super::types::QuantumTrainingConfig;

impl Default for QuantumTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 1e-4,
            learning_rate_decay: 0.99,
            log_interval: 10,
            save_interval: 50,
            early_stopping_patience: 20,
            quantum_loss_weight: 0.1,
            entanglement_preservation_weight: 0.05,
        }
    }
}
