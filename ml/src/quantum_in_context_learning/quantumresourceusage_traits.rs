//! # QuantumResourceUsage - Trait Implementations
//!
//! This module contains trait implementations for `QuantumResourceUsage`.
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
use std::collections::HashMap;
use std::f64::consts::PI;

use super::types::QuantumResourceUsage;

impl Default for QuantumResourceUsage {
    fn default() -> Self {
        Self {
            circuit_depth: 0,
            gate_count: HashMap::new(),
            entanglement_operations: 0,
            measurement_operations: 0,
            coherence_time_used: 0.0,
            quantum_volume_required: 0.0,
        }
    }
}
