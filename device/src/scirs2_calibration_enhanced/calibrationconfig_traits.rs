//! # CalibrationConfig - Trait Implementations
//!
//! This module contains trait implementations for `CalibrationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::{CalibrationConfig, CalibrationProtocols, HardwareSpec};

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_shots: 10000,
            sequence_length: 100,
            convergence_threshold: 1e-4,
            max_iterations: 100,
            hardware_spec: HardwareSpec::default(),
            protocols: CalibrationProtocols::default(),
        }
    }
}
