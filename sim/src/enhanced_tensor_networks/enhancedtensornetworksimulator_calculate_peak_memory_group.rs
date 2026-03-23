//! # EnhancedTensorNetworkSimulator - calculate_peak_memory_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::ContractionStep;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn calculate_peak_memory(&self, steps: &[ContractionStep]) -> Result<usize> {
        let mut peak = 0;
        let mut current = 0;
        for step in steps {
            current += step.memory_required;
            peak = peak.max(current);
            current = (current as f64 * 0.8) as usize;
        }
        Ok(peak)
    }
}
