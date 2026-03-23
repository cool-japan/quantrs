//! # EnhancedTensorNetworkSimulator - accessors Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array, Array2, ArrayD, IxDyn};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    /// Get final result tensor
    pub fn get_result_tensor(&self) -> Result<ArrayD<Complex64>> {
        if self.network.tensors.len() != 1 {
            return Err(SimulatorError::InvalidInput(format!(
                "Expected single result tensor, found {}",
                self.network.tensors.len()
            )));
        }
        let result_tensor = self
            .network
            .tensors
            .values()
            .next()
            .ok_or_else(|| SimulatorError::InvalidInput("No tensors in network".to_string()))?;
        Ok(result_tensor.data.clone())
    }
}
