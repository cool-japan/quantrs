//! # EnhancedTensorNetworkSimulator - convert_from_scirs2_tensor_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::types::EnhancedTensor;
#[cfg(feature = "advanced_math")]
use super::types::SciRS2Tensor;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    #[cfg(feature = "advanced_math")]
    pub(super) fn convert_from_scirs2_tensor(
        &self,
        scirs2_tensor: &SciRS2Tensor,
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
    ) -> Result<EnhancedTensor> {
        let result_indices = Self::calculate_result_indices(tensor1, tensor2, common_indices);
        Ok(EnhancedTensor {
            data: scirs2_tensor.data.clone(),
            indices: result_indices,
            bond_dimensions: scirs2_tensor.shape.clone(),
            id: 0,
            memory_size: scirs2_tensor.data.len() * std::mem::size_of::<Complex64>(),
            contraction_cost: 1.0,
            priority: 1.0,
        })
    }
}
