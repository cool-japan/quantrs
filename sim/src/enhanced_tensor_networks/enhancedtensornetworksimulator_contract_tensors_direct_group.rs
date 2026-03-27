//! # EnhancedTensorNetworkSimulator - contract_tensors_direct_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array, Array2, ArrayD, IxDyn};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::types::EnhancedTensor;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn contract_tensors_direct(
        &self,
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
    ) -> Result<EnhancedTensor> {
        let result_shape = vec![2, 2];
        let result_data = Array::zeros(IxDyn(&result_shape));
        let result_indices = Self::calculate_result_indices(tensor1, tensor2, common_indices);
        let memory_size = result_data.len() * std::mem::size_of::<Complex64>();
        Ok(EnhancedTensor {
            data: result_data,
            indices: result_indices,
            bond_dimensions: vec![2, 2],
            id: 0,
            memory_size,
            contraction_cost: 1.0,
            priority: 1.0,
        })
    }
}
