//! # EnhancedTensorNetworkSimulator - estimate_contraction_cost_group Methods
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
    pub(super) fn estimate_contraction_cost(
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
    ) -> f64 {
        let size1: usize = tensor1.bond_dimensions.iter().product();
        let size2: usize = tensor2.bond_dimensions.iter().product();
        let common_size: usize = common_indices.len();
        (size1 as f64) * (size2 as f64) * (common_size as f64)
    }
    pub(super) fn contract_tensors_blocked(
        &self,
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
    ) -> Result<EnhancedTensor> {
        let block_size = self.config.memory_limit / (8 * std::mem::size_of::<Complex64>());
        let num_blocks = ((tensor1.data.len() + tensor2.data.len()) / block_size).max(1);
        let result_indices = Self::calculate_result_indices(tensor1, tensor2, common_indices);
        let result_shape = Self::calculate_result_shape(&result_indices)?;
        let mut result_data = ArrayD::zeros(IxDyn(&result_shape));
        for block_idx in 0..num_blocks {
            let start_idx = block_idx * (tensor1.data.len() / num_blocks);
            let end_idx =
                ((block_idx + 1) * (tensor1.data.len() / num_blocks)).min(tensor1.data.len());
            if start_idx < end_idx {
                let block1 = Self::extract_tensor_block(tensor1, start_idx, end_idx)?;
                let block2 = Self::extract_tensor_block(tensor2, start_idx, end_idx)?;
                let block_result = Self::contract_tensor_blocks(&block1, &block2, common_indices)?;
                Self::accumulate_block_result(&result_data, &block_result, block_idx)?;
            }
        }
        let memory_size = result_data.len() * std::mem::size_of::<Complex64>();
        Ok(EnhancedTensor {
            data: result_data,
            indices: result_indices,
            bond_dimensions: result_shape,
            id: 0,
            memory_size,
            contraction_cost: Self::estimate_contraction_cost(tensor1, tensor2, common_indices),
            priority: 1.0,
        })
    }
}
