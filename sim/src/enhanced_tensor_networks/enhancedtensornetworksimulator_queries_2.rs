//! # EnhancedTensorNetworkSimulator - queries Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::{EnhancedTensor, OptimalIndexOrder};

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn contract_tensors_multi_index(
        &self,
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
    ) -> Result<EnhancedTensor> {
        let optimal_index_order = Self::find_optimal_index_order(tensor1, tensor2, common_indices)?;
        let reordered_tensor1 =
            Self::reorder_tensor_indices(tensor1, &optimal_index_order.tensor1_order)?;
        let reordered_tensor2 =
            Self::reorder_tensor_indices(tensor2, &optimal_index_order.tensor2_order)?;
        self.contract_tensors_direct_optimized(
            &reordered_tensor1,
            &reordered_tensor2,
            common_indices,
        )
    }
    pub(super) fn find_optimal_index_order(
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        _common_indices: &[String],
    ) -> Result<OptimalIndexOrder> {
        Ok(OptimalIndexOrder {
            tensor1_order: (0..tensor1.indices.len()).collect(),
            tensor2_order: (0..tensor2.indices.len()).collect(),
        })
    }
    pub(super) fn reorder_tensor_indices(
        tensor: &EnhancedTensor,
        _order: &[usize],
    ) -> Result<EnhancedTensor> {
        Ok(tensor.clone())
    }
}
