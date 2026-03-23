//! # EnhancedTensorNetworkSimulator - extract_tensor_block_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::types::EnhancedTensor;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn extract_tensor_block(
        tensor: &EnhancedTensor,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<EnhancedTensor> {
        let block_data = tensor
            .data
            .slice(scirs2_core::ndarray::s![start_idx..end_idx])
            .to_owned();
        Ok(EnhancedTensor {
            data: block_data.into_dyn(),
            indices: tensor.indices.clone(),
            bond_dimensions: tensor.bond_dimensions.clone(),
            id: tensor.id,
            memory_size: (end_idx - start_idx) * std::mem::size_of::<Complex64>(),
            contraction_cost: tensor.contraction_cost,
            priority: tensor.priority,
        })
    }
}
