//! # EnhancedTensorNetworkSimulator - calculate_result_shape_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array, Array2, ArrayD, IxDyn};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::types::{EnhancedTensor, TensorIndex};

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn calculate_result_shape(indices: &[TensorIndex]) -> Result<Vec<usize>> {
        Ok(indices.iter().map(|idx| idx.dimension).collect())
    }
    pub(super) fn contract_tensor_blocks(
        block1: &EnhancedTensor,
        block2: &EnhancedTensor,
        common_indices: &[String],
    ) -> Result<ArrayD<Complex64>> {
        let result_indices = Self::calculate_result_indices(block1, block2, common_indices);
        let result_shape = Self::calculate_result_shape(&result_indices)?;
        Ok(ArrayD::zeros(IxDyn(&result_shape)))
    }
}
