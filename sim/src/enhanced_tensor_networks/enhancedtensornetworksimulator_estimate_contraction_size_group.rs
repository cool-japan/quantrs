//! # EnhancedTensorNetworkSimulator - estimate_contraction_size_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::EnhancedTensor;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn estimate_contraction_size(
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
    ) -> f64 {
        let size1 = tensor1.data.len() as f64;
        let size2 = tensor2.data.len() as f64;
        let common_size = common_indices.len() as f64;
        size1 * size2 * common_size
    }
}
