//! # EnhancedTensorNetworkSimulator - prepare_contraction_indices_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

#[cfg(feature = "advanced_math")]
use super::types::ContractionIndices;
use super::types::EnhancedTensor;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    #[cfg(feature = "advanced_math")]
    pub(super) fn prepare_contraction_indices(
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
    ) -> Result<ContractionIndices> {
        Ok(ContractionIndices {
            tensor1_indices: tensor1.indices.iter().map(|i| i.label.clone()).collect(),
            tensor2_indices: tensor2.indices.iter().map(|i| i.label.clone()).collect(),
            common_indices: common_indices.to_vec(),
        })
    }
}
