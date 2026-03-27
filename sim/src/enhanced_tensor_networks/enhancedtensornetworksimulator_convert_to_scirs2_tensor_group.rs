//! # EnhancedTensorNetworkSimulator - convert_to_scirs2_tensor_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::EnhancedTensor;
#[cfg(feature = "advanced_math")]
use super::types::SciRS2Tensor;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    #[cfg(feature = "advanced_math")]
    pub(super) fn convert_to_scirs2_tensor(tensor: &EnhancedTensor) -> Result<SciRS2Tensor> {
        Ok(SciRS2Tensor {
            data: tensor.data.clone(),
            shape: tensor.bond_dimensions.clone(),
        })
    }
}
