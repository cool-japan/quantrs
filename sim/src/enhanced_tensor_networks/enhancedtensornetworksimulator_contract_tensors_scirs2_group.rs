//! # EnhancedTensorNetworkSimulator - contract_tensors_scirs2_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use scirs2_core::random::prelude::*;

use super::types::EnhancedTensor;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    #[cfg(feature = "advanced_math")]
    pub(super) fn contract_tensors_scirs2(
        &self,
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
    ) -> Result<EnhancedTensor> {
        if let Some(ref backend) = self.backend {
            return self.contract_with_scirs2_backend(tensor1, tensor2, common_indices, backend);
        }
        self.contract_tensors_optimized(tensor1, tensor2, common_indices)
    }
    #[cfg(feature = "advanced_math")]
    pub(super) fn contract_with_scirs2_backend(
        &self,
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
        backend: &SciRS2Backend,
    ) -> Result<EnhancedTensor> {
        let scirs2_tensor1 = Self::convert_to_scirs2_tensor(tensor1)?;
        let scirs2_tensor2 = Self::convert_to_scirs2_tensor(tensor2)?;
        let contraction_indices =
            Self::prepare_contraction_indices(tensor1, tensor2, common_indices)?;
        let result_scirs2 =
            backend.einsum_contract(&scirs2_tensor1, &scirs2_tensor2, &contraction_indices)?;
        self.convert_from_scirs2_tensor(&result_scirs2, tensor1, tensor2, common_indices)
    }
    pub(super) fn contract_tensors_optimized(
        &self,
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
        common_indices: &[String],
    ) -> Result<EnhancedTensor> {
        let contraction_size = Self::estimate_contraction_size(tensor1, tensor2, common_indices);
        if contraction_size > 1e6 {
            self.contract_tensors_blocked(tensor1, tensor2, common_indices)
        } else if common_indices.len() > 4 {
            self.contract_tensors_multi_index(tensor1, tensor2, common_indices)
        } else {
            self.contract_tensors_direct_optimized(tensor1, tensor2, common_indices)
        }
    }
}
