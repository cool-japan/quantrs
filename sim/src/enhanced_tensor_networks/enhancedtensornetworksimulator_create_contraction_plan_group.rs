//! # EnhancedTensorNetworkSimulator - create_contraction_plan_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::{ContractionOpType, ContractionOperation, ContractionPlan, EnhancedTensor};

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn create_contraction_plan(
        _tensor1: &EnhancedTensor,
        _tensor2: &EnhancedTensor,
        _common_indices: &[String],
    ) -> Result<ContractionPlan> {
        Ok(ContractionPlan {
            operations: vec![ContractionOperation {
                tensor1_indices: vec![0, 1],
                tensor2_indices: vec![0, 1],
                result_indices: vec![0],
                operation_type: ContractionOpType::EinsumContraction,
            }],
        })
    }
}
