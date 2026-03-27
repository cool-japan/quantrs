//! # EnhancedTensorNetworkSimulator - optimize_path_scirs2_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

#[cfg(feature = "advanced_math")]
use super::types::ContractionOptimizer;
use super::types::EnhancedContractionPath;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    #[cfg(feature = "advanced_math")]
    pub(super) fn optimize_path_scirs2(
        &self,
        tensor_ids1: &[usize],
        tensor_ids2: &[usize],
        tensor_ids3: &[usize],
        optimizer: &ContractionOptimizer,
    ) -> Result<EnhancedContractionPath> {
        let all_ids: Vec<usize> = tensor_ids1
            .iter()
            .chain(tensor_ids2.iter())
            .chain(tensor_ids3.iter())
            .copied()
            .collect();
        self.optimize_path_adaptive(&all_ids)
    }
}
