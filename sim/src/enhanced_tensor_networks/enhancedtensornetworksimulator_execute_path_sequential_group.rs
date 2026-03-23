//! # EnhancedTensorNetworkSimulator - execute_path_sequential_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::EnhancedContractionPath;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn execute_path_sequential(&mut self, path: &EnhancedContractionPath) -> Result<()> {
        for step in &path.steps {
            self.contract_tensors(step.tensor_ids.0, step.tensor_ids.1)?;
        }
        Ok(())
    }
    pub(super) fn execute_path_parallel(&mut self, path: &EnhancedContractionPath) -> Result<()> {
        for section in &path.parallel_sections {
            let parallel_results: Result<Vec<_>> = section
                .parallel_steps
                .iter()
                .map(|&step_idx| {
                    let step = &path.steps[step_idx];
                    Ok(())
                })
                .collect();
            parallel_results?;
        }
        self.execute_path_sequential(path)
    }
}
