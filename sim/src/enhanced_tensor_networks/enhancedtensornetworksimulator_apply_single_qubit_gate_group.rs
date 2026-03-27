//! # EnhancedTensorNetworkSimulator - apply_single_qubit_gate_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array, Array2, ArrayD, IxDyn};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    /// Apply single-qubit gate as tensor
    pub fn apply_single_qubit_gate(
        &mut self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        let gate_tensor = Self::create_gate_tensor(gate_matrix, vec![qubit], None)?;
        let gate_id = self.network.add_tensor(gate_tensor);
        let qubit_label = format!("q{qubit}");
        let connected_tensors = self.network.find_connected_tensors(&qubit_label);
        if let Some(&qubit_tensor_id) = connected_tensors.first() {
            self.contract_tensors(gate_id, qubit_tensor_id)?;
        }
        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    /// Contract two tensors using advanced algorithms
    pub fn contract_tensors(&mut self, id1: usize, id2: usize) -> Result<usize> {
        let start_time = std::time::Instant::now();
        let tensor1 = self
            .network
            .get_tensor(id1)
            .ok_or_else(|| SimulatorError::InvalidInput(format!("Tensor {id1} not found")))?
            .clone();
        let tensor2 = self
            .network
            .get_tensor(id2)
            .ok_or_else(|| SimulatorError::InvalidInput(format!("Tensor {id2} not found")))?
            .clone();
        let common_indices = Self::find_common_indices(&tensor1, &tensor2);
        let cost_estimate = Self::estimate_contraction_cost(&tensor1, &tensor2, &common_indices);
        let result = if cost_estimate > 1e9 && self.config.enable_slicing {
            self.contract_tensors_sliced(&tensor1, &tensor2, &common_indices)?
        } else {
            self.contract_tensors_direct(&tensor1, &tensor2, &common_indices)?
        };
        self.network.remove_tensor(id1);
        self.network.remove_tensor(id2);
        let result_id = self.network.add_tensor(result);
        self.stats.total_contractions += 1;
        self.stats.total_flops += cost_estimate;
        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(result_id)
    }
}
