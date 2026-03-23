//! # EnhancedTensorNetworkSimulator - apply_two_qubit_gate_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array, Array2, ArrayD, IxDyn};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::types::{ContractionStrategy, EnhancedContractionPath, TreeDecomposition};

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    /// Apply two-qubit gate as tensor
    pub fn apply_two_qubit_gate(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        let gate_tensor = Self::create_gate_tensor(gate_matrix, vec![control, target], None)?;
        let gate_id = self.network.add_tensor(gate_tensor);
        let control_label = format!("q{control}");
        let target_label = format!("q{target}");
        let control_tensors = self.network.find_connected_tensors(&control_label);
        let target_tensors = self.network.find_connected_tensors(&target_label);
        let contraction_path =
            self.optimize_contraction_path(&[gate_id], &control_tensors, &target_tensors)?;
        self.execute_contraction_path(&contraction_path)?;
        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    /// Optimize contraction path for multiple tensors
    pub fn optimize_contraction_path(
        &self,
        tensor_ids1: &[usize],
        tensor_ids2: &[usize],
        tensor_ids3: &[usize],
    ) -> Result<EnhancedContractionPath> {
        let start_time = std::time::Instant::now();
        #[cfg(feature = "advanced_math")]
        {
            if let Some(ref optimizer) = self.optimizer {
                return self.optimize_path_scirs2(tensor_ids1, tensor_ids2, tensor_ids3, optimizer);
            }
        }
        let all_ids: Vec<usize> = tensor_ids1
            .iter()
            .chain(tensor_ids2.iter())
            .chain(tensor_ids3.iter())
            .copied()
            .collect();
        let path = match self.config.contraction_strategy {
            ContractionStrategy::Greedy => self.optimize_path_greedy(&all_ids)?,
            ContractionStrategy::DynamicProgramming => self.optimize_path_dp(&all_ids)?,
            ContractionStrategy::SimulatedAnnealing => self.optimize_path_sa(&all_ids)?,
            ContractionStrategy::TreeDecomposition => self.optimize_path_tree(&all_ids)?,
            ContractionStrategy::Adaptive => self.optimize_path_adaptive(&all_ids)?,
            ContractionStrategy::MLGuided => self.optimize_path_ml(&all_ids)?,
        };
        let optimization_time = start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(path)
    }
    /// Execute a contraction path
    pub fn execute_contraction_path(&mut self, path: &EnhancedContractionPath) -> Result<()> {
        let start_time = std::time::Instant::now();
        if self.config.parallel_contractions {
            self.execute_path_parallel(path)?;
        } else {
            self.execute_path_sequential(path)?;
        }
        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
}
