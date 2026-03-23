//! # EnhancedTensorNetworkSimulator - queries Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet};

use super::types::EnhancedTensor;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn find_common_indices(
        tensor1: &EnhancedTensor,
        tensor2: &EnhancedTensor,
    ) -> Vec<String> {
        let indices1: HashSet<_> = tensor1.indices.iter().map(|i| &i.label).collect();
        let indices2: HashSet<_> = tensor2.indices.iter().map(|i| &i.label).collect();
        indices1.intersection(&indices2).copied().cloned().collect()
    }
    pub(super) fn find_best_contraction_pair(
        &self,
        tensor_ids: &[usize],
    ) -> Result<(usize, usize, f64)> {
        let mut best_cost = f64::INFINITY;
        let mut best_pair = (0, 1);
        for i in 0..tensor_ids.len() {
            for j in i + 1..tensor_ids.len() {
                if let (Some(tensor1), Some(tensor2)) = (
                    self.network.get_tensor(tensor_ids[i]),
                    self.network.get_tensor(tensor_ids[j]),
                ) {
                    let common_indices = Self::find_common_indices(tensor1, tensor2);
                    let cost = Self::estimate_contraction_cost(tensor1, tensor2, &common_indices);
                    if cost < best_cost {
                        best_cost = cost;
                        best_pair = (i, j);
                    }
                }
            }
        }
        Ok((best_pair.0, best_pair.1, best_cost))
    }
    pub(super) fn calculate_network_density(&self, tensor_ids: &[usize]) -> f64 {
        let num_tensors = tensor_ids.len();
        if num_tensors <= 1 {
            return 0.0;
        }
        let mut total_connections = 0;
        let max_connections = num_tensors * (num_tensors - 1) / 2;
        for i in 0..tensor_ids.len() {
            for j in i + 1..tensor_ids.len() {
                if let (Some(tensor1), Some(tensor2)) = (
                    self.network.get_tensor(tensor_ids[i]),
                    self.network.get_tensor(tensor_ids[j]),
                ) {
                    if !Self::find_common_indices(tensor1, tensor2).is_empty() {
                        total_connections += 1;
                    }
                }
            }
        }
        f64::from(total_connections) / max_connections as f64
    }
}
