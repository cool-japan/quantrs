//! # EnhancedTensorNetworkSimulator - extract_network_features_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::NetworkFeatures;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn extract_network_features(&self, tensor_ids: &[usize]) -> Result<NetworkFeatures> {
        let num_tensors = tensor_ids.len();
        let connectivity_density = self.calculate_network_density(tensor_ids);
        let mut max_bond_dimension = 0;
        let mut total_rank = 0;
        for &id in tensor_ids {
            if let Some(tensor) = self.network.get_tensor(id) {
                max_bond_dimension = max_bond_dimension
                    .max(tensor.bond_dimensions.iter().max().copied().unwrap_or(0));
                total_rank += tensor.indices.len();
            }
        }
        let avg_tensor_rank = if num_tensors > 0 {
            total_rank as f64 / num_tensors as f64
        } else {
            0.0
        };
        let circuit_depth_estimate = (num_tensors as f64).log2().ceil() as usize;
        let locality_score = if connectivity_density > 0.5 { 0.8 } else { 0.3 };
        let symmetry_score = if num_tensors % 2 == 0 { 0.6 } else { 0.4 };
        Ok(NetworkFeatures {
            num_tensors,
            connectivity_density,
            max_bond_dimension,
            avg_tensor_rank,
            circuit_depth_estimate,
            locality_score,
            symmetry_score,
        })
    }
}
