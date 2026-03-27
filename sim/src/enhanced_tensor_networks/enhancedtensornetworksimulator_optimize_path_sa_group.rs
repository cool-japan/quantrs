//! # EnhancedTensorNetworkSimulator - optimize_path_sa_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::{EnhancedContractionPath, MLPredictedStrategy, TreeDecomposition};

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn optimize_path_sa(&self, tensor_ids: &[usize]) -> Result<EnhancedContractionPath> {
        let mut current_path = self.optimize_path_greedy(tensor_ids)?;
        let mut best_path = current_path.clone();
        let mut temperature = 1000.0;
        let cooling_rate = 0.95;
        let min_temperature = 1.0;
        while temperature > min_temperature {
            let neighbor_path = self.generate_neighbor_path(&current_path)?;
            let cost_diff = neighbor_path.total_flops - current_path.total_flops;
            if cost_diff < 0.0 || thread_rng().random::<f64>() < (-cost_diff / temperature).exp() {
                current_path = neighbor_path;
                if current_path.total_flops < best_path.total_flops {
                    best_path = current_path.clone();
                }
            }
            temperature *= cooling_rate;
        }
        Ok(best_path)
    }
    pub(super) fn optimize_path_adaptive(
        &self,
        tensor_ids: &[usize],
    ) -> Result<EnhancedContractionPath> {
        let network_density = self.calculate_network_density(tensor_ids);
        let network_size = tensor_ids.len();
        if network_size <= 10 {
            self.optimize_path_dp(tensor_ids)
        } else if network_density > 0.8 {
            self.optimize_path_sa(tensor_ids)
        } else {
            self.optimize_path_greedy(tensor_ids)
        }
    }
    pub(super) fn optimize_path_ml(&self, tensor_ids: &[usize]) -> Result<EnhancedContractionPath> {
        let network_features = self.extract_network_features(tensor_ids)?;
        let predicted_strategy = self.ml_predict_strategy(&network_features)?;
        let primary_path = match predicted_strategy.strategy {
            MLPredictedStrategy::DynamicProgramming => self.optimize_path_dp(tensor_ids)?,
            MLPredictedStrategy::SimulatedAnnealing => self.optimize_path_sa(tensor_ids)?,
            MLPredictedStrategy::TreeDecomposition => self.optimize_path_tree(tensor_ids)?,
            MLPredictedStrategy::Greedy => self.optimize_path_greedy(tensor_ids)?,
        };
        if predicted_strategy.confidence < 0.8 {
            let alternative_strategy = match predicted_strategy.strategy {
                MLPredictedStrategy::DynamicProgramming => MLPredictedStrategy::SimulatedAnnealing,
                MLPredictedStrategy::SimulatedAnnealing => MLPredictedStrategy::Greedy,
                MLPredictedStrategy::TreeDecomposition => MLPredictedStrategy::DynamicProgramming,
                MLPredictedStrategy::Greedy => MLPredictedStrategy::SimulatedAnnealing,
            };
            let alternative_path = match alternative_strategy {
                MLPredictedStrategy::DynamicProgramming => self.optimize_path_dp(tensor_ids)?,
                MLPredictedStrategy::SimulatedAnnealing => self.optimize_path_sa(tensor_ids)?,
                MLPredictedStrategy::TreeDecomposition => self.optimize_path_tree(tensor_ids)?,
                MLPredictedStrategy::Greedy => self.optimize_path_greedy(tensor_ids)?,
            };
            if alternative_path.total_flops < primary_path.total_flops {
                return Ok(alternative_path);
            }
        }
        self.update_ml_model(&network_features, &primary_path)?;
        Ok(primary_path)
    }
    pub(super) fn generate_neighbor_path(
        &self,
        path: &EnhancedContractionPath,
    ) -> Result<EnhancedContractionPath> {
        let mut new_path = path.clone();
        if new_path.steps.len() >= 2 {
            let i = thread_rng().random_range(0..new_path.steps.len());
            let j = thread_rng().random_range(0..new_path.steps.len());
            if i != j {
                new_path.steps.swap(i, j);
                new_path.total_flops = new_path.steps.iter().map(|s| s.flops).sum();
            }
        }
        Ok(new_path)
    }
}
