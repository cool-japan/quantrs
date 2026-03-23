//! # EnhancedTensorNetworkSimulator - ml_predict_strategy_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::{MLPredictedStrategy, MLPrediction, NetworkFeatures, TreeDecomposition};

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) fn ml_predict_strategy(&self, features: &NetworkFeatures) -> Result<MLPrediction> {
        let (strategy, confidence) = if features.num_tensors <= 10 {
            (MLPredictedStrategy::DynamicProgramming, 0.9)
        } else if features.connectivity_density < 0.3 {
            (MLPredictedStrategy::TreeDecomposition, 0.8)
        } else if features.max_bond_dimension > 64 {
            (MLPredictedStrategy::SimulatedAnnealing, 0.7)
        } else {
            (MLPredictedStrategy::Greedy, 0.6)
        };
        let expected_performance = match strategy {
            MLPredictedStrategy::DynamicProgramming => 0.95,
            MLPredictedStrategy::TreeDecomposition => 0.85,
            MLPredictedStrategy::SimulatedAnnealing => 0.75,
            MLPredictedStrategy::Greedy => 0.6,
        };
        Ok(MLPrediction {
            strategy,
            confidence,
            expected_performance,
        })
    }
}
