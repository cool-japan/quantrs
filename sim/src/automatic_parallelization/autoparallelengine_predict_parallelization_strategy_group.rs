//! # AutoParallelEngine - predict_parallelization_strategy_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{MLFeatures, MLPredictedStrategy};

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Predict optimal parallelization strategy based on ML features
    pub(super) fn predict_parallelization_strategy(features: &MLFeatures) -> MLPredictedStrategy {
        if features.parallelism_factor > 0.7 && features.avg_connectivity < 2.0 {
            return MLPredictedStrategy::HighParallelism;
        }
        if features.circuit_depth < (features.num_gates as f64 * 0.3) as usize {
            return MLPredictedStrategy::LayerOptimized;
        }
        if features.avg_connectivity > 3.5 || features.dependency_density > 0.6 {
            return MLPredictedStrategy::ConservativeParallelism;
        }
        MLPredictedStrategy::BalancedParallelism
    }
}
