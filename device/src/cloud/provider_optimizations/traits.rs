//! Provider optimizer trait definitions

use super::types::{
    CostEstimate, ExecutionConfig, OptimizationRecommendation, OptimizationStrategy,
    PerformancePrediction, WorkloadSpec,
};
use crate::prelude::CloudProvider;
use crate::DeviceResult;

/// Trait for provider-specific optimization strategies
pub trait ProviderOptimizer {
    /// Optimize a workload for this provider
    fn optimize_workload(
        &self,
        workload: &WorkloadSpec,
    ) -> DeviceResult<OptimizationRecommendation>;

    /// Get the cloud provider this optimizer targets
    fn get_provider(&self) -> CloudProvider;

    /// Get available optimization strategies for this provider
    fn get_optimization_strategies(&self) -> Vec<OptimizationStrategy>;

    /// Predict performance for a given workload and configuration
    fn predict_performance(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<PerformancePrediction>;

    /// Estimate cost for a given workload and configuration
    fn estimate_cost(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<CostEstimate>;
}

// Plug-in marker traits used as `Box<dyn Trait + Send + Sync>` storage in the
// provider-optimizations type hierarchy.  No methods are called through these
// trait objects today — they serve as future extension points so that external
// crates can register custom strategies without forking this crate.  Adding
// required methods here would break every existing impl; prefer default-bodied
// methods if functionality ever needs to be surfaced through the trait.

/// Marker trait for feature-extraction plug-ins.
pub trait FeatureExtractor: Send + Sync {}

/// Marker trait for clustering-engine plug-ins.
pub trait ClusteringEngine: Send + Sync {}

/// Marker trait for similarity-metric plug-ins.
pub trait SimilarityMetric: Send + Sync {}

/// Marker trait for nearest-neighbor-engine plug-ins.
pub trait NearestNeighborEngine: Send + Sync {}

/// Marker trait for pattern-analysis-algorithm plug-ins.
pub trait PatternAnalysisAlgorithm: Send + Sync {}

/// Marker trait for recommendation-algorithm plug-ins.
pub trait RecommendationAlgorithm: Send + Sync {}

/// Marker trait for learning-algorithm plug-ins.
pub trait LearningAlgorithm: Send + Sync {}

/// Marker trait for update-strategy plug-ins.
pub trait UpdateStrategy: Send + Sync {}

/// Marker trait for feedback-validator plug-ins.
pub trait FeedbackValidator: Send + Sync {}

/// Marker trait for feedback-analyzer plug-ins.
pub trait FeedbackAnalyzer: Send + Sync {}

/// Marker trait for feedback-aggregator plug-ins.
pub trait FeedbackAggregator: Send + Sync {}
