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

// Stub traits for internal use (TODO: Implement properly)

/// Feature extraction trait
pub trait FeatureExtractor: Send + Sync {}

/// Clustering engine trait
pub trait ClusteringEngine: Send + Sync {}

/// Similarity metric trait
pub trait SimilarityMetric: Send + Sync {}

/// Nearest neighbor engine trait
pub trait NearestNeighborEngine: Send + Sync {}

/// Pattern analysis algorithm trait
pub trait PatternAnalysisAlgorithm: Send + Sync {}

/// Recommendation algorithm trait
pub trait RecommendationAlgorithm: Send + Sync {}

/// Learning algorithm trait
pub trait LearningAlgorithm: Send + Sync {}

/// Update strategy trait
pub trait UpdateStrategy: Send + Sync {}

/// Feedback validator trait
pub trait FeedbackValidator: Send + Sync {}

/// Feedback analyzer trait
pub trait FeedbackAnalyzer: Send + Sync {}

/// Feedback aggregator trait
pub trait FeedbackAggregator: Send + Sync {}
