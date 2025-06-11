//! Algorithm Discovery Configuration Types

use serde::{Deserialize, Serialize};

/// Algorithm discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmDiscoveryConfig {
    /// Enable intelligent discovery
    pub enable_discovery: bool,
    /// Recommendation engine settings
    pub recommendation_engine: RecommendationEngineConfig,
    /// Similarity analysis configuration
    pub similarity_analysis: SimilarityAnalysisConfig,
    /// Trend analysis and prediction
    pub trend_analysis: TrendAnalysisConfig,
    /// Personalization settings
    pub personalization_config: PersonalizationConfig,
    /// Discovery algorithms
    pub discovery_algorithms: Vec<DiscoveryAlgorithm>,
}

/// Recommendation engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationEngineConfig {
    pub enable_recommendations: bool,
    pub algorithm_type: String,
}

/// Similarity analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityAnalysisConfig {
    pub similarity_threshold: f64,
    pub analysis_methods: Vec<String>,
}

/// Trend analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisConfig {
    pub enable_trend_analysis: bool,
    pub analysis_window: u64,
}

/// Personalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationConfig {
    pub enable_personalization: bool,
    pub user_preferences: Vec<String>,
}

/// Discovery algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DiscoveryAlgorithm {
    HybridRecommendation,
    GraphBasedDiscovery,
    MachineLearningDiscovery,
    ContentBasedFiltering,
    CollaborativeFiltering,
    SemanticSearch,
    PerformanceBased,
}
