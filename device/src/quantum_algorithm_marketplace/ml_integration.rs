//! ML Integration Configuration Types

use serde::{Deserialize, Serialize};

/// Marketplace ML configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceMLConfig {
    /// Enable ML-driven features
    pub enable_ml: bool,
    /// Algorithm recommendation models
    pub recommendation_models: Vec<MLRecommendationModel>,
    /// Performance prediction models
    pub performance_models: Vec<PerformancePredictionModel>,
    /// Optimization models
    pub optimization_models: Vec<MLOptimizationModel>,
    /// Automated analysis models
    pub analysis_models: Vec<AnalysisModel>,
    /// Model training configuration
    pub training_config: MLTrainingConfig,
}

/// ML training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLTrainingConfig {
    pub training_enabled: bool,
    pub training_datasets: Vec<String>,
}

/// ML recommendation models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MLRecommendationModel {
    GraphNeuralNetworks,
    TransformerModels,
    EnsembleMethods,
    CollaborativeFiltering,
    ContentBasedFiltering,
    HybridModels,
}

/// Performance prediction models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PerformancePredictionModel {
    NeuralNetworks,
    EnsemblePredictors,
    PhysicsInformedModels,
    TimeSeriesModels,
    RegressiveModels,
}

/// ML optimization models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MLOptimizationModel {
    BayesianOptimization,
    GeneticAlgorithms,
    ReinforcementLearning,
    NeuralNetworks,
    RandomForest,
    SupportVectorMachine,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
}

/// Analysis models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnalysisModel {
    SemanticAnalysis,
    PerformanceAnalysis,
    ComplexityAnalysis,
    SecurityAnalysis,
    UsagePatternAnalysis,
}
