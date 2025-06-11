//! Marketplace Optimization Configuration Types

use serde::{Deserialize, Serialize};

/// Marketplace optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceOptimizationConfig {
    /// Enable automated optimization
    pub enable_optimization: bool,
    /// Optimization strategies
    pub optimization_strategies: Vec<AlgorithmOptimizationStrategy>,
    /// Hardware-specific optimization
    pub hardware_optimization: HardwareOptimizationConfig,
    /// Multi-objective optimization settings
    pub multi_objective_config: MultiObjectiveOptimizationConfig,
    /// Optimization pipeline configuration
    pub pipeline_config: OptimizationPipelineConfig,
    /// Collaborative optimization settings
    pub collaborative_optimization: CollaborativeOptimizationConfig,
}

/// Hardware optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimizationConfig {
    pub enable_hardware_optimization: bool,
    pub target_hardware: Vec<String>,
}

/// Multi-objective optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveOptimizationConfig {
    pub objectives: Vec<String>,
    pub weights: Vec<f64>,
}

/// Optimization pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPipelineConfig {
    pub pipeline_stages: Vec<String>,
    pub parallel_execution: bool,
}

/// Collaborative optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeOptimizationConfig {
    pub enable_collaboration: bool,
    pub sharing_policies: Vec<String>,
}

/// Algorithm optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlgorithmOptimizationStrategy {
    GeneticAlgorithm,
    SimulatedAnnealing,
    ParticleSwarmOptimization,
    BayesianOptimization,
    GradientBased,
    HybridApproach,
    MachineLearning,
    ReinforcementLearning,
}
