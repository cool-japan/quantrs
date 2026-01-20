//! Meta-Learning Optimization Engine for Quantum Annealing Systems
//!
//! This module implements a sophisticated meta-learning optimization engine that learns
//! from historical optimization experiences to automatically improve performance across
//! different problem types and configurations. It employs advanced machine learning
//! techniques including transfer learning, few-shot learning, and neural architecture
//! search to optimize quantum annealing strategies.
//!
//! Key Features:
//! - Experience-based optimization strategy learning
//! - Transfer learning across problem domains
//! - Adaptive hyperparameter optimization
//! - Neural architecture search for annealing schedules
//! - Few-shot learning for new problem types
//! - Multi-objective optimization with Pareto frontiers
//! - Automated feature engineering and selection
//! - Dynamic algorithm portfolio management

pub mod config;
pub mod feature_extraction;
pub mod meta_learning;
pub mod multi_objective;
pub mod neural_architecture_search;
pub mod portfolio_management;
pub mod transfer_learning;

// Re-export main types for public API
pub use config::*;
pub use feature_extraction::{
    AlgorithmType, ArchitectureSpec, ConvergenceMetrics, ExperienceDatabase, FeatureExtractor,
    FeatureVector, OptimizationConfiguration, OptimizationExperience, OptimizationResults,
    ProblemDomain, ProblemFeatures, QualityMetrics, ResourceAllocation, ResourceUsage,
    SuccessMetrics,
};
pub use meta_learning::{
    CrossValidationStrategy, EvaluationMetric, MetaLearner, MetaLearningAlgorithm,
    MetaLearningOptimizer, MetaOptimizationResult, PerformanceEvaluator, StatisticalTest,
    TrainingEpisode,
};
pub use multi_objective::{
    DecisionMaker, FrontierStatistics, FrontierUpdate, MultiObjectiveOptimizer,
    MultiObjectiveSolution, ParetoFrontier, UpdateReason, UserPreferences,
};
pub use neural_architecture_search::{
    ArchitectureCandidate, GenerationMethod, NeuralArchitectureSearch, PerformancePredictor,
    ResourceRequirements, SearchIteration,
};
pub use portfolio_management::{
    Algorithm, AlgorithmPerformanceStats, AlgorithmPortfolio, ApplicabilityConditions,
    GuaranteeType, PerformanceGuarantee, PerformanceRecord, PortfolioComposition,
};
pub use transfer_learning::{
    AdaptationMechanism, DomainCharacteristics, Knowledge, ModelType, SimilarityMethod,
    SimilarityMetric, SourceDomain, TransferLearner, TransferRecord, TransferStrategy,
    TransferableModel,
};

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::applications::{ApplicationError, ApplicationResult};
use crate::ising::{IsingModel, QuboModel};
use crate::simulator::{AnnealingParams, AnnealingResult, QuantumAnnealingSimulator};

/// Recommended optimization strategy
#[derive(Debug, Clone)]
pub struct RecommendedStrategy {
    /// Primary optimization algorithm
    pub algorithm: String,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, f64>,
    /// Confidence score
    pub confidence: f64,
    /// Expected performance
    pub expected_performance: f64,
    /// Alternative strategies
    pub alternatives: Vec<AlternativeStrategy>,
}

/// Alternative strategy option
#[derive(Debug, Clone)]
pub struct AlternativeStrategy {
    /// Algorithm name
    pub algorithm: String,
    /// Relative performance
    pub relative_performance: f64,
}

/// Meta-learning statistics
#[derive(Debug, Clone)]
pub struct MetaLearningStatistics {
    /// Total optimization episodes
    pub total_episodes: usize,
    /// Average improvement over baseline
    pub average_improvement: f64,
    /// Transfer learning success rate
    pub transfer_success_rate: f64,
    /// Feature extraction time
    pub feature_extraction_time: Duration,
    /// Model training time
    pub model_training_time: Duration,
    /// Prediction time
    pub prediction_time: Duration,
}

/// Create example meta-learning optimizer
#[must_use]
pub fn create_meta_learning_optimizer() -> MetaLearningOptimizer {
    let config = MetaLearningConfig::default();
    MetaLearningOptimizer::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_learning_optimizer_creation() {
        let optimizer = create_meta_learning_optimizer();
        // Basic creation test
        assert!(optimizer.config.enable_transfer_learning);
    }

    #[test]
    fn test_recommended_strategy() {
        let strategy = RecommendedStrategy {
            algorithm: "SimulatedAnnealing".to_string(),
            hyperparameters: HashMap::new(),
            confidence: 0.8,
            expected_performance: 0.95,
            alternatives: vec![],
        };

        assert_eq!(strategy.algorithm, "SimulatedAnnealing");
        assert_eq!(strategy.confidence, 0.8);
    }
}
