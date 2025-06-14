//! Strategy learning components for active learning decomposition

use std::collections::HashMap;
use std::time::{Duration, Instant};
use ndarray::Array1;

use crate::ising::IsingModel;
use super::*;

/// Decomposition strategy learner
#[derive(Debug, Clone)]
pub struct DecompositionStrategyLearner {
    /// Strategy selection model
    pub selection_model: StrategySelectionModel,
    /// Strategy performance history
    pub performance_history: HashMap<String, Vec<PerformanceRecord>>,
    /// Active learning query selector
    pub query_selector: QuerySelector,
    /// Transfer learning manager
    pub transfer_learning: TransferLearningManager,
    /// Learning statistics
    pub learning_stats: LearningStatistics,
}

impl DecompositionStrategyLearner {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            selection_model: StrategySelectionModel::new(),
            performance_history: HashMap::new(),
            query_selector: QuerySelector::new(),
            transfer_learning: TransferLearningManager::new(),
            learning_stats: LearningStatistics::new(),
        })
    }

    pub fn recommend_strategy(
        &self,
        problem: &IsingModel,
        analysis: &ProblemAnalysis,
    ) -> Result<DecompositionStrategy, String> {
        // Simplified strategy recommendation based on problem size
        if problem.num_qubits < 10 {
            Ok(DecompositionStrategy::NoDecomposition)
        } else if problem.num_qubits < 50 {
            Ok(DecompositionStrategy::GraphPartitioning)
        } else {
            Ok(DecompositionStrategy::CommunityDetection)
        }
    }
}

/// Strategy selection model
#[derive(Debug, Clone)]
pub struct StrategySelectionModel {
    /// Model type
    pub model_type: ModelType,
    /// Feature weights
    pub feature_weights: Array1<f64>,
    /// Strategy preferences
    pub strategy_preferences: HashMap<DecompositionStrategy, f64>,
    /// Uncertainty estimates
    pub uncertainty_estimates: HashMap<String, f64>,
    /// Model parameters
    pub model_parameters: ModelParameters,
}

impl StrategySelectionModel {
    pub fn new() -> Self {
        Self {
            model_type: ModelType::Linear,
            feature_weights: Array1::ones(20),
            strategy_preferences: HashMap::new(),
            uncertainty_estimates: HashMap::new(),
            model_parameters: ModelParameters::default(),
        }
    }

    pub fn get_uncertainty(&self, features: &Array1<f64>) -> Result<f64, String> {
        // Simplified uncertainty calculation
        let feature_sum = features.sum();
        Ok(1.0 / (1.0 + feature_sum.abs()))
    }

    pub fn get_strategy_uncertainty(
        &self,
        strategy: &DecompositionStrategy,
        features: &Array1<f64>,
    ) -> Result<f64, String> {
        let base_uncertainty = self.get_uncertainty(features)?;
        let strategy_key = format!("{:?}", strategy);

        if let Some(&stored_uncertainty) = self.uncertainty_estimates.get(&strategy_key) {
            Ok((base_uncertainty + stored_uncertainty) / 2.0)
        } else {
            Ok(base_uncertainty)
        }
    }
}

/// Model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Model-specific parameters
    pub parameters: HashMap<String, f64>,
    /// Regularization parameters
    pub regularization: RegularizationParameters,
    /// Training configuration
    pub training_config: ModelTrainingConfig,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            regularization: RegularizationParameters {
                l1_weight: 0.01,
                l2_weight: 0.01,
                dropout_rate: 0.1,
                early_stopping_patience: 10,
            },
            training_config: ModelTrainingConfig {
                num_epochs: 100,
                batch_size: 32,
                learning_rate: 0.001,
                validation_split: 0.2,
            },
        }
    }
}

/// Regularization parameters
#[derive(Debug, Clone)]
pub struct RegularizationParameters {
    /// L1 regularization weight
    pub l1_weight: f64,
    /// L2 regularization weight
    pub l2_weight: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

/// Model training configuration
#[derive(Debug, Clone)]
pub struct ModelTrainingConfig {
    /// Number of training epochs
    pub num_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Validation split
    pub validation_split: f64,
}

/// Query selector for active learning
#[derive(Debug, Clone)]
pub struct QuerySelector {
    /// Query strategy
    pub query_strategy: QueryStrategy,
    /// Uncertainty threshold
    pub uncertainty_threshold: f64,
    /// Diversity constraint
    pub diversity_constraint: DiversityConstraint,
    /// Query history
    pub query_history: Vec<QueryRecord>,
}

impl QuerySelector {
    pub fn new() -> Self {
        Self {
            query_strategy: QueryStrategy::UncertaintySampling,
            uncertainty_threshold: 0.5,
            diversity_constraint: DiversityConstraint {
                min_distance: 0.1,
                diversity_metric: DiversityMetric::Euclidean,
                max_similarity: 0.8,
            },
            query_history: Vec::new(),
        }
    }
}

/// Diversity constraint for query selection
#[derive(Debug, Clone)]
pub struct DiversityConstraint {
    /// Minimum distance between queries
    pub min_distance: f64,
    /// Diversity metric
    pub diversity_metric: DiversityMetric,
    /// Maximum similarity allowed
    pub max_similarity: f64,
}

/// Query record
#[derive(Debug, Clone)]
pub struct QueryRecord {
    /// Query timestamp
    pub timestamp: Instant,
    /// Queried problem features
    pub problem_features: Array1<f64>,
    /// Recommended strategy
    pub recommended_strategy: DecompositionStrategy,
    /// Query outcome
    pub query_outcome: QueryOutcome,
    /// Performance feedback
    pub performance_feedback: Option<PerformanceRecord>,
}

/// Query outcome
#[derive(Debug, Clone)]
pub struct QueryOutcome {
    /// Strategy actually used
    pub strategy_used: DecompositionStrategy,
    /// User accepted recommendation
    pub accepted_recommendation: bool,
    /// Performance achieved
    pub performance_achieved: f64,
    /// Feedback quality
    pub feedback_quality: f64,
}

/// Transfer learning manager
#[derive(Debug, Clone)]
pub struct TransferLearningManager {
    /// Source domain models
    pub source_models: Vec<SourceDomainModel>,
    /// Domain adaptation strategy
    pub adaptation_strategy: DomainAdaptationStrategy,
    /// Knowledge transfer weights
    pub transfer_weights: Array1<f64>,
    /// Transfer learning statistics
    pub transfer_stats: TransferStatistics,
}

impl TransferLearningManager {
    pub fn new() -> Self {
        Self {
            source_models: Vec::new(),
            adaptation_strategy: DomainAdaptationStrategy::FineTuning,
            transfer_weights: Array1::ones(5),
            transfer_stats: TransferStatistics {
                successful_transfers: 0,
                failed_transfers: 0,
                avg_transfer_benefit: 0.0,
                transfer_time_overhead: Duration::from_secs(0),
            },
        }
    }
}

/// Source domain model
#[derive(Debug, Clone)]
pub struct SourceDomainModel {
    /// Domain identifier
    pub domain_id: String,
    /// Model for this domain
    pub model: StrategySelectionModel,
    /// Domain characteristics
    pub domain_characteristics: DomainCharacteristics,
    /// Transfer applicability score
    pub applicability_score: f64,
}

/// Domain characteristics
#[derive(Debug, Clone)]
pub struct DomainCharacteristics {
    /// Problem types in domain
    pub problem_types: Vec<String>,
    /// Average problem size
    pub avg_problem_size: f64,
    /// Problem complexity distribution
    pub complexity_distribution: Array1<f64>,
    /// Common structures
    pub common_structures: Vec<StructureType>,
}

/// Transfer learning statistics
#[derive(Debug, Clone)]
pub struct TransferStatistics {
    /// Successful transfers
    pub successful_transfers: usize,
    /// Failed transfers
    pub failed_transfers: usize,
    /// Average transfer benefit
    pub avg_transfer_benefit: f64,
    /// Transfer time overhead
    pub transfer_time_overhead: Duration,
}

/// Learning statistics
#[derive(Debug, Clone)]
pub struct LearningStatistics {
    /// Total queries made
    pub total_queries: usize,
    /// Successful predictions
    pub successful_predictions: usize,
    /// Average prediction accuracy
    pub avg_prediction_accuracy: f64,
    /// Learning curve data
    pub learning_curve: Vec<(usize, f64)>, // (query_count, accuracy)
    /// Exploration vs exploitation ratio
    pub exploration_exploitation_ratio: f64,
}

impl LearningStatistics {
    pub fn new() -> Self {
        Self {
            total_queries: 0,
            successful_predictions: 0,
            avg_prediction_accuracy: 0.0,
            learning_curve: Vec::new(),
            exploration_exploitation_ratio: 0.5,
        }
    }
}

/// Performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Timestamp
    pub timestamp: Instant,
    /// Problem identifier
    pub problem_id: String,
    /// Strategy used
    pub strategy_used: DecompositionStrategy,
    /// Performance metrics
    pub metrics: HashMap<EvaluationMetric, f64>,
    /// Overall performance score
    pub overall_score: f64,
}