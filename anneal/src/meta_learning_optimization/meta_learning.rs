//! Core meta-learning optimization engine

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::applications::{ApplicationError, ApplicationResult};
use crate::ising::{IsingModel, QuboModel};
use crate::simulator::{AnnealingParams, AnnealingResult, QuantumAnnealingSimulator};

use super::config::MetaLearningConfig;
use super::feature_extraction::{
    AlgorithmType, DistributionStats, ExperienceDatabase, FeatureExtractor,
    OptimizationConfiguration, OptimizationExperience, ProblemDomain, ProblemFeatures,
    ResourceAllocation,
};
use super::multi_objective::MultiObjectiveOptimizer;
use super::neural_architecture_search::NeuralArchitectureSearch;
use super::portfolio_management::{AlgorithmPortfolio, PerformanceRecord};
use super::transfer_learning::{
    DomainCharacteristics, SourceDomain, TransferLearner, TransferableModel,
};
use super::{AlternativeStrategy, MetaLearningStatistics, RecommendedStrategy};

/// Main meta-learning optimization engine
pub struct MetaLearningOptimizer {
    /// Configuration
    pub config: MetaLearningConfig,
    /// Experience database
    pub experience_db: Arc<RwLock<ExperienceDatabase>>,
    /// Feature extractor
    pub feature_extractor: Arc<Mutex<FeatureExtractor>>,
    /// Meta-learner
    pub meta_learner: Arc<Mutex<MetaLearner>>,
    /// Neural architecture search engine
    pub nas_engine: Arc<Mutex<NeuralArchitectureSearch>>,
    /// Algorithm portfolio manager
    pub portfolio_manager: Arc<Mutex<AlgorithmPortfolio>>,
    /// Multi-objective optimizer
    pub multi_objective_optimizer: Arc<Mutex<MultiObjectiveOptimizer>>,
    /// Transfer learning system
    pub transfer_learner: Arc<Mutex<TransferLearner>>,
}

impl MetaLearningOptimizer {
    #[must_use]
    pub fn new(config: MetaLearningConfig) -> Self {
        Self {
            experience_db: Arc::new(RwLock::new(ExperienceDatabase::new())),
            feature_extractor: Arc::new(Mutex::new(FeatureExtractor::new(
                config.feature_config.clone(),
            ))),
            meta_learner: Arc::new(Mutex::new(MetaLearner::new())),
            nas_engine: Arc::new(Mutex::new(NeuralArchitectureSearch::new(
                config.nas_config.clone(),
            ))),
            portfolio_manager: Arc::new(Mutex::new(AlgorithmPortfolio::new(
                config.portfolio_config.clone(),
            ))),
            multi_objective_optimizer: Arc::new(Mutex::new(MultiObjectiveOptimizer::new(
                config.multi_objective_config.clone(),
            ))),
            transfer_learner: Arc::new(Mutex::new(TransferLearner::new())),
            config,
        }
    }

    /// Recommend optimization strategy for a given problem
    pub fn recommend_strategy(
        &mut self,
        problem: &IsingModel,
    ) -> ApplicationResult<RecommendedStrategy> {
        // Extract problem features
        let features = {
            let mut extractor = self.feature_extractor.lock().map_err(|_| {
                ApplicationError::ConfigurationError("Failed to lock feature extractor".to_string())
            })?;
            extractor.extract_features(problem)?
        };

        // Find similar experiences
        let similar_experiences = {
            let db = self.experience_db.read().map_err(|_| {
                ApplicationError::ConfigurationError(
                    "Failed to lock experience database".to_string(),
                )
            })?;
            db.find_similar_experiences(&features, 10)?
        };

        // Apply transfer learning if enabled
        let transferred_knowledge = if self.config.enable_transfer_learning {
            let mut transfer_learner = self.transfer_learner.lock().map_err(|_| {
                ApplicationError::ConfigurationError("Failed to lock transfer learner".to_string())
            })?;
            let domain = self.infer_problem_domain(&features);
            transfer_learner
                .transfer_knowledge(&features, &domain)
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Get meta-learning recommendation
        let meta_recommendation = {
            let mut meta_learner = self.meta_learner.lock().map_err(|_| {
                ApplicationError::ConfigurationError("Failed to lock meta learner".to_string())
            })?;
            meta_learner.recommend_strategy(&features, &similar_experiences)?
        };

        // Select algorithm from portfolio
        let algorithm_id = {
            let mut portfolio = self.portfolio_manager.lock().map_err(|_| {
                ApplicationError::ConfigurationError("Failed to lock portfolio manager".to_string())
            })?;
            portfolio
                .select_algorithm(&features)
                .map_err(|e| ApplicationError::ConfigurationError(e))?
        };

        // Get algorithm configuration
        let algorithm_config = {
            let portfolio = self.portfolio_manager.lock().map_err(|_| {
                ApplicationError::ConfigurationError("Failed to lock portfolio manager".to_string())
            })?;
            portfolio
                .algorithms
                .get(&algorithm_id)
                .map(|alg| alg.default_config.clone())
                .ok_or_else(|| {
                    ApplicationError::ConfigurationError(
                        "Algorithm not found in portfolio".to_string(),
                    )
                })?
        };

        // Apply neural architecture search if needed
        let optimized_config =
            if self.config.nas_config.enable_nas && algorithm_config.architecture.is_some() {
                let mut nas = self.nas_engine.lock().map_err(|_| {
                    ApplicationError::ConfigurationError("Failed to lock NAS engine".to_string())
                })?;
                if let Ok(architecture_candidate) = nas.search_architecture(&features) {
                    let mut config = algorithm_config;
                    config.architecture = Some(architecture_candidate.architecture);
                    config
                } else {
                    algorithm_config
                }
            } else {
                algorithm_config
            };

        // Apply multi-objective optimization if enabled
        let final_config = if self.config.multi_objective_config.enable_multi_objective {
            let mut mo_optimizer = self.multi_objective_optimizer.lock().map_err(|_| {
                ApplicationError::ConfigurationError(
                    "Failed to lock multi-objective optimizer".to_string(),
                )
            })?;
            let candidates = vec![optimized_config.clone()];
            if let Ok(solutions) = mo_optimizer.optimize(candidates) {
                if let Ok(best_solution) = mo_optimizer.make_decision(None) {
                    best_solution.decision_variables
                } else {
                    optimized_config
                }
            } else {
                optimized_config
            }
        } else {
            optimized_config
        };

        // Combine recommendations and create final strategy
        let mut final_hyperparameters = meta_recommendation.hyperparameters.clone();

        // Merge with transferred knowledge
        for model in &transferred_knowledge {
            for (param_name, param_value) in &model.parameters {
                // Use transferred parameter if confidence is high
                if model.confidence > 0.7 {
                    final_hyperparameters.insert(param_name.clone(), *param_value);
                }
            }
        }

        // Merge with final configuration
        for (param_name, param_value) in &final_config.hyperparameters {
            final_hyperparameters.insert(param_name.clone(), *param_value);
        }

        let recommended_strategy = RecommendedStrategy {
            algorithm: algorithm_id,
            hyperparameters: final_hyperparameters,
            confidence: self
                .calculate_recommendation_confidence(&similar_experiences, &transferred_knowledge),
            expected_performance: meta_recommendation.expected_performance,
            alternatives: self.generate_alternative_strategies(&features)?,
        };

        Ok(recommended_strategy)
    }

    /// Record optimization experience for learning
    pub fn record_experience(
        &mut self,
        experience: OptimizationExperience,
    ) -> ApplicationResult<()> {
        // Add to experience database
        {
            let mut db = self.experience_db.write().map_err(|_| {
                ApplicationError::ConfigurationError(
                    "Failed to lock experience database".to_string(),
                )
            })?;
            db.add_experience(experience.clone());
        }

        // Update meta-learner
        {
            let mut meta_learner = self.meta_learner.lock().map_err(|_| {
                ApplicationError::ConfigurationError("Failed to lock meta learner".to_string())
            })?;
            meta_learner.add_training_episode(experience.clone())?;
        }

        // Update portfolio performance
        if let Some(algorithm_name) =
            self.algorithm_type_to_name(&experience.configuration.algorithm)
        {
            let mut portfolio = self.portfolio_manager.lock().map_err(|_| {
                ApplicationError::ConfigurationError("Failed to lock portfolio manager".to_string())
            })?;

            let performance_record = PerformanceRecord {
                timestamp: experience.timestamp,
                problem_features: experience.problem_features.clone(),
                performance: experience.results.quality_metrics.objective_value,
                resource_usage: experience.results.resource_usage.clone(),
                context: HashMap::new(),
            };

            portfolio.record_performance(&algorithm_name, performance_record);
            portfolio.update_composition();
        }

        // Update transfer learning
        if self.config.enable_transfer_learning {
            let mut transfer_learner = self.transfer_learner.lock().map_err(|_| {
                ApplicationError::ConfigurationError("Failed to lock transfer learner".to_string())
            })?;
            let domain_characteristics = self.create_domain_characteristics(&experience);
            let source_domain = SourceDomain {
                id: format!("domain_{:?}", experience.domain),
                characteristics: domain_characteristics,
                experiences: vec![experience],
                models: Vec::new(),
                last_updated: Instant::now(),
            };
            transfer_learner.add_source_domain(source_domain);
        }

        Ok(())
    }

    /// Get meta-learning statistics
    pub fn get_statistics(&self) -> ApplicationResult<MetaLearningStatistics> {
        let db_stats = {
            let db = self.experience_db.read().map_err(|_| {
                ApplicationError::ConfigurationError(
                    "Failed to lock experience database".to_string(),
                )
            })?;
            db.statistics.clone()
        };

        let transfer_success_rate = if self.config.enable_transfer_learning {
            let transfer_learner = self.transfer_learner.lock().map_err(|_| {
                ApplicationError::ConfigurationError("Failed to lock transfer learner".to_string())
            })?;
            transfer_learner.evaluate_transfer_success()
        } else {
            0.0
        };

        Ok(MetaLearningStatistics {
            total_episodes: db_stats.total_experiences,
            average_improvement: db_stats.avg_performance,
            transfer_success_rate,
            feature_extraction_time: Duration::from_millis(10), // Simplified
            model_training_time: Duration::from_millis(100),
            prediction_time: Duration::from_millis(5),
        })
    }

    const fn infer_problem_domain(&self, features: &ProblemFeatures) -> ProblemDomain {
        // Simple domain inference based on problem characteristics
        if features.graph_features.num_edges > 0 {
            ProblemDomain::Graph
        } else if features.size > 1000 {
            ProblemDomain::Combinatorial
        } else {
            ProblemDomain::Combinatorial
        }
    }

    fn calculate_recommendation_confidence(
        &self,
        similar_experiences: &[OptimizationExperience],
        transferred_knowledge: &[TransferableModel],
    ) -> f64 {
        let experience_confidence = if similar_experiences.is_empty() {
            0.3
        } else {
            0.3f64.mul_add((similar_experiences.len() as f64 / 10.0).min(1.0), 0.7)
        };

        let transfer_confidence = if transferred_knowledge.is_empty() {
            0.0
        } else {
            transferred_knowledge
                .iter()
                .map(|m| m.confidence)
                .sum::<f64>()
                / transferred_knowledge.len() as f64
        };

        (experience_confidence + transfer_confidence * 0.3).min(1.0)
    }

    fn generate_alternative_strategies(
        &self,
        features: &ProblemFeatures,
    ) -> ApplicationResult<Vec<AlternativeStrategy>> {
        let mut alternatives = Vec::new();

        // Generate alternatives based on problem size
        if features.size < 100 {
            alternatives.push(AlternativeStrategy {
                algorithm: "simulated_annealing".to_string(),
                relative_performance: 0.9,
            });
        } else if features.size < 500 {
            alternatives.push(AlternativeStrategy {
                algorithm: "quantum_annealing".to_string(),
                relative_performance: 0.95,
            });
        } else {
            alternatives.push(AlternativeStrategy {
                algorithm: "tabu_search".to_string(),
                relative_performance: 0.85,
            });
        }

        Ok(alternatives)
    }

    fn algorithm_type_to_name(&self, algorithm_type: &AlgorithmType) -> Option<String> {
        match algorithm_type {
            AlgorithmType::SimulatedAnnealing => Some("simulated_annealing".to_string()),
            AlgorithmType::QuantumAnnealing => Some("quantum_annealing".to_string()),
            AlgorithmType::TabuSearch => Some("tabu_search".to_string()),
            AlgorithmType::GeneticAlgorithm => Some("genetic_algorithm".to_string()),
            _ => None,
        }
    }

    fn create_domain_characteristics(
        &self,
        experience: &OptimizationExperience,
    ) -> DomainCharacteristics {
        DomainCharacteristics {
            domain: experience.domain.clone(),
            avg_problem_size: experience.problem_features.size as f64,
            avg_density: experience.problem_features.density,
            typical_algorithms: vec![experience.configuration.algorithm.clone()],
            performance_distribution: DistributionStats {
                mean: experience.results.quality_metrics.objective_value,
                std_dev: 0.1,
                min: experience.results.quality_metrics.objective_value * 0.8,
                max: experience.results.quality_metrics.objective_value * 1.2,
                skewness: 0.0,
                kurtosis: 3.0,
            },
            feature_importance: HashMap::new(),
        }
    }
}

/// Meta-learning system
pub struct MetaLearner {
    /// Learning algorithm
    pub algorithm: MetaLearningAlgorithm,
    /// Model parameters
    pub parameters: Vec<f64>,
    /// Training history
    pub training_history: VecDeque<TrainingEpisode>,
    /// Performance evaluator
    pub evaluator: PerformanceEvaluator,
}

impl MetaLearner {
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithm: MetaLearningAlgorithm::MAML,
            parameters: Vec::new(),
            training_history: VecDeque::new(),
            evaluator: PerformanceEvaluator {
                metrics: vec![
                    EvaluationMetric::MeanSquaredError,
                    EvaluationMetric::Accuracy,
                ],
                cv_strategy: CrossValidationStrategy::KFold(5),
                statistical_tests: vec![StatisticalTest::TTest],
            },
        }
    }

    pub fn recommend_strategy(
        &mut self,
        features: &ProblemFeatures,
        experiences: &[OptimizationExperience],
    ) -> ApplicationResult<RecommendedStrategy> {
        // Simple strategy recommendation based on problem size
        let algorithm = if features.size < 100 {
            AlgorithmType::SimulatedAnnealing
        } else if features.size < 500 {
            AlgorithmType::QuantumAnnealing
        } else {
            AlgorithmType::TabuSearch
        };

        let mut hyperparameters = HashMap::new();

        // Set hyperparameters based on experiences
        if experiences.is_empty() {
            // Default hyperparameters
            hyperparameters.insert("initial_temperature".to_string(), 10.0);
            hyperparameters.insert("final_temperature".to_string(), 0.1);
        } else {
            let avg_initial_temp = experiences
                .iter()
                .filter_map(|exp| exp.configuration.hyperparameters.get("initial_temperature"))
                .sum::<f64>()
                / experiences.len() as f64;
            hyperparameters.insert("initial_temperature".to_string(), avg_initial_temp.max(1.0));

            let avg_final_temp = experiences
                .iter()
                .filter_map(|exp| exp.configuration.hyperparameters.get("final_temperature"))
                .sum::<f64>()
                / experiences.len() as f64;
            hyperparameters.insert("final_temperature".to_string(), avg_final_temp.max(0.01));
        }

        hyperparameters.insert(
            "num_sweeps".to_string(),
            (features.size as f64 * 10.0).min(10_000.0),
        );

        let configuration = OptimizationConfiguration {
            algorithm,
            hyperparameters,
            architecture: None,
            resources: ResourceAllocation {
                cpu: 1.0,
                memory: 512,
                gpu: 0.0,
                time: Duration::from_secs(60),
            },
        };

        let confidence = if experiences.len() >= 5 { 0.9 } else { 0.6 };

        Ok(RecommendedStrategy {
            algorithm: "meta_learner_recommendation".to_string(),
            hyperparameters: configuration.hyperparameters,
            confidence,
            expected_performance: 0.8,
            alternatives: Vec::new(),
        })
    }

    pub fn add_training_episode(
        &mut self,
        experience: OptimizationExperience,
    ) -> ApplicationResult<()> {
        let episode = TrainingEpisode {
            id: experience.id.clone(),
            support_set: vec![experience.clone()],
            query_set: vec![experience.clone()],
            loss: 1.0 - experience.results.quality_metrics.objective_value,
            accuracy: experience.results.quality_metrics.objective_value,
            timestamp: experience.timestamp,
        };

        self.training_history.push_back(episode);

        // Limit history size
        while self.training_history.len() > 1000 {
            self.training_history.pop_front();
        }

        Ok(())
    }
}

/// Meta-learning algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// Prototypical Networks
    PrototypicalNetworks,
    /// Matching Networks
    MatchingNetworks,
    /// Relation Networks
    RelationNetworks,
    /// Memory-Augmented Networks
    MemoryAugmented,
    /// Gradient-Based Meta-Learning
    GradientBased,
}

/// Training episode
#[derive(Debug, Clone)]
pub struct TrainingEpisode {
    /// Episode identifier
    pub id: String,
    /// Support set
    pub support_set: Vec<OptimizationExperience>,
    /// Query set
    pub query_set: Vec<OptimizationExperience>,
    /// Loss achieved
    pub loss: f64,
    /// Accuracy achieved
    pub accuracy: f64,
    /// Timestamp
    pub timestamp: Instant,
}

/// Performance evaluator
#[derive(Debug)]
pub struct PerformanceEvaluator {
    /// Evaluation metrics
    pub metrics: Vec<EvaluationMetric>,
    /// Cross-validation strategy
    pub cv_strategy: CrossValidationStrategy,
    /// Statistical tests
    pub statistical_tests: Vec<StatisticalTest>,
}

/// Evaluation metrics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationMetric {
    /// Mean squared error
    MeanSquaredError,
    /// Mean absolute error
    MeanAbsoluteError,
    /// R-squared
    RSquared,
    /// Accuracy
    Accuracy,
    /// Precision
    Precision,
    /// Recall
    Recall,
    /// F1 score
    F1Score,
    /// Custom metric
    Custom(String),
}

/// Cross-validation strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossValidationStrategy {
    /// K-fold cross-validation
    KFold(usize),
    /// Leave-one-out
    LeaveOneOut,
    /// Time series split
    TimeSeriesSplit,
    /// Stratified K-fold
    StratifiedKFold(usize),
    /// Custom strategy
    Custom(String),
}

/// Statistical tests
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatisticalTest {
    /// t-test
    TTest,
    /// Wilcoxon signed-rank test
    WilcoxonSignedRank,
    /// Mann-Whitney U test
    MannWhitneyU,
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Chi-square test
    ChiSquare,
}

/// Meta-optimization result
#[derive(Debug, Clone)]
pub struct MetaOptimizationResult {
    /// Recommended strategy
    pub strategy: RecommendedStrategy,
    /// Alternative strategies
    pub alternatives: Vec<AlternativeStrategy>,
    /// Confidence in recommendation
    pub confidence: f64,
    /// Expected performance gain
    pub expected_gain: f64,
    /// Reasoning for recommendation
    pub reasoning: String,
    /// Meta-learning statistics
    pub statistics: MetaLearningStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_learning_optimizer_creation() {
        let config = MetaLearningConfig::default();
        let optimizer = MetaLearningOptimizer::new(config);
        assert!(optimizer.config.enable_transfer_learning);
    }

    #[test]
    fn test_meta_learner() {
        let meta_learner = MetaLearner::new();
        assert_eq!(meta_learner.algorithm, MetaLearningAlgorithm::MAML);
        assert_eq!(meta_learner.training_history.len(), 0);
    }

    #[test]
    fn test_training_episode() {
        let episode = TrainingEpisode {
            id: "test_episode".to_string(),
            support_set: Vec::new(),
            query_set: Vec::new(),
            loss: 0.5,
            accuracy: 0.8,
            timestamp: Instant::now(),
        };

        assert_eq!(episode.id, "test_episode");
        assert_eq!(episode.loss, 0.5);
        assert_eq!(episode.accuracy, 0.8);
    }
}
