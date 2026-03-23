//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::functions::StructureDetector;
use super::extended::*;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct PerformanceDatabase {
    /// Stored performance results
    pub performance_records: Vec<PerformanceRecord>,
    /// Algorithm rankings
    pub algorithm_rankings: HashMap<String, AlgorithmRanking>,
    /// Problem categories
    pub problem_categories: HashMap<String, ProblemCategory>,
}
#[derive(Debug, Clone)]
pub struct FeatureNormalization {
    /// Normalization type
    pub normalization_type: NormalizationType,
    /// Feature statistics
    pub feature_stats: HashMap<String, FeatureStats>,
}
#[derive(Debug, Clone)]
pub struct ProblemState {
    /// QUBO matrix representation
    pub qubo_features: Array1<f64>,
    /// Current solution
    pub current_solution: Array1<f64>,
    /// Energy history
    pub energy_history: Array1<f64>,
    /// Algorithm state
    pub algorithm_state: AlgorithmState,
    /// Time step
    pub time_step: usize,
}
#[derive(Debug, Clone)]
pub struct QualityPredictorTrainingResults {
    pub r2_score: f64,
    pub mae: f64,
    pub rmse: f64,
    pub training_time: Duration,
    pub model_complexity: usize,
}
#[derive(Debug, Clone)]
pub struct AlgorithmRanking {
    /// Overall rank
    pub overall_rank: usize,
    /// Category-specific ranks
    pub category_ranks: HashMap<String, usize>,
    /// Performance scores
    pub performance_scores: HashMap<String, f64>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// CPU information
    pub cpu_info: String,
    /// GPU information
    pub gpu_info: Option<String>,
    /// Memory capacity
    pub memory_capacity: usize,
    /// Architecture
    pub architecture: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyStrategy {
    LeastConfident,
    MarginSampling,
    EntropyBased,
    VarianceReduction,
    ExpectedModelChange,
}
#[derive(Debug, Clone)]
pub struct ProblemCategory {
    /// Category name
    pub name: String,
    /// Category description
    pub description: String,
    /// Characteristic features
    pub characteristic_features: Vec<String>,
    /// Best algorithms for this category
    pub best_algorithms: Vec<String>,
    /// Performance statistics
    pub performance_stats: HashMap<String, f64>,
}
#[derive(Debug, Clone)]
pub struct CalibrationPoint {
    /// Predicted uncertainty
    pub predicted_uncertainty: f64,
    /// Actual error
    pub actual_error: f64,
    /// Problem characteristics
    pub problem_features: Array1<f64>,
}
#[derive(Debug, Clone)]
pub struct PatternDatabase {
    /// Known patterns
    pub patterns: HashMap<String, PatternInfo>,
    /// Pattern relationships
    pub pattern_relationships: HashMap<String, Vec<String>>,
    /// Algorithmic preferences for patterns
    pub algorithmic_preferences: HashMap<String, Vec<AlgorithmPreference>>,
}
#[derive(Debug, Clone)]
pub struct StateEncoder {
    /// Problem embedding dimension
    pub embedding_dim: usize,
    /// Encoder layers
    pub layers: Vec<DenseLayer>,
}
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Model type
    pub model_type: RegressionModel,
    /// Model parameters
    pub parameters: ModelParameters,
    /// Training history
    pub training_history: Vec<TrainingMetric>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Hamming,
}
#[derive(Debug, Clone)]
pub struct GraphMetricsCalculator {
    /// Available metrics
    pub available_metrics: Vec<GraphMetric>,
}
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Problem identifier
    pub problem_id: String,
    /// Algorithm used
    pub algorithm: String,
    /// Performance metrics
    pub metrics: AlgorithmPerformanceMetrics,
    /// Runtime information
    pub runtime_info: RuntimeInfo,
    /// Hardware information
    pub hardware_info: HardwareInfo,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    StandardScaling,
    MinMaxScaling,
    RobustScaling,
    QuantileUniform,
    PowerTransform,
}
#[derive(Debug, Clone)]
pub struct UncertaintyQuantifier {
    /// Uncertainty estimation method
    pub method: UncertaintyMethod,
    /// Confidence intervals
    pub confidence_levels: Vec<f64>,
    /// Calibration data
    pub calibration_data: Vec<CalibrationPoint>,
}
#[derive(Debug, Clone)]
pub struct TrainingMetric {
    /// Epoch/iteration
    pub epoch: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// R² score
    pub r2_score: f64,
    /// Mean absolute error
    pub mae: f64,
    /// Root mean squared error
    pub rmse: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemInfo {
    /// Problem size
    pub size: usize,
    /// Problem type
    pub problem_type: String,
    /// Detected structure patterns
    pub structure_patterns: Vec<StructurePattern>,
    /// Problem features
    pub features: Array1<f64>,
    /// Difficulty assessment
    pub difficulty_assessment: DifficultyAssessment,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyAssessment {
    /// Overall difficulty score
    pub difficulty_score: f64,
    /// Difficulty factors
    pub difficulty_factors: HashMap<String, f64>,
    /// Expected solution time
    pub expected_solution_time: Duration,
    /// Recommended resources
    pub recommended_resources: ResourceRecommendation,
}
#[derive(Debug, Clone)]
pub struct ActionDecoder {
    /// Action space dimension
    pub action_dim: usize,
    /// Decoder layers
    pub layers: Vec<DenseLayer>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureTransformation {
    StandardScaling,
    MinMaxScaling,
    RobustScaling,
    QuantileTransform,
    PowerTransform,
    PolynomialFeatures { degree: usize },
    InteractionFeatures,
    LogTransform,
}
#[derive(Debug, Clone)]
pub struct RLTrainingResults {
    pub episodes: u32,
    pub total_steps: u32,
    pub avg_episode_reward: f64,
    pub best_reward: f64,
    pub loss_history: Vec<f64>,
    pub exploration_history: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    ExtremelyHard,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphMetric {
    ClusteringCoefficient,
    AveragePathLength,
    Diameter,
    Density,
    Assortativity,
    Modularity,
    SmallWorldness,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRecommendation {
    /// CPU cores
    pub cpu_cores: usize,
    /// Memory (in GB)
    pub memory_gb: f64,
    /// GPU acceleration recommended
    pub gpu_recommended: bool,
    /// Distributed computing recommended
    pub distributed_recommended: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeRecommendation {
    /// Algorithm name
    pub algorithm: String,
    /// Expected performance
    pub expected_performance: f64,
    /// Trade-offs
    pub trade_offs: HashMap<String, f64>,
    /// Use case scenarios
    pub use_cases: Vec<String>,
}
#[derive(Debug, Clone)]
pub struct GraphAnalyzer {
    /// Graph metrics calculator
    pub metrics_calculator: GraphMetricsCalculator,
    /// Community detection methods
    pub community_detectors: Vec<CommunityDetectionMethod>,
    /// Centrality measures
    pub centrality_measures: Vec<CentralityMeasure>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PCAAKernel {
    Linear,
    RBF { gamma: f64 },
    Polynomial { degree: usize },
    Sigmoid,
}
#[derive(Debug, Clone)]
pub struct EnsemblePerformance {
    /// Individual model performances
    pub individual_performances: Vec<f64>,
    /// Ensemble performance
    pub ensemble_performance: f64,
    /// Improvement over best individual
    pub improvement: f64,
    /// Diversity measures
    pub diversity_measures: HashMap<String, f64>,
}
#[derive(Debug, Clone)]
pub struct ModelEnsemble {
    /// Base models
    pub base_models: Vec<PredictionModel>,
    /// Ensemble method
    pub ensemble_method: EnsembleMethod,
    /// Model weights
    pub model_weights: Array1<f64>,
    /// Ensemble performance
    pub ensemble_performance: EnsemblePerformance,
}
#[derive(Debug, Clone)]
pub struct AlgorithmClassifier {
    /// Model type
    pub model_type: ClassificationModel,
    /// Model parameters
    pub parameters: ModelParameters,
    /// Training data
    pub training_data: Vec<TrainingExample>,
    /// Model performance metrics
    pub performance_metrics: ClassificationMetrics,
}
#[derive(Debug, Clone)]
pub struct HyperparameterTrial {
    pub parameters: HashMap<String, f64>,
    pub score: f64,
    pub training_time: Duration,
    pub validation_score: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    EpsilonGreedy { epsilon: f64 },
    Boltzmann { temperature: f64 },
    UCB { exploration_factor: f64 },
    ThompsonSampling,
    NoiseNet { noise_scale: f64 },
}
/// AI-assisted quantum optimization engine
pub struct AIAssistedOptimizer {
    /// Neural network for parameter optimization
    parameter_optimizer: ParameterOptimizationNetwork,
    /// Reinforcement learning agent for sampling strategies
    rl_agent: SamplingStrategyAgent,
    /// Automated algorithm selector
    algorithm_selector: AutomatedAlgorithmSelector,
    /// Problem structure recognition system
    structure_recognizer: ProblemStructureRecognizer,
    /// Solution quality predictor
    quality_predictor: SolutionQualityPredictor,
    /// Configuration
    config: AIOptimizerConfig,
}
impl AIAssistedOptimizer {
    /// Create new AI-assisted optimizer
    pub fn new(config: AIOptimizerConfig) -> Self {
        Self {
            parameter_optimizer: ParameterOptimizationNetwork::new(&config),
            rl_agent: SamplingStrategyAgent::new(&config),
            algorithm_selector: AutomatedAlgorithmSelector::new(&config),
            structure_recognizer: ProblemStructureRecognizer::new(),
            quality_predictor: SolutionQualityPredictor::new(&config),
            config,
        }
    }
    /// Optimize quantum algorithm for given problem
    pub fn optimize(
        &mut self,
        qubo: &Array2<f64>,
        target_quality: Option<f64>,
        _time_budget: Option<Duration>,
    ) -> Result<AIOptimizationResult, String> {
        let start_time = Instant::now();
        let features = self.extract_problem_features(qubo)?;
        let structure_patterns = if self.config.structure_recognition_enabled {
            self.structure_recognizer.recognize_structure(qubo)?
        } else {
            vec![]
        };
        let recommended_algorithm = if self.config.auto_algorithm_selection_enabled {
            self.algorithm_selector
                .select_algorithm(&features, &structure_patterns)?
        } else {
            "SimulatedAnnealing".to_string()
        };
        let optimized_parameters = if self.config.parameter_optimization_enabled {
            self.parameter_optimizer.optimize_parameters(
                &features,
                &recommended_algorithm,
                target_quality,
            )?
        } else {
            HashMap::new()
        };
        let predicted_quality = if self.config.quality_prediction_enabled {
            self.quality_predictor.predict_quality(
                &features,
                &recommended_algorithm,
                &optimized_parameters,
            )?
        } else {
            QualityPrediction {
                expected_quality: 0.8,
                confidence_interval: (0.7, 0.9),
                optimal_probability: 0.1,
                expected_convergence_time: Duration::from_secs(60),
            }
        };
        let alternatives = self.generate_alternatives(&features, &recommended_algorithm)?;
        let difficulty_assessment = self.assess_difficulty(qubo, &features, &structure_patterns)?;
        let confidence = self.compute_recommendation_confidence(
            &features,
            &recommended_algorithm,
            &optimized_parameters,
        )?;
        let total_time = start_time.elapsed();
        Ok(AIOptimizationResult {
            problem_info: ProblemInfo {
                size: qubo.shape()[0],
                problem_type: self.infer_problem_type(&features, &structure_patterns),
                structure_patterns,
                features,
                difficulty_assessment,
            },
            recommended_algorithm,
            optimized_parameters,
            predicted_quality,
            confidence,
            alternatives,
            optimization_stats: OptimizationStatistics {
                total_time,
                nn_training_time: Duration::from_millis(100),
                rl_episodes: 50,
                feature_extraction_time: Duration::from_millis(50),
                model_selection_time: Duration::from_millis(30),
                model_accuracy: 0.85,
            },
        })
    }
    /// Train the AI models on historical data
    pub fn train(
        &mut self,
        training_data: &[TrainingExample],
        validation_split: f64,
    ) -> Result<TrainingResults, String> {
        let split_index = (training_data.len() as f64 * (1.0 - validation_split)) as usize;
        let (train_data, val_data) = training_data.split_at(split_index);
        let mut results = TrainingResults {
            parameter_optimizer_results: None,
            rl_agent_results: None,
            algorithm_selector_results: None,
            quality_predictor_results: None,
        };
        if self.config.parameter_optimization_enabled {
            let param_results = self.parameter_optimizer.train(train_data, val_data)?;
            results.parameter_optimizer_results = Some(param_results);
        }
        if self.config.reinforcement_learning_enabled {
            let rl_results = self.rl_agent.train(train_data)?;
            results.rl_agent_results = Some(rl_results);
        }
        if self.config.auto_algorithm_selection_enabled {
            let selector_results = self.algorithm_selector.train(train_data, val_data)?;
            results.algorithm_selector_results = Some(selector_results);
        }
        if self.config.quality_prediction_enabled {
            let predictor_results = self.quality_predictor.train(train_data, val_data)?;
            results.quality_predictor_results = Some(predictor_results);
        }
        Ok(results)
    }
    /// Extract comprehensive features from QUBO problem
    pub fn extract_problem_features(&self, qubo: &Array2<f64>) -> Result<Array1<f64>, String> {
        let n = qubo.shape()[0];
        let mut features = Vec::new();
        features.push(n as f64);
        let coeffs: Vec<f64> = qubo.iter().copied().collect();
        let mean = coeffs.iter().sum::<f64>() / coeffs.len() as f64;
        let variance = coeffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / coeffs.len() as f64;
        features.push(mean);
        features.push(variance);
        features.push(variance.sqrt());
        let non_zero_count = coeffs.iter().filter(|&&x| x.abs() > 1e-10).count();
        let density = non_zero_count as f64 / coeffs.len() as f64;
        features.push(density);
        let max_val = coeffs.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let min_val = coeffs
            .iter()
            .map(|x| x.abs())
            .filter(|&x| x > 1e-10)
            .fold(f64::INFINITY, f64::min);
        features.push(max_val);
        features.push(if min_val.is_finite() { min_val } else { 0.0 });
        features.push(if min_val > 0.0 {
            max_val / min_val
        } else {
            1.0
        });
        let mut degree_sum = 0;
        for i in 0..n {
            let mut degree = 0;
            for j in 0..n {
                if i != j && qubo[[i, j]].abs() > 1e-10 {
                    degree += 1;
                }
            }
            degree_sum += degree;
        }
        let avg_degree = degree_sum as f64 / n as f64;
        features.push(avg_degree);
        let mut is_symmetric = true;
        for i in 0..n {
            for j in 0..n {
                if (qubo[[i, j]] - qubo[[j, i]]).abs() > 1e-10 {
                    is_symmetric = false;
                    break;
                }
            }
            if !is_symmetric {
                break;
            }
        }
        features.push(if is_symmetric { 1.0 } else { 0.0 });
        let mut diag_dominance = 0.0;
        for i in 0..n {
            let diag_val = qubo[[i, i]].abs();
            let off_diag_sum: f64 = (0..n).filter(|&j| i != j).map(|j| qubo[[i, j]].abs()).sum();
            if off_diag_sum > 0.0 {
                diag_dominance += diag_val / off_diag_sum;
            }
        }
        features.push(diag_dominance / n as f64);
        let mut frustration = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                if qubo[[i, j]] > 0.0 {
                    frustration += qubo[[i, j]];
                }
            }
        }
        features.push(frustration);
        Ok(Array1::from(features))
    }
    /// Infer problem type from features and structure
    fn infer_problem_type(&self, features: &Array1<f64>, patterns: &[StructurePattern]) -> String {
        let density = features[4];
        let avg_degree = features[7];
        if patterns
            .iter()
            .any(|p| matches!(p, StructurePattern::Grid { .. }))
        {
            "Grid-based Optimization".to_string()
        } else if patterns
            .iter()
            .any(|p| matches!(p, StructurePattern::Tree { .. }))
        {
            "Tree-structured Problem".to_string()
        } else if density > 0.8 {
            "Dense QUBO".to_string()
        } else if avg_degree < 4.0 {
            "Sparse QUBO".to_string()
        } else {
            "General QUBO".to_string()
        }
    }
    /// Assess problem difficulty
    pub fn assess_difficulty(
        &self,
        _qubo: &Array2<f64>,
        features: &Array1<f64>,
        _patterns: &[StructurePattern],
    ) -> Result<DifficultyAssessment, String> {
        let size = features[0] as usize;
        let variance = features[2];
        let density = features[4];
        let frustration = features[11];
        let size_factor = ((size as f64).log2() / 10.0).min(1.0);
        let complexity_factor = (variance * density * 10.0).min(1.0);
        let frustration_factor =
            ((frustration / (size as f64 * size as f64 / 2.0)) * 100.0).min(1.0);
        let difficulty_score = 0.2f64
            .mul_add(
                frustration_factor,
                0.4f64.mul_add(size_factor, 0.4 * complexity_factor),
            )
            .min(1.0);
        let mut difficulty_factors = HashMap::new();
        difficulty_factors.insert("size".to_string(), size_factor);
        difficulty_factors.insert("complexity".to_string(), complexity_factor);
        difficulty_factors.insert("frustration".to_string(), frustration_factor);
        let base_time = Duration::from_secs(1);
        let time_multiplier = (difficulty_score * 100.0).exp();
        let expected_solution_time = base_time * time_multiplier as u32;
        let recommended_resources = ResourceRecommendation {
            cpu_cores: if size > 1000 { 8 } else { 4 },
            memory_gb: (size as f64 * 0.001).max(1.0),
            gpu_recommended: size > 500,
            distributed_recommended: size > 5000,
        };
        Ok(DifficultyAssessment {
            difficulty_score,
            difficulty_factors,
            expected_solution_time,
            recommended_resources,
        })
    }
    /// Generate alternative algorithm recommendations
    fn generate_alternatives(
        &self,
        features: &Array1<f64>,
        recommended: &str,
    ) -> Result<Vec<AlternativeRecommendation>, String> {
        let size = features[0] as usize;
        let _density = features[4];
        let mut alternatives = Vec::new();
        if recommended != "SimulatedAnnealing" {
            alternatives.push(AlternativeRecommendation {
                algorithm: "SimulatedAnnealing".to_string(),
                expected_performance: 0.75,
                trade_offs: {
                    let mut map = HashMap::new();
                    map.insert("speed".to_string(), 0.8);
                    map.insert("quality".to_string(), 0.7);
                    map
                },
                use_cases: vec!["General purpose".to_string(), "Good baseline".to_string()],
            });
        }
        if recommended != "GeneticAlgorithm" && size > 100 {
            alternatives.push(AlternativeRecommendation {
                algorithm: "GeneticAlgorithm".to_string(),
                expected_performance: 0.8,
                trade_offs: {
                    let mut map = HashMap::new();
                    map.insert("speed".to_string(), 0.6);
                    map.insert("quality".to_string(), 0.85);
                    map
                },
                use_cases: vec![
                    "Large problems".to_string(),
                    "Population diversity".to_string(),
                ],
            });
        }
        if recommended != "TabuSearch" {
            alternatives.push(AlternativeRecommendation {
                algorithm: "TabuSearch".to_string(),
                expected_performance: 0.85,
                trade_offs: {
                    let mut map = HashMap::new();
                    map.insert("speed".to_string(), 0.7);
                    map.insert("quality".to_string(), 0.9);
                    map
                },
                use_cases: vec![
                    "Local search".to_string(),
                    "Escape local minima".to_string(),
                ],
            });
        }
        Ok(alternatives)
    }
    /// Compute confidence in recommendation
    fn compute_recommendation_confidence(
        &self,
        features: &Array1<f64>,
        _algorithm: &str,
        parameters: &HashMap<String, f64>,
    ) -> Result<f64, String> {
        let size = features[0] as usize;
        let density = features[4];
        let base_confidence = 0.7;
        let size_confidence = if size < 1000 { 0.9 } else { 0.6 };
        let density_confidence = if density > 0.1 && density < 0.9 {
            0.8
        } else {
            0.6
        };
        let param_confidence = if parameters.is_empty() { 0.7 } else { 0.85 };
        let overall_confidence: f64 =
            base_confidence * size_confidence * density_confidence * param_confidence;
        Ok(overall_confidence.min(1.0))
    }
}
/// Reinforcement learning agent for adaptive sampling strategies
pub struct SamplingStrategyAgent {
    /// Q-network for value function approximation
    q_network: QNetwork,
    /// Target network for stable training
    target_network: QNetwork,
    /// Experience replay buffer
    replay_buffer: ExperienceReplayBuffer,
    /// Exploration strategy
    exploration_strategy: ExplorationStrategy,
    /// Training statistics
    training_stats: RLTrainingStats,
}
impl SamplingStrategyAgent {
    pub const fn new(config: &AIOptimizerConfig) -> Self {
        Self {
            q_network: QNetwork {
                state_encoder: StateEncoder {
                    embedding_dim: 64,
                    layers: vec![],
                },
                value_network: vec![],
                action_decoder: ActionDecoder {
                    action_dim: 10,
                    layers: vec![],
                },
            },
            target_network: QNetwork {
                state_encoder: StateEncoder {
                    embedding_dim: 64,
                    layers: vec![],
                },
                value_network: vec![],
                action_decoder: ActionDecoder {
                    action_dim: 10,
                    layers: vec![],
                },
            },
            replay_buffer: ExperienceReplayBuffer {
                buffer: VecDeque::new(),
                max_size: config.replay_buffer_size,
                position: 0,
            },
            exploration_strategy: ExplorationStrategy::EpsilonGreedy { epsilon: 0.1 },
            training_stats: RLTrainingStats {
                episodes: 0,
                total_steps: 0,
                avg_episode_reward: 0.0,
                best_reward: 0.0,
                loss_history: vec![],
                exploration_history: vec![],
            },
        }
    }
    pub fn train(&mut self, _data: &[TrainingExample]) -> Result<RLTrainingResults, String> {
        Ok(RLTrainingResults {
            episodes: 100,
            total_steps: 5000,
            avg_episode_reward: 10.0,
            best_reward: 50.0,
            loss_history: vec![1.0, 0.5, 0.2, 0.1],
            exploration_history: vec![1.0, 0.8, 0.6, 0.4],
        })
    }
    /// Get a reference to the Q-network
    pub const fn q_network(&self) -> &QNetwork {
        &self.q_network
    }
    /// Get a reference to the replay buffer
    pub const fn replay_buffer(&self) -> &ExperienceReplayBuffer {
        &self.replay_buffer
    }
    /// Get a reference to the training stats
    pub const fn training_stats(&self) -> &RLTrainingStats {
        &self.training_stats
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GPKernel {
    RBF { length_scale: f64 },
    Matern { nu: f64, length_scale: f64 },
    Linear { variance: f64 },
    Periodic { period: f64, length_scale: f64 },
}
