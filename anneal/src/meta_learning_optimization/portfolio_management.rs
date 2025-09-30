//! Algorithm portfolio management for meta-learning optimization

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::config::*;
use super::feature_extraction::*;

/// Algorithm portfolio manager
pub struct AlgorithmPortfolio {
    /// Available algorithms
    pub algorithms: HashMap<String, Algorithm>,
    /// Portfolio composition
    pub composition: PortfolioComposition,
    /// Selection strategy
    pub selection_strategy: AlgorithmSelectionStrategy,
    /// Performance history
    pub performance_history: HashMap<String, VecDeque<PerformanceRecord>>,
    /// Diversity analyzer
    pub diversity_analyzer: DiversityAnalyzer,
}

impl AlgorithmPortfolio {
    pub fn new(config: PortfolioManagementConfig) -> Self {
        let mut portfolio = Self {
            algorithms: HashMap::new(),
            composition: PortfolioComposition::new(),
            selection_strategy: config.selection_strategy,
            performance_history: HashMap::new(),
            diversity_analyzer: DiversityAnalyzer::new(config.diversity_criteria),
        };

        // Initialize with default algorithms
        portfolio.add_default_algorithms();
        portfolio
    }

    fn add_default_algorithms(&mut self) {
        // Add simulated annealing
        let sa_algorithm = Algorithm {
            id: "simulated_annealing".to_string(),
            algorithm_type: AlgorithmType::SimulatedAnnealing,
            default_config: OptimizationConfiguration {
                algorithm: AlgorithmType::SimulatedAnnealing,
                hyperparameters: [
                    ("initial_temperature".to_string(), 10.0),
                    ("final_temperature".to_string(), 0.1),
                    ("cooling_rate".to_string(), 0.95),
                ].iter().cloned().collect(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 1.0,
                    memory: 256,
                    gpu: 0.0,
                    time: Duration::from_secs(300),
                },
            },
            performance_stats: AlgorithmPerformanceStats::default(),
            applicability: ApplicabilityConditions {
                size_range: (1, 10000),
                suitable_domains: vec![
                    ProblemDomain::Combinatorial,
                    ProblemDomain::Graph,
                    ProblemDomain::Scheduling,
                ],
                required_resources: ResourceRequirements {
                    memory: 128,
                    compute_time: Duration::from_secs(60),
                    parameters: 1000,
                    flops: 1000000,
                },
                performance_guarantees: vec![],
            },
        };

        self.algorithms.insert("simulated_annealing".to_string(), sa_algorithm);

        // Add quantum annealing
        let qa_algorithm = Algorithm {
            id: "quantum_annealing".to_string(),
            algorithm_type: AlgorithmType::QuantumAnnealing,
            default_config: OptimizationConfiguration {
                algorithm: AlgorithmType::QuantumAnnealing,
                hyperparameters: [
                    ("annealing_time".to_string(), 20.0),
                    ("num_reads".to_string(), 1000.0),
                    ("chain_strength".to_string(), 1.0),
                ].iter().cloned().collect(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 0.5,
                    memory: 512,
                    gpu: 0.0,
                    time: Duration::from_secs(60),
                },
            },
            performance_stats: AlgorithmPerformanceStats::default(),
            applicability: ApplicabilityConditions {
                size_range: (1, 5000),
                suitable_domains: vec![
                    ProblemDomain::Combinatorial,
                    ProblemDomain::Physics,
                    ProblemDomain::MachineLearning,
                ],
                required_resources: ResourceRequirements {
                    memory: 256,
                    compute_time: Duration::from_secs(30),
                    parameters: 5000,
                    flops: 10000000,
                },
                performance_guarantees: vec![],
            },
        };

        self.algorithms.insert("quantum_annealing".to_string(), qa_algorithm);

        // Add tabu search
        let ts_algorithm = Algorithm {
            id: "tabu_search".to_string(),
            algorithm_type: AlgorithmType::TabuSearch,
            default_config: OptimizationConfiguration {
                algorithm: AlgorithmType::TabuSearch,
                hyperparameters: [
                    ("tabu_size".to_string(), 50.0),
                    ("max_iterations".to_string(), 1000.0),
                    ("aspiration_criteria".to_string(), 1.0),
                ].iter().cloned().collect(),
                architecture: None,
                resources: ResourceAllocation {
                    cpu: 1.0,
                    memory: 512,
                    gpu: 0.0,
                    time: Duration::from_secs(600),
                },
            },
            performance_stats: AlgorithmPerformanceStats::default(),
            applicability: ApplicabilityConditions {
                size_range: (10, 50000),
                suitable_domains: vec![
                    ProblemDomain::Combinatorial,
                    ProblemDomain::Scheduling,
                    ProblemDomain::Graph,
                ],
                required_resources: ResourceRequirements {
                    memory: 256,
                    compute_time: Duration::from_secs(120),
                    parameters: 10000,
                    flops: 50000000,
                },
                performance_guarantees: vec![],
            },
        };

        self.algorithms.insert("tabu_search".to_string(), ts_algorithm);
    }

    pub fn select_algorithm(&mut self, problem_features: &ProblemFeatures) -> Result<String, String> {
        match &self.selection_strategy {
            AlgorithmSelectionStrategy::MultiArmedBandit => self.multi_armed_bandit_selection(problem_features),
            AlgorithmSelectionStrategy::UpperConfidenceBound => self.ucb_selection(problem_features),
            AlgorithmSelectionStrategy::ThompsonSampling => self.thompson_sampling_selection(problem_features),
            AlgorithmSelectionStrategy::EpsilonGreedy(epsilon) => self.epsilon_greedy_selection(problem_features, *epsilon),
            AlgorithmSelectionStrategy::CollaborativeFiltering => self.collaborative_filtering_selection(problem_features),
            AlgorithmSelectionStrategy::MetaLearningBased => self.meta_learning_selection(problem_features),
        }
    }

    fn multi_armed_bandit_selection(&self, problem_features: &ProblemFeatures) -> Result<String, String> {
        // Simplified multi-armed bandit using average performance
        let applicable_algorithms = self.get_applicable_algorithms(problem_features);

        if applicable_algorithms.is_empty() {
            return Err("No applicable algorithms found".to_string());
        }

        let best_algorithm = applicable_algorithms.iter()
            .max_by(|a, b| {
                let perf_a = self.algorithms.get(*a).map(|alg| alg.performance_stats.mean_performance).unwrap_or(0.0);
                let perf_b = self.algorithms.get(*b).map(|alg| alg.performance_stats.mean_performance).unwrap_or(0.0);
                perf_a.partial_cmp(&perf_b).unwrap()
            })
            .ok_or("Failed to select algorithm")?;

        Ok(best_algorithm.clone())
    }

    fn ucb_selection(&self, problem_features: &ProblemFeatures) -> Result<String, String> {
        let applicable_algorithms = self.get_applicable_algorithms(problem_features);

        if applicable_algorithms.is_empty() {
            return Err("No applicable algorithms found".to_string());
        }

        // Simplified UCB calculation
        let total_trials: f64 = self.performance_history.values()
            .map(|history| history.len() as f64)
            .sum();

        let best_algorithm = applicable_algorithms.iter()
            .max_by(|a, b| {
                let history_a = self.performance_history.get(*a).map(|h| h.len()).unwrap_or(0);
                let history_b = self.performance_history.get(*b).map(|h| h.len()).unwrap_or(0);

                let mean_a = self.algorithms.get(*a).map(|alg| alg.performance_stats.mean_performance).unwrap_or(0.0);
                let mean_b = self.algorithms.get(*b).map(|alg| alg.performance_stats.mean_performance).unwrap_or(0.0);

                let confidence_a = if history_a > 0 {
                    (2.0 * total_trials.ln() / history_a as f64).sqrt()
                } else {
                    f64::INFINITY
                };

                let confidence_b = if history_b > 0 {
                    (2.0 * total_trials.ln() / history_b as f64).sqrt()
                } else {
                    f64::INFINITY
                };

                let ucb_a = mean_a + confidence_a;
                let ucb_b = mean_b + confidence_b;

                ucb_a.partial_cmp(&ucb_b).unwrap()
            })
            .ok_or("Failed to select algorithm")?;

        Ok(best_algorithm.clone())
    }

    fn thompson_sampling_selection(&self, problem_features: &ProblemFeatures) -> Result<String, String> {
        // Simplified Thompson sampling - just return the algorithm with highest variance
        let applicable_algorithms = self.get_applicable_algorithms(problem_features);

        if applicable_algorithms.is_empty() {
            return Err("No applicable algorithms found".to_string());
        }

        let best_algorithm = applicable_algorithms.iter()
            .max_by(|a, b| {
                let var_a = self.algorithms.get(*a).map(|alg| alg.performance_stats.performance_variance).unwrap_or(0.0);
                let var_b = self.algorithms.get(*b).map(|alg| alg.performance_stats.performance_variance).unwrap_or(0.0);
                var_a.partial_cmp(&var_b).unwrap()
            })
            .ok_or("Failed to select algorithm")?;

        Ok(best_algorithm.clone())
    }

    fn epsilon_greedy_selection(&self, problem_features: &ProblemFeatures, epsilon: f64) -> Result<String, String> {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();

        let applicable_algorithms = self.get_applicable_algorithms(problem_features);

        if applicable_algorithms.is_empty() {
            return Err("No applicable algorithms found".to_string());
        }

        if rng.gen_bool(epsilon) {
            // Explore: random selection
            let algorithm = applicable_algorithms.choose(&mut rng)
                .ok_or("Failed to randomly select algorithm")?;
            Ok(algorithm.clone())
        } else {
            // Exploit: best known algorithm
            self.multi_armed_bandit_selection(problem_features)
        }
    }

    fn collaborative_filtering_selection(&self, problem_features: &ProblemFeatures) -> Result<String, String> {
        // Simplified collaborative filtering based on problem similarity
        self.multi_armed_bandit_selection(problem_features)
    }

    fn meta_learning_selection(&self, problem_features: &ProblemFeatures) -> Result<String, String> {
        // Simplified meta-learning selection
        self.multi_armed_bandit_selection(problem_features)
    }

    fn get_applicable_algorithms(&self, problem_features: &ProblemFeatures) -> Vec<String> {
        self.algorithms.iter()
            .filter(|(_, algorithm)| {
                // Check size range
                problem_features.size >= algorithm.applicability.size_range.0 &&
                problem_features.size <= algorithm.applicability.size_range.1
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn record_performance(&mut self, algorithm_id: &str, record: PerformanceRecord) {
        self.performance_history
            .entry(algorithm_id.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(record.clone());

        // Update algorithm statistics
        if let Some(algorithm) = self.algorithms.get_mut(algorithm_id) {
            self.update_algorithm_stats(algorithm, &record);
        }

        // Limit history size
        if let Some(history) = self.performance_history.get_mut(algorithm_id) {
            while history.len() > 1000 {
                history.pop_front();
            }
        }
    }

    fn update_algorithm_stats(&mut self, algorithm: &mut Algorithm, record: &PerformanceRecord) {
        let history = self.performance_history.get(&algorithm.id).unwrap();

        if !history.is_empty() {
            // Update mean performance
            let total_performance: f64 = history.iter().map(|r| r.performance).sum();
            algorithm.performance_stats.mean_performance = total_performance / history.len() as f64;

            // Update variance
            let variance: f64 = history.iter()
                .map(|r| (r.performance - algorithm.performance_stats.mean_performance).powi(2))
                .sum::<f64>() / history.len() as f64;
            algorithm.performance_stats.performance_variance = variance;

            // Update success rate (simplified: performance > 0.5)
            let successes = history.iter().filter(|r| r.performance > 0.5).count();
            algorithm.performance_stats.success_rate = successes as f64 / history.len() as f64;
        }
    }

    pub fn get_portfolio_diversity(&self) -> f64 {
        self.diversity_analyzer.current_diversity
    }

    pub fn update_composition(&mut self) {
        // Simple composition update based on recent performance
        let mut new_weights = HashMap::new();
        let mut total_performance = 0.0;

        for (algorithm_id, algorithm) in &self.algorithms {
            let weight = algorithm.performance_stats.mean_performance.max(0.1);
            new_weights.insert(algorithm_id.clone(), weight);
            total_performance += weight;
        }

        // Normalize weights
        if total_performance > 0.0 {
            for (_, weight) in new_weights.iter_mut() {
                *weight /= total_performance;
            }
        }

        self.composition.weights = new_weights;
        self.composition.last_update = Instant::now();

        // Update selection probabilities (same as weights for simplicity)
        self.composition.selection_probabilities = self.composition.weights.clone();
    }
}

/// Algorithm representation
#[derive(Debug)]
pub struct Algorithm {
    /// Algorithm identifier
    pub id: String,
    /// Algorithm type
    pub algorithm_type: AlgorithmType,
    /// Default configuration
    pub default_config: OptimizationConfiguration,
    /// Performance statistics
    pub performance_stats: AlgorithmPerformanceStats,
    /// Applicability conditions
    pub applicability: ApplicabilityConditions,
}

/// Portfolio composition
#[derive(Debug, Clone)]
pub struct PortfolioComposition {
    /// Algorithm weights
    pub weights: HashMap<String, f64>,
    /// Selection probabilities
    pub selection_probabilities: HashMap<String, f64>,
    /// Last update time
    pub last_update: Instant,
    /// Composition quality
    pub quality_score: f64,
}

impl PortfolioComposition {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            selection_probabilities: HashMap::new(),
            last_update: Instant::now(),
            quality_score: 0.0,
        }
    }
}

/// Performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Timestamp
    pub timestamp: Instant,
    /// Problem characteristics
    pub problem_features: ProblemFeatures,
    /// Performance achieved
    pub performance: f64,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Context information
    pub context: HashMap<String, String>,
}

/// Algorithm performance statistics
#[derive(Debug, Clone)]
pub struct AlgorithmPerformanceStats {
    /// Mean performance
    pub mean_performance: f64,
    /// Performance variance
    pub performance_variance: f64,
    /// Success rate
    pub success_rate: f64,
    /// Average runtime
    pub avg_runtime: Duration,
    /// Scalability factor
    pub scalability_factor: f64,
}

impl Default for AlgorithmPerformanceStats {
    fn default() -> Self {
        Self {
            mean_performance: 0.5,
            performance_variance: 0.1,
            success_rate: 0.5,
            avg_runtime: Duration::from_secs(60),
            scalability_factor: 1.0,
        }
    }
}

/// Applicability conditions
#[derive(Debug, Clone)]
pub struct ApplicabilityConditions {
    /// Problem size range
    pub size_range: (usize, usize),
    /// Suitable domains
    pub suitable_domains: Vec<ProblemDomain>,
    /// Required resources
    pub required_resources: ResourceRequirements,
    /// Performance guarantees
    pub performance_guarantees: Vec<PerformanceGuarantee>,
}

/// Performance guarantee
#[derive(Debug, Clone)]
pub struct PerformanceGuarantee {
    /// Guarantee type
    pub guarantee_type: GuaranteeType,
    /// Confidence level
    pub confidence: f64,
    /// Conditions
    pub conditions: Vec<String>,
}

/// Types of performance guarantees
#[derive(Debug, Clone, PartialEq)]
pub enum GuaranteeType {
    /// Minimum performance level
    MinimumPerformance(f64),
    /// Maximum runtime
    MaximumRuntime(Duration),
    /// Resource bounds
    ResourceBounds(ResourceRequirements),
    /// Quality bounds
    QualityBounds(f64, f64),
}

/// Diversity analyzer
#[derive(Debug)]
pub struct DiversityAnalyzer {
    /// Diversity metrics
    pub metrics: Vec<DiversityMetric>,
    /// Analysis methods
    pub methods: Vec<DiversityMethod>,
    /// Current diversity score
    pub current_diversity: f64,
    /// Target diversity
    pub target_diversity: f64,
}

impl DiversityAnalyzer {
    pub fn new(criteria: DiversityCriteria) -> Self {
        Self {
            metrics: vec![
                DiversityMetric::AlgorithmDiversity,
                DiversityMetric::PerformanceDiversity,
            ],
            methods: vec![criteria.diversity_method],
            current_diversity: 0.5,
            target_diversity: criteria.min_algorithmic_diversity,
        }
    }

    pub fn analyze_diversity(&mut self, portfolio: &AlgorithmPortfolio) -> f64 {
        // Simplified diversity calculation
        let num_algorithms = portfolio.algorithms.len() as f64;
        let max_diversity = (num_algorithms * (num_algorithms - 1.0) / 2.0).max(1.0);

        // Calculate algorithmic diversity
        let algorithmic_diversity = if num_algorithms > 1.0 {
            1.0 / num_algorithms
        } else {
            0.0
        };

        // Calculate performance diversity
        let performances: Vec<f64> = portfolio.algorithms.values()
            .map(|alg| alg.performance_stats.mean_performance)
            .collect();

        let performance_diversity = if performances.len() > 1 {
            let mean_perf = performances.iter().sum::<f64>() / performances.len() as f64;
            let variance = performances.iter()
                .map(|p| (p - mean_perf).powi(2))
                .sum::<f64>() / performances.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        self.current_diversity = (algorithmic_diversity + performance_diversity) / 2.0;
        self.current_diversity
    }
}

/// Diversity metrics
#[derive(Debug, Clone, PartialEq)]
pub enum DiversityMetric {
    /// Algorithm diversity
    AlgorithmDiversity,
    /// Performance diversity
    PerformanceDiversity,
    /// Feature diversity
    FeatureDiversity,
    /// Error diversity
    ErrorDiversity,
    /// Prediction diversity
    PredictionDiversity,
}

use super::neural_architecture_search::ResourceRequirements;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_creation() {
        let config = PortfolioManagementConfig::default();
        let portfolio = AlgorithmPortfolio::new(config);
        assert!(!portfolio.algorithms.is_empty());
    }

    #[test]
    fn test_algorithm_selection() {
        let config = PortfolioManagementConfig::default();
        let mut portfolio = AlgorithmPortfolio::new(config);

        let features = ProblemFeatures {
            size: 100,
            density: 0.5,
            graph_features: GraphFeatures::default(),
            statistical_features: StatisticalFeatures::default(),
            spectral_features: SpectralFeatures::default(),
            domain_features: HashMap::new(),
        };

        let result = portfolio.select_algorithm(&features);
        assert!(result.is_ok());
    }

    #[test]
    fn test_diversity_analyzer() {
        let criteria = DiversityCriteria::default();
        let analyzer = DiversityAnalyzer::new(criteria);
        assert_eq!(analyzer.target_diversity, 0.2);
    }
}