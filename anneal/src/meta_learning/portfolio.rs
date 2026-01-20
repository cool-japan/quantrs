//! Algorithm Portfolio Management for Meta-Learning Optimization
//!
//! This module contains all Algorithm Portfolio Management types and implementations
//! used by the meta-learning optimization system.

use super::config::{
    ActivationFunction, AlgorithmSelectionStrategy, AlgorithmType, ArchitectureSpec,
    ConnectionPattern, DiversityCriteria, DiversityMethod, LayerSpec, LayerType,
    OptimizationConfiguration, OptimizationSettings, OptimizerType, RegularizationConfig,
    ResourceAllocation,
};
use super::features::ProblemFeatures;
use crate::applications::ApplicationResult;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

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

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Peak CPU usage
    pub peak_cpu: f64,
    /// Peak memory usage (MB)
    pub peak_memory: usize,
    /// GPU utilization
    pub gpu_utilization: f64,
    /// Energy consumption
    pub energy_consumption: f64,
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

/// Problem domains
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemDomain {
    /// Combinatorial optimization
    Combinatorial,
    /// Portfolio optimization
    Portfolio,
    /// Scheduling
    Scheduling,
    /// Graph problems
    Graph,
    /// Machine learning
    MachineLearning,
    /// Physics simulation
    Physics,
    /// Chemistry
    Chemistry,
    /// Custom domain
    Custom(String),
}

/// Resource requirements for algorithms
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceRequirements {
    /// Memory requirements (MB)
    pub memory: usize,
    /// Computational requirements (FLOPS)
    pub computation: f64,
    /// Training time estimate
    pub training_time: Duration,
    /// Model size (parameters)
    pub model_size: usize,
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

/// Diversity metrics
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl AlgorithmPortfolio {
    #[must_use]
    pub fn new(config: super::config::PortfolioManagementConfig) -> Self {
        Self {
            algorithms: HashMap::new(),
            composition: PortfolioComposition {
                weights: HashMap::new(),
                selection_probabilities: HashMap::new(),
                last_update: Instant::now(),
                quality_score: 0.8,
            },
            selection_strategy: config.selection_strategy,
            performance_history: HashMap::new(),
            diversity_analyzer: DiversityAnalyzer {
                metrics: vec![DiversityMetric::AlgorithmDiversity],
                methods: vec![DiversityMethod::KullbackLeibler],
                current_diversity: 0.7,
                target_diversity: 0.8,
            },
        }
    }

    /// Select algorithm for given problem features
    pub fn select_algorithm(&self, features: &ProblemFeatures) -> ApplicationResult<String> {
        // Simple algorithm selection based on problem size
        let algorithm_id = if features.size < 100 {
            "simulated_annealing"
        } else if features.size < 500 {
            "quantum_annealing"
        } else {
            "hybrid_approach"
        };

        Ok(algorithm_id.to_string())
    }

    /// Update portfolio based on performance feedback
    pub fn update_portfolio(
        &mut self,
        algorithm_id: &str,
        performance: f64,
        features: &ProblemFeatures,
    ) {
        // Record performance
        let record = PerformanceRecord {
            timestamp: Instant::now(),
            problem_features: features.clone(),
            performance,
            resource_usage: ResourceUsage {
                peak_cpu: 0.8,
                peak_memory: 512,
                gpu_utilization: 0.0,
                energy_consumption: 100.0,
            },
            context: HashMap::new(),
        };

        self.performance_history
            .entry(algorithm_id.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(record);

        // Limit history size
        if let Some(history) = self.performance_history.get_mut(algorithm_id) {
            if history.len() > 1000 {
                history.pop_front();
            }
        }

        // Update composition weights based on performance
        self.update_composition_weights();
    }

    /// Update composition weights based on performance history
    fn update_composition_weights(&mut self) {
        for (algorithm_id, history) in &self.performance_history {
            if !history.is_empty() {
                let avg_performance: f64 =
                    history.iter().map(|record| record.performance).sum::<f64>()
                        / history.len() as f64;

                self.composition
                    .weights
                    .insert(algorithm_id.clone(), avg_performance);
            }
        }

        // Normalize weights
        let total_weight: f64 = self.composition.weights.values().sum();
        if total_weight > 0.0 {
            for weight in self.composition.weights.values_mut() {
                *weight /= total_weight;
            }
        }

        self.composition.last_update = Instant::now();
    }

    /// Get portfolio statistics
    pub fn get_statistics(&self) -> PortfolioStatistics {
        let total_algorithms = self.algorithms.len();
        let active_algorithms = self.composition.weights.len();
        let avg_performance = if self.performance_history.is_empty() {
            0.0
        } else {
            let total_records: usize = self
                .performance_history
                .values()
                .map(std::collections::VecDeque::len)
                .sum();

            if total_records > 0 {
                let total_performance: f64 = self
                    .performance_history
                    .values()
                    .flat_map(|history| history.iter())
                    .map(|record| record.performance)
                    .sum();
                total_performance / total_records as f64
            } else {
                0.0
            }
        };

        PortfolioStatistics {
            total_algorithms,
            active_algorithms,
            avg_performance,
            diversity_score: self.diversity_analyzer.current_diversity,
            last_update: self.composition.last_update,
        }
    }
}

/// Portfolio statistics
#[derive(Debug, Clone)]
pub struct PortfolioStatistics {
    /// Total number of algorithms
    pub total_algorithms: usize,
    /// Number of active algorithms
    pub active_algorithms: usize,
    /// Average performance
    pub avg_performance: f64,
    /// Diversity score
    pub diversity_score: f64,
    /// Last update timestamp
    pub last_update: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_learning::config::*;

    #[test]
    fn test_portfolio_creation() {
        let config = PortfolioManagementConfig::default();
        let portfolio = AlgorithmPortfolio::new(config);

        assert_eq!(portfolio.algorithms.len(), 0);
        assert!(portfolio.composition.quality_score > 0.0);
    }

    #[test]
    fn test_algorithm_selection() {
        let config = PortfolioManagementConfig::default();
        let portfolio = AlgorithmPortfolio::new(config);

        let features = ProblemFeatures {
            size: 50,
            density: 0.3,
            graph_features: crate::meta_learning::features::GraphFeatures::default(),
            statistical_features: crate::meta_learning::features::StatisticalFeatures::default(),
            spectral_features: crate::meta_learning::features::SpectralFeatures::default(),
            domain_features: HashMap::new(),
        };

        let algorithm_id = portfolio.select_algorithm(&features);
        assert!(algorithm_id.is_ok());
        assert!(!algorithm_id
            .expect("Algorithm selection should succeed")
            .is_empty());
    }

    #[test]
    fn test_portfolio_update() {
        let config = PortfolioManagementConfig::default();
        let mut portfolio = AlgorithmPortfolio::new(config);

        let features = ProblemFeatures {
            size: 100,
            density: 0.5,
            graph_features: crate::meta_learning::features::GraphFeatures::default(),
            statistical_features: crate::meta_learning::features::StatisticalFeatures::default(),
            spectral_features: crate::meta_learning::features::SpectralFeatures::default(),
            domain_features: HashMap::new(),
        };

        portfolio.update_portfolio("test_algorithm", 0.9, &features);

        assert!(portfolio.performance_history.contains_key("test_algorithm"));
        assert_eq!(portfolio.performance_history["test_algorithm"].len(), 1);
    }

    #[test]
    fn test_portfolio_statistics() {
        let config = PortfolioManagementConfig::default();
        let portfolio = AlgorithmPortfolio::new(config);

        let stats = portfolio.get_statistics();
        assert_eq!(stats.total_algorithms, 0);
        assert_eq!(stats.active_algorithms, 0);
    }
}
