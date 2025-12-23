//! Feature extraction system for meta-learning optimization

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Instant;

use super::config::*;
use crate::applications::ApplicationResult;
use crate::ising::IsingModel;

/// Problem features representation
#[derive(Debug, Clone)]
pub struct ProblemFeatures {
    /// Problem size (number of variables)
    pub size: usize,
    /// Graph density
    pub density: f64,
    /// Graph-based features
    pub graph_features: GraphFeatures,
    /// Statistical features
    pub statistical_features: StatisticalFeatures,
    /// Spectral features
    pub spectral_features: SpectralFeatures,
    /// Domain-specific features
    pub domain_features: HashMap<String, f64>,
}

/// Feature vector representation
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Feature values
    pub values: Vec<f64>,
    /// Feature names
    pub names: Vec<String>,
    /// Normalization parameters
    pub normalization: Option<NormalizationParams>,
}

/// Normalization parameters
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    /// Mean values
    pub mean: Vec<f64>,
    /// Standard deviation values
    pub std: Vec<f64>,
    /// Min values
    pub min: Vec<f64>,
    /// Max values
    pub max: Vec<f64>,
}

/// Distribution statistics
#[derive(Debug, Clone)]
pub struct DistributionStats {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
}

impl Default for DistributionStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
            min: 0.0,
            max: 1.0,
            skewness: 0.0,
            kurtosis: 3.0,
        }
    }
}

/// Graph-based features
#[derive(Debug, Clone)]
pub struct GraphFeatures {
    /// Number of vertices
    pub num_vertices: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Average degree
    pub avg_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Path length statistics
    pub path_length_stats: PathLengthStats,
    /// Centrality measures
    pub centrality_measures: CentralityMeasures,
}

impl Default for GraphFeatures {
    fn default() -> Self {
        Self {
            num_vertices: 0,
            num_edges: 0,
            avg_degree: 0.0,
            clustering_coefficient: 0.0,
            path_length_stats: PathLengthStats {
                avg_shortest_path: 0.0,
                diameter: 0,
                radius: 0,
                eccentricity_stats: DistributionStats::default(),
            },
            centrality_measures: CentralityMeasures {
                degree_centrality: DistributionStats::default(),
                betweenness_centrality: DistributionStats::default(),
                closeness_centrality: DistributionStats::default(),
                eigenvector_centrality: DistributionStats::default(),
            },
        }
    }
}

/// Path length statistics
#[derive(Debug, Clone)]
pub struct PathLengthStats {
    /// Average shortest path length
    pub avg_shortest_path: f64,
    /// Diameter
    pub diameter: usize,
    /// Radius
    pub radius: usize,
    /// Eccentricity statistics
    pub eccentricity_stats: DistributionStats,
}

/// Centrality measures
#[derive(Debug, Clone)]
pub struct CentralityMeasures {
    /// Degree centrality
    pub degree_centrality: DistributionStats,
    /// Betweenness centrality
    pub betweenness_centrality: DistributionStats,
    /// Closeness centrality
    pub closeness_centrality: DistributionStats,
    /// Eigenvector centrality
    pub eigenvector_centrality: DistributionStats,
}

/// Statistical features
#[derive(Debug, Clone)]
pub struct StatisticalFeatures {
    /// Bias statistics
    pub bias_stats: DistributionStats,
    /// Coupling statistics
    pub coupling_stats: DistributionStats,
    /// Energy landscape features
    pub energy_landscape: EnergyLandscapeFeatures,
    /// Correlation features
    pub correlation_features: CorrelationFeatures,
}

impl Default for StatisticalFeatures {
    fn default() -> Self {
        Self {
            bias_stats: DistributionStats::default(),
            coupling_stats: DistributionStats::default(),
            energy_landscape: EnergyLandscapeFeatures {
                local_minima_estimate: 0,
                energy_barriers: Vec::new(),
                ruggedness: 0.0,
                basin_sizes: DistributionStats::default(),
            },
            correlation_features: CorrelationFeatures {
                autocorrelation: Vec::new(),
                cross_correlation: HashMap::new(),
                mutual_information: 0.0,
            },
        }
    }
}

/// Energy landscape features
#[derive(Debug, Clone)]
pub struct EnergyLandscapeFeatures {
    /// Number of local minima estimate
    pub local_minima_estimate: usize,
    /// Energy barrier estimates
    pub energy_barriers: Vec<f64>,
    /// Landscape ruggedness
    pub ruggedness: f64,
    /// Basin size distribution
    pub basin_sizes: DistributionStats,
}

/// Correlation features
#[derive(Debug, Clone)]
pub struct CorrelationFeatures {
    /// Autocorrelation function
    pub autocorrelation: Vec<f64>,
    /// Cross-correlation features
    pub cross_correlation: HashMap<String, f64>,
    /// Mutual information
    pub mutual_information: f64,
}

/// Spectral features
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Eigenvalue statistics
    pub eigenvalue_stats: DistributionStats,
    /// Spectral gap
    pub spectral_gap: f64,
    /// Spectral radius
    pub spectral_radius: f64,
    /// Trace
    pub trace: f64,
    /// Condition number
    pub condition_number: f64,
}

impl Default for SpectralFeatures {
    fn default() -> Self {
        Self {
            eigenvalue_stats: DistributionStats::default(),
            spectral_gap: 0.0,
            spectral_radius: 0.0,
            trace: 0.0,
            condition_number: 1.0,
        }
    }
}

/// Feature extraction system
pub struct FeatureExtractor {
    /// Configuration
    pub config: FeatureExtractionConfig,
    /// Feature transformers
    pub transformers: Vec<FeatureTransformer>,
    /// Feature selectors
    pub selectors: Vec<FeatureSelector>,
    /// Dimensionality reducers
    pub reducers: Vec<DimensionalityReducer>,
}

impl FeatureExtractor {
    pub fn new(config: FeatureExtractionConfig) -> Self {
        Self {
            config,
            transformers: Vec::new(),
            selectors: Vec::new(),
            reducers: Vec::new(),
        }
    }

    pub fn extract_features(&mut self, problem: &IsingModel) -> ApplicationResult<ProblemFeatures> {
        let graph_features = if self.config.enable_graph_features {
            self.extract_graph_features(problem)?
        } else {
            GraphFeatures::default()
        };

        let statistical_features = if self.config.enable_statistical_features {
            self.extract_statistical_features(problem)?
        } else {
            StatisticalFeatures::default()
        };

        let spectral_features = if self.config.enable_spectral_features {
            self.extract_spectral_features(problem)?
        } else {
            SpectralFeatures::default()
        };

        Ok(ProblemFeatures {
            size: problem.num_qubits,
            density: self.calculate_density(problem),
            graph_features,
            statistical_features,
            spectral_features,
            domain_features: HashMap::new(),
        })
    }

    fn extract_graph_features(&self, problem: &IsingModel) -> ApplicationResult<GraphFeatures> {
        let num_vertices = problem.num_qubits;
        let mut num_edges = 0;

        // Count edges (non-zero couplings)
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if problem.get_coupling(i, j).unwrap_or(0.0).abs() > 1e-10 {
                    num_edges += 1;
                }
            }
        }

        let avg_degree = if num_vertices > 0 {
            2.0 * num_edges as f64 / num_vertices as f64
        } else {
            0.0
        };

        Ok(GraphFeatures {
            num_vertices,
            num_edges,
            avg_degree,
            clustering_coefficient: 0.1, // Simplified
            path_length_stats: PathLengthStats {
                avg_shortest_path: avg_degree.ln().max(1.0),
                diameter: num_vertices / 2,
                radius: num_vertices / 4,
                eccentricity_stats: DistributionStats::default(),
            },
            centrality_measures: CentralityMeasures {
                degree_centrality: DistributionStats::default(),
                betweenness_centrality: DistributionStats::default(),
                closeness_centrality: DistributionStats::default(),
                eigenvector_centrality: DistributionStats::default(),
            },
        })
    }

    fn extract_statistical_features(
        &self,
        problem: &IsingModel,
    ) -> ApplicationResult<StatisticalFeatures> {
        let mut bias_values = Vec::new();
        let mut coupling_values = Vec::new();

        // Collect bias values
        for i in 0..problem.num_qubits {
            bias_values.push(problem.get_bias(i).unwrap_or(0.0));
        }

        // Collect coupling values
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                let coupling = problem.get_coupling(i, j).unwrap_or(0.0);
                if coupling.abs() > 1e-10 {
                    coupling_values.push(coupling);
                }
            }
        }

        Ok(StatisticalFeatures {
            bias_stats: self.calculate_distribution_stats(&bias_values),
            coupling_stats: self.calculate_distribution_stats(&coupling_values),
            energy_landscape: EnergyLandscapeFeatures {
                local_minima_estimate: (problem.num_qubits as f64).sqrt() as usize,
                energy_barriers: vec![1.0, 2.0, 3.0],
                ruggedness: 0.5,
                basin_sizes: DistributionStats::default(),
            },
            correlation_features: CorrelationFeatures {
                autocorrelation: vec![1.0, 0.8, 0.6, 0.4, 0.2],
                cross_correlation: HashMap::new(),
                mutual_information: 0.3,
            },
        })
    }

    fn extract_spectral_features(
        &self,
        problem: &IsingModel,
    ) -> ApplicationResult<SpectralFeatures> {
        // Simplified spectral analysis
        let n = problem.num_qubits as f64;
        let spectral_gap_estimate = 1.0 / n.sqrt();

        Ok(SpectralFeatures {
            eigenvalue_stats: DistributionStats {
                mean: 0.0,
                std_dev: 1.0,
                min: -n,
                max: n,
                skewness: 0.0,
                kurtosis: 3.0,
            },
            spectral_gap: spectral_gap_estimate,
            spectral_radius: n,
            trace: 0.0,
            condition_number: n,
        })
    }

    fn calculate_density(&self, problem: &IsingModel) -> f64 {
        let mut num_edges = 0;
        let max_edges = problem.num_qubits * (problem.num_qubits - 1) / 2;

        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if problem.get_coupling(i, j).unwrap_or(0.0).abs() > 1e-10 {
                    num_edges += 1;
                }
            }
        }

        if max_edges > 0 {
            num_edges as f64 / max_edges as f64
        } else {
            0.0
        }
    }

    fn calculate_distribution_stats(&self, values: &[f64]) -> DistributionStats {
        if values.is_empty() {
            return DistributionStats::default();
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        DistributionStats {
            mean,
            std_dev,
            min,
            max,
            skewness: 0.0, // Simplified
            kurtosis: 3.0, // Simplified
        }
    }

    pub fn vectorize_features(&self, features: &ProblemFeatures) -> FeatureVector {
        let mut values = Vec::new();
        let mut names = Vec::new();

        // Basic features
        values.push(features.size as f64);
        names.push("size".to_string());
        values.push(features.density);
        names.push("density".to_string());

        // Graph features
        if self.config.enable_graph_features {
            values.push(features.graph_features.num_vertices as f64);
            names.push("num_vertices".to_string());
            values.push(features.graph_features.num_edges as f64);
            names.push("num_edges".to_string());
            values.push(features.graph_features.avg_degree);
            names.push("avg_degree".to_string());
            values.push(features.graph_features.clustering_coefficient);
            names.push("clustering_coefficient".to_string());
        }

        // Statistical features
        if self.config.enable_statistical_features {
            values.push(features.statistical_features.bias_stats.mean);
            names.push("bias_mean".to_string());
            values.push(features.statistical_features.bias_stats.std_dev);
            names.push("bias_std".to_string());
            values.push(features.statistical_features.coupling_stats.mean);
            names.push("coupling_mean".to_string());
            values.push(features.statistical_features.coupling_stats.std_dev);
            names.push("coupling_std".to_string());
        }

        // Spectral features
        if self.config.enable_spectral_features {
            values.push(features.spectral_features.spectral_gap);
            names.push("spectral_gap".to_string());
            values.push(features.spectral_features.spectral_radius);
            names.push("spectral_radius".to_string());
            values.push(features.spectral_features.condition_number);
            names.push("condition_number".to_string());
        }

        FeatureVector {
            values,
            names,
            normalization: None,
        }
    }
}

/// Feature transformer
#[derive(Debug)]
pub struct FeatureTransformer {
    /// Transformer type
    pub transformer_type: TransformerType,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Fitted state
    pub is_fitted: bool,
}

/// Transformer types
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerType {
    /// Polynomial features
    Polynomial,
    /// Interaction features
    Interaction,
    /// Logarithmic transform
    Logarithmic,
    /// Box-Cox transform
    BoxCox,
    /// Custom transform
    Custom(String),
}

/// Feature selector
#[derive(Debug)]
pub struct FeatureSelector {
    /// Selection method
    pub method: FeatureSelectionMethod,
    /// Selected features
    pub selected_features: Vec<usize>,
    /// Feature importance scores
    pub importance_scores: Vec<f64>,
}

/// Dimensionality reducer
#[derive(Debug)]
pub struct DimensionalityReducer {
    /// Reduction method
    pub method: DimensionalityReduction,
    /// Target dimensions
    pub target_dims: usize,
    /// Transformation matrix
    pub transformation_matrix: Option<Vec<Vec<f64>>>,
    /// Explained variance
    pub explained_variance: Vec<f64>,
}

/// Experience database for storing optimization experiences
pub struct ExperienceDatabase {
    /// Stored experiences
    pub experiences: VecDeque<OptimizationExperience>,
    /// Index for fast retrieval
    pub index: ExperienceIndex,
    /// Similarity cache
    pub similarity_cache: HashMap<String, Vec<(String, f64)>>,
    /// Statistics
    pub statistics: DatabaseStatistics,
}

/// Optimization experience record
#[derive(Debug, Clone)]
pub struct OptimizationExperience {
    /// Unique experience identifier
    pub id: String,
    /// Problem characteristics
    pub problem_features: ProblemFeatures,
    /// Configuration used
    pub configuration: OptimizationConfiguration,
    /// Results achieved
    pub results: OptimizationResults,
    /// Timestamp
    pub timestamp: Instant,
    /// Problem domain
    pub domain: ProblemDomain,
    /// Success metrics
    pub success_metrics: SuccessMetrics,
}

use std::time::Duration;

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfiguration {
    /// Algorithm used
    pub algorithm: AlgorithmType,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, f64>,
    /// Architecture specification
    pub architecture: Option<ArchitectureSpec>,
    /// Resource allocation
    pub resources: ResourceAllocation,
}

/// Algorithm types
#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmType {
    /// Simulated annealing
    SimulatedAnnealing,
    /// Quantum annealing
    QuantumAnnealing,
    /// Tabu search
    TabuSearch,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Ant colony optimization
    AntColony,
    /// Variable neighborhood search
    VariableNeighborhood,
    /// Hybrid algorithm
    Hybrid(Vec<AlgorithmType>),
}

/// Architecture specification
#[derive(Debug, Clone)]
pub struct ArchitectureSpec {
    /// Layer specifications
    pub layers: Vec<LayerSpec>,
    /// Connection pattern
    pub connections: ConnectionPattern,
    /// Optimization settings
    pub optimization: OptimizationSettings,
}

/// Layer specification
#[derive(Debug, Clone)]
pub struct LayerSpec {
    /// Layer type
    pub layer_type: LayerType,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationFunction,
    /// Dropout rate
    pub dropout: f64,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

/// Connection patterns
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionPattern {
    /// Sequential connections
    Sequential,
    /// Skip connections
    SkipConnections,
    /// Dense connections
    DenseConnections,
    /// Residual connections
    ResidualConnections,
    /// Custom pattern
    Custom(Vec<(usize, usize)>),
}

/// Optimization settings
#[derive(Debug, Clone)]
pub struct OptimizationSettings {
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Regularization
    pub regularization: RegularizationConfig,
}

/// Optimizer types
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    LBFGS,
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L1 regularization weight
    pub l1_weight: f64,
    /// L2 regularization weight
    pub l2_weight: f64,
    /// Dropout rate
    pub dropout: f64,
    /// Batch normalization
    pub batch_norm: bool,
    /// Early stopping
    pub early_stopping: bool,
}

/// Resource allocation
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// CPU allocation
    pub cpu: f64,
    /// Memory allocation (MB)
    pub memory: usize,
    /// GPU allocation
    pub gpu: f64,
    /// Time allocation
    pub time: Duration,
}

/// Optimization results
#[derive(Debug, Clone)]
pub struct OptimizationResults {
    /// Final objective values
    pub objective_values: Vec<f64>,
    /// Execution time
    pub execution_time: Duration,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Convergence metrics
    pub convergence: ConvergenceMetrics,
    /// Solution quality metrics
    pub quality_metrics: QualityMetrics,
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

/// Convergence metrics
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Number of iterations
    pub iterations: usize,
    /// Final convergence rate
    pub convergence_rate: f64,
    /// Plateau detection
    pub plateau_detected: bool,
    /// Convergence confidence
    pub confidence: f64,
}

/// Solution quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Objective function value
    pub objective_value: f64,
    /// Constraint violation
    pub constraint_violation: f64,
    /// Robustness score
    pub robustness: f64,
    /// Diversity score
    pub diversity: f64,
}

/// Problem domains
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

/// Success metrics
#[derive(Debug, Clone)]
pub struct SuccessMetrics {
    /// Overall success score
    pub success_score: f64,
    /// Performance relative to baseline
    pub relative_performance: f64,
    /// User satisfaction score
    pub user_satisfaction: f64,
    /// Recommendation confidence
    pub recommendation_confidence: f64,
}

/// Experience indexing system
#[derive(Debug)]
pub struct ExperienceIndex {
    /// Domain-based index
    pub domain_index: HashMap<ProblemDomain, Vec<String>>,
    /// Size-based index
    pub size_index: BTreeMap<usize, Vec<String>>,
    /// Performance-based index
    pub performance_index: BTreeMap<String, Vec<String>>,
    /// Feature-based index
    pub feature_index: HashMap<String, Vec<String>>,
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStatistics {
    /// Total experiences
    pub total_experiences: usize,
    /// Experiences per domain
    pub domain_distribution: HashMap<ProblemDomain, usize>,
    /// Average performance
    pub avg_performance: f64,
    /// Coverage statistics
    pub coverage_stats: CoverageStatistics,
}

/// Coverage statistics
#[derive(Debug, Clone)]
pub struct CoverageStatistics {
    /// Feature space coverage
    pub feature_coverage: f64,
    /// Problem size coverage
    pub size_coverage: (usize, usize),
    /// Domain coverage
    pub domain_coverage: f64,
    /// Performance range coverage
    pub performance_range: (f64, f64),
}

impl ExperienceDatabase {
    pub fn new() -> Self {
        Self {
            experiences: VecDeque::new(),
            index: ExperienceIndex {
                domain_index: HashMap::new(),
                size_index: BTreeMap::new(),
                performance_index: BTreeMap::new(),
                feature_index: HashMap::new(),
            },
            similarity_cache: HashMap::new(),
            statistics: DatabaseStatistics {
                total_experiences: 0,
                domain_distribution: HashMap::new(),
                avg_performance: 0.0,
                coverage_stats: CoverageStatistics {
                    feature_coverage: 0.0,
                    size_coverage: (0, 0),
                    domain_coverage: 0.0,
                    performance_range: (0.0, 1.0),
                },
            },
        }
    }

    pub fn add_experience(&mut self, experience: OptimizationExperience) {
        self.experiences.push_back(experience.clone());
        self.update_index(&experience);
        self.update_statistics();

        // Limit buffer size
        if self.experiences.len() > 10000 {
            if let Some(removed) = self.experiences.pop_front() {
                self.remove_from_index(&removed);
            }
        }
    }

    fn update_index(&mut self, experience: &OptimizationExperience) {
        // Update domain index
        self.index
            .domain_index
            .entry(experience.domain.clone())
            .or_insert_with(Vec::new)
            .push(experience.id.clone());

        // Update size index
        self.index
            .size_index
            .entry(experience.problem_features.size)
            .or_insert_with(Vec::new)
            .push(experience.id.clone());
    }

    fn remove_from_index(&mut self, experience: &OptimizationExperience) {
        // Remove from domain index
        if let Some(ids) = self.index.domain_index.get_mut(&experience.domain) {
            ids.retain(|id| id != &experience.id);
        }

        // Remove from size index
        if let Some(ids) = self
            .index
            .size_index
            .get_mut(&experience.problem_features.size)
        {
            ids.retain(|id| id != &experience.id);
        }
    }

    fn update_statistics(&mut self) {
        self.statistics.total_experiences = self.experiences.len();

        if !self.experiences.is_empty() {
            let total_performance: f64 = self
                .experiences
                .iter()
                .map(|exp| exp.results.quality_metrics.objective_value)
                .sum();
            self.statistics.avg_performance = total_performance / self.experiences.len() as f64;
        }

        // Update domain distribution
        self.statistics.domain_distribution.clear();
        for experience in &self.experiences {
            *self
                .statistics
                .domain_distribution
                .entry(experience.domain.clone())
                .or_insert(0) += 1;
        }
    }

    pub fn find_similar_experiences(
        &self,
        features: &ProblemFeatures,
        limit: usize,
    ) -> ApplicationResult<Vec<OptimizationExperience>> {
        let mut similarities = Vec::new();

        for experience in &self.experiences {
            let similarity = self.calculate_similarity(features, &experience.problem_features);
            similarities.push((experience.clone(), similarity));
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities
            .into_iter()
            .take(limit)
            .map(|(exp, _)| exp)
            .collect())
    }

    fn calculate_similarity(
        &self,
        features1: &ProblemFeatures,
        features2: &ProblemFeatures,
    ) -> f64 {
        // Simple similarity calculation based on size and density
        let size_diff = (features1.size as f64 - features2.size as f64).abs()
            / features1.size.max(features2.size) as f64;
        let density_diff = (features1.density - features2.density).abs();

        let size_similarity = 1.0 - size_diff;
        let density_similarity = 1.0 - density_diff;

        (size_similarity + density_similarity) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureExtractionConfig::default();
        let extractor = FeatureExtractor::new(config);
        assert!(extractor.config.enable_graph_features);
    }

    #[test]
    fn test_distribution_stats() {
        let stats = DistributionStats::default();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std_dev, 1.0);
    }

    #[test]
    fn test_experience_database() {
        let db = ExperienceDatabase::new();
        assert_eq!(db.statistics.total_experiences, 0);
    }
}
