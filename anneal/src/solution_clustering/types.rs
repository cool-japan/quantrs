//! Core data structures for solution clustering

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Solution representation for clustering
#[derive(Debug, Clone)]
pub struct SolutionPoint {
    /// Solution vector (spin configuration)
    pub solution: Vec<i8>,
    /// Energy of the solution
    pub energy: f64,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
    /// Solution metadata
    pub metadata: SolutionMetadata,
    /// Feature vector for clustering
    pub features: Option<Vec<f64>>,
}

/// Solution metadata
#[derive(Debug, Clone)]
pub struct SolutionMetadata {
    /// Solution ID
    pub id: usize,
    /// Source algorithm or run
    pub source: String,
    /// Timestamp when solution was found
    pub timestamp: Instant,
    /// Number of iterations to find this solution
    pub iterations: usize,
    /// Quality rank among all solutions
    pub quality_rank: Option<usize>,
    /// Feasibility status
    pub is_feasible: bool,
}

/// Cluster representation
#[derive(Debug, Clone)]
pub struct SolutionCluster {
    /// Cluster ID
    pub id: usize,
    /// Solutions in this cluster
    pub solutions: Vec<SolutionPoint>,
    /// Cluster centroid
    pub centroid: Vec<f64>,
    /// Representative solution (closest to centroid)
    pub representative: Option<SolutionPoint>,
    /// Cluster statistics
    pub statistics: ClusterStatistics,
    /// Cluster quality metrics
    pub quality_metrics: ClusterQualityMetrics,
}

/// Cluster statistics
#[derive(Debug, Clone)]
pub struct ClusterStatistics {
    /// Number of solutions in cluster
    pub size: usize,
    /// Mean energy
    pub mean_energy: f64,
    /// Energy standard deviation
    pub energy_std: f64,
    /// Minimum energy in cluster
    pub min_energy: f64,
    /// Maximum energy in cluster
    pub max_energy: f64,
    /// Intra-cluster distance (compactness)
    pub intra_cluster_distance: f64,
    /// Cluster diameter (maximum distance between any two points)
    pub diameter: f64,
    /// Cluster density
    pub density: f64,
}

/// Cluster quality metrics
#[derive(Debug, Clone)]
pub struct ClusterQualityMetrics {
    /// Silhouette coefficient
    pub silhouette_coefficient: f64,
    /// Inertia (within-cluster sum of squares)
    pub inertia: f64,
    /// Calinski-Harabasz index
    pub calinski_harabasz_index: f64,
    /// Davies-Bouldin index
    pub davies_bouldin_index: f64,
    /// Cluster stability measure
    pub stability: f64,
}

/// Clustering results containing all clusters and analysis
#[derive(Debug, Clone)]
pub struct ClusteringResults {
    /// All clusters found
    pub clusters: Vec<SolutionCluster>,
    /// Clustering algorithm used
    pub algorithm: super::algorithms::ClusteringAlgorithm,
    /// Distance metric used
    pub distance_metric: super::algorithms::DistanceMetric,
    /// Overall clustering quality
    pub overall_quality: OverallClusteringQuality,
    /// Landscape analysis
    pub landscape_analysis: LandscapeAnalysis,
    /// Statistical summary
    pub statistical_summary: StatisticalSummary,
    /// Clustering performance metrics
    pub performance_metrics: ClusteringPerformanceMetrics,
    /// Recommendations for optimization
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Overall clustering quality assessment
#[derive(Debug, Clone)]
pub struct OverallClusteringQuality {
    /// Overall silhouette score
    pub silhouette_score: f64,
    /// Adjusted Rand Index (if ground truth available)
    pub adjusted_rand_index: Option<f64>,
    /// Normalized Mutual Information
    pub normalized_mutual_information: Option<f64>,
    /// Inter-cluster separation
    pub inter_cluster_separation: f64,
    /// Cluster cohesion
    pub cluster_cohesion: f64,
    /// Number of clusters found
    pub num_clusters: usize,
    /// Optimal number of clusters estimate
    pub optimal_num_clusters: usize,
}

/// Landscape analysis results
#[derive(Debug, Clone)]
pub struct LandscapeAnalysis {
    /// Energy landscape statistics
    pub energy_statistics: EnergyStatistics,
    /// Basin detection results
    pub basins: Vec<EnergyBasin>,
    /// Connectivity analysis
    pub connectivity: ConnectivityAnalysis,
    /// Multi-modality assessment
    pub multi_modality: MultiModalityAnalysis,
    /// Ruggedness measures
    pub ruggedness: RuggednessMetrics,
    /// Funnel structure analysis
    pub funnel_analysis: FunnelAnalysis,
}

/// Energy statistics across the solution set
#[derive(Debug, Clone)]
pub struct EnergyStatistics {
    /// Mean energy
    pub mean: f64,
    /// Energy standard deviation
    pub std_dev: f64,
    /// Minimum energy found
    pub min: f64,
    /// Maximum energy found
    pub max: f64,
    /// Energy distribution percentiles
    pub percentiles: Vec<f64>,
    /// Skewness of energy distribution
    pub skewness: f64,
    /// Kurtosis of energy distribution
    pub kurtosis: f64,
    /// Number of distinct energy levels
    pub num_distinct_energies: usize,
}

/// Energy basin in the landscape
#[derive(Debug, Clone)]
pub struct EnergyBasin {
    /// Basin ID
    pub id: usize,
    /// Solutions in this basin
    pub solutions: Vec<usize>,
    /// Basin minimum energy
    pub min_energy: f64,
    /// Basin size (number of solutions)
    pub size: usize,
    /// Basin depth (relative to global minimum)
    pub depth: f64,
    /// Basin width (energy range)
    pub width: f64,
    /// Escape barrier height
    pub escape_barrier: f64,
}

/// Connectivity analysis of the solution landscape
#[derive(Debug, Clone)]
pub struct ConnectivityAnalysis {
    /// Number of connected components
    pub num_components: usize,
    /// Largest connected component size
    pub largest_component_size: usize,
    /// Average path length between solutions
    pub average_path_length: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Network diameter
    pub diameter: usize,
}

/// Multi-modality analysis
#[derive(Debug, Clone)]
pub struct MultiModalityAnalysis {
    /// Number of modes detected
    pub num_modes: usize,
    /// Mode locations (energy values)
    pub mode_energies: Vec<f64>,
    /// Mode strengths (relative populations)
    pub mode_strengths: Vec<f64>,
    /// Inter-mode distances
    pub inter_mode_distances: Vec<Vec<f64>>,
    /// Multi-modality index
    pub multi_modality_index: f64,
}

/// Ruggedness metrics for the landscape
#[derive(Debug, Clone)]
pub struct RuggednessMetrics {
    /// Autocorrelation function
    pub autocorrelation: Vec<f64>,
    /// Ruggedness coefficient
    pub ruggedness_coefficient: f64,
    /// Number of local optima
    pub num_local_optima: usize,
    /// Epistasis measure
    pub epistasis: f64,
    /// Neutrality measure
    pub neutrality: f64,
}

/// Funnel structure analysis
#[derive(Debug, Clone)]
pub struct FunnelAnalysis {
    /// Number of funnels detected
    pub num_funnels: usize,
    /// Funnel depths
    pub funnel_depths: Vec<f64>,
    /// Funnel widths
    pub funnel_widths: Vec<f64>,
    /// Global funnel identification
    pub global_funnel: Option<usize>,
    /// Funnel competition index
    pub competition_index: f64,
}

/// Statistical summary of clustering results
#[derive(Debug, Clone)]
pub struct StatisticalSummary {
    /// Distribution of cluster sizes
    pub cluster_size_distribution: Vec<usize>,
    /// Energy distribution analysis
    pub energy_distribution: DistributionAnalysis,
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
    /// Correlation analysis
    pub correlation_analysis: CorrelationAnalysis,
    /// Outlier detection results
    pub outliers: Vec<OutlierInfo>,
}

/// Distribution analysis results
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    /// Distribution type detected
    pub distribution_type: DistributionType,
    /// Distribution parameters
    pub parameters: HashMap<String, f64>,
    /// Goodness of fit score
    pub goodness_of_fit: f64,
    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
}

/// Distribution types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionType {
    /// Normal distribution
    Normal,
    /// Exponential distribution
    Exponential,
    /// Gamma distribution
    Gamma,
    /// Beta distribution
    Beta,
    /// Weibull distribution
    Weibull,
    /// Log-normal distribution
    LogNormal,
    /// Uniform distribution
    Uniform,
    /// Multimodal distribution
    Multimodal,
    /// Unknown/custom distribution
    Unknown,
}

/// Convergence analysis results
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis {
    /// Convergence trajectory clusters
    pub trajectory_clusters: Vec<TrajectoryCluster>,
    /// Convergence rates by cluster
    pub convergence_rates: Vec<f64>,
    /// Plateau analysis
    pub plateau_analysis: PlateauAnalysis,
    /// Premature convergence detection
    pub premature_convergence: bool,
    /// Diversity evolution
    pub diversity_evolution: Vec<f64>,
}

/// Trajectory cluster for convergence analysis
#[derive(Debug, Clone)]
pub struct TrajectoryCluster {
    /// Cluster ID
    pub id: usize,
    /// Trajectory patterns in this cluster
    pub trajectories: Vec<Vec<f64>>,
    /// Representative trajectory
    pub representative_trajectory: Vec<f64>,
    /// Convergence characteristics
    pub convergence_characteristics: ConvergenceCharacteristics,
}

/// Convergence characteristics
#[derive(Debug, Clone)]
pub struct ConvergenceCharacteristics {
    /// Convergence speed
    pub speed: f64,
    /// Final convergence quality
    pub final_quality: f64,
    /// Stability measure
    pub stability: f64,
    /// Exploration vs exploitation balance
    pub exploration_exploitation_ratio: f64,
}

/// Plateau analysis in convergence trajectories
#[derive(Debug, Clone)]
pub struct PlateauAnalysis {
    /// Number of plateaus detected
    pub num_plateaus: usize,
    /// Plateau durations
    pub plateau_durations: Vec<usize>,
    /// Plateau energy levels
    pub plateau_energies: Vec<f64>,
    /// Escape probabilities from plateaus
    pub escape_probabilities: Vec<f64>,
}

/// Correlation analysis results
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    /// Variable correlation matrix
    pub variable_correlations: Vec<Vec<f64>>,
    /// Energy-variable correlations
    pub energy_correlations: Vec<f64>,
    /// Significant correlations
    pub significant_correlations: Vec<(usize, usize, f64)>,
    /// Correlation patterns
    pub correlation_patterns: Vec<CorrelationPattern>,
}

/// Correlation patterns
#[derive(Debug, Clone)]
pub struct CorrelationPattern {
    /// Pattern description
    pub description: String,
    /// Variables involved
    pub variables: Vec<usize>,
    /// Pattern strength
    pub strength: f64,
    /// Pattern type
    pub pattern_type: PatternType,
}

/// Types of correlation patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternType {
    /// Positive correlation
    Positive,
    /// Negative correlation
    Negative,
    /// Non-linear correlation
    NonLinear,
    /// Conditional correlation
    Conditional,
    /// Cluster-specific correlation
    ClusterSpecific,
}

/// Outlier information
#[derive(Debug, Clone)]
pub struct OutlierInfo {
    /// Solution ID
    pub solution_id: usize,
    /// Outlier score
    pub outlier_score: f64,
    /// Outlier type
    pub outlier_type: OutlierType,
    /// Distance to nearest cluster
    pub distance_to_cluster: f64,
}

/// Types of outliers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutlierType {
    /// Energy outlier (unusually high/low energy)
    Energy,
    /// Structural outlier (unusual solution structure)
    Structural,
    /// Performance outlier (unusual algorithm performance)
    Performance,
    /// Global outlier (outlier in multiple dimensions)
    Global,
}

/// Clustering performance metrics
#[derive(Debug, Clone)]
pub struct ClusteringPerformanceMetrics {
    /// Clustering time
    pub clustering_time: Duration,
    /// Analysis time
    pub analysis_time: Duration,
    /// Memory usage
    pub memory_usage: usize,
    /// Scalability metrics
    pub scalability_metrics: ScalabilityMetrics,
    /// Algorithm efficiency
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Scalability metrics
#[derive(Debug, Clone)]
pub struct ScalabilityMetrics {
    /// Time complexity estimate
    pub time_complexity: String,
    /// Space complexity estimate
    pub space_complexity: String,
    /// Performance vs data size relationship
    pub scaling_factor: f64,
    /// Parallelization efficiency
    pub parallelization_efficiency: f64,
}

/// Algorithm efficiency metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// Convergence efficiency
    pub convergence_efficiency: f64,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Quality vs time trade-off
    pub quality_time_ratio: f64,
    /// Robustness measure
    pub robustness: f64,
}

/// Optimization recommendations based on clustering analysis
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommendation description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Implementation difficulty
    pub difficulty: DifficultyLevel,
    /// Priority level
    pub priority: PriorityLevel,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendationType {
    /// Parameter tuning recommendation
    ParameterTuning,
    /// Algorithm modification
    AlgorithmModification,
    /// Problem reformulation
    ProblemReformulation,
    /// Initialization strategy
    InitializationStrategy,
    /// Termination criteria
    TerminationCriteria,
    /// Hybrid approach
    HybridApproach,
    /// Multi-start strategy
    MultiStart,
    /// Constraint handling
    ConstraintHandling,
}

/// Difficulty levels for implementing recommendations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DifficultyLevel {
    /// Easy to implement
    Easy,
    /// Moderate implementation effort
    Moderate,
    /// Difficult implementation
    Difficult,
    /// Very difficult, requires significant changes
    VeryDifficult,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PriorityLevel {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Analysis statistics
#[derive(Debug, Clone)]
pub struct AnalysisStatistics {
    /// Total solutions analyzed
    pub total_solutions: usize,
    /// Total analysis time
    pub total_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Memory usage peak
    pub peak_memory: usize,
}
