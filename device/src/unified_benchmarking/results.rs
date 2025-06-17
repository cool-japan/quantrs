//! Result types for the unified benchmarking system

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::config::{
    CircuitType, MultiQubitGate, SingleQubitGate, TwoQubitGate, UnifiedBenchmarkConfig,
};
use super::types::QuantumPlatform;

/// Main unified benchmarking result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedBenchmarkResult {
    /// Unique benchmark execution ID
    pub execution_id: String,
    /// Execution timestamp
    pub timestamp: SystemTime,
    /// Configuration used
    pub config: UnifiedBenchmarkConfig,
    /// Platform-specific results
    pub platform_results: HashMap<QuantumPlatform, PlatformBenchmarkResult>,
    /// Cross-platform analysis
    pub cross_platform_analysis: CrossPlatformAnalysis,
    /// SciRS2 analysis results
    pub scirs2_analysis: SciRS2AnalysisResult,
    /// Resource utilization analysis
    pub resource_analysis: ResourceAnalysisResult,
    /// Cost analysis
    pub cost_analysis: CostAnalysisResult,
    /// Performance optimization recommendations
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    /// Historical comparison
    pub historical_comparison: Option<HistoricalComparisonResult>,
    /// Execution metadata
    pub execution_metadata: ExecutionMetadata,
}

/// Platform-specific benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformBenchmarkResult {
    pub platform: QuantumPlatform,
    pub device_info: DeviceInfo,
    pub gate_level_results: GateLevelResults,
    pub circuit_level_results: CircuitLevelResults,
    pub algorithm_level_results: AlgorithmLevelResults,
    pub system_level_results: SystemLevelResults,
    pub performance_metrics: PlatformPerformanceMetrics,
    pub reliability_metrics: ReliabilityMetrics,
    pub cost_metrics: CostMetrics,
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub device_id: String,
    pub provider: String,
    pub technology: QuantumTechnology,
    pub specifications: DeviceSpecifications,
    pub current_status: DeviceStatus,
    pub calibration_date: Option<SystemTime>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumTechnology {
    Superconducting,
    TrappedIon,
    Photonic,
    NeutralAtom,
    Topological,
    SpinQubit,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSpecifications {
    pub num_qubits: usize,
    pub connectivity: ConnectivityInfo,
    pub gate_set: Vec<String>,
    pub coherence_times: CoherenceTimes,
    pub gate_times: HashMap<String, Duration>,
    pub error_rates: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityInfo {
    pub topology_type: TopologyType,
    pub coupling_map: Vec<(usize, usize)>,
    pub connectivity_matrix: Array2<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopologyType {
    Linear,
    Ring,
    Grid,
    Heavy,
    AllToAll,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceTimes {
    pub t1: HashMap<usize, Duration>,
    pub t2: HashMap<usize, Duration>,
    pub t2_echo: HashMap<usize, Duration>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeviceStatus {
    Online,
    Offline,
    Maintenance,
    Degraded,
    Calibrating,
    Unknown,
}

/// Gate-level benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateLevelResults {
    pub single_qubit_results: HashMap<SingleQubitGate, GatePerformanceResult>,
    pub two_qubit_results: HashMap<TwoQubitGate, GatePerformanceResult>,
    pub multi_qubit_results: HashMap<MultiQubitGate, GatePerformanceResult>,
    pub randomized_benchmarking: RandomizedBenchmarkingResult,
    pub process_tomography: Option<ProcessTomographyResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatePerformanceResult {
    pub gate_type: String,
    pub fidelity: StatisticalSummary,
    pub execution_time: StatisticalSummary,
    pub error_rate: StatisticalSummary,
    pub success_rate: f64,
    pub measurements: Vec<GateMeasurement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: HashMap<u8, f64>,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateMeasurement {
    pub timestamp: SystemTime,
    pub fidelity: f64,
    pub execution_time: Duration,
    pub error_type: Option<String>,
    pub additional_data: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomizedBenchmarkingResult {
    pub clifford_fidelity: f64,
    pub decay_parameter: f64,
    pub confidence_interval: (f64, f64),
    pub sequence_lengths: Vec<usize>,
    pub survival_probabilities: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessTomographyResult {
    pub process_matrix: Array2<f64>,
    pub process_fidelity: f64,
    pub diamond_distance: f64,
    pub reconstruction_error: f64,
}

/// Circuit-level benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitLevelResults {
    pub depth_scaling: DepthScalingResult,
    pub width_scaling: WidthScalingResult,
    pub circuit_type_results: HashMap<CircuitType, CircuitTypeResult>,
    pub parametric_results: HashMap<String, ParametricResult>,
    pub volume_benchmarks: VolumeBenchmarkResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthScalingResult {
    pub depth_vs_fidelity: Vec<(usize, f64)>,
    pub depth_vs_execution_time: Vec<(usize, Duration)>,
    pub scaling_exponent: f64,
    pub coherence_limited_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidthScalingResult {
    pub width_vs_fidelity: Vec<(usize, f64)>,
    pub width_vs_execution_time: Vec<(usize, Duration)>,
    pub scaling_exponent: f64,
    pub connectivity_limited_width: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitTypeResult {
    pub circuit_type: CircuitType,
    pub performance_metrics: CircuitPerformanceMetrics,
    pub optimization_effectiveness: f64,
    pub resource_utilization: CircuitResourceUtilization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitPerformanceMetrics {
    pub fidelity: StatisticalSummary,
    pub execution_time: StatisticalSummary,
    pub success_probability: f64,
    pub depth_overhead: f64,
    pub gate_overhead: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitResourceUtilization {
    pub cpu_time: Duration,
    pub memory_usage: f64,
    pub shots_used: usize,
    pub queue_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricResult {
    pub parameter_name: String,
    pub parameter_values: Vec<f64>,
    pub performance_data: Vec<CircuitPerformanceMetrics>,
    pub optimal_parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeBenchmarkResult {
    pub heavy_output: HeavyOutputResult,
    pub cross_entropy: CrossEntropyResult,
    pub quantum_volume: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeavyOutputResult {
    pub heavy_output_probability: f64,
    pub theoretical_threshold: f64,
    pub statistical_significance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEntropyResult {
    pub cross_entropy_benchmarking_fidelity: f64,
    pub linear_xeb_fidelity: f64,
    pub log_xeb_fidelity: f64,
}

/// Algorithm-level benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmLevelResults {
    pub algorithm_results: HashMap<String, AlgorithmResult>,
    pub nisq_performance: NISQPerformanceResult,
    pub quantum_advantage: QuantumAdvantageResult,
    pub classical_comparison: ClassicalComparisonResult,
    pub variational_algorithm_performance: VariationalAlgorithmResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmResult {
    pub algorithm_name: String,
    pub problem_sizes: Vec<usize>,
    pub success_rates: Vec<f64>,
    pub execution_times: Vec<Duration>,
    pub fidelities: Vec<f64>,
    pub resource_requirements: Vec<ResourceRequirement>,
    pub scalability_analysis: ScalabilityAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub problem_size: usize,
    pub qubits_required: usize,
    pub circuit_depth: usize,
    pub gate_count: HashMap<String, usize>,
    pub shots_needed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityAnalysis {
    pub polynomial_fit: PolynomialFit,
    pub exponential_fit: ExponentialFit,
    pub complexity_class: ComplexityClass,
    pub quantum_advantage_threshold: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialFit {
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub predicted_scaling: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialFit {
    pub base: f64,
    pub exponent: f64,
    pub r_squared: f64,
    pub predicted_scaling: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComplexityClass {
    Constant,
    Logarithmic,
    Linear,
    Quadratic,
    Polynomial(u32),
    Exponential,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NISQPerformanceResult {
    pub noise_resilience: f64,
    pub error_mitigation_effectiveness: f64,
    pub depth_limited_performance: HashMap<usize, f64>,
    pub variational_optimization_convergence: ConvergenceAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantageResult {
    pub advantage_demonstrated: bool,
    pub speedup_factor: Option<f64>,
    pub confidence_level: f64,
    pub problem_instances_tested: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalComparisonResult {
    pub classical_runtime: Duration,
    pub quantum_runtime: Duration,
    pub speedup_ratio: f64,
    pub accuracy_comparison: AccuracyComparison,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyComparison {
    pub classical_accuracy: f64,
    pub quantum_accuracy: f64,
    pub relative_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalAlgorithmResult {
    pub optimization_landscapes: HashMap<String, OptimizationLandscape>,
    pub convergence_analysis: ConvergenceAnalysis,
    pub parameter_sensitivity: ParameterSensitivityAnalysis,
    pub barren_plateau_analysis: BarrenPlateauAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationLandscape {
    pub parameter_space: Array2<f64>,
    pub cost_surface: Array2<f64>,
    pub local_minima: Vec<(Vec<f64>, f64)>,
    pub global_minimum: (Vec<f64>, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysis {
    pub convergence_achieved: bool,
    pub iterations_to_convergence: Option<usize>,
    pub final_cost: f64,
    pub cost_history: Vec<f64>,
    pub gradient_norms: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSensitivityAnalysis {
    pub sensitivity_matrix: Array2<f64>,
    pub most_sensitive_parameters: Vec<usize>,
    pub robustness_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrenPlateauAnalysis {
    pub plateau_detected: bool,
    pub gradient_variance: f64,
    pub effective_dimension: f64,
    pub mitigation_strategies: Vec<String>,
}

/// System-level benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemLevelResults {
    pub cross_platform_comparison: CrossPlatformComparison,
    pub resource_utilization: SystemResourceUtilization,
    pub reliability_analysis: SystemReliabilityAnalysis,
    pub scalability_analysis: SystemScalabilityAnalysis,
    pub cost_efficiency: SystemCostEfficiency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformComparison {
    pub platform_rankings: Vec<PlatformRanking>,
    pub relative_performance: HashMap<String, f64>,
    pub statistical_significance: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformRanking {
    pub platform: QuantumPlatform,
    pub overall_score: f64,
    pub category_scores: HashMap<String, f64>,
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceUtilization {
    pub average_queue_time: Duration,
    pub throughput: f64,
    pub utilization_rate: f64,
    pub peak_usage_times: Vec<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemReliabilityAnalysis {
    pub uptime: f64,
    pub error_frequency: f64,
    pub recovery_time: Duration,
    pub failure_patterns: Vec<FailurePattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePattern {
    pub failure_type: String,
    pub frequency: f64,
    pub typical_duration: Duration,
    pub impact_severity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemScalabilityAnalysis {
    pub max_supported_qubits: usize,
    pub max_circuit_depth: usize,
    pub performance_scaling: HashMap<String, ScalingMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingMetric {
    pub metric_name: String,
    pub scaling_function: String,
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemCostEfficiency {
    pub cost_per_shot: f64,
    pub cost_per_gate: f64,
    pub cost_efficiency_score: f64,
    pub roi_analysis: ROIAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROIAnalysis {
    pub investment_cost: f64,
    pub operational_cost: f64,
    pub performance_benefit: f64,
    pub roi_ratio: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformPerformanceMetrics {
    pub overall_fidelity: f64,
    pub average_execution_time: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub availability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    pub uptime: f64,
    pub mtbf: Duration, // Mean Time Between Failures
    pub mttr: Duration, // Mean Time To Recovery
    pub availability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMetrics {
    pub total_cost: f64,
    pub cost_per_shot: f64,
    pub cost_per_hour: f64,
    pub cost_breakdown: HashMap<String, f64>,
}

/// Analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformAnalysis {
    pub platform_comparison: HashMap<String, f64>,
    pub best_platform_per_metric: HashMap<String, QuantumPlatform>,
    pub statistical_significance_tests: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciRS2AnalysisResult {
    pub statistical_analysis: StatisticalAnalysisResult,
    pub ml_analysis: MLAnalysisResult,
    pub optimization_analysis: OptimizationAnalysisResult,
    pub graph_analysis: GraphAnalysisResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisResult {
    pub hypothesis_tests: Vec<HypothesisTestResult>,
    pub correlation_analysis: CorrelationAnalysisResult,
    pub regression_analysis: RegressionAnalysisResult,
    pub time_series_analysis: TimeSeriesAnalysisResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    pub test_name: String,
    pub p_value: f64,
    pub statistic: f64,
    pub critical_value: f64,
    pub significant: bool,
    pub effect_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysisResult {
    pub correlation_matrix: Array2<f64>,
    pub significant_correlations: Vec<CorrelationPair>,
    pub partial_correlations: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPair {
    pub variable1: String,
    pub variable2: String,
    pub correlation: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysisResult {
    pub linear_regression: LinearRegressionResult,
    pub nonlinear_regression: NonlinearRegressionResult,
    pub model_comparison: ModelComparisonResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionResult {
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
    pub p_values: Vec<f64>,
    pub residuals: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonlinearRegressionResult {
    pub model_type: String,
    pub parameters: Vec<f64>,
    pub r_squared: f64,
    pub mse: f64,
    pub convergence_achieved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonResult {
    pub aic_scores: HashMap<String, f64>,
    pub bic_scores: HashMap<String, f64>,
    pub cross_validation_scores: HashMap<String, f64>,
    pub best_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesAnalysisResult {
    pub trend_analysis: TrendAnalysisResult,
    pub seasonality_analysis: SeasonalityAnalysisResult,
    pub stationarity_tests: StationarityTestResults,
    pub forecasting: ForecastingResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisResult {
    pub trend_detected: bool,
    pub trend_direction: String,
    pub trend_strength: f64,
    pub trend_coefficients: Vec<f64>,
    pub change_points: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityAnalysisResult {
    pub seasonal_components: Vec<f64>,
    pub seasonal_period: usize,
    pub seasonal_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityTestResults {
    pub adf_test: HypothesisTestResult,
    pub kpss_test: HypothesisTestResult,
    pub is_stationary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingResults {
    pub forecasts: Vec<f64>,
    pub confidence_intervals: Vec<(f64, f64)>,
    pub forecast_horizon: usize,
    pub model_performance: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLAnalysisResult {
    pub clustering_results: ClusteringResults,
    pub classification_results: ClassificationResults,
    pub regression_results: MLRegressionResults,
    pub anomaly_detection: AnomalyDetectionResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResults {
    pub cluster_assignments: Vec<usize>,
    pub cluster_centers: Array2<f64>,
    pub silhouette_score: f64,
    pub inertia: f64,
    pub optimal_clusters: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResults {
    pub model_accuracy: f64,
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub f1_score: Vec<f64>,
    pub confusion_matrix: Array2<f64>,
    pub feature_importance: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLRegressionResults {
    pub models: Vec<MLModelResult>,
    pub ensemble_result: EnsembleResult,
    pub cross_validation: CrossValidationResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelResult {
    pub model_type: String,
    pub mse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub predictions: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    pub ensemble_mse: f64,
    pub ensemble_mae: f64,
    pub ensemble_r_squared: f64,
    pub model_weights: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResult {
    pub cv_scores: Vec<f64>,
    pub mean_cv_score: f64,
    pub std_cv_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResults {
    pub anomaly_scores: Vec<f64>,
    pub anomaly_labels: Vec<bool>,
    pub anomaly_count: usize,
    pub feature_importance: FeatureImportanceResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceResults {
    pub importance_scores: Vec<f64>,
    pub feature_names: Vec<String>,
    pub ranked_features: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAnalysisResult {
    pub optimization_results: Vec<OptimizationResult>,
    pub pareto_analysis: ParetoAnalysisResult,
    pub sensitivity_analysis: SensitivityAnalysisResult,
    pub robustness_analysis: RobustnessAnalysisResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub objective_function: String,
    pub optimal_solution: Vec<f64>,
    pub optimal_value: f64,
    pub convergence_history: Vec<f64>,
    pub optimization_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoAnalysisResult {
    pub pareto_front: Vec<Vec<f64>>,
    pub pareto_solutions: Vec<Vec<f64>>,
    pub hypervolume: f64,
    pub spread_metric: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysisResult {
    pub sensitivity_indices: Vec<f64>,
    pub total_sensitivity_indices: Vec<f64>,
    pub interaction_effects: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessAnalysisResult {
    pub robustness_score: f64,
    pub stability_analysis: StabilityAnalysis,
    pub uncertainty_propagation: UncertaintyPropagation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityAnalysis {
    pub stability_score: f64,
    pub perturbation_analysis: Vec<PerturbationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationResult {
    pub perturbation_magnitude: f64,
    pub output_change: f64,
    pub stability_metric: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyPropagation {
    pub input_uncertainties: Vec<f64>,
    pub output_uncertainty: f64,
    pub uncertainty_contributions: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysisResult {
    pub connectivity_analysis: ConnectivityAnalysisResult,
    pub centrality_analysis: CentralityAnalysisResult,
    pub community_detection: CommunityDetectionResult,
    pub topology_optimization: TopologyOptimizationResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityAnalysisResult {
    pub connectivity_matrix: Array2<f64>,
    pub path_lengths: Array2<f64>,
    pub clustering_coefficient: f64,
    pub graph_density: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityAnalysisResult {
    pub betweenness_centrality: Vec<f64>,
    pub closeness_centrality: Vec<f64>,
    pub eigenvector_centrality: Vec<f64>,
    pub pagerank: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionResult {
    pub community_assignments: Vec<usize>,
    pub modularity: f64,
    pub num_communities: usize,
    pub community_sizes: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyOptimizationResult {
    pub optimal_topology: Array2<f64>,
    pub optimization_objective: f64,
    pub improvement_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAnalysisResult {
    pub cpu_utilization: ResourceUtilizationMetrics,
    pub memory_utilization: ResourceUtilizationMetrics,
    pub network_utilization: ResourceUtilizationMetrics,
    pub storage_utilization: ResourceUtilizationMetrics,
    pub capacity_planning: CapacityPlanningResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    pub average_utilization: f64,
    pub peak_utilization: f64,
    pub utilization_distribution: Vec<f64>,
    pub efficiency_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityPlanningResult {
    pub current_capacity: f64,
    pub projected_demand: Vec<f64>,
    pub capacity_recommendations: Vec<CapacityRecommendation>,
    pub scaling_timeline: Vec<(SystemTime, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityRecommendation {
    pub resource_type: String,
    pub recommended_capacity: f64,
    pub timeline: Duration,
    pub cost_estimate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysisResult {
    pub total_cost: f64,
    pub cost_breakdown: HashMap<String, f64>,
    pub cost_per_metric: HashMap<String, f64>,
    pub cost_optimization: CostOptimizationAnalysisResult,
    pub roi_analysis: ROIAnalysisResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationAnalysisResult {
    pub potential_savings: f64,
    pub optimization_strategies: Vec<CostOptimizationStrategy>,
    pub implementation_roadmap: Vec<OptimizationStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationStrategy {
    pub strategy_name: String,
    pub potential_savings: f64,
    pub implementation_cost: f64,
    pub payback_period: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStep {
    pub step_description: String,
    pub timeline: Duration,
    pub cost: f64,
    pub expected_savings: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROIAnalysisResult {
    pub roi_percentage: f64,
    pub payback_period: Duration,
    pub net_present_value: f64,
    pub break_even_analysis: BreakEvenAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakEvenAnalysis {
    pub break_even_point: Duration,
    pub break_even_volume: f64,
    pub sensitivity_analysis: Vec<SensitivityFactor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityFactor {
    pub factor_name: String,
    pub impact_on_breakeven: f64,
    pub uncertainty_range: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_type: String,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_effort: String,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalComparisonResult {
    pub baseline_comparison: Vec<MetricComparison>,
    pub trend_analysis: HashMap<String, TrendAnalysisResult>,
    pub performance_evolution: Vec<PerformanceSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub metric_name: String,
    pub current_value: f64,
    pub baseline_value: f64,
    pub percentage_change: f64,
    pub statistical_significance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: SystemTime,
    pub metrics: HashMap<String, f64>,
    pub configuration: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    pub execution_start_time: SystemTime,
    pub execution_end_time: SystemTime,
    pub total_duration: Duration,
    pub platforms_tested: Vec<QuantumPlatform>,
    pub benchmarks_executed: usize,
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub hostname: String,
    pub operating_system: String,
    pub cpu_info: String,
    pub memory_total: u64,
    pub disk_space: u64,
    pub network_info: String,
}
