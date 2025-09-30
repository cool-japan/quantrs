//! Result types and data structures for mid-circuit measurements

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use scirs2_core::ndarray::{Array1, Array2};
use quantrs2_core::qubit::QubitId;
use serde::{Deserialize, Serialize};

/// Enhanced mid-circuit measurement execution result with SciRS2 analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidCircuitExecutionResult {
    /// Final quantum measurement results
    pub final_measurements: HashMap<String, usize>,
    /// Classical register states
    pub classical_registers: HashMap<String, Vec<u8>>,
    /// Mid-circuit measurement history
    pub measurement_history: Vec<MeasurementEvent>,
    /// Execution statistics
    pub execution_stats: ExecutionStats,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Error analysis
    pub error_analysis: Option<ErrorAnalysis>,
    /// Advanced analytics results
    pub analytics_results: AdvancedAnalyticsResults,
    /// Prediction results
    pub prediction_results: Option<MeasurementPredictionResults>,
    /// Optimization recommendations
    pub optimization_recommendations: OptimizationRecommendations,
    /// Adaptive learning insights
    pub adaptive_insights: AdaptiveLearningInsights,
}

/// Individual measurement event during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementEvent {
    /// Timestamp (microseconds from start)
    pub timestamp: f64,
    /// Measured qubit
    pub qubit: QubitId,
    /// Measurement result (0 or 1)
    pub result: u8,
    /// Classical bit/register where result was stored
    pub storage_location: StorageLocation,
    /// Measurement latency (microseconds)
    pub latency: f64,
    /// Confidence/fidelity of measurement
    pub confidence: f64,
}

/// Location where measurement result is stored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageLocation {
    /// Classical bit index
    ClassicalBit(usize),
    /// Classical register and bit index
    ClassicalRegister(String, usize),
    /// Temporary buffer
    Buffer(usize),
}

/// Execution statistics for mid-circuit measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Total execution time
    pub total_execution_time: Duration,
    /// Time spent on quantum operations
    pub quantum_time: Duration,
    /// Time spent on measurements
    pub measurement_time: Duration,
    /// Time spent on classical processing
    pub classical_time: Duration,
    /// Number of mid-circuit measurements
    pub num_measurements: usize,
    /// Number of conditional operations
    pub num_conditional_ops: usize,
    /// Average measurement latency
    pub avg_measurement_latency: f64,
    /// Maximum measurement latency
    pub max_measurement_latency: f64,
}

/// Performance metrics for mid-circuit measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Measurement success rate
    pub measurement_success_rate: f64,
    /// Classical processing efficiency
    pub classical_efficiency: f64,
    /// Overall circuit fidelity
    pub circuit_fidelity: f64,
    /// Measurement error rate
    pub measurement_error_rate: f64,
    /// Timing overhead compared to no measurements
    pub timing_overhead: f64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// Quantum resource usage (0-1)
    pub quantum_utilization: f64,
    /// Classical resource usage (0-1)
    pub classical_utilization: f64,
    /// Memory usage for classical data
    pub memory_usage: usize,
    /// Communication overhead
    pub communication_overhead: f64,
}

/// Error analysis for mid-circuit measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    /// Measurement errors by qubit
    pub measurement_errors: HashMap<QubitId, MeasurementErrorStats>,
    /// Classical processing errors
    pub classical_errors: Vec<ClassicalError>,
    /// Timing violations
    pub timing_violations: Vec<TimingViolation>,
    /// Correlation analysis
    pub error_correlations: Array2<f64>,
}

/// Measurement error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementErrorStats {
    /// Readout error rate
    pub readout_error_rate: f64,
    /// State preparation and measurement (SPAM) error
    pub spam_error: f64,
    /// Thermal relaxation during measurement
    pub thermal_relaxation: f64,
    /// Dephasing during measurement
    pub dephasing: f64,
}

/// Classical processing error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalError {
    /// Error type
    pub error_type: ClassicalErrorType,
    /// Timestamp when error occurred
    pub timestamp: f64,
    /// Error description
    pub description: String,
    /// Affected operations
    pub affected_operations: Vec<usize>,
}

/// Types of classical errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassicalErrorType {
    /// Timeout in classical condition evaluation
    Timeout,
    /// Invalid register access
    InvalidRegisterAccess,
    /// Condition evaluation error
    ConditionEvaluationError,
    /// Buffer overflow
    BufferOverflow,
    /// Communication error
    CommunicationError,
}

/// Timing constraint violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingViolation {
    /// Operation that violated timing
    pub operation_index: usize,
    /// Expected timing (microseconds)
    pub expected_timing: f64,
    /// Actual timing (microseconds)
    pub actual_timing: f64,
    /// Violation severity (0-1)
    pub severity: f64,
}

/// Advanced analytics results for mid-circuit measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedAnalyticsResults {
    /// Statistical analysis results
    pub statistical_analysis: StatisticalAnalysisResults,
    /// Correlation analysis results
    pub correlation_analysis: CorrelationAnalysisResults,
    /// Time series analysis results
    pub time_series_analysis: Option<TimeSeriesAnalysisResults>,
    /// Anomaly detection results
    pub anomaly_detection: Option<AnomalyDetectionResults>,
    /// Distribution analysis results
    pub distribution_analysis: DistributionAnalysisResults,
    /// Causal inference results
    pub causal_analysis: Option<CausalAnalysisResults>,
}

/// Statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisResults {
    /// Descriptive statistics
    pub descriptive_stats: DescriptiveStatistics,
    /// Hypothesis test results
    pub hypothesis_tests: HypothesisTestResults,
    /// Confidence intervals
    pub confidence_intervals: ConfidenceIntervals,
    /// Effect size measurements
    pub effect_sizes: EffectSizeAnalysis,
}

/// Descriptive statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStatistics {
    /// Mean measurement latency
    pub mean_latency: f64,
    /// Standard deviation of latency
    pub std_latency: f64,
    /// Median latency
    pub median_latency: f64,
    /// Percentiles (25th, 75th, 95th, 99th)
    pub latency_percentiles: Vec<f64>,
    /// Measurement success rate statistics
    pub success_rate_stats: MeasurementSuccessStats,
    /// Error rate distribution
    pub error_rate_distribution: ErrorRateDistribution,
}

/// Measurement success statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementSuccessStats {
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Success rate by qubit
    pub per_qubit_success_rate: HashMap<QubitId, f64>,
    /// Success rate over time
    pub temporal_success_rate: Vec<(f64, f64)>, // (timestamp, success_rate)
    /// Success rate confidence interval
    pub success_rate_ci: (f64, f64),
}

/// Error rate distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRateDistribution {
    /// Error rate histogram
    pub histogram: Vec<(f64, usize)>, // (error_rate, count)
    /// Best-fit distribution
    pub best_fit_distribution: String,
    /// Distribution parameters
    pub distribution_parameters: Vec<f64>,
    /// Goodness-of-fit statistic
    pub goodness_of_fit: f64,
}

/// Hypothesis test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestResults {
    /// Tests for measurement independence
    pub independence_tests: HashMap<String, StatisticalTest>,
    /// Tests for stationarity
    pub stationarity_tests: HashMap<String, StatisticalTest>,
    /// Tests for normality
    pub normality_tests: HashMap<String, StatisticalTest>,
    /// Comparison tests between different conditions
    pub comparison_tests: HashMap<String, ComparisonTest>,
}

/// Individual statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value
    pub critical_value: f64,
    /// Test conclusion
    pub is_significant: bool,
    /// Effect size
    pub effect_size: Option<f64>,
}

impl Default for StatisticalTest {
    fn default() -> Self {
        Self {
            statistic: 0.0,
            p_value: 0.1,
            critical_value: 1.96,
            is_significant: false,
            effect_size: None,
        }
    }
}

/// Comparison test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonTest {
    /// Test type
    pub test_type: String,
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Mean difference
    pub mean_difference: f64,
    /// Confidence interval for difference
    pub difference_ci: (f64, f64),
    /// Cohen's d effect size
    pub cohens_d: f64,
}

/// Confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// Confidence level used
    pub confidence_level: f64,
    /// Confidence intervals for means
    pub mean_intervals: HashMap<String, (f64, f64)>,
    /// Bootstrap confidence intervals
    pub bootstrap_intervals: HashMap<String, (f64, f64)>,
    /// Prediction intervals
    pub prediction_intervals: HashMap<String, (f64, f64)>,
}

/// Effect size analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeAnalysis {
    /// Cohen's d for measurement differences
    pub cohens_d: HashMap<String, f64>,
    /// Correlation coefficients
    pub correlations: HashMap<String, f64>,
    /// R-squared values for relationships
    pub r_squared: HashMap<String, f64>,
    /// Practical significance indicators
    pub practical_significance: HashMap<String, bool>,
}

/// Correlation analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysisResults {
    /// Pearson correlation matrix
    pub pearson_correlations: Array2<f64>,
    /// Spearman correlation matrix
    pub spearman_correlations: Array2<f64>,
    /// Kendall's tau correlations
    pub kendall_correlations: Array2<f64>,
    /// Significant correlations
    pub significant_correlations: Vec<CorrelationPair>,
    /// Partial correlations
    pub partial_correlations: Array2<f64>,
    /// Correlation network analysis
    pub network_analysis: CorrelationNetworkAnalysis,
}

/// Correlation pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPair {
    /// Variable 1
    pub variable1: String,
    /// Variable 2
    pub variable2: String,
    /// Correlation coefficient
    pub correlation: f64,
    /// P-value
    pub p_value: f64,
    /// Correlation type
    pub correlation_type: CorrelationType,
}

/// Types of correlation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CorrelationType {
    Pearson,
    Spearman,
    Kendall,
    Partial,
}

/// Correlation network analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationNetworkAnalysis {
    /// Graph adjacency matrix
    pub adjacency_matrix: Array2<f64>,
    /// Node centrality measures
    pub centrality_measures: NodeCentralityMeasures,
    /// Community detection results
    pub communities: Vec<Vec<usize>>,
    /// Network density
    pub network_density: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
}

/// Node centrality measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCentralityMeasures {
    /// Betweenness centrality
    pub betweenness: Vec<f64>,
    /// Closeness centrality
    pub closeness: Vec<f64>,
    /// Eigenvector centrality
    pub eigenvector: Vec<f64>,
    /// Degree centrality
    pub degree: Vec<f64>,
}

/// Time series analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesAnalysisResults {
    /// Trend analysis
    pub trend_analysis: TrendAnalysis,
    /// Seasonality analysis
    pub seasonality_analysis: Option<SeasonalityAnalysis>,
    /// Autocorrelation analysis
    pub autocorrelation: AutocorrelationAnalysis,
    /// Change point detection
    pub change_points: Vec<ChangePoint>,
    /// Stationarity test results
    pub stationarity: StationarityTestResults,
}

/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend strength
    pub trend_strength: f64,
    /// Trend slope
    pub trend_slope: f64,
    /// Trend significance
    pub trend_significance: f64,
    /// Trend confidence interval
    pub trend_ci: (f64, f64),
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Cyclical,
}

/// Seasonality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityAnalysis {
    /// Detected seasonal periods
    pub periods: Vec<usize>,
    /// Seasonal strength
    pub seasonal_strength: f64,
    /// Seasonal components
    pub seasonal_components: Array1<f64>,
    /// Residual components
    pub residual_components: Array1<f64>,
}

/// Autocorrelation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationAnalysis {
    /// Autocorrelation function
    pub acf: Array1<f64>,
    /// Partial autocorrelation function
    pub pacf: Array1<f64>,
    /// Significant lags
    pub significant_lags: Vec<usize>,
    /// Ljung-Box test statistic
    pub ljung_box_statistic: f64,
    /// Ljung-Box p-value
    pub ljung_box_p_value: f64,
}

/// Change point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    /// Change point index
    pub index: usize,
    /// Change point timestamp
    pub timestamp: f64,
    /// Change point confidence
    pub confidence: f64,
    /// Change magnitude
    pub magnitude: f64,
    /// Change type
    pub change_type: ChangePointType,
}

/// Change point types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChangePointType {
    MeanShift,
    VarianceChange,
    TrendChange,
    DistributionChange,
}

/// Stationarity test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityTestResults {
    /// Augmented Dickey-Fuller test
    pub adf_test: StatisticalTest,
    /// KPSS test
    pub kpss_test: StatisticalTest,
    /// Phillips-Perron test
    pub pp_test: StatisticalTest,
    /// Overall stationarity conclusion
    pub is_stationary: bool,
}

/// Anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResults {
    /// Detected anomalies
    pub anomalies: Vec<AnomalyEvent>,
    /// Anomaly scores
    pub anomaly_scores: Array1<f64>,
    /// Detection thresholds
    pub thresholds: HashMap<String, f64>,
    /// Method performance
    pub method_performance: AnomalyMethodPerformance,
}

/// Anomaly event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    /// Event index
    pub index: usize,
    /// Event timestamp
    pub timestamp: f64,
    /// Anomaly score
    pub anomaly_score: f64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Affected measurements
    pub affected_measurements: Vec<usize>,
    /// Severity level
    pub severity: AnomalySeverity,
}

/// Anomaly types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyType {
    PointAnomaly,
    ContextualAnomaly,
    CollectiveAnomaly,
    TrendAnomaly,
    SeasonalAnomaly,
}

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Anomaly method performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyMethodPerformance {
    /// Precision scores
    pub precision: HashMap<String, f64>,
    /// Recall scores
    pub recall: HashMap<String, f64>,
    /// F1 scores
    pub f1_scores: HashMap<String, f64>,
    /// False positive rates
    pub false_positive_rates: HashMap<String, f64>,
}

/// Distribution analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysisResults {
    /// Best-fit distributions
    pub best_fit_distributions: HashMap<String, DistributionFit>,
    /// Distribution comparison results
    pub distribution_comparisons: Vec<DistributionComparison>,
    /// Mixture model results
    pub mixture_models: Option<MixtureModelResults>,
    /// Normality assessment
    pub normality_assessment: NormalityAssessment,
}

/// Distribution fit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionFit {
    /// Distribution name
    pub distribution_name: String,
    /// Distribution parameters
    pub parameters: Vec<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC score
    pub aic: f64,
    /// BIC score
    pub bic: f64,
    /// Kolmogorov-Smirnov test statistic
    pub ks_statistic: f64,
    /// KS test p-value
    pub ks_p_value: f64,
}

/// Distribution comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionComparison {
    /// Distribution 1
    pub distribution1: String,
    /// Distribution 2
    pub distribution2: String,
    /// AIC difference
    pub aic_difference: f64,
    /// BIC difference
    pub bic_difference: f64,
    /// Likelihood ratio test
    pub likelihood_ratio_test: StatisticalTest,
    /// Better fitting distribution
    pub better_fit: String,
}

/// Mixture model results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtureModelResults {
    /// Number of components
    pub n_components: usize,
    /// Component weights
    pub weights: Array1<f64>,
    /// Component parameters
    pub component_parameters: Vec<Vec<f64>>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// BIC score
    pub bic: f64,
    /// Component assignments
    pub assignments: Array1<usize>,
}

/// Normality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalityAssessment {
    /// Shapiro-Wilk test
    pub shapiro_wilk: StatisticalTest,
    /// Anderson-Darling test
    pub anderson_darling: StatisticalTest,
    /// Jarque-Bera test
    pub jarque_bera: StatisticalTest,
    /// Overall normality conclusion
    pub is_normal: bool,
    /// Normality confidence
    pub normality_confidence: f64,
}

/// Causal analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalAnalysisResults {
    /// Causal graph
    pub causal_graph: CausalGraph,
    /// Causal effects
    pub causal_effects: Vec<CausalEffect>,
    /// Confounding analysis
    pub confounding_analysis: ConfoundingAnalysis,
    /// Causal strength measures
    pub causal_strength: HashMap<String, f64>,
}

/// Causal graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    /// Adjacency matrix
    pub adjacency_matrix: Array2<f64>,
    /// Node names
    pub node_names: Vec<String>,
    /// Edge weights
    pub edge_weights: HashMap<(usize, usize), f64>,
    /// Graph confidence
    pub graph_confidence: f64,
}

/// Causal effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEffect {
    /// Cause variable
    pub cause: String,
    /// Effect variable
    pub effect: String,
    /// Effect size
    pub effect_size: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// P-value
    pub p_value: f64,
    /// Causal mechanism
    pub mechanism: CausalMechanism,
}

/// Causal mechanisms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CausalMechanism {
    Direct,
    Indirect,
    Mediated,
    Confounded,
    Spurious,
}

/// Confounding analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfoundingAnalysis {
    /// Detected confounders
    pub confounders: Vec<String>,
    /// Confounder strength
    pub confounder_strength: HashMap<String, f64>,
    /// Backdoor criteria satisfaction
    pub backdoor_satisfied: bool,
    /// Frontdoor criteria satisfaction
    pub frontdoor_satisfied: bool,
}

/// Measurement prediction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementPredictionResults {
    /// Predicted measurement outcomes
    pub predictions: Array1<f64>,
    /// Prediction confidence intervals
    pub confidence_intervals: Array2<f64>,
    /// Prediction timestamps
    pub timestamps: Vec<f64>,
    /// Model performance metrics
    pub model_performance: PredictionModelPerformance,
    /// Uncertainty quantification
    pub uncertainty: PredictionUncertainty,
}

/// Prediction model performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModelPerformance {
    /// Mean absolute error
    pub mae: f64,
    /// Mean squared error
    pub mse: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// Mean absolute percentage error
    pub mape: f64,
    /// R-squared score
    pub r2_score: f64,
    /// Prediction accuracy
    pub accuracy: f64,
}

/// Prediction uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionUncertainty {
    /// Aleatoric uncertainty
    pub aleatoric_uncertainty: Array1<f64>,
    /// Epistemic uncertainty
    pub epistemic_uncertainty: Array1<f64>,
    /// Total uncertainty
    pub total_uncertainty: Array1<f64>,
    /// Uncertainty bounds
    pub uncertainty_bounds: Array2<f64>,
}

/// Optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendations {
    /// Scheduling optimizations
    pub scheduling_optimizations: Vec<SchedulingOptimization>,
    /// Protocol optimizations
    pub protocol_optimizations: Vec<ProtocolOptimization>,
    /// Resource optimizations
    pub resource_optimizations: Vec<ResourceOptimization>,
    /// Performance improvements
    pub performance_improvements: Vec<PerformanceImprovement>,
}

/// Scheduling optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingOptimization {
    /// Optimization type
    pub optimization_type: SchedulingOptimizationType,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Implementation difficulty
    pub difficulty: OptimizationDifficulty,
    /// Recommendation description
    pub description: String,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
}

/// Scheduling optimization types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchedulingOptimizationType {
    MeasurementBatching,
    TemporalReordering,
    ParallelExecution,
    ConditionalOptimization,
    LatencyReduction,
}

/// Protocol optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolOptimization {
    /// Protocol type
    pub protocol_type: String,
    /// Optimization description
    pub description: String,
    /// Expected benefit
    pub expected_benefit: f64,
    /// Risk assessment
    pub risk_level: RiskLevel,
    /// Validation requirements
    pub validation_requirements: Vec<String>,
}

/// Resource optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimization {
    /// Resource type
    pub resource_type: ResourceType,
    /// Current utilization
    pub current_utilization: f64,
    /// Optimal utilization
    pub optimal_utilization: f64,
    /// Optimization strategy
    pub strategy: String,
    /// Expected savings
    pub expected_savings: f64,
}

/// Resource types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceType {
    QuantumProcessor,
    ClassicalProcessor,
    Memory,
    NetworkBandwidth,
    StorageCapacity,
}

/// Performance improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovement {
    /// Improvement area
    pub area: PerformanceArea,
    /// Current performance
    pub current_performance: f64,
    /// Target performance
    pub target_performance: f64,
    /// Improvement strategy
    pub strategy: String,
    /// Implementation priority
    pub priority: Priority,
}

/// Performance areas
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PerformanceArea {
    MeasurementLatency,
    MeasurementAccuracy,
    ThroughputRate,
    ErrorRate,
    ResourceEfficiency,
}

/// Priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Risk levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Optimization difficulty levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationDifficulty {
    Easy,
    Moderate,
    Difficult,
    VeryDifficult,
}

/// Adaptive learning insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningInsights {
    /// Learning progress metrics
    pub learning_progress: LearningProgress,
    /// Model adaptation history
    pub adaptation_history: Vec<AdaptationEvent>,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
    /// Concept drift detection
    pub drift_detection: DriftDetectionResults,
    /// Knowledge transfer insights
    pub transfer_learning: TransferLearningInsights,
}

/// Learning progress metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProgress {
    /// Training iterations completed
    pub iterations_completed: usize,
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Training loss history
    pub loss_history: Array1<f64>,
    /// Validation accuracy history
    pub accuracy_history: Array1<f64>,
    /// Convergence status
    pub convergence_status: ConvergenceStatus,
}

/// Convergence status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    NotStarted,
    InProgress,
    Converged,
    Diverged,
    Plateaued,
    Stuck,
    Improving,
    Diverging,
    Oscillating,
}

/// Adaptation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Adaptation type
    pub adaptation_type: AdaptationType,
    /// Trigger condition
    pub trigger: String,
    /// Performance before adaptation
    pub performance_before: f64,
    /// Performance after adaptation
    pub performance_after: f64,
    /// Performance snapshot at time of adaptation
    pub performance_snapshot: PerformanceMetrics,
    /// Magnitude of adaptation
    pub adaptation_magnitude: f64,
    /// Success indicator
    pub success_indicator: f64,
    /// Adaptation success
    pub success: bool,
}

/// Adaptation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdaptationType {
    ParameterTuning,
    ArchitectureChange,
    FeatureSelection,
    HyperparameterOptimization,
    ModelRetrained,
    ThresholdAdjustment,
    PerformanceOptimization,
}

/// Performance trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Short-term trend
    pub short_term_trend: TrendDirection,
    /// Long-term trend
    pub long_term_trend: TrendDirection,
    /// Trend strength
    pub trend_strength: f64,
    /// Seasonal patterns
    pub seasonal_patterns: Option<SeasonalityAnalysis>,
    /// Performance volatility
    pub volatility: f64,
}

/// Drift detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionResults {
    /// Drift detected
    pub drift_detected: bool,
    /// Drift type
    pub drift_type: Option<DriftType>,
    /// Drift magnitude
    pub drift_magnitude: f64,
    /// Detection confidence
    pub detection_confidence: f64,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Drift types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DriftType {
    Gradual,
    Sudden,
    Incremental,
    Recurring,
    Virtual,
    PerformanceDegradation,
    QualityDrift,
    ConceptDrift,
}

/// Transfer learning insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningInsights {
    /// Knowledge transfer effectiveness
    pub transfer_effectiveness: f64,
    /// Source domain similarity
    pub domain_similarity: f64,
    /// Feature transferability
    pub feature_transferability: Array1<f64>,
    /// Adaptation requirements
    pub adaptation_requirements: Vec<String>,
    /// Transfer learning recommendations
    pub recommendations: Vec<String>,
}

/// Device capabilities for mid-circuit measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidCircuitCapabilities {
    /// Maximum number of mid-circuit measurements
    pub max_measurements: Option<usize>,
    /// Supported measurement types
    pub supported_measurement_types: Vec<MeasurementType>,
    /// Classical register capacity
    pub classical_register_capacity: usize,
    /// Maximum classical processing time
    pub max_classical_processing_time: f64,
    /// Real-time feedback support
    pub realtime_feedback: bool,
    /// Parallel measurement support
    pub parallel_measurements: bool,
    /// Native measurement protocols
    pub native_protocols: Vec<String>,
    /// Timing constraints
    pub timing_constraints: TimingConstraints,
}

/// Supported measurement types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeasurementType {
    /// Standard Z-basis measurement
    ZBasis,
    /// X-basis measurement
    XBasis,
    /// Y-basis measurement
    YBasis,
    /// Custom Pauli measurement
    Pauli(String),
    /// Joint measurement of multiple qubits
    Joint,
    /// Non-destructive measurement
    NonDestructive,
}

/// Timing constraints for mid-circuit measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConstraints {
    /// Minimum time between measurements (nanoseconds)
    pub min_measurement_spacing: f64,
    /// Maximum measurement duration (nanoseconds)
    pub max_measurement_duration: f64,
    /// Classical processing deadline (nanoseconds)
    pub classical_deadline: f64,
    /// Coherence time limits
    pub coherence_limits: HashMap<QubitId, f64>,
}

// Additional types for analytics modules

/// Statistical anomaly information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnomaly {
    /// Index in the data series
    pub index: usize,
    /// Anomalous value
    pub value: f64,
    /// Z-score of the anomaly
    pub z_score: f64,
    /// Statistical p-value
    pub p_value: f64,
    /// Metric type
    pub metric_type: String,
    /// Severity of the anomaly
    pub anomaly_severity: AnomalySeverity,
}

/// Temporal anomaly information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnomaly {
    /// Start index of anomalous period
    pub start_index: usize,
    /// End index of anomalous period
    pub end_index: usize,
    /// Change point index
    pub change_point: usize,
    /// Magnitude of change
    pub magnitude: f64,
    /// Direction of change
    pub direction: ChangeDirection,
    /// Confidence in anomaly detection
    pub confidence: f64,
}

/// Direction of change in temporal anomaly
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChangeDirection {
    Increase,
    Decrease,
    Oscillation,
}

/// Pattern anomaly information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnomaly {
    /// Type of pattern anomaly
    pub pattern_type: PatternType,
    /// Description of the anomaly
    pub description: String,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Indices affected by the anomaly
    pub affected_indices: Vec<usize>,
    /// Confidence in detection
    pub confidence: f64,
}

/// Types of pattern anomalies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternType {
    CorrelationAnomaly,
    ConstantSequence,
    PeriodicityBreak,
    TrendReversal,
    VariabilityAnomaly,
}

/// Summary of all anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalySummary {
    /// Total number of anomalies detected
    pub total_anomalies: usize,
    /// Overall anomaly rate
    pub anomaly_rate: f64,
    /// Distribution by severity
    pub severity_distribution: Vec<(String, usize)>,
    /// Types of anomalies found
    pub anomaly_types: Vec<String>,
    /// Recommendations for addressing anomalies
    pub recommendations: Vec<String>,
}

/// Causal relationship information (only missing fields if any)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRelationship {
    /// Cause variable
    pub cause: String,
    /// Effect variable
    pub effect: String,
    /// Strength of causal relationship
    pub causal_strength: f64,
    /// Direction of causality
    pub causal_direction: CausalDirection,
    /// P-value of causal test
    pub p_value: f64,
    /// Confidence interval for effect
    pub confidence_interval: (f64, f64),
    /// Causal mechanism
    pub mechanism: CausalMechanism,
}

/// Direction of causal relationship
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CausalDirection {
    Forward,
    Backward,
    Bidirectional,
    None,
}

/// Edge type in causal graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeType {
    Directed,
    Undirected,
    Bidirected,
}

/// Intervention analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionAnalysis {
    /// Type of intervention
    pub intervention_type: String,
    /// Target variable
    pub target_variable: String,
    /// Magnitude of intervention
    pub intervention_magnitude: f64,
    /// Predicted effects
    pub predicted_effects: Vec<PredictedEffect>,
    /// Cost of intervention
    pub intervention_cost: f64,
    /// Benefit-cost ratio
    pub benefit_ratio: f64,
}

/// Predicted effect of intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedEffect {
    /// Variable affected
    pub variable: String,
    /// Effect size
    pub effect_size: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// P-value
    pub p_value: f64,
}

/// Confounding assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfoundingAssessment {
    /// Identified confounders
    pub confounders: Vec<ConfoundingVariable>,
    /// Overall confounding risk
    pub overall_confounding_risk: String,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Confounding variable information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfoundingVariable {
    /// Variable name
    pub variable: String,
    /// Confounding strength
    pub confounding_strength: f64,
    /// Adjustment method
    pub adjustment_method: String,
    /// P-value
    pub p_value: f64,
}

/// Prediction model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModel {
    /// Model type
    pub model_type: String,
    /// Model coefficients
    pub coefficients: Vec<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Training accuracy
    pub training_accuracy: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Last training time
    pub last_trained: SystemTime,
    /// Hyperparameters
    pub hyperparameters: Vec<(String, f64)>,
}

impl Default for PredictionModel {
    fn default() -> Self {
        Self {
            model_type: "linear".to_string(),
            coefficients: vec![],
            feature_names: vec![],
            training_accuracy: 0.0,
            validation_accuracy: 0.0,
            last_trained: SystemTime::now(),
            hyperparameters: vec![],
        }
    }
}

/// ML features for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLFeatures {
    /// Statistical features
    pub statistical_features: StatisticalFeatures,
    /// Temporal features
    pub temporal_features: TemporalFeatures,
    /// Pattern features
    pub pattern_features: PatternFeatures,
    /// Feature importance
    pub feature_importance: Vec<FeatureImportance>,
}

impl Default for MLFeatures {
    fn default() -> Self {
        Self {
            statistical_features: StatisticalFeatures::default(),
            temporal_features: TemporalFeatures::default(),
            pattern_features: PatternFeatures::default(),
            feature_importance: vec![],
        }
    }
}

/// Statistical features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFeatures {
    /// Mean latency
    pub mean_latency: f64,
    /// Standard deviation of latency
    pub std_latency: f64,
    /// Mean confidence
    pub mean_confidence: f64,
    /// Standard deviation of confidence
    pub std_confidence: f64,
    /// Skewness of latency
    pub skewness_latency: f64,
    /// Kurtosis of latency
    pub kurtosis_latency: f64,
}

impl Default for StatisticalFeatures {
    fn default() -> Self {
        Self {
            mean_latency: 0.0,
            std_latency: 0.0,
            mean_confidence: 0.0,
            std_confidence: 0.0,
            skewness_latency: 0.0,
            kurtosis_latency: 0.0,
        }
    }
}

/// Temporal features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFeatures {
    /// Measurement rate
    pub measurement_rate: f64,
    /// Temporal autocorrelation
    pub temporal_autocorrelation: f64,
    /// Trend slope
    pub trend_slope: f64,
    /// Periodicity strength
    pub periodicity_strength: f64,
}

impl Default for TemporalFeatures {
    fn default() -> Self {
        Self {
            measurement_rate: 0.0,
            temporal_autocorrelation: 0.0,
            trend_slope: 0.0,
            periodicity_strength: 0.0,
        }
    }
}

/// Pattern features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternFeatures {
    /// Correlation between latency and confidence
    pub latency_confidence_correlation: f64,
    /// Measurement consistency
    pub measurement_consistency: f64,
    /// Outlier ratio
    pub outlier_ratio: f64,
    /// Pattern complexity
    pub pattern_complexity: f64,
}

impl Default for PatternFeatures {
    fn default() -> Self {
        Self {
            latency_confidence_correlation: 0.0,
            measurement_consistency: 0.0,
            outlier_ratio: 0.0,
            pattern_complexity: 0.0,
        }
    }
}

/// Feature importance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature name
    pub feature_name: String,
    /// Importance score
    pub importance: f64,
}

/// Training epoch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEpoch {
    /// Epoch number
    pub epoch_number: usize,
    /// Features used
    pub features: MLFeatures,
    /// Target metrics
    pub target_metrics: PerformanceMetrics,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Learning rate
    pub learning_rate: f64,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Training accuracy
    pub training_accuracy: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Cross-validation score
    pub cross_validation_score: f64,
    /// Overfitting score
    pub overfitting_score: f64,
}

/// Optimization model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationModel {
    /// Model type
    pub model_type: String,
    /// Model parameters
    pub parameters: Vec<f64>,
    /// Training features
    pub training_features: MLFeatures,
    /// Model performance
    pub model_performance: ModelPerformance,
    /// Last update time
    pub last_updated: SystemTime,
}

/// Performance improvements prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovements {
    /// Latency reduction
    pub latency_reduction: f64,
    /// Confidence increase
    pub confidence_increase: f64,
    /// Throughput increase
    pub throughput_increase: f64,
    /// Error rate reduction
    pub error_rate_reduction: f64,
    /// Overall score improvement
    pub overall_score_improvement: f64,
}

impl Default for PerformanceImprovements {
    fn default() -> Self {
        Self {
            latency_reduction: 0.0,
            confidence_increase: 0.0,
            throughput_increase: 0.0,
            error_rate_reduction: 0.0,
            overall_score_improvement: 0.0,
        }
    }
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Parameter to optimize
    pub parameter: String,
    /// Current value
    pub current_value: f64,
    /// Recommended value
    pub recommended_value: f64,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Confidence in recommendation
    pub confidence: f64,
    /// Rationale for recommendation
    pub rationale: String,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Predicted improvements
    pub predicted_improvements: PerformanceImprovements,
    /// Confidence in optimization
    pub confidence: f64,
    /// Model version
    pub model_version: String,
}

impl Default for OptimizationResult {
    fn default() -> Self {
        Self {
            recommendations: vec![],
            predicted_improvements: PerformanceImprovements::default(),
            confidence: 0.5,
            model_version: "1.0".to_string(),
        }
    }
}
