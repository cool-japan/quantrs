//! Configuration types for the unified benchmarking system

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};

use super::types::{QuantumPlatform, BaselineMetric};

/// Unified benchmarking system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedBenchmarkConfig {
    /// Target quantum platforms
    pub target_platforms: Vec<QuantumPlatform>,
    /// Benchmark suite configuration
    pub benchmark_suite: BenchmarkSuiteConfig,
    /// SciRS2 analysis configuration
    pub scirs2_config: SciRS2AnalysisConfig,
    /// Reporting and visualization configuration
    pub reporting_config: ReportingConfig,
    /// Resource optimization configuration
    pub optimization_config: ResourceOptimizationConfig,
    /// Historical tracking configuration
    pub tracking_config: HistoricalTrackingConfig,
    /// Custom benchmark configuration
    pub custom_benchmarks: Vec<CustomBenchmarkDefinition>,
    /// Performance targets and thresholds
    pub performance_targets: PerformanceTargets,
}

/// Benchmark suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteConfig {
    /// Gate-level benchmarks
    pub gate_benchmarks: GateBenchmarkConfig,
    /// Circuit-level benchmarks
    pub circuit_benchmarks: CircuitBenchmarkConfig,
    /// Algorithm-level benchmarks
    pub algorithm_benchmarks: AlgorithmBenchmarkConfig,
    /// System-level benchmarks
    pub system_benchmarks: SystemBenchmarkConfig,
    /// Execution parameters
    pub execution_params: BenchmarkExecutionParams,
}

/// Gate-level benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateBenchmarkConfig {
    /// Single-qubit gates to benchmark
    pub single_qubit_gates: Vec<SingleQubitGate>,
    /// Two-qubit gates to benchmark
    pub two_qubit_gates: Vec<TwoQubitGate>,
    /// Multi-qubit gates to benchmark
    pub multi_qubit_gates: Vec<MultiQubitGate>,
    /// Number of repetitions per gate
    pub repetitions_per_gate: usize,
    /// Randomized gate sequences
    pub enable_random_sequences: bool,
    /// Gate fidelity measurement methods
    pub fidelity_methods: Vec<FidelityMeasurementMethod>,
}

/// Circuit-level benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBenchmarkConfig {
    /// Circuit depth range to test
    pub depth_range: (usize, usize),
    /// Circuit width range to test
    pub width_range: (usize, usize),
    /// Circuit types to benchmark
    pub circuit_types: Vec<CircuitType>,
    /// Number of random circuits per configuration
    pub random_circuits_per_config: usize,
    /// Parametric circuit configurations
    pub parametric_configs: Vec<ParametricCircuitConfig>,
}

/// Algorithm-level benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmBenchmarkConfig {
    /// Quantum algorithms to benchmark
    pub algorithms: Vec<QuantumAlgorithm>,
    /// Problem sizes for each algorithm
    pub problem_sizes: HashMap<String, Vec<usize>>,
    /// Algorithm-specific parameters
    pub algorithm_params: HashMap<String, AlgorithmParams>,
    /// Enable noisy intermediate-scale quantum (NISQ) optimizations
    pub enable_nisq_optimizations: bool,
}

/// System-level benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBenchmarkConfig {
    /// Cross-platform comparison benchmarks
    pub enable_cross_platform: bool,
    /// Resource utilization benchmarks
    pub enable_resource_benchmarks: bool,
    /// Cost efficiency benchmarks
    pub enable_cost_benchmarks: bool,
    /// Scalability benchmarks
    pub enable_scalability_benchmarks: bool,
    /// Reliability and uptime benchmarks
    pub enable_reliability_benchmarks: bool,
}

/// SciRS2 analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciRS2AnalysisConfig {
    /// Statistical analysis configuration
    pub statistical_analysis: StatisticalAnalysisConfig,
    /// Machine learning analysis configuration
    pub ml_analysis: MLAnalysisConfig,
    /// Optimization analysis configuration
    pub optimization_analysis: OptimizationAnalysisConfig,
    /// Graph analysis configuration
    pub graph_analysis: GraphAnalysisConfig,
}

/// Statistical analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisConfig {
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Enable Bayesian analysis
    pub enable_bayesian: bool,
    /// Enable non-parametric tests
    pub enable_nonparametric: bool,
    /// Enable multivariate analysis
    pub enable_multivariate: bool,
    /// Bootstrap configuration
    pub bootstrap_samples: usize,
    /// Hypothesis testing configuration
    pub hypothesis_testing: HypothesisTestingConfig,
}

/// Machine learning analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLAnalysisConfig {
    /// Enable ML-based performance prediction
    pub enable_prediction: bool,
    /// Enable clustering analysis
    pub enable_clustering: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Model types to use
    pub model_types: Vec<MLModelType>,
    /// Feature engineering configuration
    pub feature_engineering: FeatureEngineeringConfig,
}

/// Optimization analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAnalysisConfig {
    /// Enable performance optimization
    pub enable_optimization: bool,
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    /// Optimization algorithms
    pub algorithms: Vec<OptimizationAlgorithm>,
    /// Multi-objective optimization
    pub enable_multi_objective: bool,
}

/// Graph analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysisConfig {
    /// Enable connectivity analysis
    pub enable_connectivity: bool,
    /// Enable topology optimization
    pub enable_topology_optimization: bool,
    /// Enable community detection
    pub enable_community_detection: bool,
    /// Graph metrics to compute
    pub metrics: Vec<GraphMetric>,
}

/// Reporting and visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Report formats to generate
    pub formats: Vec<ReportFormat>,
    /// Visualization types
    pub visualizations: Vec<VisualizationType>,
    /// Export destinations
    pub export_destinations: Vec<ExportDestination>,
    /// Real-time dashboard configuration
    pub dashboard_config: DashboardConfig,
    /// Automated report generation
    pub automated_reports: AutomatedReportConfig,
}

/// Resource optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationConfig {
    /// Enable intelligent resource allocation
    pub enable_intelligent_allocation: bool,
    /// Cost optimization strategies
    pub cost_optimization: CostOptimizationConfig,
    /// Performance optimization strategies
    pub performance_optimization: PerformanceOptimizationConfig,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
    /// Scheduling optimization
    pub scheduling_optimization: SchedulingOptimizationConfig,
}

/// Historical tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalTrackingConfig {
    /// Enable historical data collection
    pub enable_tracking: bool,
    /// Data retention period (days)
    pub retention_period_days: u32,
    /// Trend analysis configuration
    pub trend_analysis: TrendAnalysisConfig,
    /// Performance baseline tracking
    pub baseline_tracking: BaselineTrackingConfig,
    /// Comparative analysis configuration
    pub comparative_analysis: ComparativeAnalysisConfig,
}

/// Gate types for benchmarking

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SingleQubitGate {
    X, Y, Z, H, S, T, SqrtX, RX(f64), RY(f64), RZ(f64),
}

// Custom implementations for SingleQubitGate
impl PartialEq for SingleQubitGate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SingleQubitGate::X, SingleQubitGate::X) => true,
            (SingleQubitGate::Y, SingleQubitGate::Y) => true,
            (SingleQubitGate::Z, SingleQubitGate::Z) => true,
            (SingleQubitGate::H, SingleQubitGate::H) => true,
            (SingleQubitGate::S, SingleQubitGate::S) => true,
            (SingleQubitGate::T, SingleQubitGate::T) => true,
            (SingleQubitGate::SqrtX, SingleQubitGate::SqrtX) => true,
            (SingleQubitGate::RX(a), SingleQubitGate::RX(b)) => (a - b).abs() < 1e-10,
            (SingleQubitGate::RY(a), SingleQubitGate::RY(b)) => (a - b).abs() < 1e-10,
            (SingleQubitGate::RZ(a), SingleQubitGate::RZ(b)) => (a - b).abs() < 1e-10,
            _ => false,
        }
    }
}

impl Eq for SingleQubitGate {}

impl std::hash::Hash for SingleQubitGate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            SingleQubitGate::X => 0u8.hash(state),
            SingleQubitGate::Y => 1u8.hash(state),
            SingleQubitGate::Z => 2u8.hash(state),
            SingleQubitGate::H => 3u8.hash(state),
            SingleQubitGate::S => 4u8.hash(state),
            SingleQubitGate::T => 5u8.hash(state),
            SingleQubitGate::SqrtX => 6u8.hash(state),
            SingleQubitGate::RX(f) => {
                7u8.hash(state);
                (*f as u64).hash(state);
            },
            SingleQubitGate::RY(f) => {
                8u8.hash(state);
                (*f as u64).hash(state);
            },
            SingleQubitGate::RZ(f) => {
                9u8.hash(state);
                (*f as u64).hash(state);
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TwoQubitGate {
    CNOT, CZ, SWAP, iSWAP, CRX(f64), CRY(f64), CRZ(f64),
}

// Custom implementations to work around f64 issues
impl PartialEq for TwoQubitGate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TwoQubitGate::CNOT, TwoQubitGate::CNOT) => true,
            (TwoQubitGate::CZ, TwoQubitGate::CZ) => true,
            (TwoQubitGate::SWAP, TwoQubitGate::SWAP) => true,
            (TwoQubitGate::iSWAP, TwoQubitGate::iSWAP) => true,
            (TwoQubitGate::CRX(a), TwoQubitGate::CRX(b)) => (a - b).abs() < 1e-10,
            (TwoQubitGate::CRY(a), TwoQubitGate::CRY(b)) => (a - b).abs() < 1e-10,
            (TwoQubitGate::CRZ(a), TwoQubitGate::CRZ(b)) => (a - b).abs() < 1e-10,
            _ => false,
        }
    }
}

impl Eq for TwoQubitGate {}

impl std::hash::Hash for TwoQubitGate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            TwoQubitGate::CNOT => 0u8.hash(state),
            TwoQubitGate::CZ => 1u8.hash(state),
            TwoQubitGate::SWAP => 2u8.hash(state),
            TwoQubitGate::iSWAP => 3u8.hash(state),
            TwoQubitGate::CRX(f) => {
                4u8.hash(state);
                (*f as u64).hash(state); // Approximate hash for f64
            },
            TwoQubitGate::CRY(f) => {
                5u8.hash(state);
                (*f as u64).hash(state);
            },
            TwoQubitGate::CRZ(f) => {
                6u8.hash(state);
                (*f as u64).hash(state);
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MultiQubitGate {
    Toffoli, Fredkin, CCZ, Controlled(Box<SingleQubitGate>, usize),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FidelityMeasurementMethod {
    ProcessTomography,
    RandomizedBenchmarking,
    SimultaneousRandomizedBenchmarking,
    CycleBenchmarking,
    GateSetTomography,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CircuitType {
    Random,
    QFT,
    Grover,
    Supremacy,
    QAOA,
    VQE,
    Arithmetic,
    ErrorCorrection,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricCircuitConfig {
    pub circuit_type: CircuitType,
    pub parameter_ranges: HashMap<String, (f64, f64)>,
    pub parameter_steps: HashMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumAlgorithm {
    Shor { bit_length: usize },
    Grover { database_size: usize },
    QFT { num_qubits: usize },
    VQE { molecule: String },
    QAOA { graph_size: usize },
    QuantumWalk { graph_type: String },
    HHL { matrix_size: usize },
    QuantumCounting { target_states: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmParams {
    pub parameters: HashMap<String, f64>,
    pub options: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkExecutionParams {
    /// Number of shots per circuit
    pub shots: usize,
    /// Maximum execution time per benchmark
    pub max_execution_time: Duration,
    /// Number of repetitions for statistical significance
    pub repetitions: usize,
    /// Parallel execution configuration
    pub parallelism: ParallelismConfig,
    /// Error handling configuration
    pub error_handling: ErrorHandlingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelismConfig {
    /// Enable parallel execution across platforms
    pub enable_parallel: bool,
    /// Maximum concurrent executions
    pub max_concurrent: usize,
    /// Batch size for grouped executions
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Timeout handling
    pub timeout_handling: TimeoutHandling,
    /// Error recovery strategies
    pub recovery_strategies: Vec<ErrorRecoveryStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub retry_delay: Duration,
    pub exponential_backoff: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeoutHandling {
    AbortOnTimeout,
    ContinueWithPartialResults,
    ExtendTimeoutOnce,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorRecoveryStrategy {
    RetryOnDifferentDevice,
    ReduceCircuitComplexity,
    FallbackToSimulator,
    SkipFailedBenchmark,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestingConfig {
    pub tests: Vec<StatisticalTest>,
    pub multiple_comparisons_correction: MultipleComparisonsCorrection,
    pub effect_size_measures: Vec<EffectSizeMeasure>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StatisticalTest {
    TTest,
    MannWhitneyU,
    KolmogorovSmirnov,
    ChiSquare,
    ANOVA,
    KruskalWallis,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MultipleComparisonsCorrection {
    Bonferroni,
    FDR,
    Holm,
    Hochberg,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EffectSizeMeasure {
    CohenD,
    HedgeG,
    GlassD,
    EtaSquared,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MLModelType {
    LinearRegression,
    RandomForest,
    GradientBoosting,
    SupportVectorMachine,
    NeuralNetwork,
    GaussianProcess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    pub polynomial_features: bool,
    pub interaction_features: bool,
    pub feature_selection: bool,
    pub dimensionality_reduction: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MinimizeExecutionTime,
    MaximizeFidelity,
    MinimizeCost,
    MaximizeReliability,
    MinimizeErrorRate,
    MaximizeThroughput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    GradientDescent,
    ParticleSwarm,
    GeneticAlgorithm,
    DifferentialEvolution,
    BayesianOptimization,
    SimulatedAnnealing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphMetric {
    Betweenness,
    Closeness,
    Eigenvector,
    PageRank,
    ClusteringCoefficient,
    Diameter,
    AveragePathLength,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportFormat {
    PDF,
    HTML,
    JSON,
    CSV,
    LaTeX,
    Markdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VisualizationType {
    PerformanceCharts,
    StatisticalPlots,
    TopologyGraphs,
    CostAnalysis,
    TrendAnalysis,
    ComparisonMatrices,
    Heatmaps,
    TimeSeries,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExportDestination {
    LocalFile(String),
    S3Bucket(String),
    Database(String),
    APIEndpoint(String),
    Email(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub enable_realtime: bool,
    pub update_interval: Duration,
    pub dashboard_port: u16,
    pub authentication: DashboardAuth,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DashboardAuth {
    None,
    Basic { username: String, password: String },
    Token { token: String },
    OAuth { provider: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedReportConfig {
    pub enable_automated: bool,
    pub report_schedule: ReportSchedule,
    pub recipients: Vec<String>,
    pub report_types: Vec<AutomatedReportType>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportSchedule {
    Daily,
    Weekly,
    Monthly,
    Custom(String), // Cron expression
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AutomatedReportType {
    PerformanceSummary,
    CostAnalysis,
    TrendReport,
    AnomalyReport,
    ComparisonReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationConfig {
    pub enable_cost_optimization: bool,
    pub cost_targets: CostTargets,
    pub optimization_strategies: Vec<CostOptimizationStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTargets {
    pub max_cost_per_shot: Option<f64>,
    pub max_daily_cost: Option<f64>,
    pub max_monthly_cost: Option<f64>,
    pub cost_efficiency_target: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CostOptimizationStrategy {
    PreferLowerCostPlatforms,
    OptimizeShotAllocation,
    BatchExecutions,
    UseSpotInstances,
    ScheduleForOffPeakHours,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimizationConfig {
    pub enable_performance_optimization: bool,
    pub performance_targets: PerformanceTargets,
    pub optimization_strategies: Vec<PerformanceOptimizationStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub min_fidelity: f64,
    pub max_error_rate: f64,
    pub max_execution_time: Duration,
    pub min_throughput: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PerformanceOptimizationStrategy {
    OptimizeCircuitMapping,
    UseErrorMitigation,
    ImplementDynamicalDecoupling,
    OptimizeGateSequences,
    AdaptiveCalibration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    pub enable_load_balancing: bool,
    pub balancing_strategy: LoadBalancingStrategy,
    pub health_checks: HealthCheckConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    ResourceBased,
    PerformanceBased,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub enable_health_checks: bool,
    pub check_interval: Duration,
    pub timeout: Duration,
    pub failure_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingOptimizationConfig {
    pub enable_scheduling: bool,
    pub scheduling_strategy: SchedulingStrategy,
    pub priority_handling: PriorityHandling,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    FIFO,
    SJF, // Shortest Job First
    Priority,
    Deadline,
    ResourceAware,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PriorityHandling {
    Strict,
    WeightedFair,
    TimeSlicing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisConfig {
    pub enable_trend_analysis: bool,
    pub analysis_window: Duration,
    pub trend_detection_methods: Vec<TrendDetectionMethod>,
    pub forecast_horizon: Duration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDetectionMethod {
    LinearRegression,
    ARIMA,
    ExponentialSmoothing,
    ChangePointDetection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineTrackingConfig {
    pub enable_baseline_tracking: bool,
    pub baseline_update_frequency: Duration,
    pub baseline_metrics: Vec<BaselineMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysisConfig {
    pub enable_comparative_analysis: bool,
    pub comparison_methods: Vec<ComparisonMethod>,
    pub significance_testing: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonMethod {
    PairwiseComparison,
    RankingAnalysis,
    PerformanceMatrix,
    CostBenefitAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomBenchmarkDefinition {
    pub name: String,
    pub description: String,
    pub circuit_definition: CustomCircuitDefinition,
    pub execution_parameters: CustomExecutionParameters,
    pub success_criteria: SuccessCriteria,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomCircuitDefinition {
    pub circuit_type: CustomCircuitType,
    pub parameters: HashMap<String, f64>,
    pub constraints: Vec<CircuitConstraint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CustomCircuitType {
    QASM(String),
    PythonFunction(String),
    ParametricTemplate(String),
    CircuitGenerator(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CircuitConstraint {
    MaxDepth(usize),
    MaxQubits(usize),
    AllowedGates(Vec<String>),
    ConnectivityConstraint(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomExecutionParameters {
    pub shots: usize,
    pub repetitions: usize,
    pub timeout: Duration,
    pub platforms: Vec<QuantumPlatform>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub min_fidelity: Option<f64>,
    pub max_error_rate: Option<f64>,
    pub max_execution_time: Option<Duration>,
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for SystemBenchmarkConfig {
    fn default() -> Self {
        Self {
            enable_cross_platform: false,
            enable_resource_benchmarks: false,
            enable_cost_benchmarks: false,
            enable_scalability_benchmarks: false,
            enable_reliability_benchmarks: false,
        }
    }
}