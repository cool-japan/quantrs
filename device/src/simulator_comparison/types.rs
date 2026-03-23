//! Advanced Simulator Comparison Framework
//!
//! This module provides a comprehensive framework for comparing quantum simulators
//! across multiple dimensions including performance, accuracy, resource usage,
//! and specialized capabilities. Features sophisticated benchmarking, ML-powered
//! analysis, and automated recommendation systems for optimal simulator selection.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::circuit_integration::CircuitInterface;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    circuit_integration::{ExecutionResult, UniversalCircuitInterface},
    topology::HardwareTopology,
    DeviceError, DeviceResult,
};

/// Comprehensive simulator comparison and benchmarking framework
#[derive(Debug)]
pub struct SimulatorComparisonFramework {
    /// Configuration for the comparison framework
    pub(super) config: ComparisonConfig,
    /// Registered simulators
    pub(super) simulators: Arc<RwLock<HashMap<String, SimulatorProfile>>>,
    /// Benchmark suite
    pub(super) benchmark_suite: Arc<RwLock<BenchmarkSuite>>,
    /// Comparison results cache
    pub(super) results_cache: Arc<RwLock<HashMap<String, ComparisonResult>>>,
    /// Performance analytics engine
    pub(super) analytics: Arc<RwLock<PerformanceAnalytics>>,
    /// ML-powered recommendation engine
    pub(super) recommendation_engine: Arc<RwLock<RecommendationEngine>>,
    /// Benchmark executor
    pub(super) executor: Arc<RwLock<BenchmarkExecutor>>,
}

/// Benchmark configuration (local — distinct from `quantrs2_circuit::BenchmarkConfig`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub enabled: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "Default Benchmark".to_string(),
            enabled: true,
        }
    }
}

/// Configuration for simulator comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    /// Enable automatic benchmarking
    pub auto_benchmarking: bool,
    /// Benchmarking frequency
    pub benchmark_frequency: Duration,
    /// Enable performance tracking
    pub enable_performance_tracking: bool,
    /// Enable ML-powered recommendations
    pub enable_ml_recommendations: bool,
    /// Maximum benchmark execution time
    pub max_benchmark_time: Duration,
    /// Benchmark configurations
    pub benchmark_configs: Vec<BenchmarkConfig>,
    /// Comparison criteria
    pub comparison_criteria: ComparisonCriteria,
    /// Resource monitoring settings
    pub resource_monitoring: ResourceMonitoringConfig,
    /// Output and reporting settings
    pub reporting_config: ReportingConfig,
}

/// Comparison criteria for ranking simulators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonCriteria {
    /// Weight for execution speed (0.0-1.0)
    pub speed_weight: f64,
    /// Weight for accuracy (0.0-1.0)
    pub accuracy_weight: f64,
    /// Weight for memory efficiency (0.0-1.0)
    pub memory_weight: f64,
    /// Weight for scalability (0.0-1.0)
    pub scalability_weight: f64,
    /// Weight for feature completeness (0.0-1.0)
    pub features_weight: f64,
    /// Weight for stability (0.0-1.0)
    pub stability_weight: f64,
    /// Minimum acceptable accuracy
    pub min_accuracy: f64,
    /// Maximum acceptable memory usage (MB)
    pub max_memory_usage: f64,
    /// Required features
    pub required_features: Vec<SimulatorFeature>,
    /// Preferred simulator types
    pub preferred_types: Vec<SimulatorType>,
}

/// Simulator feature enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimulatorFeature {
    NoiseModeling,
    ErrorCorrection,
    StateVectorSimulation,
    DensityMatrixSimulation,
    MonteCarlo,
    TensorNetwork,
    MatrixProductState,
    Stabilizer,
    CliffordSimulation,
    VariationalAlgorithms,
    QAOA,
    VQE,
    QuantumMachineLearning,
    GPUAcceleration,
    DistributedSimulation,
    CustomGates,
    PulseSimulation,
    AdiabaticSimulation,
    OpenSystems,
    ParameterSweep,
}

/// Simulator type classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimulatorType {
    StateVector,
    DensityMatrix,
    StabilizerTableau,
    TensorNetwork,
    MatrixProductState,
    MonteCarloWaveFunction,
    QuantumTrajectory,
    MasterEquation,
    StochasticSchrodinger,
    Hybrid,
    Specialized(String),
}

/// Comprehensive simulator profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorProfile {
    /// Simulator identifier
    pub simulator_id: String,
    /// Simulator name
    pub name: String,
    /// Simulator version
    pub version: String,
    /// Simulator type
    pub simulator_type: SimulatorType,
    /// Supported features
    pub features: Vec<SimulatorFeature>,
    /// Technical specifications
    pub specifications: SimulatorSpecs,
    /// Capabilities
    pub capabilities: SimulatorCapabilities,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Configuration options
    pub configuration_options: ConfigurationOptions,
    /// Integration interface
    pub integration_interface: IntegrationInterface,
}

/// Technical specifications of a simulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorSpecs {
    /// Maximum number of qubits supported
    pub max_qubits: usize,
    /// Maximum circuit depth
    pub max_circuit_depth: usize,
    /// Supported gate types
    pub supported_gates: Vec<String>,
    /// Precision (bits)
    pub precision: u32,
    /// Memory architecture
    pub memory_architecture: MemoryArchitecture,
    /// Parallelization support
    pub parallelization: ParallelizationSupport,
    /// Hardware acceleration
    pub hardware_acceleration: Vec<HardwareAcceleration>,
}

/// Memory architecture details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryArchitecture {
    SingleNode,
    DistributedMemory,
    SharedMemory,
    HybridMemory,
    GPU,
    Custom(String),
}

/// Parallelization support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationSupport {
    /// Thread-level parallelism
    pub threading: bool,
    /// Process-level parallelism
    pub multiprocessing: bool,
    /// GPU parallelism
    pub gpu_acceleration: bool,
    /// Distributed computing
    pub distributed: bool,
    /// Vector operations
    pub vectorization: bool,
}

/// Hardware acceleration types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HardwareAcceleration {
    CPU,
    GPU,
    TPU,
    FPGA,
    QuantumHardware,
    Custom(String),
}

/// Simulator capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorCapabilities {
    /// Noise modeling capabilities
    pub noise_modeling: NoiseModelingCapabilities,
    /// Measurement capabilities
    pub measurement_capabilities: MeasurementCapabilities,
    /// Optimization capabilities
    pub optimization_capabilities: OptimizationCapabilities,
    /// Analysis capabilities
    pub analysis_capabilities: AnalysisCapabilities,
    /// Export capabilities
    pub export_capabilities: ExportCapabilities,
}

/// Noise modeling capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseModelingCapabilities {
    /// Supported noise models
    pub noise_models: Vec<NoiseModel>,
    /// Custom noise support
    pub custom_noise: bool,
    /// Correlated noise support
    pub correlated_noise: bool,
    /// Time-dependent noise
    pub time_dependent_noise: bool,
    /// Realistic device noise
    pub device_noise_profiles: bool,
}

/// Noise model types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NoiseModel {
    Depolarizing,
    BitFlip,
    PhaseFlip,
    PhaseDamping,
    AmplitudeDamping,
    Thermal,
    Coherent,
    Pauli,
    Kraus,
    Custom(String),
}

/// Measurement capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCapabilities {
    /// Computational basis measurements
    pub computational_basis: bool,
    /// Pauli measurements
    pub pauli_measurements: bool,
    /// General POVM measurements
    pub povm_measurements: bool,
    /// Weak measurements
    pub weak_measurements: bool,
    /// Mid-circuit measurements
    pub mid_circuit_measurements: bool,
    /// Conditional operations
    pub conditional_operations: bool,
}

/// Optimization capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationCapabilities {
    /// Circuit optimization
    pub circuit_optimization: bool,
    /// Memory optimization
    pub memory_optimization: bool,
    /// Execution optimization
    pub execution_optimization: bool,
    /// Parallel optimization
    pub parallel_optimization: bool,
    /// Custom optimization algorithms
    pub custom_algorithms: Vec<String>,
}

/// Analysis capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCapabilities {
    /// State analysis
    pub state_analysis: bool,
    /// Entanglement analysis
    pub entanglement_analysis: bool,
    /// Fidelity calculations
    pub fidelity_calculations: bool,
    /// Process tomography
    pub process_tomography: bool,
    /// Statistical analysis
    pub statistical_analysis: bool,
    /// Visualization support
    pub visualization: bool,
}

/// Export capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportCapabilities {
    /// Supported output formats
    pub output_formats: Vec<OutputFormat>,
    /// Data streaming
    pub data_streaming: bool,
    /// Real-time export
    pub realtime_export: bool,
    /// Compression support
    pub compression: bool,
}

/// Output format types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputFormat {
    JSON,
    HDF5,
    CSV,
    Binary,
    NumPy,
    MATLAB,
    Custom(String),
}

/// Performance profile characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Execution speed characteristics
    pub speed_profile: SpeedProfile,
    /// Memory usage characteristics
    pub memory_profile: MemoryProfile,
    /// Accuracy characteristics
    pub accuracy_profile: AccuracyProfile,
    /// Scalability characteristics
    pub scalability_profile: ScalabilityProfile,
    /// Stability metrics
    pub stability_metrics: StabilityMetrics,
}

/// Speed performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedProfile {
    /// Initialization time
    pub initialization_time: Duration,
    /// Gate execution rate (gates/sec)
    pub gate_execution_rate: f64,
    /// Circuit compilation time
    pub compilation_time: Duration,
    /// Measurement time
    pub measurement_time: Duration,
    /// Time complexity scaling
    pub time_complexity: ComplexityScaling,
}

/// Memory usage profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Base memory usage (MB)
    pub base_memory: f64,
    /// Memory per qubit (MB)
    pub memory_per_qubit: f64,
    /// Peak memory usage (MB)
    pub peak_memory: f64,
    /// Memory complexity scaling
    pub memory_complexity: ComplexityScaling,
    /// Memory efficiency rating
    pub efficiency_rating: f64,
}

/// Accuracy profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyProfile {
    /// Numerical precision
    pub numerical_precision: f64,
    /// Fidelity with exact results
    pub exact_fidelity: f64,
    /// Error accumulation rate
    pub error_accumulation: f64,
    /// Accuracy vs circuit depth
    pub depth_accuracy_scaling: ComplexityScaling,
    /// Accuracy consistency
    pub consistency_score: f64,
}

/// Scalability profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityProfile {
    /// Qubit scalability
    pub qubit_scaling: ComplexityScaling,
    /// Depth scalability
    pub depth_scaling: ComplexityScaling,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Maximum practical size
    pub max_practical_qubits: usize,
    /// Resource scaling efficiency
    pub resource_efficiency: f64,
}

/// Stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// Crash rate
    pub crash_rate: f64,
    /// Result consistency
    pub result_consistency: f64,
    /// Error handling quality
    pub error_handling: f64,
    /// Memory leak rate
    pub memory_leak_rate: f64,
    /// Long-run stability
    pub long_run_stability: f64,
}

/// Complexity scaling characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityScaling {
    Constant,
    Linear,
    Quadratic,
    Exponential,
    Factorial,
    Polynomial { degree: f64 },
    Logarithmic,
    Unknown,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum CPU cores
    pub min_cpu_cores: u32,
    /// Minimum RAM (GB)
    pub min_ram_gb: f64,
    /// Minimum disk space (GB)
    pub min_disk_gb: f64,
    /// GPU requirements
    pub gpu_requirements: Option<GPURequirements>,
    /// Network requirements
    pub network_requirements: Option<NetworkRequirements>,
    /// Operating system requirements
    pub os_requirements: Vec<String>,
}

/// GPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPURequirements {
    /// Minimum GPU memory (GB)
    pub min_gpu_memory_gb: f64,
    /// Required compute capability
    pub compute_capability: String,
    /// Preferred GPU architecture
    pub preferred_architecture: Vec<String>,
    /// Multiple GPU support
    pub multi_gpu_support: bool,
}

/// Network requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRequirements {
    /// Bandwidth requirements (Mbps)
    pub bandwidth_mbps: f64,
    /// Latency requirements (ms)
    pub max_latency_ms: f64,
    /// Internet connectivity required
    pub internet_required: bool,
    /// Cluster networking
    pub cluster_networking: bool,
}

/// Configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationOptions {
    /// Available precision levels
    pub precision_levels: Vec<u32>,
    /// Optimization levels
    pub optimization_levels: Vec<String>,
    /// Memory management options
    pub memory_options: Vec<String>,
    /// Parallelization options
    pub parallel_options: Vec<String>,
    /// Custom parameters
    pub custom_parameters: HashMap<String, ParameterSpec>,
}

/// Parameter specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpec {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Default value
    pub default_value: String,
    /// Valid range or options
    pub valid_range: Option<ParameterRange>,
    /// Description
    pub description: String,
}

/// Parameter types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterType {
    Integer,
    Float,
    Boolean,
    String,
    Enum(Vec<String>),
}

/// Parameter range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    IntegerRange { min: i64, max: i64 },
    FloatRange { min: f64, max: f64 },
    StringOptions(Vec<String>),
}

/// Integration interface specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationInterface {
    /// API type
    pub api_type: APIType,
    /// Connection parameters
    pub connection_params: ConnectionParameters,
    /// Authentication requirements
    pub auth_requirements: AuthenticationSpec,
    /// Data format specifications
    pub data_formats: DataFormatSpec,
    /// Error handling
    pub error_handling: ErrorHandlingSpec,
}

/// API type specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum APIType {
    REST,
    GraphQL,
    GRpc,
    WebSocket,
    Native,
    Library,
    CommandLine,
    Custom(String),
}

/// Connection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionParameters {
    /// Endpoint URL
    pub endpoint: Option<String>,
    /// Port number
    pub port: Option<u16>,
    /// Protocol version
    pub protocol_version: String,
    /// Connection timeout
    pub timeout: Duration,
    /// SSL/TLS requirements
    pub ssl_required: bool,
}

/// Authentication specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationSpec {
    /// Authentication method
    pub auth_method: AuthMethod,
    /// Required credentials
    pub required_credentials: Vec<String>,
    /// Token validity
    pub token_validity: Option<Duration>,
    /// Refresh mechanism
    pub refresh_mechanism: bool,
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthMethod {
    None,
    APIKey,
    OAuth2,
    JWT,
    Basic,
    Certificate,
    Custom(String),
}

/// Data format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFormatSpec {
    /// Input formats
    pub input_formats: Vec<DataFormat>,
    /// Output formats
    pub output_formats: Vec<DataFormat>,
    /// Streaming support
    pub streaming_support: bool,
    /// Compression options
    pub compression_options: Vec<String>,
}

/// Data format types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataFormat {
    QASM,
    Cirq,
    Qiskit,
    Braket,
    OpenQASM,
    JSON,
    Protocol,
    Custom(String),
}

/// Error handling specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingSpec {
    /// Error reporting format
    pub error_format: ErrorFormat,
    /// Retry mechanisms
    pub retry_support: bool,
    /// Error recovery
    pub recovery_mechanisms: Vec<String>,
    /// Debugging support
    pub debugging_support: bool,
}

/// Error format types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorFormat {
    Standard,
    Structured,
    Detailed,
    Custom(String),
}

/// Benchmark suite for comprehensive testing
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    /// Benchmark test cases
    pub benchmarks: Vec<BenchmarkTest>,
    /// Suite configuration
    pub config: BenchmarkSuiteConfig,
    /// Reference results
    pub reference_results: HashMap<String, ReferenceResult>,
}

/// Individual benchmark test
#[derive(Debug, Clone)]
pub struct BenchmarkTest {
    /// Test identifier
    pub test_id: String,
    /// Test name
    pub name: String,
    /// Test description
    pub description: String,
    /// Test category
    pub category: BenchmarkCategory,
    /// Test circuits
    pub circuits: Vec<BenchmarkCircuit>,
    /// Test parameters
    pub parameters: BenchmarkParameters,
    /// Success criteria
    pub success_criteria: SuccessCriteria,
    /// Expected outcomes
    pub expected_outcomes: Option<ExpectedOutcomes>,
}

/// Benchmark categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    Performance,
    Accuracy,
    Scalability,
    Memory,
    Features,
    Stability,
    Integration,
    Noise,
    Algorithms,
    Custom(String),
}

/// Benchmark circuit specification
#[derive(Debug, Clone)]
pub struct BenchmarkCircuit {
    /// Circuit identifier
    pub circuit_id: String,
    /// Circuit generator function
    pub generator: CircuitGenerator,
    /// Circuit parameters
    pub parameters: CircuitParameters,
    /// Circuit metadata
    pub metadata: CircuitMetadata,
}

/// Circuit generator types
#[derive(Clone)]
pub enum CircuitGenerator {
    Predefined(String),
    Random(RandomCircuitSpec),
    Algorithmic(AlgorithmicCircuitSpec),
    /// Custom generator using a shared closure.
    Custom(
        std::sync::Arc<
            dyn Fn(&CircuitParameters) -> DeviceResult<Box<dyn CircuitInterface>> + Send + Sync,
        >,
    ),
}

impl std::fmt::Debug for CircuitGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Predefined(s) => write!(f, "CircuitGenerator::Predefined({s:?})"),
            Self::Random(r) => write!(f, "CircuitGenerator::Random({r:?})"),
            Self::Algorithmic(a) => write!(f, "CircuitGenerator::Algorithmic({a:?})"),
            Self::Custom(_) => write!(f, "CircuitGenerator::Custom(<fn>)"),
        }
    }
}

/// Random circuit specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomCircuitSpec {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Gate distribution
    pub gate_distribution: HashMap<String, f64>,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

/// Algorithmic circuit specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmicCircuitSpec {
    /// Algorithm type
    pub algorithm: AlgorithmType,
    /// Algorithm parameters
    pub parameters: HashMap<String, String>,
    /// Problem instance
    pub problem_instance: Option<String>,
}

/// Algorithm types for benchmarking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlgorithmType {
    QFT,
    Grover,
    Shor,
    VQE,
    QAOA,
    QuantumWalk,
    QuantumSupremacy,
    BernsteinVazirani,
    DeutschJozsa,
    SimonsAlgorithm,
    Custom(String),
}

/// Circuit parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitParameters {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Number of measurements
    pub num_measurements: usize,
    /// Noise parameters
    pub noise_params: Option<NoiseParameters>,
    /// Custom parameters
    pub custom_params: HashMap<String, String>,
}

/// Noise parameters for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseParameters {
    /// Gate error rate
    pub gate_error_rate: f64,
    /// Readout error rate
    pub readout_error_rate: f64,
    /// Decoherence time
    pub decoherence_time: f64,
    /// Noise model type
    pub noise_model: NoiseModel,
}

/// Circuit metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetadata {
    /// Expected execution time
    pub expected_time: Duration,
    /// Expected memory usage
    pub expected_memory: f64,
    /// Difficulty level
    pub difficulty: DifficultyLevel,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Difficulty levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Trivial,
    Easy,
    Medium,
    Hard,
    Expert,
    Extreme,
}

/// Benchmark parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkParameters {
    /// Number of repetitions
    pub repetitions: usize,
    /// Timeout per test
    pub timeout: Duration,
    /// Shots per circuit
    pub shots: usize,
    /// Warm-up runs
    pub warmup_runs: usize,
    /// Statistical confidence level
    pub confidence_level: f64,
}

/// Success criteria for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Minimum accuracy
    pub min_accuracy: Option<f64>,
    /// Maximum memory usage
    pub max_memory_usage: Option<f64>,
    /// Success rate threshold
    pub min_success_rate: f64,
    /// Custom criteria
    pub custom_criteria: HashMap<String, f64>,
}

/// Expected outcomes for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcomes {
    /// Expected measurement probabilities
    pub measurement_probabilities: Option<HashMap<String, f64>>,
    /// Expected final state
    pub final_state: Option<Vec<scirs2_core::Complex64>>,
    /// Expected observables
    pub observables: Option<HashMap<String, f64>>,
    /// Tolerance for comparisons
    pub tolerance: f64,
}

/// Benchmark suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteConfig {
    /// Parallel execution
    pub parallel_execution: bool,
    /// Resource monitoring
    pub resource_monitoring: bool,
    /// Detailed logging
    pub detailed_logging: bool,
    /// Result caching
    pub result_caching: bool,
    /// Comparison with reference
    pub reference_comparison: bool,
}

/// Reference results for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceResult {
    /// Simulator that generated reference
    pub reference_simulator: String,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage
    pub memory_usage: f64,
    /// Accuracy metrics
    pub accuracy_metrics: HashMap<String, f64>,
    /// Output results
    pub results: HashMap<String, String>,
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    /// Monitor CPU usage
    pub monitor_cpu: bool,
    /// Monitor memory usage
    pub monitor_memory: bool,
    /// Monitor disk I/O
    pub monitor_disk: bool,
    /// Monitor network usage
    pub monitor_network: bool,
    /// Monitor GPU usage
    pub monitor_gpu: bool,
    /// Monitoring frequency
    pub monitoring_frequency: Duration,
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Output formats
    pub output_formats: Vec<OutputFormat>,
    /// Report detail level
    pub detail_level: ReportDetailLevel,
    /// Include charts
    pub include_charts: bool,
    /// Export raw data
    pub export_raw_data: bool,
    /// Comparison tables
    pub comparison_tables: bool,
}

/// Report detail levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportDetailLevel {
    Summary,
    Standard,
    Detailed,
    Comprehensive,
}

/// Comprehensive comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Comparison timestamp
    pub timestamp: SystemTime,
    /// Simulators compared
    pub simulators: Vec<String>,
    /// Overall rankings
    pub overall_rankings: Vec<SimulatorRanking>,
    /// Category-specific rankings
    pub category_rankings: HashMap<BenchmarkCategory, Vec<SimulatorRanking>>,
    /// Detailed metrics
    pub detailed_metrics: HashMap<String, SimulatorMetrics>,
    /// Performance analysis
    pub performance_analysis: PerformanceAnalysis,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
    /// Statistical analysis
    pub statistical_analysis: StatisticalAnalysis,
}

/// Simulator ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorRanking {
    /// Simulator ID
    pub simulator_id: String,
    /// Overall score
    pub overall_score: f64,
    /// Category scores
    pub category_scores: HashMap<String, f64>,
    /// Rank position
    pub rank: usize,
    /// Strengths
    pub strengths: Vec<String>,
    /// Weaknesses
    pub weaknesses: Vec<String>,
}

/// Detailed simulator metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorMetrics {
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Accuracy metrics
    pub accuracy: AccuracyMetrics,
    /// Resource metrics
    pub resources: ResourceMetrics,
    /// Reliability metrics
    pub reliability: ReliabilityMetrics,
    /// Feature coverage metrics
    pub features: FeatureMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Throughput (circuits/sec)
    pub throughput: f64,
    /// Initialization overhead
    pub initialization_time: Duration,
    /// Scaling efficiency
    pub scaling_efficiency: f64,
    /// Performance consistency
    pub consistency_score: f64,
}

/// Accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Fidelity with reference
    pub reference_fidelity: f64,
    /// Numerical precision
    pub numerical_precision: f64,
    /// Error rates
    pub error_rates: HashMap<String, f64>,
    /// Accuracy degradation with scale
    pub scale_degradation: f64,
    /// Result reproducibility
    pub reproducibility_score: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
    /// Average CPU usage (%)
    pub avg_cpu_usage: f64,
    /// Disk I/O (MB/s)
    pub disk_io_rate: f64,
    /// Network usage (MB/s)
    pub network_usage: f64,
    /// GPU utilization (%)
    pub gpu_utilization: Option<f64>,
}

/// Reliability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    /// Success rate
    pub success_rate: f64,
    /// Crash frequency
    pub crash_frequency: f64,
    /// Error recovery rate
    pub error_recovery_rate: f64,
    /// Memory leak detection
    pub memory_leak_score: f64,
    /// Long-run stability
    pub long_run_stability: f64,
}

/// Feature coverage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetrics {
    /// Supported features percentage
    pub feature_coverage: f64,
    /// Feature quality scores
    pub feature_quality: HashMap<String, f64>,
    /// Compatibility scores
    pub compatibility_scores: HashMap<String, f64>,
    /// Innovation score
    pub innovation_score: f64,
}

/// Performance analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    /// Best performer by category
    pub best_performers: HashMap<String, String>,
    /// Performance trends
    pub trends: HashMap<String, TrendAnalysis>,
    /// Correlation analysis
    pub correlations: HashMap<String, f64>,
    /// Outlier detection
    pub outliers: Vec<OutlierDetection>,
    /// Scaling analysis
    pub scaling_analysis: ScalingAnalysis,
}

/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f64,
    /// R-squared value
    pub r_squared: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Trend directions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Outlier detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetection {
    /// Simulator with outlier behavior
    pub simulator_id: String,
    /// Metric with outlier values
    pub metric: String,
    /// Outlier score
    pub outlier_score: f64,
    /// Explanation
    pub explanation: String,
}

/// Scaling analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    /// Qubit scaling analysis
    pub qubit_scaling: HashMap<String, ComplexityScaling>,
    /// Depth scaling analysis
    pub depth_scaling: HashMap<String, ComplexityScaling>,
    /// Memory scaling analysis
    pub memory_scaling: HashMap<String, ComplexityScaling>,
    /// Performance predictions
    pub predictions: HashMap<String, PerformancePrediction>,
}

/// Performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePrediction {
    /// Predicted execution time
    pub execution_time: Duration,
    /// Predicted memory usage
    pub memory_usage: f64,
    /// Confidence level
    pub confidence: f64,
    /// Prediction range
    pub range: (f64, f64),
}

/// Recommendation for simulator selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommended simulator
    pub simulator_id: String,
    /// Use case description
    pub use_case: String,
    /// Confidence score
    pub confidence: f64,
    /// Reasoning
    pub reasoning: String,
    /// Alternative options
    pub alternatives: Vec<String>,
}

/// Recommendation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecommendationType {
    BestOverall,
    BestForSpeed,
    BestForAccuracy,
    BestForMemory,
    BestForScalability,
    BestForFeatures,
    BestForBeginners,
    BestForExperts,
    Custom(String),
}

/// Statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// ANOVA results
    pub anova_results: HashMap<String, ANOVAResult>,
    /// Correlation matrix
    pub correlationmatrix: HashMap<String, HashMap<String, f64>>,
    /// Significance tests
    pub significance_tests: HashMap<String, SignificanceTest>,
    /// Effect sizes
    pub effect_sizes: HashMap<String, f64>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

/// ANOVA analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ANOVAResult {
    /// F-statistic
    pub f_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: (u32, u32),
    /// Significant difference detected
    pub significant: bool,
}

/// Statistical significance test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceTest {
    /// Test type
    pub test_type: String,
    /// Test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Significant result
    pub significant: bool,
    /// Effect size
    pub effect_size: f64,
}

/// Performance analytics engine
#[derive(Debug)]
pub struct PerformanceAnalytics {
    /// Historical data
    pub(super) historical_data: Vec<ComparisonResult>,
    /// Performance trends
    pub(super) trends: HashMap<String, Vec<DataPoint>>,
    /// Prediction models
    pub(super) prediction_models: HashMap<String, Box<dyn PredictionModel>>,
    /// Anomaly detection
    pub(super) anomaly_detector: Box<dyn AnomalyDetector>,
}

/// Data point for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Value
    pub value: f64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Prediction model trait
pub trait PredictionModel: Send + Sync + std::fmt::Debug {
    /// Make a prediction
    fn predict(&self, features: &[f64]) -> f64;
    /// Update model with new data
    fn update(&mut self, features: &[f64], target: f64);
    /// Get model accuracy
    fn accuracy(&self) -> f64;
}

/// Anomaly detection trait
pub trait AnomalyDetector: Send + Sync + std::fmt::Debug {
    /// Detect anomalies in data
    fn detect_anomalies(&self, data: &[f64]) -> Vec<usize>;
    /// Update detector with new data
    fn update(&mut self, data: &[f64]);
    /// Get anomaly threshold
    fn threshold(&self) -> f64;
}

/// ML-powered recommendation engine
#[derive(Debug)]
pub struct RecommendationEngine {
    /// Feature extractors
    pub(super) feature_extractors: Vec<Box<dyn FeatureExtractor>>,
    /// Recommendation models
    pub(super) models: HashMap<RecommendationType, Box<dyn RecommendationModel>>,
    /// Training data
    pub(super) training_data: Vec<TrainingExample>,
    /// Model performance metrics
    pub(super) model_metrics: HashMap<String, f64>,
}

/// Feature extractor trait
pub trait FeatureExtractor: Send + Sync + std::fmt::Debug {
    /// Extract features from simulator profile
    fn extract_features(&self, profile: &SimulatorProfile) -> Vec<f64>;
    /// Get feature names
    fn feature_names(&self) -> Vec<String>;
}

/// Recommendation model trait
pub trait RecommendationModel: Send + Sync + std::fmt::Debug {
    /// Generate recommendations
    fn recommend(&self, features: &[f64], context: &RecommendationContext) -> Vec<Recommendation>;
    /// Train the model
    fn train(&mut self, examples: &[TrainingExample]);
    /// Evaluate model performance
    fn evaluate(&self, test_data: &[TrainingExample]) -> f64;
}

/// Training example for ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Features
    pub features: Vec<f64>,
    /// Target recommendation
    pub target: String,
    /// Context information
    pub context: RecommendationContext,
    /// User feedback
    pub feedback: Option<f64>,
}

/// Recommendation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationContext {
    /// Circuit characteristics
    pub circuit_info: CircuitInfo,
    /// Performance requirements
    pub requirements: PerformanceRequirements,
    /// Resource constraints
    pub constraints: ResourceConstraints,
    /// User preferences
    pub preferences: UserPreferences,
}

/// Circuit information for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitInfo {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Gate types used
    pub gate_types: Vec<String>,
    /// Algorithm type
    pub algorithm_type: Option<AlgorithmType>,
    /// Noise requirements
    pub noise_modeling: bool,
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Minimum accuracy
    pub min_accuracy: Option<f64>,
    /// Maximum memory usage
    pub max_memory_usage: Option<f64>,
    /// Required features
    pub required_features: Vec<SimulatorFeature>,
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Available CPU cores
    pub cpu_cores: u32,
    /// Available RAM (GB)
    pub ram_gb: f64,
    /// GPU availability
    pub gpu_available: bool,
    /// Network bandwidth (Mbps)
    pub bandwidth_mbps: f64,
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Preferred simulator types
    pub preferred_types: Vec<SimulatorType>,
    /// Importance weights
    pub importance_weights: HashMap<String, f64>,
    /// Excluded simulators
    pub excluded_simulators: Vec<String>,
    /// Experience level
    pub experience_level: ExperienceLevel,
}

/// User experience levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExperienceLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Benchmark executor
#[derive(Debug)]
pub struct BenchmarkExecutor {
    /// Execution configuration
    pub(super) config: ExecutorConfig,
    /// Resource monitor
    pub(super) resource_monitor: ResourceMonitor,
    /// Result collector
    pub(super) result_collector: ResultCollector,
    /// Parallel execution pool
    pub(super) execution_pool: Option<tokio::runtime::Runtime>,
}

/// Executor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// Maximum parallel executions
    pub max_parallel: usize,
    /// Timeout per benchmark
    pub benchmark_timeout: Duration,
    /// Retry on failure
    pub retry_on_failure: bool,
    /// Maximum retries
    pub max_retries: u32,
    /// Isolation mode
    pub isolation_mode: IsolationMode,
}

/// Execution isolation modes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IsolationMode {
    None,
    Process,
    Container,
    VM,
}

/// Resource monitor
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Monitoring channels
    pub(super) monitoring_channels: Vec<mpsc::Sender<ResourceUpdate>>,
    /// Current measurements
    pub(super) current_measurements: Arc<RwLock<ResourceMeasurements>>,
    /// Historical measurements
    pub(super) historical_measurements: Vec<ResourceMeasurements>,
}

/// Resource update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUpdate {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Simulator ID
    pub simulator_id: String,
    /// Resource measurements
    pub measurements: ResourceMeasurements,
}

/// Resource measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMeasurements {
    /// CPU usage (%)
    pub cpu_usage: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
    /// GPU usage (%)
    pub gpu_usage: Option<f64>,
    /// Disk I/O rate (MB/s)
    pub disk_io_rate: f64,
    /// Network I/O rate (MB/s)
    pub network_io_rate: f64,
    /// Temperature (°C)
    pub temperature: Option<f64>,
}

/// Result collector
#[derive(Debug)]
pub struct ResultCollector {
    /// Collected results
    pub(super) results: Arc<RwLock<HashMap<String, BenchmarkResult>>>,
    /// Result processing pipeline
    pub(super) processing_pipeline: Vec<Box<dyn ResultProcessor>>,
    /// Export handlers
    pub(super) export_handlers: HashMap<OutputFormat, Box<dyn ExportHandler>>,
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark ID
    pub benchmark_id: String,
    /// Simulator ID
    pub simulator_id: String,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage
    pub memory_usage: f64,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Detailed metrics
    pub metrics: HashMap<String, f64>,
    /// Resource usage during execution
    pub resource_usage: ResourceMeasurements,
    /// Output data
    pub output_data: HashMap<String, String>,
    /// Metadata
    pub metadata: BenchmarkMetadata,
}

/// Benchmark metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// Execution timestamp
    pub timestamp: SystemTime,
    /// Environment information
    pub environment: EnvironmentInfo,
    /// Configuration used
    pub configuration: HashMap<String, String>,
    /// Validation results
    pub validation: Option<ValidationResult>,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Operating system
    pub os: String,
    /// CPU information
    pub cpu_info: String,
    /// Memory information
    pub memory_info: String,
    /// GPU information
    pub gpu_info: Option<String>,
    /// Network information
    pub network_info: String,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation passed
    pub passed: bool,
    /// Accuracy score
    pub accuracy_score: f64,
    /// Detailed validation metrics
    pub validation_metrics: HashMap<String, f64>,
    /// Comparison with expected results
    pub expected_comparison: Option<ExpectedComparison>,
}

/// Expected comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedComparison {
    /// Fidelity with expected result
    pub fidelity: f64,
    /// Statistical distance
    pub statistical_distance: f64,
    /// Measurement differences
    pub measurement_differences: HashMap<String, f64>,
}

/// Result processor trait
pub trait ResultProcessor: Send + Sync + std::fmt::Debug {
    /// Process benchmark result
    fn process(&self, result: &mut BenchmarkResult);
    /// Get processor name
    fn name(&self) -> &str;
}

/// Export handler trait
pub trait ExportHandler: Send + Sync + std::fmt::Debug {
    /// Export results in specific format
    fn export(&self, results: &[BenchmarkResult], output_path: &str) -> DeviceResult<()>;
    /// Get supported format
    fn format(&self) -> OutputFormat;
}
