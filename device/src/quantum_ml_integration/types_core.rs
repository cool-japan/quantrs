//! QML Integration Types - Part 1 of 3
//!
//! Auto-generated module - split from large types.rs file
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{VecDeque, HashMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use scirs2_core::Complex64;

// Import from crate modules
use crate::{DeviceError, DeviceResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Percentiles
    pub percentiles: HashMap<u8, f64>,
    /// Sample count
    pub count: usize,
}
/// Classical processing components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalComponent {
    /// Component type
    pub component_type: ClassicalComponentType,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Parameters
    pub parameters: Array1<f64>,
    /// Activation function
    pub activation: ActivationFunction,
}
/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Minimum gate fidelity
    pub min_gate_fidelity: f64,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Minimum coherence time
    pub min_coherence_time: Duration,
    /// Maximum error rate
    pub max_error_rate: f64,
}
/// Idle pattern
#[derive(Debug, Clone)]
pub struct IdlePattern {
    /// Pattern start time
    pub start_time: Instant,
    /// Pattern duration
    pub duration: Duration,
    /// Resources affected
    pub affected_resources: Vec<String>,
    /// Pattern type
    pub pattern_type: IdlePatternType,
}
/// Cached dataset
#[derive(Debug, Clone)]
pub struct CachedDataset {
    /// Dataset
    pub dataset: QMLDataset,
    /// Cache timestamp
    pub cached_at: Instant,
    /// Access count
    pub access_count: usize,
    /// Cache size (bytes)
    pub size_bytes: usize,
}
/// Conversion result
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Converted model
    pub model: QMLModel,
    /// Conversion time
    pub conversion_time: Duration,
    /// Conversion accuracy
    pub accuracy: f64,
    /// Cache timestamp
    pub cached_at: Instant,
}
/// QML resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLResourceConfig {
    /// Maximum quantum circuit executions per training step
    pub max_circuits_per_step: usize,
    /// Memory limit for classical computation (MB)
    pub memory_limit_mb: usize,
    /// Parallel execution configuration
    pub parallel_config: ParallelExecutionConfig,
    /// Caching strategy
    pub caching_strategy: CachingStrategy,
    /// Resource allocation priorities
    pub resource_priorities: ResourcePriorities,
}
/// Simple ML anomaly detector implementation
#[derive(Debug)]
pub struct SimpleMLAnomalyDetector {
    threshold: f64,
}
impl SimpleMLAnomalyDetector {
    pub fn new() -> Self {
        Self { threshold: 2.0 }
    }
}
/// Bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Resource affected
    pub resource: String,
    /// Severity
    pub severity: f64,
    /// Duration
    pub duration: Duration,
    /// Impact on performance
    pub performance_impact: f64,
}
/// Training history tracking
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Loss history
    pub loss_history: Vec<f64>,
    /// Validation loss history
    pub val_loss_history: Vec<f64>,
    /// Metric history
    pub metric_history: HashMap<String, Vec<f64>>,
    /// Learning rate history
    pub lr_history: Vec<f64>,
    /// Gradient norm history
    pub gradient_norm_history: Vec<f64>,
    /// Parameter norm history
    pub parameter_norm_history: Vec<f64>,
}
/// Resource analytics
#[derive(Debug, Clone)]
pub struct ResourceAnalytics {
    /// Utilization analytics
    pub utilization_analytics: UtilizationAnalytics,
    /// Cost analytics
    pub cost_analytics: CostAnalytics,
    /// Efficiency analytics
    pub efficiency_analytics: EfficiencyMetrics,
    /// Bottleneck analysis
    pub bottleneck_analysis: BottleneckAnalysis,
}
/// Quantum encoding types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumEncodingType {
    Amplitude,
    Angle,
    Basis,
    Displacement,
    Squeezed,
    Custom(String),
}
/// Forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Forecasted values
    pub values: Vec<f64>,
    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Forecast horizon
    pub horizon: Duration,
    /// Forecast accuracy
    pub accuracy: f64,
}
/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    /// Accuracy metrics
    pub accuracy: f64,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// AUC-ROC
    pub auc_roc: f64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}
/// Progress metrics for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressMetrics {
    /// Current epoch
    pub current_epoch: usize,
    /// Total epochs
    pub total_epochs: usize,
    /// Progress percentage
    pub progress_percentage: f64,
    /// Estimated time remaining
    pub estimated_time_remaining: Duration,
    /// Current loss
    pub current_loss: f64,
    /// Best loss achieved
    pub best_loss: f64,
    /// Learning rate
    pub learning_rate: f64,
}
/// Recommendation priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}
/// Types of classical components
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassicalComponentType {
    Dense,
    Convolutional,
    Recurrent,
    Attention,
    Normalization,
    Dropout,
    Custom(String),
}
/// Resource efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEfficiencyAnalysis {
    /// Overall efficiency score
    pub overall_efficiency: f64,
    /// Quantum efficiency
    pub quantum_efficiency: f64,
    /// Classical efficiency
    pub classical_efficiency: f64,
    /// Memory efficiency
    pub memory_efficiency: f64,
    /// Cost efficiency
    pub cost_efficiency: f64,
    /// Efficiency trends
    pub efficiency_trends: HashMap<String, TrendAnalysis>,
}
/// QML monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLMonitoringConfig {
    /// Enable comprehensive monitoring
    pub enable_monitoring: bool,
    /// Metrics collection frequency
    pub collection_frequency: Duration,
    /// Performance tracking
    pub performance_tracking: PerformanceTrackingConfig,
    /// Resource monitoring
    pub resource_monitoring: ResourceMonitoringConfig,
    /// Alert configuration
    pub alert_config: AlertConfig,
}
/// Data formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataFormat {
    CSV,
    JSON,
    HDF5,
    NumPy,
    Parquet,
    Custom(String),
}
/// Quantum encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEncoding {
    /// Encoding type
    pub encoding_type: QuantumEncodingType,
    /// Encoding parameters
    pub parameters: HashMap<String, f64>,
    /// Number of qubits used
    pub qubits_used: usize,
    /// Encoding efficiency
    pub efficiency: f64,
}
/// Convergence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    /// Convergence rate
    pub convergence_rate: f64,
    /// Stability measure
    pub stability: f64,
    /// Plateau detection
    pub plateau_detected: bool,
    /// Oscillation measure
    pub oscillation: f64,
    /// Final gradient norm
    pub final_gradient_norm: f64,
}
/// QML training state
#[derive(Debug, Clone)]
pub struct QMLTrainingState {
    /// Current epoch
    pub current_epoch: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: Option<f64>,
    /// Learning rate
    pub learning_rate: f64,
    /// Optimizer state
    pub optimizer_state: OptimizerState,
    /// Training history
    pub training_history: TrainingHistory,
    /// Early stopping state
    pub early_stopping_state: EarlyStoppingState,
}
/// QML dataset
#[derive(Debug, Clone)]
pub struct QMLDataset {
    /// Features
    pub features: Array2<f64>,
    /// Labels
    pub labels: Array1<f64>,
    /// Metadata
    pub metadata: DatasetMetadata,
    /// Quantum encoding
    pub quantum_encoding: Option<QuantumEncoding>,
}
/// Implementation effort levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}
/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_lambda: f64,
    /// L2 regularization strength
    pub l2_lambda: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Quantum noise regularization
    pub quantum_noise: f64,
    /// Parameter constraint enforcement
    pub parameter_constraints: ParameterConstraints,
}
/// QML Data Pipeline
pub struct QMLDataPipeline {
    /// Data sources
    data_sources: HashMap<String, Box<dyn QMLDataSource>>,
    /// Data processors
    data_processors: Vec<Box<dyn QMLDataProcessor>>,
    /// Data cache
    data_cache: Arc<RwLock<HashMap<String, CachedDataset>>>,
    /// Pipeline configuration
    config: DataPipelineConfig,
}
impl QMLDataPipeline {
    pub fn new(config: QMLIntegrationConfig) -> DeviceResult<Self> {
        Ok(Self {
            data_sources: HashMap::new(),
            data_processors: Vec::new(),
            data_cache: Arc::new(RwLock::new(HashMap::new())),
            config: DataPipelineConfig {
                enable_caching: true,
                cache_size_limit_mb: 1024,
                preprocessing_steps: Vec::new(),
                enable_parallel_processing: true,
                processing_batch_size: 1000,
            },
        })
    }
}
/// Parallel execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionConfig {
    /// Enable parallel circuit execution
    pub enable_parallel_circuits: bool,
    /// Maximum parallel workers
    pub max_workers: usize,
    /// Batch processing configuration
    pub batch_processing: BatchProcessingConfig,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}
/// Bridge performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgePerformanceMetrics {
    /// Conversion time
    pub avg_conversion_time: Duration,
    /// Execution time
    pub avg_execution_time: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Accuracy preservation
    pub accuracy_preservation: f64,
}
/// Classical resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalResourceRequirements {
    /// CPU cores needed
    pub cpu_cores: usize,
    /// Memory needed (MB)
    pub memory_mb: usize,
    /// GPU requirements
    pub gpu_requirements: Option<GPURequirements>,
    /// Storage requirements (MB)
    pub storage_mb: usize,
    /// Network bandwidth (Mbps)
    pub network_bandwidth: Option<f64>,
}
/// Types of quantum ML models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QMLModelType {
    QuantumNeuralNetwork,
    VariationalQuantumEigensolver,
    QuantumApproximateOptimization,
    QuantumClassifier,
    QuantumRegressor,
    QuantumGAN,
    QuantumAutoencoder,
    QuantumReinforcement,
    HybridClassical,
    Custom(String),
}
/// QML layer definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLLayer {
    /// Layer type
    pub layer_type: QMLLayerType,
    /// Layer parameters
    pub parameters: HashMap<String, f64>,
    /// Qubit connectivity
    pub connectivity: Vec<(usize, usize)>,
    /// Gate sequence
    pub gate_sequence: Vec<QMLGate>,
}
/// Supported ML frameworks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MLFramework {
    TensorFlow,
    PyTorch,
    PennyLane,
    Qiskit,
    Cirq,
    JAX,
    Custom(String),
}
/// Measurement strategies for QML
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeasurementStrategy {
    Computational,
    Pauli,
    Bell,
    Custom(String),
}
/// Entanglement patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntanglementPattern {
    Linear,
    Circular,
    AllToAll,
    Random,
    Hardware,
    Custom(Vec<(usize, usize)>),
}
/// GPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPURequirements {
    /// Minimum GPU memory (GB)
    pub min_memory_gb: f64,
    /// Compute capability required
    pub compute_capability: String,
    /// Number of GPUs
    pub num_gpus: usize,
    /// Preferred GPU type
    pub preferred_type: Option<String>,
}
/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingConfig {
    /// Dynamic batch sizing
    pub dynamic_batch_size: bool,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch size adaptation strategy
    pub adaptation_strategy: BatchAdaptationStrategy,
}
/// Active alert
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    /// Alert ID
    pub alert_id: String,
    /// Alert type
    pub alert_type: String,
    /// Severity
    pub severity: f64,
    /// Start time
    pub start_time: Instant,
    /// Description
    pub description: String,
    /// Escalation level
    pub escalation_level: usize,
    /// Acknowledged
    pub acknowledged: bool,
}
/// QML model architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLArchitecture {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit layers
    pub layers: Vec<QMLLayer>,
    /// Measurement strategy
    pub measurement_strategy: MeasurementStrategy,
    /// Entanglement pattern
    pub entanglement_pattern: EntanglementPattern,
    /// Classical processing components
    pub classical_components: Vec<ClassicalComponent>,
}
/// Allocation record
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Timestamp
    pub timestamp: Instant,
    /// Session ID
    pub session_id: String,
    /// Resources allocated
    pub allocation: ResourceAllocation,
    /// Allocation duration
    pub duration: Duration,
    /// Allocation efficiency
    pub efficiency: f64,
}
/// Batch size adaptation strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BatchAdaptationStrategy {
    Fixed,
    Linear,
    Exponential,
    Performance,
    Memory,
}
/// Quantum resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResourceRequirements {
    /// Number of qubits needed
    pub qubits_needed: usize,
    /// Circuit executions per epoch
    pub circuits_per_epoch: usize,
    /// Required gate fidelity
    pub required_fidelity: f64,
    /// Required coherence time
    pub required_coherence: Duration,
    /// Preferred quantum backend
    pub preferred_backend: Option<String>,
}
/// Early stopping state
#[derive(Debug, Clone)]
pub struct EarlyStoppingState {
    /// Best metric value
    pub best_metric: f64,
    /// Epochs without improvement
    pub patience_counter: usize,
    /// Best parameters
    pub best_parameters: Option<Array1<f64>>,
    /// Should stop training
    pub should_stop: bool,
}
/// Anomaly flag
#[derive(Debug, Clone)]
pub struct AnomalyFlag {
    /// Timestamp
    pub timestamp: Instant,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity
    pub severity: f64,
    /// Description
    pub description: String,
}
/// Comparative analytics
#[derive(Debug, Clone)]
pub struct ComparativeAnalytics {
    /// Model comparisons
    pub model_comparisons: HashMap<String, ModelComparison>,
    /// Algorithm comparisons
    pub algorithm_comparisons: HashMap<String, AlgorithmComparison>,
    /// Framework comparisons
    pub framework_comparisons: HashMap<MLFramework, FrameworkComparison>,
    /// Benchmark results
    pub benchmark_results: BenchmarkResults,
}
/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,
    /// Number of samples
    pub num_samples: usize,
    /// Number of features
    pub num_features: usize,
    /// Data type
    pub data_type: DataType,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Preprocessing applied
    pub preprocessing: Vec<String>,
}
/// Resource utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// GPU utilization
    pub gpu_utilization: HashMap<usize, f64>,
    /// Storage utilization
    pub storage_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
}
/// Data processor types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataProcessorType {
    Normalization,
    Encoding,
    Augmentation,
    Filtering,
    Transformation,
    Custom(String),
}
/// Data source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceInfo {
    /// Source name
    pub name: String,
    /// Source type
    pub source_type: DataSourceType,
    /// Supported formats
    pub supported_formats: Vec<DataFormat>,
    /// Description
    pub description: String,
}
/// Optimizer types for QML
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizerType {
    Adam,
    SGD,
    RMSprop,
    Adagrad,
    LBFGS,
    NaturalGradient,
    SPSA,
    GradientDescent,
    QuantumNaturalGradient,
    Rotosolve,
    Custom(String),
}
/// QML data batch
#[derive(Debug, Clone)]
pub struct QMLDataBatch {
    /// Batch features
    pub features: Array2<f64>,
    /// Batch labels
    pub labels: Array1<f64>,
    /// Batch size
    pub batch_size: usize,
    /// Batch index
    pub batch_index: usize,
    /// Quantum states (if pre-computed)
    pub quantum_states: Option<Array2<Complex64>>,
}
/// Hybrid ML Optimizer
pub struct HybridMLOptimizer {
    /// Optimization configuration
    config: QMLOptimizationConfig,
    /// Active optimizers
    optimizers: HashMap<String, Box<dyn QMLOptimizer>>,
    /// Optimization history
    optimization_history: VecDeque<OptimizationRecord>,
    /// Performance analytics
    performance_analytics: OptimizationAnalytics,
}
impl HybridMLOptimizer {
    pub fn new(config: QMLOptimizationConfig) -> DeviceResult<Self> {
        Ok(Self {
            config,
            optimizers: HashMap::new(),
            optimization_history: VecDeque::new(),
            performance_analytics: OptimizationAnalytics {
                convergence_analysis: ConvergenceAnalysis {
                    status: ConvergenceStatus::NotStarted,
                    rate: 0.0,
                    stability: 0.0,
                    predicted_convergence: None,
                    confidence: 0.0,
                },
                performance_trends: HashMap::new(),
                resource_efficiency: ResourceEfficiencyAnalysis {
                    overall_efficiency: 0.0,
                    quantum_efficiency: 0.0,
                    classical_efficiency: 0.0,
                    memory_efficiency: 0.0,
                    cost_efficiency: 0.0,
                    efficiency_trends: HashMap::new(),
                },
                anomaly_detection: AnomalyDetectionResults {
                    anomalies: Vec::new(),
                    anomaly_score: 0.0,
                    threshold: 0.95,
                    confidence: 0.0,
                },
            },
        })
    }
}
/// Idle time analysis
#[derive(Debug, Clone)]
pub struct IdleTimeAnalysis {
    /// Total idle time
    pub total_idle_time: Duration,
    /// Idle time percentage
    pub idle_percentage: f64,
    /// Idle time patterns
    pub idle_patterns: Vec<IdlePattern>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
}
/// Anomaly detection results
#[derive(Debug, Clone)]
pub struct AnomalyDetectionResults {
    /// Anomalies detected
    pub anomalies: Vec<DetectedAnomaly>,
    /// Anomaly score
    pub anomaly_score: f64,
    /// Threshold used
    pub threshold: f64,
    /// Detection confidence
    pub confidence: f64,
}
/// Constraint handling methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintHandling {
    Penalty,
    Barrier,
    Lagrangian,
    Adaptive,
}
/// Optimizer state
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Optimizer type
    pub optimizer_type: OptimizerType,
    /// Momentum terms
    pub momentum: Option<Array1<f64>>,
    /// Velocity terms
    pub velocity: Option<Array1<f64>>,
    /// Second moment estimates
    pub second_moment: Option<Array1<f64>>,
    /// Accumulated gradients
    pub accumulated_gradients: Option<Array1<f64>>,
    /// Step count
    pub step_count: usize,
}
/// Convergence status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    NotStarted,
    Improving,
    Converged,
    Plateaued,
    Diverging,
    Oscillating,
}
/// Framework information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkInfo {
    /// Framework name
    pub name: String,
    /// Framework version
    pub version: String,
    /// Supported features
    pub supported_features: Vec<String>,
    /// Integration quality
    pub integration_quality: f64,
}
/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    /// GPU ID
    pub gpu_id: usize,
    /// GPU name
    pub name: String,
    /// Memory (GB)
    pub memory_gb: f64,
    /// Compute capability
    pub compute_capability: String,
    /// Current utilization
    pub utilization: f64,
    /// Available
    pub available: bool,
}
/// Performance metrics collection
#[derive(Debug, Clone)]
pub struct PerformanceMetricsCollection {
    /// Metric name
    pub name: String,
    /// Values over time
    pub values: VecDeque<(Instant, f64)>,
    /// Statistical summary
    pub statistics: StatisticalSummary,
    /// Trend analysis
    pub trend: TrendAnalysis,
    /// Anomaly flags
    pub anomalies: Vec<AnomalyFlag>,
}
/// Connectivity requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityRequirements {
    /// Required connectivity graph
    pub connectivity_graph: Vec<(usize, usize)>,
    /// Minimum connectivity degree
    pub min_connectivity: usize,
    /// Topology constraints
    pub topology_constraints: Vec<TopologyConstraint>,
}

// Trait implementations moved to separate trait files:
// - QMLResourceConfig::Default -> qmlresourceconfig_traits.rs
// - QMLMonitoringConfig::Default -> qmlmonitoringconfig_traits.rs
// - SimpleMLAnomalyDetector::AnomalyDetector -> simplemlanomalydetector_traits.rs
// - HybridMLOptimizer::Debug -> hybridmloptimizer_traits.rs
// - QMLDataPipeline::Debug -> qmldatapipeline_traits.rs


