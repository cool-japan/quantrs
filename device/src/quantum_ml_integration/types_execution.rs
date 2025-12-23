//! QML Integration Types - Part 2 of 3
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

// Import types from sibling modules (loaded before this module)
use super::types_core::*;

// Import from crate modules
use crate::{DeviceError, DeviceResult};

/// Hardware requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    /// Minimum qubits required
    pub min_qubits: usize,
    /// Required gate set
    pub required_gates: Vec<String>,
    /// Connectivity requirements
    pub connectivity_requirements: ConnectivityRequirements,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
}
/// Caching strategies for QML
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CachingStrategy {
    None,
    LRU,
    LFU,
    FIFO,
    Adaptive,
    Custom(String),
}
/// Trend component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendComponent {
    /// Component name
    pub name: String,
    /// Trend strength
    pub strength: f64,
    /// Trend direction
    pub direction: TrendDirection,
    /// Confidence
    pub confidence: f64,
}
/// Resource allocation priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePriorities {
    /// Priority weights for different resources
    pub weights: HashMap<String, f64>,
    /// Dynamic priority adjustment
    pub dynamic_adjustment: bool,
    /// Performance-based reallocation
    pub performance_reallocation: bool,
}
/// Quantum ML model representation
#[derive(Debug, Clone)]
pub struct QMLModel {
    /// Model identifier
    pub model_id: String,
    /// Model type
    pub model_type: QMLModelType,
    /// Model architecture
    pub architecture: QMLArchitecture,
    /// Model parameters
    pub parameters: QMLParameters,
    /// Training state
    pub training_state: QMLTrainingState,
    /// Performance metrics
    pub performance_metrics: QMLPerformanceMetrics,
    /// Metadata
    pub metadata: QMLModelMetadata,
}
/// Convergence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysis {
    /// Convergence status
    pub status: ConvergenceStatus,
    /// Convergence rate
    pub rate: f64,
    /// Stability score
    pub stability: f64,
    /// Predicted convergence time
    pub predicted_convergence: Option<Duration>,
    /// Convergence confidence
    pub confidence: f64,
}
/// QML resource manager
#[derive(Debug)]
pub struct QMLResourceManager {
    /// Available quantum resources
    quantum_resources: HashMap<String, QuantumResourcePool>,
    /// Available classical resources
    classical_resources: ClassicalResourcePool,
    /// Resource allocation history
    allocation_history: VecDeque<AllocationRecord>,
    /// Resource utilization tracking
    utilization_tracker: ResourceUtilizationTracker,
}
impl QMLResourceManager {
    pub fn new() -> Self {
        Self {
            quantum_resources: HashMap::new(),
            classical_resources: ClassicalResourcePool {
                available_cpu_cores: 8,
                available_memory_mb: 16384,
                available_gpus: Vec::new(),
                available_storage_mb: 102400,
                utilization: ResourceUtilization {
                    cpu_utilization: 0.0,
                    memory_utilization: 0.0,
                    gpu_utilization: HashMap::new(),
                    storage_utilization: 0.0,
                    network_utilization: 0.0,
                },
            },
            allocation_history: VecDeque::new(),
            utilization_tracker: ResourceUtilizationTracker {
                utilization_history: VecDeque::new(),
                current_metrics: ResourceUtilization {
                    cpu_utilization: 0.0,
                    memory_utilization: 0.0,
                    gpu_utilization: HashMap::new(),
                    storage_utilization: 0.0,
                    network_utilization: 0.0,
                },
                trend_analysis: HashMap::new(),
                efficiency_metrics: EfficiencyMetrics {
                    overall_efficiency: 0.0,
                    resource_efficiency: HashMap::new(),
                    cost_efficiency: 0.0,
                    time_efficiency: 0.0,
                    quality_efficiency: 0.0,
                },
            },
        }
    }
}
/// QML gate representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLGate {
    /// Gate type
    pub gate_type: String,
    /// Target qubits
    pub targets: Vec<usize>,
    /// Control qubits
    pub controls: Vec<usize>,
    /// Parameters
    pub parameters: Vec<f64>,
    /// Trainable parameter indices
    pub trainable_params: Vec<usize>,
}
/// Alert record
#[derive(Debug, Clone)]
pub struct AlertRecord {
    /// Alert ID
    pub alert_id: String,
    /// Alert type
    pub alert_type: String,
    /// Severity
    pub severity: f64,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Duration
    pub duration: Duration,
    /// Resolution
    pub resolution: String,
}
/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Standard benchmarks
    pub standard_benchmarks: HashMap<String, BenchmarkResult>,
    /// Custom benchmarks
    pub custom_benchmarks: HashMap<String, BenchmarkResult>,
    /// Leaderboard rankings
    pub leaderboard: Vec<LeaderboardEntry>,
}
/// Inference metadata
#[derive(Debug, Clone)]
pub struct InferenceMetadata {
    /// Inference ID
    pub inference_id: String,
    /// Model used
    pub model_id: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Input size
    pub input_size: usize,
    /// Output size
    pub output_size: usize,
}
/// Model analytics
#[derive(Debug, Clone)]
pub struct ModelAnalytics {
    /// Model performance metrics
    pub performance_metrics: ModelPerformanceMetrics,
    /// Model complexity analysis
    pub complexity_analysis: ModelComplexityAnalysis,
    /// Interpretability metrics
    pub interpretability_metrics: InterpretabilityMetrics,
    /// Robustness analysis
    pub robustness_analysis: RobustnessAnalysis,
}
/// Parameter constraints for quantum models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraints {
    /// Minimum parameter value
    pub min_value: Option<f64>,
    /// Maximum parameter value
    pub max_value: Option<f64>,
    /// Enforce unitary constraints
    pub enforce_unitarity: bool,
    /// Enforce hermiticity
    pub enforce_hermiticity: bool,
    /// Custom constraint functions
    pub custom_constraints: Vec<String>,
}
/// Alert manager
pub struct AlertManager {
    /// Alert configuration
    config: AlertConfig,
    /// Active alerts
    active_alerts: HashMap<String, ActiveAlert>,
    /// Alert history
    alert_history: VecDeque<AlertRecord>,
    /// Notification channels
    notification_channels: HashMap<QMLAlertChannel, Box<dyn NotificationChannel>>,
}
impl AlertManager {
    pub fn new() -> Self {
        Self {
            config: AlertConfig {
                enabled: true,
                thresholds: HashMap::new(),
                channels: vec![QMLAlertChannel::Log],
                escalation: AlertEscalation {
                    enabled: false,
                    levels: Vec::new(),
                    timeouts: HashMap::new(),
                },
            },
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            notification_channels: HashMap::new(),
        }
    }
}
/// Algorithm comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmComparison {
    /// Algorithms compared
    pub algorithms: Vec<String>,
    /// Convergence comparison
    pub convergence_comparison: HashMap<String, ConvergenceMetrics>,
    /// Efficiency comparison
    pub efficiency_comparison: HashMap<String, f64>,
    /// Scalability comparison
    pub scalability_comparison: HashMap<String, f64>,
    /// Recommendation
    pub recommendation: String,
}
/// QML performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLPerformanceMetrics {
    /// Training metrics
    pub training_metrics: HashMap<String, f64>,
    /// Validation metrics
    pub validation_metrics: HashMap<String, f64>,
    /// Test metrics
    pub test_metrics: HashMap<String, f64>,
    /// Circuit execution metrics
    pub circuit_metrics: CircuitExecutionMetrics,
    /// Resource utilization metrics
    pub resource_metrics: ResourceUtilizationMetrics,
    /// Convergence metrics
    pub convergence_metrics: ConvergenceMetrics,
}
/// Inference performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePerformanceMetrics {
    /// Inference time
    pub inference_time: Duration,
    /// Quantum circuit executions
    pub circuit_executions: usize,
    /// Resource usage
    pub resource_usage: ResourceUtilization,
    /// Accuracy (if ground truth available)
    pub accuracy: Option<f64>,
}
/// QML model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLModelMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last updated
    pub updated_at: SystemTime,
    /// Model version
    pub version: String,
    /// Author
    pub author: String,
    /// Description
    pub description: String,
    /// Tags
    pub tags: Vec<String>,
    /// Framework used
    pub framework: MLFramework,
    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
}
/// Gradient computation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GradientMethod {
    ParameterShift,
    FiniteDifference,
    Adjoint,
    Backpropagation,
    Natural,
    SPSA,
    Custom(String),
}
/// Resource allocation
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Allocated quantum resources
    pub quantum_allocation: AllocatedQuantumResources,
    /// Allocated classical resources
    pub classical_allocation: AllocatedClassicalResources,
    /// Allocation timestamp
    pub allocated_at: Instant,
    /// Allocation priority
    pub priority: TrainingPriority,
}
/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    /// Quantum resource usage
    pub quantum_usage: f64,
    /// Classical compute usage
    pub classical_usage: f64,
    /// Memory usage
    pub memory_usage: f64,
    /// Network usage
    pub network_usage: f64,
    /// Cost efficiency
    pub cost_efficiency: f64,
}
/// Performance tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrackingConfig {
    /// Track training metrics
    pub track_training_metrics: bool,
    /// Track inference metrics
    pub track_inference_metrics: bool,
    /// Track quantum circuit metrics
    pub track_circuit_metrics: bool,
    /// Metric aggregation window
    pub aggregation_window: Duration,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
}
/// Optimization analytics
#[derive(Debug, Clone)]
pub struct OptimizationAnalytics {
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
    /// Performance trends
    pub performance_trends: HashMap<String, TrendAnalysis>,
    /// Resource efficiency
    pub resource_efficiency: ResourceEfficiencyAnalysis,
    /// Anomaly detection
    pub anomaly_detection: AnomalyDetectionResults,
}
/// Detected anomaly
#[derive(Debug, Clone)]
pub struct DetectedAnomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity
    pub severity: f64,
    /// Description
    pub description: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
}
/// Allocated quantum resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocatedQuantumResources {
    /// Backend assigned
    pub backend_id: String,
    /// Qubits allocated
    pub qubits_allocated: Vec<usize>,
    /// Estimated queue time
    pub estimated_queue_time: Duration,
    /// Resource cost
    pub resource_cost: f64,
}
/// Training analytics
#[derive(Debug, Clone)]
pub struct TrainingAnalytics {
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
    /// Learning curve analysis
    pub learning_curve: LearningCurveAnalysis,
    /// Optimization efficiency
    pub optimization_efficiency: OptimizationEfficiencyAnalysis,
    /// Time series analysis
    pub time_series_analysis: TimeSeriesAnalysis,
}
/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Score achieved
    pub score: f64,
    /// Time taken
    pub time_taken: Duration,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Rank
    pub rank: Option<usize>,
}
/// QML optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLOptimizationConfig {
    /// Optimization algorithm
    pub optimizer_type: OptimizerType,
    /// Optimizer-specific parameters
    pub optimizer_params: HashMap<String, f64>,
    /// Enable parameter sharing
    pub enable_parameter_sharing: bool,
    /// Circuit depth optimization
    pub circuit_optimization: CircuitOptimizationConfig,
    /// Hardware-aware optimization
    pub hardware_aware: bool,
    /// Multi-objective optimization
    pub multi_objective: MultiObjectiveConfig,
}
/// Circuit optimization for QML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitOptimizationConfig {
    /// Enable gate fusion
    pub enable_gate_fusion: bool,
    /// Enable circuit compression
    pub enable_compression: bool,
    /// Maximum circuit depth
    pub max_depth: Option<usize>,
    /// Gate set restrictions
    pub allowed_gates: Option<Vec<String>>,
    /// Topology awareness
    pub topology_aware: bool,
}
/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: HashMap<String, f64>,
    /// Alert channels
    pub channels: Vec<QMLAlertChannel>,
    /// Alert escalation
    pub escalation: AlertEscalation,
}
/// Learning curve analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurveAnalysis {
    /// Learning rate effectiveness
    pub learning_rate_effectiveness: f64,
    /// Overfitting detection
    pub overfitting_score: f64,
    /// Underfitting detection
    pub underfitting_score: f64,
    /// Optimal stopping point
    pub optimal_stopping_epoch: Option<usize>,
    /// Learning curve smoothness
    pub smoothness: f64,
}
/// Optimization efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEfficiencyAnalysis {
    /// Convergence speed
    pub convergence_speed: f64,
    /// Gradient utilization efficiency
    pub gradient_efficiency: f64,
    /// Parameter update efficiency
    pub parameter_efficiency: f64,
    /// Overall optimization score
    pub optimization_score: f64,
}
/// Interpretability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityMetrics {
    /// Feature importance
    pub feature_importance: HashMap<String, f64>,
    /// Parameter sensitivity
    pub parameter_sensitivity: HashMap<String, f64>,
    /// Decision boundaries clarity
    pub decision_clarity: f64,
    /// Explanation quality
    pub explanation_quality: f64,
}
/// Escalation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level name
    pub name: String,
    /// Threshold multiplier
    pub threshold_multiplier: f64,
    /// Alert channels for this level
    pub channels: Vec<QMLAlertChannel>,
    /// Actions to take
    pub actions: Vec<EscalationAction>,
}
/// Bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    /// Bottleneck impact analysis
    pub impact_analysis: HashMap<String, f64>,
    /// Resolution recommendations
    pub recommendations: Vec<BottleneckRecommendation>,
}
/// Model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Models compared
    pub models: Vec<String>,
    /// Performance comparison
    pub performance_comparison: HashMap<String, f64>,
    /// Complexity comparison
    pub complexity_comparison: HashMap<String, f64>,
    /// Cost comparison
    pub cost_comparison: HashMap<String, f64>,
    /// Recommendation
    pub recommendation: String,
}
/// Time constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraints {
    /// Maximum training time
    pub max_training_time: Option<Duration>,
    /// Deadline
    pub deadline: Option<SystemTime>,
    /// Priority scheduling
    pub priority_scheduling: bool,
}
/// Noise characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseCharacteristics {
    /// Noise level
    pub noise_level: f64,
    /// Noise type
    pub noise_type: NoiseType,
    /// Autocorrelation
    pub autocorrelation: f64,
    /// Signal-to-noise ratio
    pub snr: f64,
}
/// Data types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Classification,
    Regression,
    Clustering,
    Reinforcement,
    Unsupervised,
}
/// Alert escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEscalation {
    /// Enable escalation
    pub enabled: bool,
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
    /// Escalation timeouts
    pub timeouts: HashMap<String, Duration>,
}
/// Escalation actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EscalationAction {
    Notify,
    Throttle,
    Pause,
    Restart,
    Fallback,
}
/// Activation functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU,
    GELU,
    Swish,
    Custom(String),
}
/// Training session
#[derive(Debug, Clone)]
pub struct TrainingSession {
    /// Session ID
    pub session_id: String,
    /// Model being trained
    pub model_id: String,
    /// Training state
    pub training_state: QMLTrainingState,
    /// Start time
    pub start_time: Instant,
    /// Estimated completion time
    pub estimated_completion: Option<Instant>,
    /// Resource allocation
    pub resource_allocation: ResourceAllocation,
    /// Progress metrics
    pub progress_metrics: ProgressMetrics,
}
/// Training performance monitor
pub struct TrainingPerformanceMonitor {
    /// Performance metrics collection
    performance_metrics: HashMap<String, PerformanceMetricsCollection>,
    /// Anomaly detector
    anomaly_detector: Box<dyn AnomalyDetector>,
    /// Alert manager
    alert_manager: AlertManager,
    /// Monitoring configuration
    config: QMLMonitoringConfig,
}
impl TrainingPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            performance_metrics: HashMap::new(),
            anomaly_detector: Box::new(SimpleMLAnomalyDetector::new()),
            alert_manager: AlertManager::new(),
            config: QMLMonitoringConfig::default(),
        }
    }
}
/// QML training result
#[derive(Debug, Clone)]
pub struct QMLTrainingResult {
    /// Training session ID
    pub session_id: String,
    /// Final model
    pub trained_model: QMLModel,
    /// Training metrics
    pub training_metrics: QMLPerformanceMetrics,
    /// Training history
    pub training_history: TrainingHistory,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Training duration
    pub training_duration: Duration,
    /// Success status
    pub success: bool,
}
/// Resource usage snapshot
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// Quantum resource usage
    pub quantum_usage: f64,
    /// Classical resource usage
    pub classical_usage: f64,
    /// Memory usage
    pub memory_usage: f64,
    /// Network bandwidth usage
    pub network_usage: f64,
}
/// Framework bridge
pub struct FrameworkBridge {
    /// Framework type
    framework_type: MLFramework,
    /// Bridge implementation
    bridge_impl: Box<dyn FrameworkBridgeImpl>,
    /// Conversion cache
    conversion_cache: HashMap<String, ConversionResult>,
    /// Performance metrics
    performance_metrics: BridgePerformanceMetrics,
}
/// Types of anomalies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyType {
    PerformanceDegradation,
    ResourceSpike,
    ConvergenceFailure,
    GradientExplosion,
    ParameterDrift,
    AccuracyDrop,
    LatencyIncrease,
    CostSpike,
}
/// QML inference result
#[derive(Debug, Clone)]
pub struct QMLInferenceResult {
    /// Predictions
    pub predictions: Array1<f64>,
    /// Prediction probabilities (if applicable)
    pub probabilities: Option<Array2<f64>>,
    /// Inference metadata
    pub metadata: InferenceMetadata,
    /// Performance metrics
    pub performance_metrics: InferencePerformanceMetrics,
}
/// Classical resource pool
#[derive(Debug, Clone)]
pub struct ClassicalResourcePool {
    /// Available CPU cores
    pub available_cpu_cores: usize,
    /// Available memory (MB)
    pub available_memory_mb: usize,
    /// Available GPUs
    pub available_gpus: Vec<GPUInfo>,
    /// Available storage (MB)
    pub available_storage_mb: usize,
    /// Current utilization
    pub utilization: ResourceUtilization,
}
/// Training priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrainingPriority {
    Low,
    Normal,
    High,
    Critical,
}
/// Framework comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkComparison {
    /// Framework performance
    pub performance: f64,
    /// Ease of use
    pub ease_of_use: f64,
    /// Feature completeness
    pub feature_completeness: f64,
    /// Integration quality
    pub integration_quality: f64,
    /// Overall score
    pub overall_score: f64,
}
/// Resource utilization tracker
#[derive(Debug)]
pub struct ResourceUtilizationTracker {
    /// Historical utilization data
    utilization_history: VecDeque<UtilizationSnapshot>,
    /// Current metrics
    current_metrics: ResourceUtilization,
    /// Trend analysis
    trend_analysis: HashMap<String, TrendAnalysis>,
    /// Efficiency metrics
    efficiency_metrics: EfficiencyMetrics,
}
/// Improvement direction for early stopping
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImprovementMode {
    Minimize,
    Maximize,
}
/// Cost optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationOpportunity {
    /// Opportunity description
    pub description: String,
    /// Potential savings
    pub potential_savings: f64,
    /// Implementation effort
    pub implementation_effort: ImplementationEffort,
    /// Confidence
    pub confidence: f64,
}
/// Circuit execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitExecutionMetrics {
    /// Average circuit depth
    pub avg_circuit_depth: f64,
    /// Total gate count
    pub total_gate_count: usize,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Circuit fidelity
    pub circuit_fidelity: f64,
    /// Shot efficiency
    pub shot_efficiency: f64,
}
/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    Performance,
    Latency,
    Cost,
}
/// Topology constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopologyConstraint {
    Linear,
    Grid,
    Ring,
    Tree,
    Complete,
    Custom(String),
}
/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f64,
    /// Trend confidence
    pub confidence: f64,
    /// Projected values
    pub projection: Vec<f64>,
}
/// Seasonal pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// Pattern period
    pub period: Duration,
    /// Pattern strength
    pub strength: f64,
    /// Pattern phase
    pub phase: f64,
}

// Trait implementations moved to separate trait files:
// - QMLOptimizationConfig::Default -> qmloptimizationconfig_traits.rs
// - TrainingPerformanceMonitor::Debug -> trainingperformancemonitor_traits.rs
// - FrameworkBridge::Debug -> frameworkbridge_traits.rs
// - AlertManager::Debug -> alertmanager_traits.rs


