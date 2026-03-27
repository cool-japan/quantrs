//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::extended::*;
use crate::applications::{ApplicationError, ApplicationResult};
use crate::ising::{IsingModel, QuboModel};
use crate::quantum_error_correction::{
    ErrorCorrectionCode, LogicalAnnealingEncoder, MitigationTechnique, SyndromeDetector,
};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

/// Adaptive resource management
pub struct AdaptiveResourceManager {
    /// Resource allocation
    pub allocation: ResourceAllocation,
    /// Resource constraints
    pub constraints: ResourceConstraints,
    /// Optimization algorithms
    pub optimizers: Vec<ResourceOptimizer>,
}
impl AdaptiveResourceManager {
    pub(crate) fn new(config: ResourceManagementConfig) -> Self {
        Self {
            allocation: ResourceAllocation {
                allocation_map: HashMap::new(),
                total_allocated: 0.0,
                available_resources: 100.0,
                allocation_history: VecDeque::new(),
            },
            constraints: ResourceConstraints {
                max_total: 100.0,
                per_component: HashMap::new(),
                min_performance: 0.8,
                enforcement_method: ConstraintEnforcement::Soft,
            },
            optimizers: vec![],
        }
    }
    pub(crate) fn update_allocation_based_on_performance(&mut self, metadata: &CorrectionMetadata) {
        let performance_score =
            metadata.errors_corrected as f64 / metadata.errors_detected.max(1) as f64;
        if performance_score < 0.8 {
            self.allocation
                .allocation_map
                .insert("error_correction".to_string(), 0.4);
        } else if performance_score > 0.95 && metadata.correction_overhead < 0.1 {
            self.allocation
                .allocation_map
                .insert("error_correction".to_string(), 0.2);
        }
    }
}
/// Correction result
#[derive(Debug, Clone)]
pub struct CorrectionResult {
    pub corrected_problem: IsingModel,
    pub correction_overhead: f64,
    pub errors_detected: usize,
    pub errors_corrected: usize,
}
/// Ensemble methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnsembleMethod {
    /// Simple averaging
    Average,
    /// Weighted averaging
    WeightedAverage,
    /// Voting
    Voting,
    /// Stacking
    Stacking,
    /// Boosting
    Boosting,
}
/// Resource allocation strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceAllocationStrategy {
    /// Fixed allocation
    Fixed,
    /// Performance-based allocation
    PerformanceBased,
    /// Adaptive allocation
    Adaptive,
    /// Predictive allocation
    Predictive,
}
/// Feature extraction configuration
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Enable temporal features
    pub enable_temporal: bool,
    /// Enable spectral analysis features
    pub enable_spectral: bool,
    /// Enable correlation features
    pub enable_correlation: bool,
    /// Feature normalization method
    pub normalization: FeatureNormalization,
    /// Feature selection method
    pub selection_method: FeatureSelection,
}
/// Resource optimization algorithms
#[derive(Debug)]
pub struct ResourceOptimizer {
    /// Optimizer identifier
    pub id: String,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Algorithm parameters
    pub parameters: HashMap<String, f64>,
}
/// Noise prediction result
#[derive(Debug, Clone)]
pub struct NoisePrediction {
    pub predicted_noise: NoiseCharacteristics,
    pub confidence: f64,
    pub horizon: Duration,
    pub uncertainty_bounds: (f64, f64),
}
/// Switching criteria for hybrid approaches
#[derive(Debug, Clone)]
pub struct SwitchingCriteria {
    /// Error rate threshold
    pub error_rate_threshold: f64,
    /// Performance degradation threshold
    pub performance_threshold: f64,
    /// Resource usage threshold
    pub resource_threshold: f64,
    /// Time-based switching
    pub time_based: Option<Duration>,
}
/// Model ensemble for improved predictions
#[derive(Debug)]
pub struct ModelEnsemble {
    /// Ensemble method
    pub method: EnsembleMethod,
    /// Model weights
    pub weights: HashMap<String, f64>,
    /// Performance history
    pub performance_history: VecDeque<EnsemblePerformance>,
}
/// Resource constraints
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum total resources
    pub max_total: f64,
    /// Per-component constraints
    pub per_component: HashMap<String, f64>,
    /// Minimum performance requirement
    pub min_performance: f64,
    /// Constraint enforcement method
    pub enforcement_method: ConstraintEnforcement,
}
/// Hierarchy coordination system
pub struct HierarchyCoordinator {
    /// Hierarchy levels
    pub levels: Vec<HierarchyLevel>,
    /// Inter-level communication
    pub communication: HierarchyCommunicationManager,
    /// Coordination algorithms
    pub coordinators: Vec<CoordinationAlgorithm>,
}
impl HierarchyCoordinator {
    pub(crate) fn new(config: HierarchyConfig) -> Self {
        let mut levels = Vec::new();
        for i in 0..config.num_levels {
            levels.push(HierarchyLevel {
                id: i,
                priority: (config.num_levels - i) as u8,
                protocols: vec![],
                resources: config.level_resources.get(i).copied().unwrap_or(0.1),
                performance: LevelPerformance {
                    accuracy: 0.9,
                    response_time: Duration::from_millis(10 * (i + 1) as u64),
                    efficiency: 0.8,
                    coordination_effectiveness: 0.85,
                },
            });
        }
        Self {
            levels,
            communication: HierarchyCommunicationManager {
                channels: HashMap::new(),
                message_queues: HashMap::new(),
                statistics: CommunicationStatistics {
                    throughput: 100.0,
                    avg_latency: Duration::from_millis(5),
                    success_rate: 0.99,
                    channel_utilization: HashMap::new(),
                },
            },
            coordinators: vec![],
        }
    }
}
/// Strategy performance tracking
#[derive(Debug, Clone)]
pub struct StrategyPerformance {
    /// Strategy identifier
    pub strategy_id: String,
    /// Performance score
    pub performance_score: f64,
    /// Resource overhead
    pub resource_overhead: f64,
    /// Success rate
    pub success_rate: f64,
    /// Timestamp
    pub timestamp: Instant,
}
/// Noise prediction system
pub struct NoisePredictionSystem {
    /// Prediction models
    pub models: HashMap<String, NoisePredictionModel>,
    /// Model ensemble
    pub ensemble: ModelEnsemble,
    /// Prediction cache
    pub prediction_cache: HashMap<String, PredictionResult>,
}
impl NoisePredictionSystem {
    pub(crate) fn new(ml_config: MLNoiseConfig) -> Self {
        Self {
            models: HashMap::new(),
            ensemble: ModelEnsemble {
                method: EnsembleMethod::WeightedAverage,
                weights: HashMap::new(),
                performance_history: VecDeque::new(),
            },
            prediction_cache: HashMap::new(),
        }
    }
}
/// Adaptation decisions
#[derive(Debug, Clone)]
pub struct AdaptationDecision {
    /// Decision timestamp
    pub timestamp: Instant,
    /// Decision rationale
    pub rationale: String,
    /// Actions taken
    pub actions: Vec<AdaptationAction>,
    /// Expected impact
    pub expected_impact: f64,
    /// Actual impact (filled later)
    pub actual_impact: Option<f64>,
}
/// Neural network architectures for noise prediction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuralArchitecture {
    /// Long Short-Term Memory networks
    LSTM,
    /// Gated Recurrent Units
    GRU,
    /// Transformer architecture
    Transformer,
    /// Convolutional Neural Network
    CNN,
    /// Hybrid CNN-LSTM
    CNNLstm,
}
/// Ensemble performance tracking
#[derive(Debug, Clone)]
pub struct EnsemblePerformance {
    /// Timestamp
    pub timestamp: Instant,
    /// Accuracy
    pub accuracy: f64,
    /// Confidence
    pub confidence: f64,
    /// Individual model contributions
    pub model_contributions: HashMap<String, f64>,
}
/// Types of noise
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseType {
    /// White noise (uncorrelated)
    White,
    /// Colored noise (correlated)
    Colored,
    /// Burst noise (intermittent)
    Burst,
    /// Drift noise (slowly varying)
    Drift,
    /// Mixed noise types
    Mixed(Vec<Self>),
}
/// Adaptive protocol management
pub struct AdaptiveProtocolManager {
    /// Currently active protocols
    pub active_protocols: HashMap<String, AdaptiveProtocol>,
    /// Protocol history
    pub protocol_history: VecDeque<ProtocolEvent>,
    /// Adaptation engine
    pub adaptation_engine: AdaptationEngine,
}
impl AdaptiveProtocolManager {
    pub(crate) fn new() -> Self {
        Self {
            active_protocols: HashMap::new(),
            protocol_history: VecDeque::new(),
            adaptation_engine: AdaptationEngine {
                algorithm: AdaptationAlgorithm::RuleBased,
                parameters: HashMap::new(),
                decision_history: VecDeque::new(),
            },
        }
    }
}
/// Resource optimization algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationAlgorithm {
    /// Gradient descent
    GradientDescent,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Bayesian optimization
    BayesianOptimization,
}
/// Message payload data
#[derive(Debug, Clone)]
pub enum MessagePayload {
    /// Error information
    ErrorInfo(ErrorInfo),
    /// Resource request details
    ResourceRequest(ResourceRequestDetails),
    /// Performance data
    PerformanceData(PerformanceData),
    /// Coordination instructions
    CoordinationInstructions(CoordinationInstructions),
    /// Generic data
    Generic(Vec<u8>),
}
/// Prediction results
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Predicted noise characteristics
    pub predicted_noise: NoiseCharacteristics,
    /// Prediction confidence
    pub confidence: f64,
    /// Prediction horizon
    pub horizon: Duration,
    /// Uncertainty bounds
    pub uncertainty_bounds: (f64, f64),
    /// Prediction timestamp
    pub timestamp: Instant,
}
/// Types of noise sensors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SensorType {
    /// Error rate sensor
    ErrorRate,
    /// Coherence time sensor
    CoherenceTime,
    /// Gate fidelity sensor
    GateFidelity,
    /// Environmental sensor
    Environmental,
    /// Process tomography
    ProcessTomography,
}
/// Resource request details
#[derive(Debug, Clone)]
pub struct ResourceRequestDetails {
    /// Requested resources
    pub requested_resources: HashMap<String, f64>,
    /// Request priority
    pub priority: u8,
    /// Request deadline
    pub deadline: Option<Instant>,
    /// Justification
    pub justification: String,
}
/// Noise assessment result
#[derive(Debug, Clone)]
pub struct NoiseAssessment {
    pub current_noise: NoiseCharacteristics,
    pub severity: NoiseSeverity,
    pub trends: NoiseTrends,
    pub confidence: f64,
    pub timestamp: Instant,
}
/// Noise monitoring system
pub struct NoiseMonitor {
    /// Current noise characteristics
    pub current_noise: NoiseCharacteristics,
    /// Noise history
    pub noise_history: VecDeque<NoiseCharacteristics>,
    /// Monitoring sensors
    pub sensors: Vec<NoiseSensor>,
    /// Analysis algorithms
    pub analyzers: Vec<NoiseAnalyzer>,
}
impl NoiseMonitor {
    pub(crate) fn new() -> Self {
        Self {
            current_noise: NoiseCharacteristics {
                timestamp: Instant::now(),
                noise_level: 0.01,
                noise_type: NoiseType::White,
                temporal_correlation: 0.1,
                spatial_correlation: 0.1,
                noise_spectrum: vec![0.01; 10],
                per_qubit_error_rates: vec![0.001; 100],
                coherence_times: vec![100.0; 100],
                gate_fidelities: HashMap::new(),
            },
            noise_history: VecDeque::new(),
            sensors: vec![],
            analyzers: vec![],
        }
    }
}
/// Noise trend analysis
#[derive(Debug, Clone)]
pub struct NoiseTrends {
    pub direction: TrendDirection,
    pub rate: f64,
    pub confidence: f64,
}
/// Final corrected problem with metadata
#[derive(Debug, Clone)]
pub struct CorrectedProblem {
    pub original_problem: IsingModel,
    pub corrected_problem: IsingModel,
    pub correction_metadata: CorrectionMetadata,
}
/// Allocation snapshot
#[derive(Debug, Clone)]
pub struct AllocationSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// Allocation state
    pub allocation: HashMap<String, f64>,
    /// Performance at this allocation
    pub performance: f64,
}
/// Noise sensor interface
#[derive(Debug)]
pub struct NoiseSensor {
    /// Sensor identifier
    pub id: String,
    /// Sensor type
    pub sensor_type: SensorType,
    /// Measurement frequency
    pub frequency: f64,
    /// Last measurement
    pub last_measurement: Option<Instant>,
    /// Calibration data
    pub calibration: SensorCalibration,
}
/// Noise prediction model
#[derive(Debug)]
pub struct NoisePredictionModel {
    /// Model type
    pub model_type: NeuralArchitecture,
    /// Model parameters
    pub parameters: Vec<f64>,
    /// Training data
    pub training_data: VecDeque<NoiseDataPoint>,
    /// Prediction accuracy
    pub accuracy: f64,
    /// Last update time
    pub last_update: Instant,
    /// Feature extractor
    pub feature_extractor: FeatureExtractor,
}
/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Performance score
    pub score: f64,
    /// Comparison baseline
    pub baseline: f64,
    /// Improvement factor
    pub improvement_factor: f64,
    /// Timestamp
    pub timestamp: Instant,
}
/// Real-time adaptive QEC configuration
#[derive(Debug, Clone)]
pub struct AdaptiveQecConfig {
    /// Noise monitoring interval
    pub monitoring_interval: Duration,
    /// Adaptation trigger threshold
    pub adaptation_threshold: f64,
    /// Performance window for assessment
    pub performance_window: Duration,
    /// Machine learning configuration
    pub ml_config: MLNoiseConfig,
    /// Hierarchical error correction settings
    pub hierarchy_config: HierarchyConfig,
    /// Resource management settings
    pub resource_config: ResourceManagementConfig,
    /// Prediction and forecasting settings
    pub prediction_config: PredictionConfig,
}
/// Noise data point for training
#[derive(Debug, Clone)]
pub struct NoiseDataPoint {
    /// Input features
    pub features: Vec<f64>,
    /// Target noise characteristics
    pub target: NoiseCharacteristics,
    /// Timestamp
    pub timestamp: Instant,
}
/// Hierarchy communication protocols
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HierarchyCommunication {
    /// Cascade from low to high levels
    Cascade,
    /// Parallel processing at all levels
    Parallel,
    /// Dynamic level selection
    Dynamic,
    /// Adaptive switching
    Adaptive,
}
/// Current resource allocation
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Allocation map
    pub allocation_map: HashMap<String, f64>,
    /// Total allocated resources
    pub total_allocated: f64,
    /// Available resources
    pub available_resources: f64,
    /// Allocation history
    pub allocation_history: VecDeque<AllocationSnapshot>,
}
/// Communication channel between levels
#[derive(Debug)]
pub struct CommunicationChannel {
    /// Source level
    pub source: usize,
    /// Target level
    pub target: usize,
    /// Channel capacity
    pub capacity: f64,
    /// Current utilization
    pub utilization: f64,
    /// Message latency
    pub latency: Duration,
}
/// Error correction strategies
#[derive(Debug, Clone)]
pub enum ErrorCorrectionStrategy {
    /// No error correction
    None,
    /// Basic error detection
    Detection(DetectionConfig),
    /// Full error correction
    Correction(CorrectionConfig),
    /// Hybrid approach
    Hybrid(HybridConfig),
    /// Adaptive strategy selection
    Adaptive(AdaptiveStrategyConfig),
}
/// Correction metadata
#[derive(Debug, Clone)]
pub struct CorrectionMetadata {
    pub strategy_used: ErrorCorrectionStrategy,
    pub execution_time: Duration,
    pub correction_overhead: f64,
    pub errors_detected: usize,
    pub errors_corrected: usize,
    pub confidence: f64,
}
/// Hierarchical error correction configuration
#[derive(Debug, Clone)]
pub struct HierarchyConfig {
    /// Enable multi-level hierarchy
    pub enable_hierarchy: bool,
    /// Number of hierarchy levels
    pub num_levels: usize,
    /// Level transition thresholds
    pub level_thresholds: Vec<f64>,
    /// Resource allocation per level
    pub level_resources: Vec<f64>,
    /// Inter-level communication protocol
    pub communication_protocol: HierarchyCommunication,
}
/// Full error correction configuration
#[derive(Debug, Clone)]
pub struct CorrectionConfig {
    /// Error correction code
    pub code: ErrorCorrectionCode,
    /// Correction threshold
    pub threshold: f64,
    /// Maximum correction attempts
    pub max_attempts: usize,
    /// Correction efficiency target
    pub efficiency_target: f64,
}
/// Messages between hierarchy levels
#[derive(Debug, Clone)]
pub struct HierarchyMessage {
    /// Message identifier
    pub id: String,
    /// Source level
    pub source: usize,
    /// Target level
    pub target: usize,
    /// Message type
    pub message_type: MessageType,
    /// Message payload
    pub payload: MessagePayload,
    /// Timestamp
    pub timestamp: Instant,
    /// Priority
    pub priority: u8,
}
/// System state representation
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Active protocols
    pub active_protocols: Vec<String>,
    /// Current noise level
    pub noise_level: f64,
    /// Resource usage
    pub resource_usage: f64,
    /// Performance score
    pub performance_score: f64,
}
/// Feature definition
#[derive(Debug, Clone)]
pub struct FeatureDefinition {
    /// Feature name
    pub name: String,
    /// Feature type
    pub feature_type: FeatureType,
    /// Extraction parameters
    pub parameters: HashMap<String, f64>,
}
/// Resource management configuration
#[derive(Debug, Clone)]
pub struct ResourceManagementConfig {
    /// Maximum overhead ratio
    pub max_overhead_ratio: f64,
    /// Resource allocation strategy
    pub allocation_strategy: ResourceAllocationStrategy,
    /// Dynamic resource adjustment
    pub enable_dynamic_adjustment: bool,
    /// Resource constraint enforcement
    pub enforce_constraints: bool,
    /// Performance vs. overhead trade-off
    pub performance_weight: f64,
}
/// Feature normalization methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureNormalization {
    /// Z-score normalization
    ZScore,
    /// Min-max normalization
    MinMax,
    /// Robust scaling
    Robust,
    /// No normalization
    None,
}
/// Error information payload
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    /// Error type
    pub error_type: String,
    /// Error location
    pub location: Vec<usize>,
    /// Error severity
    pub severity: f64,
    /// Suggested correction
    pub suggested_correction: Option<String>,
}
/// Adaptive strategy selection configuration
#[derive(Debug, Clone)]
pub struct AdaptiveStrategyConfig {
    /// Available strategies
    pub available_strategies: Vec<ErrorCorrectionStrategy>,
    /// Selection algorithm
    pub selection_algorithm: StrategySelectionAlgorithm,
    /// Performance history
    pub performance_history: Vec<StrategyPerformance>,
}
/// Protocol events for history tracking
#[derive(Debug, Clone)]
pub struct ProtocolEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Event type
    pub event_type: ProtocolEventType,
    /// Protocol involved
    pub protocol_id: String,
    /// Event details
    pub details: HashMap<String, String>,
}
/// Conditions for triggering adaptation
#[derive(Debug, Clone)]
pub enum AdaptationCondition {
    /// Noise level exceeds threshold
    NoiseThreshold(f64),
    /// Performance drops below threshold
    PerformanceThreshold(f64),
    /// Error rate exceeds threshold
    ErrorRateThreshold(f64),
    /// Resource usage exceeds threshold
    ResourceThreshold(f64),
    /// Time-based condition
    TimeBased(Duration),
    /// Composite condition
    Composite(Vec<Self>),
}
/// Comprehensive performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Error correction efficiency
    pub correction_efficiency: f64,
    /// Resource utilization efficiency
    pub resource_efficiency: f64,
    /// Adaptation responsiveness
    pub adaptation_responsiveness: f64,
    /// Prediction accuracy
    pub prediction_accuracy: f64,
    /// Overall system performance
    pub overall_performance: f64,
    /// Performance history
    pub performance_history: VecDeque<PerformanceSnapshot>,
}
/// Adaptation rules for protocol adjustment
#[derive(Debug, Clone)]
pub struct AdaptationRule {
    /// Rule identifier
    pub id: String,
    /// Condition for triggering adaptation
    pub condition: AdaptationCondition,
    /// Action to take
    pub action: AdaptationAction,
    /// Priority level
    pub priority: u8,
    /// Cooldown period
    pub cooldown: Duration,
}
/// Performance analysis system
pub struct PerformanceAnalyzer {
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Analysis algorithms
    pub analyzers: Vec<PerformanceAnalysisAlgorithm>,
    /// Benchmark comparisons
    pub benchmarks: HashMap<String, BenchmarkResult>,
}
impl PerformanceAnalyzer {
    pub(crate) fn new() -> Self {
        Self {
            metrics: PerformanceMetrics {
                correction_efficiency: 0.9,
                resource_efficiency: 0.8,
                adaptation_responsiveness: 0.85,
                prediction_accuracy: 0.8,
                overall_performance: 0.85,
                performance_history: VecDeque::new(),
            },
            analyzers: vec![],
            benchmarks: HashMap::new(),
        }
    }
    pub(crate) fn update_performance(&mut self, metadata: &CorrectionMetadata) {
        let efficiency = metadata.errors_corrected as f64 / metadata.errors_detected.max(1) as f64;
        self.metrics.correction_efficiency = self
            .metrics
            .correction_efficiency
            .mul_add(0.9, efficiency * 0.1);
        let resource_efficiency = 1.0 / (1.0 + metadata.correction_overhead);
        self.metrics.resource_efficiency = self
            .metrics
            .resource_efficiency
            .mul_add(0.9, resource_efficiency * 0.1);
        self.metrics.overall_performance = self.metrics.prediction_accuracy.mul_add(
            0.2,
            self.metrics.adaptation_responsiveness.mul_add(
                0.2,
                self.metrics
                    .correction_efficiency
                    .mul_add(0.3, self.metrics.resource_efficiency * 0.3),
            ),
        );
    }
}
/// Types of performance analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisType {
    /// Trend analysis
    Trend,
    /// Anomaly detection
    AnomalyDetection,
    /// Correlation analysis
    Correlation,
    /// Comparative analysis
    Comparative,
    /// Predictive analysis
    Predictive,
}
/// Hybrid error correction configuration
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Detection configuration
    pub detection: DetectionConfig,
    /// Correction configuration
    pub correction: CorrectionConfig,
    /// Switching criteria
    pub switching_criteria: SwitchingCriteria,
}
/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Computational resources
    pub computational: f64,
    /// Memory usage
    pub memory: f64,
    /// Communication overhead
    pub communication: f64,
    /// Total overhead
    pub total_overhead: f64,
    /// Usage history
    pub usage_history: VecDeque<ResourceSnapshot>,
}
