//! Type definitions for real-time hardware monitoring

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::braket::{BraketClient, BraketDevice};
use crate::dwave::DWaveClient;
use crate::embedding::{Embedding, HardwareGraph};
use crate::hardware_compilation::{CompilationTarget, HardwareCompiler};
use crate::ising::{IsingModel, QuboModel};
use crate::HardwareTopology;

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Monitoring interval (milliseconds)
    pub monitoring_interval: Duration,
    /// Metric collection window size
    pub metric_window_size: usize,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Adaptation sensitivity
    pub adaptation_sensitivity: f64,
    /// Predictive window size
    pub prediction_window: Duration,
    /// Enable real-time noise characterization
    pub enable_noise_characterization: bool,
    /// Enable adaptive compilation
    pub enable_adaptive_compilation: bool,
    /// Enable predictive failure detection
    pub enable_failure_prediction: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_millis(100),
            metric_window_size: 1000,
            alert_thresholds: AlertThresholds::default(),
            adaptation_sensitivity: 0.1,
            prediction_window: Duration::from_secs(300),
            enable_noise_characterization: true,
            enable_adaptive_compilation: true,
            enable_failure_prediction: true,
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Maximum error rate before alert
    pub max_error_rate: f64,
    /// Maximum temperature deviation
    pub max_temperature_deviation: f64,
    /// Minimum coherence time threshold
    pub min_coherence_time: Duration,
    /// Maximum noise level
    pub max_noise_level: f64,
    /// Minimum success rate
    pub min_success_rate: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_error_rate: 0.05,
            max_temperature_deviation: 0.1,
            min_coherence_time: Duration::from_micros(100),
            max_noise_level: 0.1,
            min_success_rate: 0.9,
        }
    }
}

/// Monitored quantum device
#[derive(Debug)]
pub struct MonitoredDevice {
    /// Device identifier
    pub device_id: String,
    /// Device type and capabilities
    pub device_info: DeviceInfo,
    /// Current device status
    pub status: DeviceStatus,
    /// Real-time performance metrics
    pub performance_metrics: Arc<RwLock<DevicePerformanceMetrics>>,
    /// Hardware topology information
    pub topology: HardwareTopology,
    /// Device connection
    pub connection: DeviceConnection,
    /// Monitoring history
    pub monitoring_history: Arc<Mutex<VecDeque<MonitoringSnapshot>>>,
    /// Current noise characterization
    pub noise_profile: Arc<RwLock<NoiseProfile>>,
    /// Calibration data
    pub calibration_data: Arc<RwLock<CalibrationData>>,
}

/// Device information and capabilities
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Maximum connectivity
    pub max_connectivity: usize,
    /// Supported operations
    pub supported_operations: Vec<QuantumOperation>,
    /// Temperature range
    pub temperature_range: (f64, f64),
    /// Coherence characteristics
    pub coherence_characteristics: CoherenceCharacteristics,
}

/// Supported quantum operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantumOperation {
    /// Ising annealing
    IsingAnnealing,
    /// QUBO optimization
    QUBOOptimization,
    /// Reverse annealing
    ReverseAnnealing,
    /// Multi-chip execution
    MultiChipExecution,
    /// Error correction
    ErrorCorrection,
}

/// Coherence characteristics
#[derive(Debug, Clone)]
pub struct CoherenceCharacteristics {
    /// T1 relaxation time
    pub t1_relaxation: Duration,
    /// T2 dephasing time
    pub t2_dephasing: Duration,
    /// Coherence preservation factor
    pub coherence_factor: f64,
    /// Decoherence sources
    pub decoherence_sources: Vec<DecoherenceSource>,
}

/// Sources of decoherence
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoherenceSource {
    /// Thermal fluctuations
    ThermalNoise,
    /// Flux noise
    FluxNoise,
    /// Charge noise
    ChargeNoise,
    /// Cross-talk
    CrossTalk,
    /// Environmental interference
    Environmental,
}

/// Device status enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceStatus {
    /// Online and available
    Online,
    /// Busy with current task
    Busy,
    /// Calibrating
    Calibrating,
    /// Maintenance mode
    Maintenance,
    /// Warning conditions detected
    Warning(Vec<String>),
    /// Error state
    Error(String),
    /// Offline
    Offline,
}

/// Real-time device performance metrics
#[derive(Debug, Clone)]
pub struct DevicePerformanceMetrics {
    /// Current error rate
    pub error_rate: f64,
    /// Current temperature
    pub temperature: f64,
    /// Coherence time measurements
    pub coherence_time: Duration,
    /// Noise level
    pub noise_level: f64,
    /// Success rate
    pub success_rate: f64,
    /// Execution speed
    pub execution_speed: f64,
    /// Queue depth
    pub queue_depth: usize,
    /// Last update timestamp
    pub last_update: Instant,
    /// Performance trend
    pub performance_trend: PerformanceTrend,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Error rate trend
    pub error_rate_trend: TrendDirection,
    /// Temperature trend
    pub temperature_trend: TrendDirection,
    /// Coherence trend
    pub coherence_trend: TrendDirection,
    /// Overall performance trend
    pub overall_trend: TrendDirection,
    /// Confidence level
    pub confidence: f64,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    /// Improving
    Improving,
    /// Stable
    Stable,
    /// Degrading
    Degrading,
    /// Uncertain
    Uncertain,
}

/// Device connection interface
#[derive(Debug)]
pub enum DeviceConnection {
    /// D-Wave connection
    DWave(Arc<Mutex<DWaveClient>>),
    /// AWS Braket connection
    Braket(Arc<Mutex<BraketClient>>),
    /// Local simulator
    Simulator(String),
    /// Custom connection
    Custom(String),
}

/// Monitoring snapshot for historical tracking
#[derive(Debug, Clone)]
pub struct MonitoringSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// Performance metrics at this time
    pub metrics: DevicePerformanceMetrics,
    /// Any alerts generated
    pub alerts: Vec<Alert>,
    /// Adaptive actions taken
    pub adaptations: Vec<AdaptiveAction>,
}

/// System alert
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert identifier
    pub id: String,
    /// Alert level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Source device
    pub device_id: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Metric that triggered alert
    pub trigger_metric: String,
    /// Metric value
    pub metric_value: f64,
    /// Threshold that was exceeded
    pub threshold: f64,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertLevel {
    /// Informational
    Info,
    /// Warning condition
    Warning,
    /// Error condition
    Error,
    /// Critical condition
    Critical,
}

/// Adaptive action taken by the system
#[derive(Debug, Clone)]
pub struct AdaptiveAction {
    /// Action identifier
    pub id: String,
    /// Action type
    pub action_type: AdaptiveActionType,
    /// Target device
    pub device_id: String,
    /// Action parameters
    pub parameters: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: Instant,
    /// Expected impact
    pub expected_impact: String,
    /// Success indicator
    pub success: Option<bool>,
}

/// Types of adaptive actions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveActionType {
    /// Adjust chain strength
    ChainStrengthAdjustment,
    /// Modify annealing schedule
    ScheduleModification,
    /// Change embedding strategy
    EmbeddingChange,
    /// Temperature compensation
    TemperatureCompensation,
    /// Noise mitigation
    NoiseMitigation,
    /// Topology reconfiguration
    TopologyReconfiguration,
    /// Calibration update
    CalibrationUpdate,
}

/// Noise characterization profile
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Per-qubit noise levels
    pub qubit_noise: Vec<f64>,
    /// Coupling noise matrix
    pub coupling_noise: Vec<Vec<f64>>,
    /// Temporal noise characteristics
    pub temporal_noise: TemporalNoiseProfile,
    /// Spectral noise analysis
    pub spectral_noise: SpectralNoiseProfile,
    /// Correlated noise patterns
    pub noise_correlations: NoiseCorrelationMatrix,
    /// Last characterization time
    pub last_update: Instant,
}

/// Temporal noise characteristics
#[derive(Debug, Clone)]
pub struct TemporalNoiseProfile {
    /// Noise autocorrelation function
    pub autocorrelation: Vec<f64>,
    /// Correlation time scales
    pub correlation_times: Vec<Duration>,
    /// Non-Markovian memory effects
    pub memory_effects: Vec<f64>,
    /// Burst noise patterns
    pub burst_patterns: Vec<BurstPattern>,
}

/// Burst noise pattern
#[derive(Debug, Clone)]
pub struct BurstPattern {
    /// Pattern duration
    pub duration: Duration,
    /// Intensity scale
    pub intensity: f64,
    /// Frequency of occurrence
    pub frequency: f64,
    /// Affected qubits
    pub affected_qubits: Vec<usize>,
}

/// Spectral noise analysis
#[derive(Debug, Clone)]
pub struct SpectralNoiseProfile {
    /// Power spectral density
    pub power_spectrum: Vec<f64>,
    /// Frequency bins
    pub frequency_bins: Vec<f64>,
    /// Dominant noise frequencies
    pub dominant_frequencies: Vec<f64>,
    /// 1/f noise characteristics
    pub flicker_noise_params: FlickerNoiseParams,
}

/// Flicker noise parameters
#[derive(Debug, Clone)]
pub struct FlickerNoiseParams {
    /// Flicker noise amplitude
    pub amplitude: f64,
    /// Frequency exponent
    pub exponent: f64,
    /// Corner frequency
    pub corner_frequency: f64,
}

/// Noise correlation matrix
#[derive(Debug, Clone)]
pub struct NoiseCorrelationMatrix {
    /// Spatial correlations between qubits
    pub spatial_correlations: Vec<Vec<f64>>,
    /// Temporal correlations
    pub temporal_correlations: Vec<f64>,
    /// Cross-correlation patterns
    pub cross_correlations: HashMap<String, Vec<f64>>,
}

/// Calibration data for device
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Per-qubit bias calibration
    pub bias_calibration: Vec<f64>,
    /// Coupling strength calibration
    pub coupling_calibration: Vec<Vec<f64>>,
    /// Annealing schedule calibration
    pub schedule_calibration: ScheduleCalibration,
    /// Temperature calibration
    pub temperature_calibration: TemperatureCalibration,
    /// Last calibration time
    pub last_calibration: Instant,
    /// Calibration validity
    pub calibration_validity: Duration,
}

/// Annealing schedule calibration
#[derive(Debug, Clone)]
pub struct ScheduleCalibration {
    /// Optimal annealing time
    pub optimal_anneal_time: Duration,
    /// Schedule shape parameters
    pub shape_parameters: Vec<f64>,
    /// Pause points
    pub pause_points: Vec<f64>,
    /// Ramp rates
    pub ramp_rates: Vec<f64>,
}

/// Temperature calibration data
#[derive(Debug, Clone)]
pub struct TemperatureCalibration {
    /// Temperature offset correction
    pub offset_correction: f64,
    /// Temperature scaling factor
    pub scaling_factor: f64,
    /// Thermal stability map
    pub stability_map: Vec<Vec<f64>>,
}

/// Metrics collection configuration
#[derive(Debug, Clone)]
pub struct MetricsCollectionConfig {
    /// Metrics to collect
    pub enabled_metrics: std::collections::HashSet<MetricType>,
    /// Collection frequency
    pub collection_frequency: Duration,
    /// Retention period
    pub retention_period: Duration,
    /// Aggregation window
    pub aggregation_window: Duration,
}

/// Types of metrics to collect
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum MetricType {
    /// Error rate
    ErrorRate,
    /// Temperature
    Temperature,
    /// Coherence time
    CoherenceTime,
    /// Noise level
    NoiseLevel,
    /// Success rate
    SuccessRate,
    /// Execution speed
    ExecutionSpeed,
    /// Queue depth
    QueueDepth,
    /// Memory usage
    MemoryUsage,
    /// CPU utilization
    CPUUtilization,
}

/// Time series data for a metric
#[derive(Debug, Clone)]
pub struct MetricTimeSeries {
    /// Metric type
    pub metric_type: MetricType,
    /// Time series data points
    pub data_points: VecDeque<MetricDataPoint>,
    /// Statistical summary
    pub statistics: MetricStatistics,
}

/// Individual metric data point
#[derive(Debug, Clone)]
pub struct MetricDataPoint {
    /// Timestamp
    pub timestamp: Instant,
    /// Metric value
    pub value: f64,
    /// Data quality indicator
    pub quality: DataQuality,
    /// Source device
    pub device_id: String,
}

/// Data quality indicator
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataQuality {
    /// High quality data
    High,
    /// Medium quality data
    Medium,
    /// Low quality data
    Low,
    /// Estimated/interpolated data
    Estimated,
}

/// Metric statistics
#[derive(Debug, Clone)]
pub struct MetricStatistics {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Trend analysis
    pub trend: TrendAnalysis,
    /// Outlier detection
    pub outliers: Vec<usize>,
}

/// Trend analysis results
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Linear trend slope
    pub slope: f64,
    /// Trend confidence
    pub confidence: f64,
    /// Seasonal patterns
    pub seasonal_patterns: Vec<SeasonalPattern>,
    /// Change points
    pub change_points: Vec<ChangePoint>,
}

/// Seasonal pattern in metrics
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    /// Pattern period
    pub period: Duration,
    /// Pattern amplitude
    pub amplitude: f64,
    /// Pattern phase
    pub phase: f64,
    /// Pattern strength
    pub strength: f64,
}

/// Change point in time series
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Change point timestamp
    pub timestamp: Instant,
    /// Change magnitude
    pub magnitude: f64,
    /// Change type
    pub change_type: ChangeType,
    /// Confidence level
    pub confidence: f64,
}

/// Types of changes in time series
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    /// Mean shift
    MeanShift,
    /// Variance change
    VarianceChange,
    /// Trend change
    TrendChange,
    /// Regime shift
    RegimeShift,
}

/// Metric aggregate data
#[derive(Debug, Clone)]
pub struct MetricAggregate {
    /// Aggregation window
    pub window: Duration,
    /// Aggregated values
    pub values: HashMap<AggregationType, f64>,
    /// Last update
    pub last_update: Instant,
}

/// Types of aggregation
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum AggregationType {
    /// Average
    Average,
    /// Maximum
    Maximum,
    /// Minimum
    Minimum,
    /// 95th percentile
    Percentile95,
    /// Standard deviation
    StandardDeviation,
}

/// Collection statistics
#[derive(Debug, Clone)]
pub struct CollectionStatistics {
    /// Total data points collected
    pub total_points: u64,
    /// Collection success rate
    pub success_rate: f64,
    /// Average collection latency
    pub avg_latency: Duration,
    /// Last collection time
    pub last_collection: Instant,
}

/// Adaptive compiler configuration
#[derive(Debug, Clone)]
pub struct AdaptiveCompilerConfig {
    /// Enable real-time recompilation
    pub enable_realtime_recompilation: bool,
    /// Adaptation threshold
    pub adaptation_threshold: f64,
    /// Maximum adaptations per hour
    pub max_adaptations_per_hour: usize,
    /// Compilation cache size
    pub cache_size: usize,
    /// Performance tracking window
    pub performance_window: Duration,
}

/// Cached compilation result
#[derive(Debug, Clone)]
pub struct CachedCompilation {
    /// Problem hash
    pub problem_hash: String,
    /// Compiled embedding
    pub embedding: Embedding,
    /// Compilation parameters
    pub parameters: CompilationParameters,
    /// Expected performance
    pub expected_performance: f64,
    /// Cache timestamp
    pub timestamp: Instant,
    /// Usage count
    pub usage_count: u64,
}

/// Compilation parameters
#[derive(Debug, Clone)]
pub struct CompilationParameters {
    /// Chain strength
    pub chain_strength: f64,
    /// Annealing schedule
    pub annealing_schedule: Vec<(f64, f64)>,
    /// Temperature compensation
    pub temperature_compensation: f64,
    /// Noise mitigation settings
    pub noise_mitigation: NoiseMitigationSettings,
}

/// Noise mitigation settings
#[derive(Debug, Clone)]
pub struct NoiseMitigationSettings {
    /// Enable error correction
    pub enable_error_correction: bool,
    /// Noise model
    pub noise_model: NoiseModel,
    /// Mitigation strategy
    pub mitigation_strategy: MitigationStrategy,
    /// Correction threshold
    pub correction_threshold: f64,
}

/// Noise model for mitigation
#[derive(Debug, Clone)]
pub enum NoiseModel {
    /// Gaussian noise model
    Gaussian { variance: f64 },
    /// Correlated noise model
    Correlated { correlation_matrix: Vec<Vec<f64>> },
    /// Markovian noise model
    Markovian { transition_rates: Vec<f64> },
    /// Non-Markovian noise model
    NonMarkovian { memory_kernel: Vec<f64> },
}

/// Mitigation strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MitigationStrategy {
    /// Zero-noise extrapolation
    ZeroNoiseExtrapolation,
    /// Probabilistic error cancellation
    ProbabilisticErrorCancellation,
    /// Symmetry verification
    SymmetryVerification,
    /// Dynamical decoupling
    DynamicalDecoupling,
}

/// Adaptation strategy
#[derive(Debug, Clone)]
pub struct AdaptationStrategy {
    /// Strategy name
    pub name: String,
    /// Trigger conditions
    pub triggers: Vec<AdaptationTrigger>,
    /// Adaptation actions
    pub actions: Vec<AdaptationAction>,
    /// Success criteria
    pub success_criteria: SuccessCriteria,
    /// Strategy priority
    pub priority: u32,
}

/// Trigger for adaptation
#[derive(Debug, Clone)]
pub struct AdaptationTrigger {
    /// Metric to monitor
    pub metric: MetricType,
    /// Trigger condition
    pub condition: TriggerCondition,
    /// Threshold value
    pub threshold: f64,
    /// Persistence requirement
    pub persistence: Duration,
}

/// Trigger condition
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerCondition {
    /// Greater than threshold
    GreaterThan,
    /// Less than threshold
    LessThan,
    /// Rapid change detected
    RapidChange,
    /// Trend detected
    TrendDetected(TrendDirection),
    /// Anomaly detected
    AnomalyDetected,
}

/// Adaptation action to take
#[derive(Debug, Clone)]
pub struct AdaptationAction {
    /// Action type
    pub action_type: AdaptiveActionType,
    /// Action parameters
    pub parameters: HashMap<String, f64>,
    /// Expected impact
    pub expected_impact: f64,
    /// Action priority
    pub priority: u32,
}

/// Success criteria for adaptations
#[derive(Debug, Clone)]
pub struct SuccessCriteria {
    /// Improvement threshold
    pub improvement_threshold: f64,
    /// Evaluation window
    pub evaluation_window: Duration,
    /// Minimum sample size
    pub min_samples: usize,
    /// Success metric
    pub success_metric: MetricType,
}

/// Compilation performance tracking
#[derive(Debug, Clone)]
pub struct CompilationPerformance {
    /// Problem identifier
    pub problem_id: String,
    /// Compilation time
    pub compilation_time: Duration,
    /// Execution performance
    pub execution_performance: f64,
    /// Solution quality
    pub solution_quality: f64,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Timestamp
    pub timestamp: Instant,
}

/// Active adaptation tracking
#[derive(Debug, Clone)]
pub struct ActiveAdaptation {
    /// Adaptation identifier
    pub id: String,
    /// Strategy used
    pub strategy: String,
    /// Start time
    pub start_time: Instant,
    /// Current status
    pub status: AdaptationStatus,
    /// Performance before adaptation
    pub baseline_performance: f64,
    /// Current performance
    pub current_performance: f64,
    /// Expected completion
    pub expected_completion: Instant,
}

/// Status of adaptation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptationStatus {
    /// In progress
    InProgress,
    /// Successful
    Successful,
    /// Failed
    Failed,
    /// Rolled back
    RolledBack,
}

/// Alert system configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Maximum active alerts
    pub max_active_alerts: usize,
    /// Alert aggregation window
    pub aggregation_window: Duration,
    /// Alert suppression rules
    pub suppression_rules: Vec<SuppressionRule>,
    /// Escalation rules
    pub escalation_rules: Vec<EscalationRule>,
}

/// Alert suppression rule
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule name
    pub name: String,
    /// Suppression conditions
    pub conditions: Vec<SuppressionCondition>,
    /// Suppression duration
    pub duration: Duration,
}

/// Suppression condition
#[derive(Debug, Clone)]
pub struct SuppressionCondition {
    /// Condition type
    pub condition_type: SuppressionType,
    /// Condition parameters
    pub parameters: HashMap<String, String>,
}

/// Types of suppression
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuppressionType {
    /// Similar alerts in time window
    SimilarAlerts,
    /// Device in maintenance
    MaintenanceMode,
    /// Scheduled downtime
    ScheduledDowntime,
    /// Alert flood protection
    FloodProtection,
}

/// Alert escalation rule
#[derive(Debug, Clone)]
pub struct EscalationRule {
    /// Rule name
    pub name: String,
    /// Escalation conditions
    pub conditions: Vec<EscalationCondition>,
    /// Escalation actions
    pub actions: Vec<EscalationAction>,
}

/// Escalation condition
#[derive(Debug, Clone)]
pub struct EscalationCondition {
    /// Time without resolution
    pub unresolved_duration: Duration,
    /// Alert level threshold
    pub level_threshold: AlertLevel,
    /// Repeat count threshold
    pub repeat_threshold: usize,
}

/// Escalation action
#[derive(Debug, Clone)]
pub struct EscalationAction {
    /// Action type
    pub action_type: EscalationType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
}

/// Types of escalation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscalationType {
    /// Notify additional contacts
    NotifyContacts,
    /// Increase alert level
    IncreaseLevel,
    /// Trigger automated response
    AutomatedResponse,
    /// Create support ticket
    CreateTicket,
}

/// Alert handler trait
pub trait AlertHandler: Send + Sync {
    /// Handle alert
    fn handle_alert(&self, alert: &Alert) -> crate::applications::ApplicationResult<()>;
    /// Get handler name
    fn get_name(&self) -> &str;
}

/// Alert statistics
#[derive(Debug, Clone)]
pub struct AlertStatistics {
    /// Total alerts generated
    pub total_alerts: u64,
    /// Alerts by level
    pub alerts_by_level: HashMap<AlertLevel, u64>,
    /// Alerts by device
    pub alerts_by_device: HashMap<String, u64>,
    /// Average resolution time
    pub avg_resolution_time: Duration,
    /// False positive rate
    pub false_positive_rate: f64,
}

/// Failure detection configuration
#[derive(Debug, Clone)]
pub struct FailureDetectionConfig {
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Model update frequency
    pub model_update_frequency: Duration,
    /// Feature extraction window
    pub feature_window: Duration,
}

/// Prediction model for failure detection
#[derive(Debug)]
pub struct PredictionModel {
    /// Model type
    pub model_type: ModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Feature extractors
    pub features: Vec<FeatureExtractor>,
    /// Model state
    pub state: ModelState,
    /// Last training time
    pub last_training: Instant,
}

/// Types of prediction models
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelType {
    /// Linear regression
    LinearRegression,
    /// Support vector machine
    SVM,
    /// Random forest
    RandomForest,
    /// Neural network
    NeuralNetwork,
    /// LSTM recurrent network
    LSTM,
    /// Gaussian process
    GaussianProcess,
}

/// Feature extractor for prediction
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Feature name
    pub name: String,
    /// Feature type
    pub feature_type: FeatureType,
    /// Extraction parameters
    pub parameters: HashMap<String, f64>,
    /// Normalization method
    pub normalization: NormalizationMethod,
}

/// Types of features for prediction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureType {
    /// Statistical features (mean, std, etc.)
    Statistical,
    /// Temporal features (trends, seasonality)
    Temporal,
    /// Spectral features (frequency domain)
    Spectral,
    /// Correlation features
    Correlation,
    /// Anomaly features
    Anomaly,
}

/// Normalization methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Z-score normalization
    ZScore,
    /// Min-max normalization
    MinMax,
    /// Robust scaling
    Robust,
    /// No normalization
    None,
}

/// Model state
#[derive(Debug, Clone)]
pub struct ModelState {
    /// Is model trained
    pub is_trained: bool,
    /// Training accuracy
    pub training_accuracy: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Model complexity
    pub complexity: f64,
    /// Training data size
    pub training_data_size: usize,
}

/// Failure event for historical tracking
#[derive(Debug, Clone)]
pub struct FailureEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Device that failed
    pub device_id: String,
    /// Failure type
    pub failure_type: FailureType,
    /// Failure severity
    pub severity: FailureSeverity,
    /// Leading indicators
    pub leading_indicators: Vec<String>,
    /// Resolution time
    pub resolution_time: Option<Duration>,
    /// Root cause
    pub root_cause: Option<String>,
}

/// Types of failures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureType {
    /// Hardware failure
    Hardware,
    /// Calibration drift
    CalibrationDrift,
    /// Noise increase
    NoiseIncrease,
    /// Temperature excursion
    TemperatureExcursion,
    /// Coherence loss
    CoherenceLoss,
    /// Communication failure
    CommunicationFailure,
}

/// Failure severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureSeverity {
    /// Minor performance degradation
    Minor,
    /// Moderate impact
    Moderate,
    /// Major impact
    Major,
    /// Complete failure
    Critical,
}

/// Failure prediction
#[derive(Debug, Clone)]
pub struct FailurePrediction {
    /// Device identifier
    pub device_id: String,
    /// Predicted failure type
    pub predicted_failure: FailureType,
    /// Prediction confidence
    pub confidence: f64,
    /// Time to failure estimate
    pub time_to_failure: Duration,
    /// Prediction timestamp
    pub prediction_time: Instant,
    /// Contributing factors
    pub contributing_factors: Vec<String>,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Model performance tracking
#[derive(Debug, Clone)]
pub struct ModelPerformance {
    /// Model identifier
    pub model_id: String,
    /// Prediction accuracy
    pub accuracy: f64,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// False negative rate
    pub false_negative_rate: f64,
    /// Last evaluation time
    pub last_evaluation: Instant,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Optimization frequency
    pub optimization_frequency: Duration,
    /// Performance improvement threshold
    pub improvement_threshold: f64,
    /// Maximum concurrent optimizations
    pub max_concurrent_optimizations: usize,
    /// Optimization timeout
    pub optimization_timeout: Duration,
}

/// Optimization strategy
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    /// Strategy name
    pub name: String,
    /// Optimization targets
    pub targets: Vec<OptimizationTarget>,
    /// Optimization method
    pub method: OptimizationMethod,
    /// Strategy parameters
    pub parameters: HashMap<String, f64>,
    /// Success rate
    pub success_rate: f64,
}

/// Optimization target
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationTarget {
    /// Minimize error rate
    MinimizeErrorRate,
    /// Maximize success rate
    MaximizeSuccessRate,
    /// Minimize execution time
    MinimizeExecutionTime,
    /// Maximize coherence time
    MaximizeCoherenceTime,
    /// Minimize noise
    MinimizeNoise,
    /// Optimize energy efficiency
    OptimizeEnergyEfficiency,
}

/// Optimization methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationMethod {
    /// Gradient descent
    GradientDescent,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Bayesian optimization
    BayesianOptimization,
    /// Reinforcement learning
    ReinforcementLearning,
}

/// Performance baseline
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Device identifier
    pub device_id: String,
    /// Baseline metrics
    pub baseline_metrics: HashMap<MetricType, f64>,
    /// Baseline timestamp
    pub baseline_time: Instant,
    /// Baseline validity period
    pub validity_period: Duration,
}

/// Active optimization tracking
#[derive(Debug, Clone)]
pub struct ActiveOptimization {
    /// Optimization identifier
    pub id: String,
    /// Target device
    pub device_id: String,
    /// Optimization strategy
    pub strategy: String,
    /// Start time
    pub start_time: Instant,
    /// Current status
    pub status: OptimizationStatus,
    /// Progress indicator
    pub progress: f64,
    /// Intermediate results
    pub intermediate_results: Vec<f64>,
}

/// Optimization status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationStatus {
    /// Initialization
    Initializing,
    /// Running
    Running,
    /// Converged
    Converged,
    /// Failed
    Failed,
    /// Terminated
    Terminated,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimization identifier
    pub id: String,
    /// Device identifier
    pub device_id: String,
    /// Strategy used
    pub strategy: String,
    /// Performance before optimization
    pub baseline_performance: f64,
    /// Performance after optimization
    pub final_performance: f64,
    /// Improvement achieved
    pub improvement: f64,
    /// Optimization duration
    pub duration: Duration,
    /// Success indicator
    pub success: bool,
    /// Timestamp
    pub timestamp: Instant,
}
