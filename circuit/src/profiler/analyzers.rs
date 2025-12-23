//! Performance analysis, anomaly detection, and prediction engines
//!
//! This module provides comprehensive performance analysis capabilities
//! including statistical analysis, anomaly detection, alert systems,
//! and performance prediction using machine learning models.

use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

// Import types from sibling modules
use super::collectors::*;
use super::metrics::*;

pub struct PerformanceAnalyzer {
    /// Analysis configuration
    pub config: AnalysisConfig,
    /// Historical performance data
    pub historical_data: HistoricalPerformanceData,
    /// Performance models
    pub performance_models: PerformanceModels,
    /// Anomaly detector
    pub anomaly_detector: AnomalyDetector,
    /// Prediction engine
    pub prediction_engine: PredictionEngine,
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Analysis depth
    pub analysis_depth: AnalysisDepth,
    /// Statistical methods to use
    pub statistical_methods: HashSet<StatisticalMethod>,
    /// Machine learning models to use
    pub ml_models: HashSet<MlModel>,
    /// Confidence level for analysis
    pub confidence_level: f64,
    /// Minimum data points for analysis
    pub min_data_points: usize,
}

/// Analysis depth levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDepth {
    /// Basic statistical analysis
    Basic,
    /// Standard analysis with trends
    Standard,
    /// Advanced analysis with predictions
    Advanced,
    /// Comprehensive analysis with ML
    Comprehensive,
}

/// Statistical methods for analysis
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum StatisticalMethod {
    /// Descriptive statistics
    Descriptive,
    /// Correlation analysis
    Correlation,
    /// Regression analysis
    Regression,
    /// Time series analysis
    TimeSeries,
    /// Hypothesis testing
    HypothesisTesting,
}

/// Machine learning models for analysis
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum MlModel {
    /// Linear regression
    LinearRegression,
    /// Random forest
    RandomForest,
    /// Neural networks
    NeuralNetwork,
    /// Support vector machines
    SupportVectorMachine,
    /// Clustering algorithms
    Clustering,
}

/// Historical performance data storage
#[derive(Debug, Clone)]
pub struct HistoricalPerformanceData {
    /// Performance snapshots over time
    pub snapshots: VecDeque<PerformanceSnapshot>,
    /// Data retention policy
    pub retention_policy: DataRetentionPolicy,
    /// Data compression settings
    pub compression_settings: CompressionSettings,
    /// Data integrity checks
    pub integrity_checks: IntegrityChecks,
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// System state
    pub system_state: SystemState,
    /// Environment information
    pub environment: EnvironmentInfo,
    /// Snapshot metadata
    pub metadata: HashMap<String, String>,
}

/// System state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// CPU state
    pub cpu_state: CpuState,
    /// Memory state
    pub memory_state: MemoryState,
    /// I/O state
    pub io_state: IoState,
    /// Network state
    pub network_state: NetworkState,
}

/// CPU state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuState {
    /// CPU utilization
    pub utilization: f64,
    /// CPU frequency
    pub frequency: f64,
    /// CPU temperature
    pub temperature: Option<f64>,
    /// Active processes
    pub active_processes: usize,
}

/// Memory state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryState {
    /// Total memory
    pub total_memory: usize,
    /// Used memory
    pub used_memory: usize,
    /// Free memory
    pub free_memory: usize,
    /// Cached memory
    pub cached_memory: usize,
}

/// I/O state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoState {
    /// Disk usage
    pub disk_usage: f64,
    /// Read IOPS
    pub read_iops: f64,
    /// Write IOPS
    pub write_iops: f64,
    /// Queue depth
    pub queue_depth: f64,
}

/// Network state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkState {
    /// Bandwidth utilization
    pub bandwidth_utilization: f64,
    /// Active connections
    pub active_connections: usize,
    /// Packet rate
    pub packet_rate: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Operating system
    pub operating_system: String,
    /// Hardware configuration
    pub hardware_config: HardwareConfig,
    /// Software versions
    pub software_versions: HashMap<String, String>,
    /// Environment variables
    pub environment_variables: HashMap<String, String>,
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// CPU model
    pub cpu_model: String,
    /// CPU cores
    pub cpu_cores: u32,
    /// Total memory
    pub total_memory: usize,
    /// GPU information
    pub gpu_info: Option<GpuInfo>,
    /// Storage information
    pub storage_info: StorageInfo,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU model
    pub model: String,
    /// GPU memory
    pub memory: usize,
    /// Compute capability
    pub compute_capability: String,
}

/// Storage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    /// Storage type
    pub storage_type: StorageType,
    /// Total capacity
    pub total_capacity: usize,
    /// Available capacity
    pub available_capacity: usize,
}

/// Storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    /// Hard disk drive
    HDD,
    /// Solid state drive
    SSD,
    /// `NVMe` SSD
    NVMe,
    /// Network storage
    Network,
}

/// Data retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionPolicy {
    /// Maximum age for data
    pub max_age: Duration,
    /// Maximum number of snapshots
    pub max_snapshots: usize,
    /// Compression threshold
    pub compression_threshold: Duration,
    /// Archival policy
    pub archival_policy: ArchivalPolicy,
}

/// Archival policy for old data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchivalPolicy {
    /// Delete old data
    Delete,
    /// Compress old data
    Compress,
    /// Archive to external storage
    Archive { location: String },
    /// Keep all data
    KeepAll,
}

/// Data compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub compression_level: u8,
    /// Enable real-time compression
    pub realtime_compression: bool,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 compression
    LZ4,
    /// Gzip compression
    Gzip,
    /// Zstd compression
    Zstd,
    /// Custom compression
    Custom { name: String },
}

/// Data integrity checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityChecks {
    /// Enable checksums
    pub enable_checksums: bool,
    /// Checksum algorithm
    pub checksum_algorithm: ChecksumAlgorithm,
    /// Verification frequency
    pub verification_frequency: Duration,
}

/// Checksum algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    /// CRC32 checksum
    CRC32,
    /// MD5 hash
    MD5,
    /// SHA256 hash
    SHA256,
    /// Blake3 hash
    Blake3,
}

/// Performance models for prediction
#[derive(Debug, Clone)]
pub struct PerformanceModels {
    /// Statistical models
    pub statistical_models: HashMap<String, StatisticalModel>,
    /// Machine learning models
    pub ml_models: HashMap<String, MachineLearningModel>,
    /// Hybrid models
    pub hybrid_models: HashMap<String, HybridModel>,
    /// Model evaluation results
    pub evaluation_results: ModelEvaluationResults,
}

/// Statistical performance model
#[derive(Debug, Clone)]
pub struct StatisticalModel {
    /// Model type
    pub model_type: StatisticalModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Model accuracy
    pub accuracy: f64,
    /// Training data size
    pub training_data_size: usize,
}

/// Types of statistical models
#[derive(Debug, Clone)]
pub enum StatisticalModelType {
    /// Linear regression
    LinearRegression,
    /// Autoregressive model
    Autoregressive,
    /// Moving average model
    MovingAverage,
    /// ARIMA model
    Arima,
    /// Exponential smoothing
    ExponentialSmoothing,
}

/// Machine learning performance model
#[derive(Debug, Clone)]
pub struct MachineLearningModel {
    /// Model type
    pub model_type: MlModelType,
    /// Model hyperparameters
    pub hyperparameters: HashMap<String, f64>,
    /// Model accuracy
    pub accuracy: f64,
    /// Training data size
    pub training_data_size: usize,
    /// Feature importance
    pub feature_importance: HashMap<String, f64>,
}

/// Types of ML models
#[derive(Debug, Clone)]
pub enum MlModelType {
    /// Random forest
    RandomForest,
    /// Gradient boosting
    GradientBoosting,
    /// Neural network
    NeuralNetwork,
    /// Support vector regression
    SupportVectorRegression,
    /// Gaussian process
    GaussianProcess,
}

/// Hybrid performance model
#[derive(Debug, Clone)]
pub struct HybridModel {
    /// Component models
    pub component_models: Vec<ComponentModel>,
    /// Ensemble weights
    pub ensemble_weights: HashMap<String, f64>,
    /// Model accuracy
    pub accuracy: f64,
    /// Combination strategy
    pub combination_strategy: CombinationStrategy,
}

/// Component model in hybrid ensemble
#[derive(Debug, Clone)]
pub struct ComponentModel {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ComponentModelType,
    /// Model weight
    pub weight: f64,
    /// Model accuracy
    pub accuracy: f64,
}

/// Types of component models
#[derive(Debug, Clone)]
pub enum ComponentModelType {
    /// Statistical model
    Statistical(StatisticalModelType),
    /// Machine learning model
    MachineLearning(MlModelType),
    /// Physics-based model
    PhysicsBased,
    /// Empirical model
    Empirical,
}

/// Strategies for combining models
#[derive(Debug, Clone)]
pub enum CombinationStrategy {
    /// Weighted average
    WeightedAverage,
    /// Voting ensemble
    Voting,
    /// Stacking ensemble
    Stacking,
    /// Bayesian model averaging
    BayesianAveraging,
}

/// Model evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvaluationResults {
    /// Cross-validation scores
    pub cv_scores: HashMap<String, f64>,
    /// Test set performance
    pub test_performance: HashMap<String, f64>,
    /// Model comparison
    pub model_comparison: ModelComparison,
    /// Feature analysis
    pub feature_analysis: FeatureAnalysis,
}

/// Model comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Best performing model
    pub best_model: String,
    /// Performance rankings
    pub performance_rankings: Vec<ModelRanking>,
    /// Statistical significance tests
    pub significance_tests: HashMap<String, f64>,
}

/// Individual model ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRanking {
    /// Model name
    pub model_name: String,
    /// Performance score
    pub performance_score: f64,
    /// Ranking position
    pub rank: u32,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Feature analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureAnalysis {
    /// Feature importance scores
    pub feature_importance: HashMap<String, f64>,
    /// Feature correlations
    pub feature_correlations: HashMap<String, f64>,
    /// Feature selection results
    pub feature_selection: FeatureSelectionResults,
}

/// Feature selection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionResults {
    /// Selected features
    pub selected_features: Vec<String>,
    /// Feature selection method
    pub selection_method: String,
    /// Selection criteria
    pub selection_criteria: HashMap<String, f64>,
}

/// Anomaly detection system
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Detection algorithms
    pub algorithms: HashMap<String, AnomalyDetectionAlgorithm>,
    /// Detected anomalies
    pub detected_anomalies: Vec<PerformanceAnomaly>,
    /// Detection configuration
    pub config: AnomalyDetectionConfig,
    /// Alert system
    pub alert_system: AlertSystem,
}

/// Anomaly detection algorithm
#[derive(Debug, Clone)]
pub struct AnomalyDetectionAlgorithm {
    /// Algorithm type
    pub algorithm_type: AnomalyAlgorithmType,
    /// Algorithm parameters
    pub parameters: HashMap<String, f64>,
    /// Detection threshold
    pub threshold: f64,
    /// False positive rate
    pub false_positive_rate: f64,
}

/// Types of anomaly detection algorithms
#[derive(Debug, Clone)]
pub enum AnomalyAlgorithmType {
    /// Statistical outlier detection
    StatisticalOutlier,
    /// Isolation forest
    IsolationForest,
    /// One-class SVM
    OneClassSVM,
    /// DBSCAN clustering
    DBSCAN,
    /// Autoencoder
    Autoencoder,
}

/// Performance anomaly information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnomaly {
    /// Anomaly ID
    pub id: String,
    /// Detection timestamp
    pub timestamp: SystemTime,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: AnomySeverity,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Anomaly score
    pub anomaly_score: f64,
    /// Root cause analysis
    pub root_cause: Option<String>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Types of performance anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Performance degradation
    PerformanceDegradation,
    /// Resource spike
    ResourceSpike,
    /// Error rate increase
    ErrorRateIncrease,
    /// Latency increase
    LatencyIncrease,
    /// Throughput decrease
    ThroughputDecrease,
    /// Memory leak
    MemoryLeak,
    /// Custom anomaly
    Custom { name: String },
}

/// Anomaly severity levels
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum AnomySeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable real-time detection
    pub enable_realtime: bool,
    /// Detection sensitivity
    pub sensitivity: f64,
    /// Minimum anomaly duration
    pub min_duration: Duration,
    /// Alert thresholds
    pub alert_thresholds: HashMap<AnomySeverity, f64>,
}

/// Alert system for anomalies
#[derive(Debug, Clone)]
pub struct AlertSystem {
    /// Alert channels
    pub alert_channels: Vec<AlertChannel>,
    /// Alert history
    pub alert_history: VecDeque<Alert>,
    /// Alert rules
    pub alert_rules: Vec<AlertRule>,
    /// Alert suppression rules
    pub suppression_rules: Vec<SuppressionRule>,
}

/// Alert channel configuration
#[derive(Debug, Clone)]
pub struct AlertChannel {
    /// Channel type
    pub channel_type: AlertChannelType,
    /// Channel configuration
    pub config: HashMap<String, String>,
    /// Enabled status
    pub enabled: bool,
}

/// Types of alert channels
#[derive(Debug, Clone)]
pub enum AlertChannelType {
    /// Email alerts
    Email,
    /// Slack notifications
    Slack,
    /// Webhook calls
    Webhook,
    /// Log file entries
    LogFile,
    /// System notifications
    SystemNotification,
}

/// Alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Associated anomaly
    pub anomaly_id: Option<String>,
    /// Alert source
    pub source: String,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    /// Info level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical level
    Critical,
}

/// Alert rule configuration
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: AlertCondition,
    /// Alert level
    pub alert_level: AlertLevel,
    /// Alert message template
    pub message_template: String,
    /// Enabled status
    pub enabled: bool,
}

/// Alert condition
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Threshold condition
    Threshold {
        metric: String,
        operator: ComparisonOperator,
        value: f64,
    },
    /// Rate condition
    Rate {
        metric: String,
        rate_threshold: f64,
        time_window: Duration,
    },
    /// Composite condition
    Composite {
        conditions: Vec<Self>,
        operator: LogicalOperator,
    },
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Equal to
    EqualTo,
    /// Not equal to
    NotEqualTo,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than or equal
    LessThanOrEqual,
}

/// Logical operators
#[derive(Debug, Clone)]
pub enum LogicalOperator {
    /// Logical AND
    And,
    /// Logical OR
    Or,
    /// Logical NOT
    Not,
}

/// Alert suppression rule
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule name
    pub name: String,
    /// Suppression condition
    pub condition: SuppressionCondition,
    /// Suppression duration
    pub duration: Duration,
    /// Enabled status
    pub enabled: bool,
}

/// Suppression condition
#[derive(Debug, Clone)]
pub enum SuppressionCondition {
    /// Time-based suppression
    TimeBased {
        start_time: SystemTime,
        end_time: SystemTime,
    },
    /// Metric-based suppression
    MetricBased { metric: String, threshold: f64 },
    /// Event-based suppression
    EventBased { event_type: String },
}

/// Prediction engine for performance forecasting
#[derive(Debug, Clone)]
pub struct PredictionEngine {
    /// Prediction models
    pub models: HashMap<String, PredictionModel>,
    /// Prediction results
    pub predictions: HashMap<String, PredictionResult>,
    /// Prediction configuration
    pub config: PredictionConfig,
    /// Forecast accuracy tracking
    pub accuracy_tracking: AccuracyTracking,
}

/// Prediction model
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: PredictionModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Training status
    pub training_status: TrainingStatus,
    /// Model accuracy
    pub accuracy: f64,
}

/// Types of prediction models
#[derive(Debug, Clone)]
pub enum PredictionModelType {
    /// Time series forecasting
    TimeSeries,
    /// Regression prediction
    Regression,
    /// Classification prediction
    Classification,
    /// Ensemble prediction
    Ensemble,
    /// Deep learning prediction
    DeepLearning,
}

/// Prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Prediction timestamp
    pub timestamp: SystemTime,
    /// Predicted value
    pub predicted_value: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Prediction accuracy
    pub accuracy: f64,
    /// Time horizon
    pub time_horizon: Duration,
    /// Prediction metadata
    pub metadata: HashMap<String, String>,
}

/// Training status of models
#[derive(Debug, Clone)]
pub enum TrainingStatus {
    /// Not trained
    NotTrained,
    /// Currently training
    Training,
    /// Trained successfully
    Trained,
    /// Training failed
    Failed { error: String },
    /// Needs retraining
    NeedsRetraining,
}

/// Prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Update frequency
    pub update_frequency: Duration,
    /// Minimum data points for prediction
    pub min_data_points: usize,
    /// Confidence level
    pub confidence_level: f64,
    /// Enable ensemble predictions
    pub enable_ensemble: bool,
}

/// Accuracy tracking for predictions
#[derive(Debug, Clone)]
pub struct AccuracyTracking {
    /// Accuracy history
    pub accuracy_history: VecDeque<AccuracyMeasurement>,
    /// Model performance comparison
    pub model_comparison: HashMap<String, f64>,
    /// Accuracy trends
    pub accuracy_trends: HashMap<String, TrendDirection>,
}

/// Accuracy measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMeasurement {
    /// Measurement timestamp
    pub timestamp: SystemTime,
    /// Model name
    pub model_name: String,
    /// Actual value
    pub actual_value: f64,
    /// Predicted value
    pub predicted_value: f64,
    /// Accuracy score
    pub accuracy_score: f64,
}
