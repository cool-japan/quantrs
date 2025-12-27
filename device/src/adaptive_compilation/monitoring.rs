//! Performance Monitoring and Anomaly Detection Configuration

use std::time::Duration;

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct PerformanceMonitoringConfig {
    /// Enable comprehensive monitoring
    pub enable_monitoring: bool,
    /// Monitoring metrics
    pub metrics: Vec<PerformanceMetric>,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Historical data retention
    pub history_retention: Duration,
    /// Anomaly detection settings
    pub anomaly_detection: AnomalyDetectionConfig,
    /// Performance prediction settings
    pub performance_prediction: PerformancePredictionConfig,
}

/// Performance metrics to monitor
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceMetric {
    ExecutionTime,
    Fidelity,
    GateErrorRate,
    DecoherenceRate,
    ThroughputRate,
    ResourceUtilization,
    EnergyConsumption,
    SuccessRate,
    LatencyVariance,
    CustomMetric(String),
}

/// Anomaly detection configuration
#[derive(Debug, Clone)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection
    pub enable_detection: bool,
    /// Detection algorithms
    pub detection_algorithms: Vec<AnomalyDetectionAlgorithm>,
    /// Detection sensitivity
    pub sensitivity: f64,
    /// Historical baseline window
    pub baseline_window: Duration,
    /// Anomaly response configuration
    pub response_config: AnomalyResponseConfig,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyDetectionAlgorithm {
    StatisticalOutlier,
    MachineLearningBased,
    ThresholdBased,
    ChangePointDetection,
    ClusteringBased,
    EnsembleBased,
}

/// Configuration for anomaly response
#[derive(Debug, Clone)]
pub struct AnomalyResponseConfig {
    /// Response strategies
    pub response_strategies: Vec<AnomalyResponse>,
    /// Response delay
    pub response_delay: Duration,
    /// Escalation thresholds
    pub escalation_thresholds: EscalationThresholds,
    /// Notification settings
    pub notification_settings: NotificationSettings,
}

/// Responses to detected anomalies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyResponse {
    LogOnly,
    RecalibrateDevice,
    SwitchOptimizationStrategy,
    IncreaseErrorMitigation,
    NotifyOperator,
    HaltExecution,
    AutomaticRecovery,
}

/// Escalation thresholds for anomaly responses
#[derive(Debug, Clone)]
pub struct EscalationThresholds {
    /// Warning threshold
    pub warning_threshold: f64,
    /// Critical threshold
    pub critical_threshold: f64,
    /// Emergency threshold
    pub emergency_threshold: f64,
    /// Escalation timeouts
    pub escalation_timeouts: EscalationTimeouts,
}

/// Timeouts for different escalation levels
#[derive(Debug, Clone)]
pub struct EscalationTimeouts {
    /// Warning level timeout
    pub warning_timeout: Duration,
    /// Critical level timeout
    pub critical_timeout: Duration,
    /// Emergency level timeout
    pub emergency_timeout: Duration,
}

/// Notification settings for monitoring
#[derive(Debug, Clone)]
pub struct NotificationSettings {
    /// Enable notifications
    pub enable_notifications: bool,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
    /// Notification frequency limits
    pub frequency_limits: NotificationFrequencyLimits,
    /// Notification content configuration
    pub content_config: NotificationContentConfig,
}

/// Notification delivery channels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotificationChannel {
    Email,
    SMS,
    Dashboard,
    API,
    Webhook,
    Log,
}

/// Frequency limits for notifications
#[derive(Debug, Clone)]
pub struct NotificationFrequencyLimits {
    /// Maximum notifications per hour
    pub max_per_hour: usize,
    /// Minimum interval between notifications
    pub min_interval: Duration,
    /// Burst allowance
    pub burst_allowance: usize,
}

/// Notification content configuration
#[derive(Debug, Clone)]
pub struct NotificationContentConfig {
    /// Include performance data
    pub include_performance_data: bool,
    /// Include suggested actions
    pub include_suggested_actions: bool,
    /// Include historical context
    pub include_historical_context: bool,
    /// Content verbosity level
    pub verbosity_level: VerbosityLevel,
}

/// Verbosity levels for notifications
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerbosityLevel {
    Minimal,
    Standard,
    Detailed,
    Comprehensive,
}

/// Performance prediction configuration
#[derive(Debug, Clone)]
pub struct PerformancePredictionConfig {
    /// Enable performance prediction
    pub enable_prediction: bool,
    /// Prediction models
    pub prediction_models: Vec<PredictionModelType>,
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Model update frequency
    pub model_update_frequency: Duration,
    /// Prediction accuracy requirements
    pub accuracy_requirements: AccuracyRequirements,
}

/// Types of prediction models
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionModelType {
    LinearRegression,
    ARIMA,
    LSTM,
    GaussianProcess,
    RandomForest,
    EnsembleModel,
}

/// Accuracy requirements for predictions
#[derive(Debug, Clone)]
pub struct AccuracyRequirements {
    /// Minimum accuracy threshold
    pub min_accuracy: f64,
    /// Confidence interval requirement
    pub confidence_interval: f64,
    /// Maximum prediction error
    pub max_prediction_error: f64,
    /// Model validation requirements
    pub validation_requirements: ValidationRequirements,
}

/// Model validation requirements
#[derive(Debug, Clone)]
pub struct ValidationRequirements {
    /// Validation method
    pub validation_method: ValidationMethod,
    /// Validation frequency
    pub validation_frequency: Duration,
    /// Cross-validation folds
    pub cross_validation_folds: usize,
    /// Test data percentage
    pub test_data_percentage: f64,
}

/// Model validation methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationMethod {
    CrossValidation,
    TimeSeriesSplit,
    WalkForward,
    Bootstrap,
    HoldOut,
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            metrics: vec![
                PerformanceMetric::ExecutionTime,
                PerformanceMetric::Fidelity,
                PerformanceMetric::GateErrorRate,
                PerformanceMetric::ThroughputRate,
            ],
            monitoring_interval: Duration::from_secs(10),
            history_retention: Duration::from_secs(86400 * 7), // 7 days
            anomaly_detection: AnomalyDetectionConfig::default(),
            performance_prediction: PerformancePredictionConfig::default(),
        }
    }
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            detection_algorithms: vec![
                AnomalyDetectionAlgorithm::StatisticalOutlier,
                AnomalyDetectionAlgorithm::ThresholdBased,
            ],
            sensitivity: 0.95,
            baseline_window: Duration::from_secs(3600), // 1 hour
            response_config: AnomalyResponseConfig::default(),
        }
    }
}

impl Default for AnomalyResponseConfig {
    fn default() -> Self {
        Self {
            response_strategies: vec![AnomalyResponse::LogOnly, AnomalyResponse::NotifyOperator],
            response_delay: Duration::from_secs(5),
            escalation_thresholds: EscalationThresholds::default(),
            notification_settings: NotificationSettings::default(),
        }
    }
}

impl Default for EscalationThresholds {
    fn default() -> Self {
        Self {
            warning_threshold: 0.1,
            critical_threshold: 0.05,
            emergency_threshold: 0.01,
            escalation_timeouts: EscalationTimeouts::default(),
        }
    }
}

impl Default for EscalationTimeouts {
    fn default() -> Self {
        Self {
            warning_timeout: Duration::from_secs(300),  // 5 minutes
            critical_timeout: Duration::from_secs(60),  // 1 minute
            emergency_timeout: Duration::from_secs(10), // 10 seconds
        }
    }
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            enable_notifications: true,
            notification_channels: vec![NotificationChannel::Dashboard, NotificationChannel::Log],
            frequency_limits: NotificationFrequencyLimits::default(),
            content_config: NotificationContentConfig::default(),
        }
    }
}

impl Default for NotificationFrequencyLimits {
    fn default() -> Self {
        Self {
            max_per_hour: 10,
            min_interval: Duration::from_secs(60),
            burst_allowance: 3,
        }
    }
}

impl Default for NotificationContentConfig {
    fn default() -> Self {
        Self {
            include_performance_data: true,
            include_suggested_actions: true,
            include_historical_context: false,
            verbosity_level: VerbosityLevel::Standard,
        }
    }
}

impl Default for PerformancePredictionConfig {
    fn default() -> Self {
        Self {
            enable_prediction: true,
            prediction_models: vec![
                PredictionModelType::LinearRegression,
                PredictionModelType::ARIMA,
            ],
            prediction_horizon: Duration::from_secs(3600), // 1 hour
            model_update_frequency: Duration::from_secs(1800), // 30 minutes
            accuracy_requirements: AccuracyRequirements::default(),
        }
    }
}

impl Default for AccuracyRequirements {
    fn default() -> Self {
        Self {
            min_accuracy: 0.8,
            confidence_interval: 0.95,
            max_prediction_error: 0.1,
            validation_requirements: ValidationRequirements::default(),
        }
    }
}

impl Default for ValidationRequirements {
    fn default() -> Self {
        Self {
            validation_method: ValidationMethod::CrossValidation,
            validation_frequency: Duration::from_secs(3600 * 6), // 6 hours
            cross_validation_folds: 5,
            test_data_percentage: 0.2,
        }
    }
}
