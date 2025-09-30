//! Real-Time Hardware Performance Monitoring Integration
//!
//! This module provides comprehensive real-time monitoring capabilities for quantum
//! hardware systems, including performance metrics collection, analysis, alerting,
//! and optimization recommendations across all major quantum computing platforms.

use crate::{
    error::QuantRS2Result,
    hardware_compilation::{HardwarePlatform, NativeGateType},
    qubit::QubitId,
};
use scirs2_core::Complex64;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
    sync::{Arc, RwLock},
    thread,
    time::{Duration, SystemTime},
};

/// Real-time monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Data retention period
    pub data_retention_period: Duration,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Enabled metrics
    pub enabled_metrics: HashSet<MetricType>,
    /// Platform-specific configurations
    pub platform_configs: HashMap<HardwarePlatform, PlatformMonitoringConfig>,
    /// Export settings
    pub export_settings: ExportSettings,
}

/// Alert threshold configurations
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Gate error rate threshold
    pub max_gate_error_rate: f64,
    /// Readout error rate threshold
    pub max_readout_error_rate: f64,
    /// Coherence time threshold (minimum)
    pub min_coherence_time: Duration,
    /// Calibration drift threshold
    pub max_calibration_drift: f64,
    /// Temperature threshold
    pub max_temperature: f64,
    /// Queue depth threshold
    pub max_queue_depth: usize,
    /// Execution time threshold
    pub max_execution_time: Duration,
}

/// Platform-specific monitoring configuration
#[derive(Debug, Clone)]
pub struct PlatformMonitoringConfig {
    /// Platform type
    pub platform: HardwarePlatform,
    /// Specific metrics to monitor
    pub monitored_metrics: HashSet<MetricType>,
    /// Sampling rates for different metrics
    pub sampling_rates: HashMap<MetricType, Duration>,
    /// Platform-specific thresholds
    pub custom_thresholds: HashMap<String, f64>,
    /// Connection settings
    pub connection_settings: HashMap<String, String>,
}

/// Export settings for monitoring data
#[derive(Debug, Clone)]
pub struct ExportSettings {
    /// Enable data export
    pub enable_export: bool,
    /// Export formats
    pub export_formats: Vec<ExportFormat>,
    /// Export destinations
    pub export_destinations: Vec<ExportDestination>,
    /// Export frequency
    pub export_frequency: Duration,
    /// Compression settings
    pub compression_enabled: bool,
}

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    JSON,
    CSV,
    Parquet,
    InfluxDB,
    Prometheus,
    Custom,
}

/// Export destination options
#[derive(Debug, Clone)]
pub enum ExportDestination {
    File(String),
    Database(DatabaseConfig),
    Cloud(CloudConfig),
    Stream(StreamConfig),
}

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub connection_string: String,
    pub table_name: String,
    pub credentials: HashMap<String, String>,
}

/// Cloud configuration
#[derive(Debug, Clone)]
pub struct CloudConfig {
    pub provider: String,
    pub endpoint: String,
    pub credentials: HashMap<String, String>,
}

/// Stream configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub stream_type: String,
    pub endpoint: String,
    pub topic: String,
}

/// Types of metrics to monitor
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    // Gate performance metrics
    GateErrorRate,
    GateFidelity,
    GateExecutionTime,
    GateCalibrationDrift,

    // Qubit metrics
    QubitCoherenceTime,
    QubitReadoutError,
    QubitTemperature,
    QubitCrosstalk,

    // System metrics
    SystemUptime,
    QueueDepth,
    Throughput,
    Latency,

    // Environmental metrics
    EnvironmentalTemperature,
    MagneticField,
    Vibration,
    ElectromagneticNoise,

    // Resource metrics
    CPUUsage,
    MemoryUsage,
    NetworkLatency,
    StorageUsage,

    // Custom metrics
    Custom(String),
}

/// Real-time monitoring engine
#[derive(Debug)]
pub struct RealtimeMonitor {
    /// Configuration
    config: MonitoringConfig,
    /// Metric collectors by platform
    collectors: Arc<RwLock<HashMap<HardwarePlatform, Box<dyn MetricCollector>>>>,
    /// Real-time data storage
    data_store: Arc<RwLock<RealtimeDataStore>>,
    /// Analytics engine
    analytics_engine: Arc<RwLock<AnalyticsEngine>>,
    /// Alert manager
    alert_manager: Arc<RwLock<AlertManager>>,
    /// Optimization advisor
    optimization_advisor: Arc<RwLock<OptimizationAdvisor>>,
    /// Performance dashboard
    dashboard: Arc<RwLock<PerformanceDashboard>>,
    /// Export manager
    export_manager: Arc<RwLock<ExportManager>>,
    /// Monitoring status
    monitoring_status: Arc<RwLock<MonitoringStatus>>,
}

/// Trait for platform-specific metric collection
pub trait MetricCollector: std::fmt::Debug + Send + Sync {
    /// Collect metrics from the platform
    fn collect_metrics(&self) -> QuantRS2Result<Vec<MetricMeasurement>>;

    /// Get supported metric types
    fn supported_metrics(&self) -> HashSet<MetricType>;

    /// Platform identifier
    fn platform(&self) -> HardwarePlatform;

    /// Initialize connection to hardware
    fn initialize(&mut self) -> QuantRS2Result<()>;

    /// Check connection status
    fn is_connected(&self) -> bool;

    /// Disconnect from hardware
    fn disconnect(&mut self) -> QuantRS2Result<()>;
}

/// Individual metric measurement
#[derive(Debug, Clone)]
pub struct MetricMeasurement {
    /// Metric type
    pub metric_type: MetricType,
    /// Measurement value
    pub value: MetricValue,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Associated qubit (if applicable)
    pub qubit: Option<QubitId>,
    /// Associated gate type (if applicable)
    pub gate_type: Option<NativeGateType>,
    /// Measurement metadata
    pub metadata: HashMap<String, String>,
    /// Measurement uncertainty
    pub uncertainty: Option<f64>,
}

/// Metric value types
#[derive(Debug, Clone)]
pub enum MetricValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
    Array(Vec<f64>),
    Complex(Complex64),
    Duration(Duration),
}

/// Real-time data storage
#[derive(Debug)]
pub struct RealtimeDataStore {
    /// Time-series data by metric type
    time_series: HashMap<MetricType, VecDeque<MetricMeasurement>>,
    /// Aggregated statistics
    aggregated_stats: HashMap<MetricType, AggregatedStats>,
    /// Data retention settings
    retention_settings: HashMap<MetricType, Duration>,
    /// Current data size
    current_data_size: usize,
    /// Maximum data size
    max_data_size: usize,
}

/// Aggregated statistics for metrics
#[derive(Debug, Clone)]
pub struct AggregatedStats {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median value
    pub median: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// Sample count
    pub sample_count: usize,
    /// Last update time
    pub last_updated: SystemTime,
}

/// Analytics engine for performance analysis
#[derive(Debug)]
pub struct AnalyticsEngine {
    /// Trend analyzers
    trend_analyzers: HashMap<MetricType, Box<dyn TrendAnalyzer>>,
    /// Anomaly detectors
    anomaly_detectors: HashMap<MetricType, Box<dyn AnomalyDetector>>,
    /// Correlation analyzers
    correlation_analyzers: Vec<Box<dyn CorrelationAnalyzer>>,
    /// Predictive models
    predictive_models: HashMap<MetricType, Box<dyn PredictiveModel>>,
    /// Analysis results cache
    analysis_cache: HashMap<String, AnalysisResult>,
}

/// Trend analysis trait
pub trait TrendAnalyzer: std::fmt::Debug + Send + Sync {
    /// Analyze trend in metric data
    fn analyze_trend(&self, data: &[MetricMeasurement]) -> QuantRS2Result<TrendAnalysis>;

    /// Get analyzer name
    fn name(&self) -> &str;
}

/// Anomaly detection trait
pub trait AnomalyDetector: std::fmt::Debug + Send + Sync {
    /// Detect anomalies in metric data
    fn detect_anomalies(&self, data: &[MetricMeasurement]) -> QuantRS2Result<Vec<Anomaly>>;

    /// Get detector name
    fn name(&self) -> &str;

    /// Get confidence threshold
    fn confidence_threshold(&self) -> f64;
}

/// Correlation analysis trait
pub trait CorrelationAnalyzer: std::fmt::Debug + Send + Sync {
    /// Analyze correlations between metrics
    fn analyze_correlations(
        &self,
        data: &HashMap<MetricType, Vec<MetricMeasurement>>,
    ) -> QuantRS2Result<Vec<Correlation>>;

    /// Get analyzer name
    fn name(&self) -> &str;
}

/// Predictive modeling trait
pub trait PredictiveModel: std::fmt::Debug + Send + Sync {
    /// Predict future values
    fn predict(
        &self,
        historical_data: &[MetricMeasurement],
        horizon: Duration,
    ) -> QuantRS2Result<Prediction>;

    /// Update model with new data
    fn update(&mut self, new_data: &[MetricMeasurement]) -> QuantRS2Result<()>;

    /// Get model name
    fn name(&self) -> &str;

    /// Get model accuracy
    fn accuracy(&self) -> f64;
}

/// Trend analysis result
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Trend duration
    pub duration: Duration,
    /// Statistical significance
    pub significance: f64,
    /// Trend extrapolation
    pub extrapolation: Option<f64>,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Unknown,
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Confidence score
    pub confidence: f64,
    /// Affected metric
    pub metric: MetricType,
    /// Anomaly timestamp
    pub timestamp: SystemTime,
    /// Anomaly description
    pub description: String,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Types of anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    Outlier,
    PatternBreak,
    Drift,
    Spike,
    PerformanceDegradation,
    SystemFailure,
}

/// Correlation analysis result
#[derive(Debug, Clone)]
pub struct Correlation {
    /// First metric
    pub metric1: MetricType,
    /// Second metric
    pub metric2: MetricType,
    /// Correlation coefficient
    pub coefficient: f64,
    /// Statistical significance
    pub significance: f64,
    /// Correlation type
    pub correlation_type: CorrelationType,
    /// Time lag (if any)
    pub time_lag: Option<Duration>,
}

/// Types of correlations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationType {
    Positive,
    Negative,
    NonLinear,
    Causal,
    Spurious,
}

/// Prediction result
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted values
    pub predicted_values: Vec<(SystemTime, f64)>,
    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Prediction accuracy estimate
    pub accuracy_estimate: f64,
    /// Model used
    pub model_name: String,
    /// Prediction horizon
    pub horizon: Duration,
}

/// Analysis result
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Analysis type
    pub analysis_type: String,
    /// Result data
    pub result_data: HashMap<String, String>,
    /// Confidence score
    pub confidence: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Alert management system
#[derive(Debug)]
pub struct AlertManager {
    /// Active alerts
    active_alerts: HashMap<String, Alert>,
    /// Alert rules
    alert_rules: Vec<AlertRule>,
    /// Alert handlers
    alert_handlers: Vec<Box<dyn AlertHandler>>,
    /// Alert history
    alert_history: VecDeque<Alert>,
    /// Alert suppression rules
    suppression_rules: Vec<SuppressionRule>,
}

/// Alert definition
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Affected metrics
    pub affected_metrics: Vec<MetricType>,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert source
    pub source: String,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
    /// Alert status
    pub status: AlertStatus,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
    Suppressed,
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Condition for triggering alert
    pub condition: AlertCondition,
    /// Alert level to generate
    pub alert_level: AlertLevel,
    /// Alert message template
    pub message_template: String,
    /// Cooldown period
    pub cooldown_period: Duration,
}

/// Alert condition
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Threshold condition
    Threshold {
        metric: MetricType,
        operator: ComparisonOperator,
        threshold: f64,
        duration: Duration,
    },
    /// Rate of change condition
    RateOfChange {
        metric: MetricType,
        rate_threshold: f64,
        time_window: Duration,
    },
    /// Anomaly detection condition
    AnomalyDetected {
        metric: MetricType,
        confidence_threshold: f64,
    },
    /// Complex condition with multiple metrics
    Complex {
        expression: String,
        required_metrics: Vec<MetricType>,
    },
}

/// Comparison operators for alert conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Alert handler trait
pub trait AlertHandler: std::fmt::Debug + Send + Sync {
    /// Handle an alert
    fn handle_alert(&self, alert: &Alert) -> QuantRS2Result<()>;

    /// Get handler name
    fn name(&self) -> &str;

    /// Check if handler can handle this alert level
    fn can_handle(&self, level: AlertLevel) -> bool;
}

/// Alert suppression rule
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule ID
    pub id: String,
    /// Condition for suppression
    pub condition: SuppressionCondition,
    /// Suppression duration
    pub duration: Duration,
    /// Rule description
    pub description: String,
}

/// Suppression condition
#[derive(Debug, Clone)]
pub enum SuppressionCondition {
    /// Suppress alerts of specific type
    AlertType(String),
    /// Suppress alerts during maintenance
    MaintenanceWindow(SystemTime, SystemTime),
    /// Suppress based on metric pattern
    MetricPattern(MetricType, String),
}

/// Optimization advisor system
#[derive(Debug)]
pub struct OptimizationAdvisor {
    /// Optimization strategies
    optimization_strategies: HashMap<String, Box<dyn OptimizationStrategy>>,
    /// Recommendation engine
    recommendation_engine: RecommendationEngine,
    /// Active recommendations
    active_recommendations: Vec<OptimizationRecommendation>,
    /// Historical recommendations
    recommendation_history: VecDeque<OptimizationRecommendation>,
}

/// Optimization strategy trait
pub trait OptimizationStrategy: std::fmt::Debug + Send + Sync {
    /// Analyze performance data and generate recommendations
    fn analyze(&self, data: &RealtimeDataStore) -> QuantRS2Result<Vec<OptimizationRecommendation>>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get strategy priority
    fn priority(&self) -> u32;
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub id: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Description
    pub description: String,
    /// Affected components
    pub affected_components: Vec<String>,
    /// Expected improvement
    pub expected_improvement: ExpectedImprovement,
    /// Implementation difficulty
    pub implementation_difficulty: DifficultyLevel,
    /// Recommendation priority
    pub priority: RecommendationPriority,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationType {
    GateOptimization,
    CalibrationAdjustment,
    CircuitOptimization,
    ResourceReallocation,
    EnvironmentalAdjustment,
    MaintenanceRequired,
    UpgradeRecommendation,
}

/// Expected improvement from recommendation
#[derive(Debug, Clone)]
pub struct ExpectedImprovement {
    /// Fidelity improvement
    pub fidelity_improvement: Option<f64>,
    /// Speed improvement
    pub speed_improvement: Option<f64>,
    /// Error rate reduction
    pub error_rate_reduction: Option<f64>,
    /// Resource savings
    pub resource_savings: Option<f64>,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    ExpertRequired,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Recommendation engine
#[derive(Debug)]
pub struct RecommendationEngine {
    /// Machine learning models for recommendations
    ml_models: HashMap<String, Box<dyn MLModel>>,
    /// Rule-based recommendation rules
    rule_based_rules: Vec<RecommendationRule>,
    /// Knowledge base
    knowledge_base: KnowledgeBase,
}

/// Machine learning model trait
pub trait MLModel: std::fmt::Debug + Send + Sync {
    /// Train model with historical data
    fn train(&mut self, training_data: &[TrainingExample]) -> QuantRS2Result<()>;

    /// Predict recommendations
    fn predict(
        &self,
        input_data: &[MetricMeasurement],
    ) -> QuantRS2Result<Vec<OptimizationRecommendation>>;

    /// Get model accuracy
    fn accuracy(&self) -> f64;

    /// Get model name
    fn name(&self) -> &str;
}

/// Training example for ML models
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input metrics
    pub input_metrics: Vec<MetricMeasurement>,
    /// Expected recommendation
    pub expected_recommendation: OptimizationRecommendation,
    /// Actual outcome
    pub actual_outcome: Option<OutcomeMetrics>,
}

/// Outcome metrics after applying recommendation
#[derive(Debug, Clone)]
pub struct OutcomeMetrics {
    /// Performance improvement achieved
    pub performance_improvement: f64,
    /// Implementation success
    pub implementation_success: bool,
    /// Time to implement
    pub implementation_time: Duration,
    /// Side effects observed
    pub side_effects: Vec<String>,
}

/// Rule-based recommendation rule
#[derive(Debug, Clone)]
pub struct RecommendationRule {
    /// Rule ID
    pub id: String,
    /// Rule condition
    pub condition: String,
    /// Recommendation template
    pub recommendation_template: OptimizationRecommendation,
    /// Rule weight
    pub weight: f64,
}

/// Knowledge base for optimization
#[derive(Debug)]
pub struct KnowledgeBase {
    /// Best practices database
    best_practices: HashMap<String, BestPractice>,
    /// Common issues and solutions
    issue_solutions: HashMap<String, Solution>,
    /// Platform-specific knowledge
    platform_knowledge: HashMap<HardwarePlatform, PlatformKnowledge>,
}

/// Best practice entry
#[derive(Debug, Clone)]
pub struct BestPractice {
    /// Practice ID
    pub id: String,
    /// Description
    pub description: String,
    /// Applicable platforms
    pub applicable_platforms: Vec<HardwarePlatform>,
    /// Expected benefits
    pub expected_benefits: Vec<String>,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
}

/// Solution entry
#[derive(Debug, Clone)]
pub struct Solution {
    /// Solution ID
    pub id: String,
    /// Problem description
    pub problem_description: String,
    /// Solution description
    pub solution_description: String,
    /// Success rate
    pub success_rate: f64,
    /// Implementation complexity
    pub complexity: DifficultyLevel,
}

/// Platform-specific knowledge
#[derive(Debug, Clone)]
pub struct PlatformKnowledge {
    /// Platform type
    pub platform: HardwarePlatform,
    /// Known limitations
    pub known_limitations: Vec<String>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
    /// Common failure modes
    pub common_failure_modes: Vec<String>,
    /// Vendor-specific tips
    pub vendor_tips: Vec<String>,
}

/// Performance dashboard
#[derive(Debug)]
pub struct PerformanceDashboard {
    /// Dashboard widgets
    widgets: HashMap<String, Box<dyn DashboardWidget>>,
    /// Dashboard layout
    layout: DashboardLayout,
    /// Update frequency
    update_frequency: Duration,
    /// Dashboard state
    dashboard_state: DashboardState,
}

/// Dashboard widget trait
pub trait DashboardWidget: std::fmt::Debug + Send + Sync {
    /// Render widget with current data
    fn render(&self, data: &RealtimeDataStore) -> QuantRS2Result<WidgetData>;

    /// Get widget configuration
    fn get_config(&self) -> WidgetConfig;

    /// Update widget configuration
    fn update_config(&mut self, config: WidgetConfig) -> QuantRS2Result<()>;
}

/// Widget data for rendering
#[derive(Debug, Clone)]
pub struct WidgetData {
    /// Widget type
    pub widget_type: String,
    /// Data payload
    pub data: HashMap<String, String>,
    /// Visualization hints
    pub visualization_hints: Vec<String>,
    /// Update timestamp
    pub timestamp: SystemTime,
}

/// Widget configuration
#[derive(Debug, Clone)]
pub struct WidgetConfig {
    /// Widget title
    pub title: String,
    /// Widget size
    pub size: (u32, u32),
    /// Widget position
    pub position: (u32, u32),
    /// Refresh rate
    pub refresh_rate: Duration,
    /// Data source
    pub data_source: String,
    /// Display options
    pub display_options: HashMap<String, String>,
}

/// Dashboard layout
#[derive(Debug, Clone)]
pub struct DashboardLayout {
    /// Layout type
    pub layout_type: LayoutType,
    /// Grid dimensions
    pub grid_dimensions: (u32, u32),
    /// Widget positions
    pub widget_positions: HashMap<String, (u32, u32)>,
}

/// Layout types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutType {
    Grid,
    Flexible,
    Stacked,
    Tabbed,
}

/// Dashboard state
#[derive(Debug, Clone)]
pub struct DashboardState {
    /// Currently active widgets
    pub active_widgets: HashSet<String>,
    /// Last update time
    pub last_update: SystemTime,
    /// Dashboard mode
    pub mode: DashboardMode,
}

/// Dashboard modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashboardMode {
    Monitoring,
    Analysis,
    Debugging,
    Maintenance,
}

/// Export manager
#[derive(Debug)]
pub struct ExportManager {
    /// Configured exporters
    exporters: HashMap<ExportFormat, Box<dyn DataExporter>>,
    /// Export queue
    export_queue: VecDeque<ExportTask>,
    /// Export statistics
    export_stats: ExportStatistics,
}

/// Data exporter trait
pub trait DataExporter: std::fmt::Debug + Send + Sync {
    /// Export data in specific format
    fn export(
        &self,
        data: &[MetricMeasurement],
        destination: &ExportDestination,
    ) -> QuantRS2Result<()>;

    /// Get supported format
    fn format(&self) -> ExportFormat;

    /// Validate export configuration
    fn validate_config(&self, destination: &ExportDestination) -> QuantRS2Result<()>;
}

/// Export task
#[derive(Debug, Clone)]
pub struct ExportTask {
    /// Task ID
    pub id: String,
    /// Data to export
    pub data_range: (SystemTime, SystemTime),
    /// Export format
    pub format: ExportFormat,
    /// Export destination
    pub destination: ExportDestination,
    /// Task priority
    pub priority: TaskPriority,
    /// Created timestamp
    pub created: SystemTime,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Urgent,
}

/// Export statistics
#[derive(Debug, Clone)]
pub struct ExportStatistics {
    /// Total exports completed
    pub total_exports: u64,
    /// Failed exports
    pub failed_exports: u64,
    /// Average export time
    pub average_export_time: Duration,
    /// Data volume exported
    pub total_data_volume: u64,
    /// Last export time
    pub last_export_time: SystemTime,
}

/// Monitoring status
#[derive(Debug, Clone)]
pub struct MonitoringStatus {
    /// Overall status
    pub overall_status: SystemStatus,
    /// Platform statuses
    pub platform_statuses: HashMap<HardwarePlatform, PlatformStatus>,
    /// Active collectors
    pub active_collectors: usize,
    /// Data points collected
    pub total_data_points: u64,
    /// Active alerts
    pub active_alerts: usize,
    /// System uptime
    pub uptime: Duration,
}

/// System status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemStatus {
    Healthy,
    Degraded,
    Critical,
    Offline,
}

/// Platform status
#[derive(Debug, Clone)]
pub struct PlatformStatus {
    /// Connection status
    pub connection_status: ConnectionStatus,
    /// Last data collection time
    pub last_data_collection: SystemTime,
    /// Data collection rate
    pub collection_rate: f64,
    /// Error rate
    pub error_rate: f64,
    /// Platform-specific metrics
    pub platform_metrics: HashMap<String, f64>,
}

/// Connection status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
    Reconnecting,
    Error,
}

impl RealtimeMonitor {
    /// Create a new real-time monitor
    pub fn new(config: MonitoringConfig) -> QuantRS2Result<Self> {
        Ok(Self {
            config: config.clone(),
            collectors: Arc::new(RwLock::new(HashMap::new())),
            data_store: Arc::new(RwLock::new(RealtimeDataStore::new(
                config.data_retention_period,
            ))),
            analytics_engine: Arc::new(RwLock::new(AnalyticsEngine::new())),
            alert_manager: Arc::new(RwLock::new(AlertManager::new(config.alert_thresholds))),
            optimization_advisor: Arc::new(RwLock::new(OptimizationAdvisor::new())),
            dashboard: Arc::new(RwLock::new(PerformanceDashboard::new())),
            export_manager: Arc::new(RwLock::new(ExportManager::new(config.export_settings))),
            monitoring_status: Arc::new(RwLock::new(MonitoringStatus::new())),
        })
    }

    /// Start monitoring
    pub fn start_monitoring(&self) -> QuantRS2Result<()> {
        // Initialize collectors for each configured platform
        self.initialize_collectors()?;

        // Start data collection threads
        self.start_data_collection_threads()?;

        // Start analytics engine
        self.start_analytics_engine()?;

        // Start alert processing
        self.start_alert_processing()?;

        // Start export processing
        self.start_export_processing()?;

        // Update monitoring status
        {
            let mut status = self.monitoring_status.write().unwrap();
            status.overall_status = SystemStatus::Healthy;
        }

        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&self) -> QuantRS2Result<()> {
        // Implementation would stop all background threads and cleanup
        Ok(())
    }

    /// Register a metric collector for a platform
    pub fn register_collector(
        &self,
        platform: HardwarePlatform,
        collector: Box<dyn MetricCollector>,
    ) -> QuantRS2Result<()> {
        let mut collectors = self.collectors.write().unwrap();
        collectors.insert(platform, collector);
        Ok(())
    }

    /// Get current metrics
    pub fn get_current_metrics(
        &self,
        metric_types: Option<Vec<MetricType>>,
    ) -> QuantRS2Result<Vec<MetricMeasurement>> {
        let data_store = self.data_store.read().unwrap();

        let mut results = Vec::new();

        match metric_types {
            Some(types) => {
                for metric_type in types {
                    if let Some(time_series) = data_store.time_series.get(&metric_type) {
                        if let Some(latest) = time_series.back() {
                            results.push(latest.clone());
                        }
                    }
                }
            }
            None => {
                for time_series in data_store.time_series.values() {
                    if let Some(latest) = time_series.back() {
                        results.push(latest.clone());
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get historical metrics
    pub fn get_historical_metrics(
        &self,
        metric_type: MetricType,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> QuantRS2Result<Vec<MetricMeasurement>> {
        let data_store = self.data_store.read().unwrap();

        if let Some(time_series) = data_store.time_series.get(&metric_type) {
            let filtered: Vec<MetricMeasurement> = time_series
                .iter()
                .filter(|measurement| {
                    measurement.timestamp >= start_time && measurement.timestamp <= end_time
                })
                .cloned()
                .collect();

            Ok(filtered)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get aggregated statistics
    pub fn get_aggregated_stats(
        &self,
        metric_type: MetricType,
    ) -> QuantRS2Result<Option<AggregatedStats>> {
        let data_store = self.data_store.read().unwrap();
        Ok(data_store.aggregated_stats.get(&metric_type).cloned())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> QuantRS2Result<Vec<Alert>> {
        let alert_manager = self.alert_manager.read().unwrap();
        Ok(alert_manager.active_alerts.values().cloned().collect())
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(
        &self,
    ) -> QuantRS2Result<Vec<OptimizationRecommendation>> {
        let optimization_advisor = self.optimization_advisor.read().unwrap();
        Ok(optimization_advisor.active_recommendations.clone())
    }

    /// Get monitoring status
    pub fn get_monitoring_status(&self) -> MonitoringStatus {
        self.monitoring_status.read().unwrap().clone()
    }

    /// Force data collection from all platforms
    pub fn collect_metrics_now(&self) -> QuantRS2Result<usize> {
        let collectors = self.collectors.read().unwrap();
        let mut total_metrics = 0;

        for collector in collectors.values() {
            let metrics = collector.collect_metrics()?;
            total_metrics += metrics.len();

            // Store metrics
            self.store_metrics(metrics)?;
        }

        Ok(total_metrics)
    }

    /// Trigger analytics update
    pub fn update_analytics(&self) -> QuantRS2Result<()> {
        let data_store = self.data_store.read().unwrap();
        let analytics = self.analytics_engine.write().unwrap();

        // Run trend analysis on all metrics
        for (metric_type, time_series) in &data_store.time_series {
            if let Some(analyzer) = analytics.trend_analyzers.get(metric_type) {
                let data: Vec<MetricMeasurement> = time_series.iter().cloned().collect();
                let _trend = analyzer.analyze_trend(&data)?;
                // Store trend analysis results
            }
        }

        Ok(())
    }

    // Private implementation methods
    fn initialize_collectors(&self) -> QuantRS2Result<()> {
        // Initialize collectors for each configured platform
        for (platform, platform_config) in &self.config.platform_configs {
            // Create appropriate collector based on platform type
            let collector = self.create_collector_for_platform(*platform, platform_config)?;
            self.register_collector(*platform, collector)?;
        }
        Ok(())
    }

    fn create_collector_for_platform(
        &self,
        platform: HardwarePlatform,
        config: &PlatformMonitoringConfig,
    ) -> QuantRS2Result<Box<dyn MetricCollector>> {
        match platform {
            HardwarePlatform::Superconducting => {
                Ok(Box::new(SuperconductingCollector::new(config.clone())))
            }
            HardwarePlatform::TrappedIon => Ok(Box::new(TrappedIonCollector::new(config.clone()))),
            HardwarePlatform::Photonic => Ok(Box::new(PhotonicCollector::new(config.clone()))),
            HardwarePlatform::NeutralAtom => {
                Ok(Box::new(NeutralAtomCollector::new(config.clone())))
            }
            _ => Ok(Box::new(GenericCollector::new(config.clone()))),
        }
    }

    fn start_data_collection_threads(&self) -> QuantRS2Result<()> {
        // Start background threads for data collection
        let collectors = Arc::clone(&self.collectors);
        let data_store = Arc::clone(&self.data_store);
        let monitoring_interval = self.config.monitoring_interval;

        thread::spawn(move || loop {
            thread::sleep(monitoring_interval);

            let collectors_guard = collectors.read().unwrap();
            for collector in collectors_guard.values() {
                if let Ok(metrics) = collector.collect_metrics() {
                    let mut store = data_store.write().unwrap();
                    for metric in metrics {
                        store.add_measurement(metric);
                    }
                }
            }
        });

        Ok(())
    }

    fn start_analytics_engine(&self) -> QuantRS2Result<()> {
        // Start analytics processing
        Ok(())
    }

    fn start_alert_processing(&self) -> QuantRS2Result<()> {
        // Start alert processing
        Ok(())
    }

    fn start_export_processing(&self) -> QuantRS2Result<()> {
        // Start export processing
        Ok(())
    }

    fn store_metrics(&self, metrics: Vec<MetricMeasurement>) -> QuantRS2Result<()> {
        let mut data_store = self.data_store.write().unwrap();
        for metric in metrics {
            data_store.add_measurement(metric);
        }
        Ok(())
    }
}

// Implementation of various collector types
#[derive(Debug)]
struct SuperconductingCollector {
    config: PlatformMonitoringConfig,
    connected: bool,
}

impl SuperconductingCollector {
    fn new(config: PlatformMonitoringConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }
}

impl MetricCollector for SuperconductingCollector {
    fn collect_metrics(&self) -> QuantRS2Result<Vec<MetricMeasurement>> {
        // Simulate collecting superconducting metrics
        let mut metrics = Vec::new();

        // Gate error rate
        metrics.push(MetricMeasurement {
            metric_type: MetricType::GateErrorRate,
            value: MetricValue::Float(0.001),
            timestamp: SystemTime::now(),
            qubit: Some(QubitId::new(0)),
            gate_type: Some(NativeGateType::CNOT),
            metadata: HashMap::new(),
            uncertainty: Some(0.0001),
        });

        // Qubit coherence time
        metrics.push(MetricMeasurement {
            metric_type: MetricType::QubitCoherenceTime,
            value: MetricValue::Duration(Duration::from_micros(100)),
            timestamp: SystemTime::now(),
            qubit: Some(QubitId::new(0)),
            gate_type: None,
            metadata: HashMap::new(),
            uncertainty: Some(0.01),
        });

        Ok(metrics)
    }

    fn supported_metrics(&self) -> HashSet<MetricType> {
        let mut metrics = HashSet::new();
        metrics.insert(MetricType::GateErrorRate);
        metrics.insert(MetricType::QubitCoherenceTime);
        metrics.insert(MetricType::QubitReadoutError);
        metrics.insert(MetricType::QubitTemperature);
        metrics
    }

    fn platform(&self) -> HardwarePlatform {
        HardwarePlatform::Superconducting
    }

    fn initialize(&mut self) -> QuantRS2Result<()> {
        self.connected = true;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn disconnect(&mut self) -> QuantRS2Result<()> {
        self.connected = false;
        Ok(())
    }
}

// Similar implementations for other collector types
#[derive(Debug)]
struct TrappedIonCollector {
    config: PlatformMonitoringConfig,
    connected: bool,
}

impl TrappedIonCollector {
    fn new(config: PlatformMonitoringConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }
}

impl MetricCollector for TrappedIonCollector {
    fn collect_metrics(&self) -> QuantRS2Result<Vec<MetricMeasurement>> {
        Ok(vec![])
    }

    fn supported_metrics(&self) -> HashSet<MetricType> {
        HashSet::new()
    }

    fn platform(&self) -> HardwarePlatform {
        HardwarePlatform::TrappedIon
    }

    fn initialize(&mut self) -> QuantRS2Result<()> {
        self.connected = true;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn disconnect(&mut self) -> QuantRS2Result<()> {
        self.connected = false;
        Ok(())
    }
}

#[derive(Debug)]
struct PhotonicCollector {
    config: PlatformMonitoringConfig,
    connected: bool,
}

impl PhotonicCollector {
    fn new(config: PlatformMonitoringConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }
}

impl MetricCollector for PhotonicCollector {
    fn collect_metrics(&self) -> QuantRS2Result<Vec<MetricMeasurement>> {
        Ok(vec![])
    }

    fn supported_metrics(&self) -> HashSet<MetricType> {
        HashSet::new()
    }

    fn platform(&self) -> HardwarePlatform {
        HardwarePlatform::Photonic
    }

    fn initialize(&mut self) -> QuantRS2Result<()> {
        self.connected = true;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn disconnect(&mut self) -> QuantRS2Result<()> {
        self.connected = false;
        Ok(())
    }
}

#[derive(Debug)]
struct NeutralAtomCollector {
    config: PlatformMonitoringConfig,
    connected: bool,
}

impl NeutralAtomCollector {
    fn new(config: PlatformMonitoringConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }
}

impl MetricCollector for NeutralAtomCollector {
    fn collect_metrics(&self) -> QuantRS2Result<Vec<MetricMeasurement>> {
        Ok(vec![])
    }

    fn supported_metrics(&self) -> HashSet<MetricType> {
        HashSet::new()
    }

    fn platform(&self) -> HardwarePlatform {
        HardwarePlatform::NeutralAtom
    }

    fn initialize(&mut self) -> QuantRS2Result<()> {
        self.connected = true;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn disconnect(&mut self) -> QuantRS2Result<()> {
        self.connected = false;
        Ok(())
    }
}

#[derive(Debug)]
struct GenericCollector {
    config: PlatformMonitoringConfig,
    connected: bool,
}

impl GenericCollector {
    fn new(config: PlatformMonitoringConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }
}

impl MetricCollector for GenericCollector {
    fn collect_metrics(&self) -> QuantRS2Result<Vec<MetricMeasurement>> {
        Ok(vec![])
    }

    fn supported_metrics(&self) -> HashSet<MetricType> {
        HashSet::new()
    }

    fn platform(&self) -> HardwarePlatform {
        HardwarePlatform::Universal
    }

    fn initialize(&mut self) -> QuantRS2Result<()> {
        self.connected = true;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn disconnect(&mut self) -> QuantRS2Result<()> {
        self.connected = false;
        Ok(())
    }
}

// Implementation of data store and other components
impl RealtimeDataStore {
    fn new(_retention_period: Duration) -> Self {
        Self {
            time_series: HashMap::new(),
            aggregated_stats: HashMap::new(),
            retention_settings: HashMap::new(),
            current_data_size: 0,
            max_data_size: 1_000_000, // 1M measurements max
        }
    }

    fn add_measurement(&mut self, measurement: MetricMeasurement) {
        let metric_type = measurement.metric_type.clone();

        // Add to time series
        let time_series = self
            .time_series
            .entry(metric_type.clone())
            .or_insert_with(VecDeque::new);
        time_series.push_back(measurement.clone());

        // Update aggregated stats
        self.update_aggregated_stats(metric_type, &measurement);

        // Clean up old data if necessary
        self.cleanup_old_data();
    }

    fn update_aggregated_stats(
        &mut self,
        metric_type: MetricType,
        measurement: &MetricMeasurement,
    ) {
        // Update statistical aggregates
        let stats = self
            .aggregated_stats
            .entry(metric_type)
            .or_insert_with(|| AggregatedStats {
                mean: 0.0,
                std_dev: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
                sample_count: 0,
                last_updated: SystemTime::now(),
            });

        if let MetricValue::Float(value) = measurement.value {
            stats.sample_count += 1;
            stats.min = stats.min.min(value);
            stats.max = stats.max.max(value);
            stats.last_updated = SystemTime::now();

            // Update mean (simplified)
            stats.mean =
                (stats.mean * (stats.sample_count - 1) as f64 + value) / stats.sample_count as f64;
        }
    }

    fn cleanup_old_data(&mut self) {
        // Remove old data based on retention settings
        // Implementation would be more sophisticated in practice
    }
}

impl AnalyticsEngine {
    fn new() -> Self {
        Self {
            trend_analyzers: HashMap::new(),
            anomaly_detectors: HashMap::new(),
            correlation_analyzers: Vec::new(),
            predictive_models: HashMap::new(),
            analysis_cache: HashMap::new(),
        }
    }
}

impl AlertManager {
    fn new(_thresholds: AlertThresholds) -> Self {
        Self {
            active_alerts: HashMap::new(),
            alert_rules: Vec::new(),
            alert_handlers: Vec::new(),
            alert_history: VecDeque::new(),
            suppression_rules: Vec::new(),
        }
    }
}

impl OptimizationAdvisor {
    fn new() -> Self {
        Self {
            optimization_strategies: HashMap::new(),
            recommendation_engine: RecommendationEngine::new(),
            active_recommendations: Vec::new(),
            recommendation_history: VecDeque::new(),
        }
    }
}

impl RecommendationEngine {
    fn new() -> Self {
        Self {
            ml_models: HashMap::new(),
            rule_based_rules: Vec::new(),
            knowledge_base: KnowledgeBase::new(),
        }
    }
}

impl KnowledgeBase {
    fn new() -> Self {
        Self {
            best_practices: HashMap::new(),
            issue_solutions: HashMap::new(),
            platform_knowledge: HashMap::new(),
        }
    }
}

impl PerformanceDashboard {
    fn new() -> Self {
        Self {
            widgets: HashMap::new(),
            layout: DashboardLayout {
                layout_type: LayoutType::Grid,
                grid_dimensions: (4, 3),
                widget_positions: HashMap::new(),
            },
            update_frequency: Duration::from_secs(1),
            dashboard_state: DashboardState {
                active_widgets: HashSet::new(),
                last_update: SystemTime::now(),
                mode: DashboardMode::Monitoring,
            },
        }
    }
}

impl ExportManager {
    fn new(_settings: ExportSettings) -> Self {
        Self {
            exporters: HashMap::new(),
            export_queue: VecDeque::new(),
            export_stats: ExportStatistics {
                total_exports: 0,
                failed_exports: 0,
                average_export_time: Duration::from_millis(0),
                total_data_volume: 0,
                last_export_time: SystemTime::now(),
            },
        }
    }
}

impl MonitoringStatus {
    fn new() -> Self {
        Self {
            overall_status: SystemStatus::Offline,
            platform_statuses: HashMap::new(),
            active_collectors: 0,
            total_data_points: 0,
            active_alerts: 0,
            uptime: Duration::from_secs(0),
        }
    }
}

// Default implementations
impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(1),
            data_retention_period: Duration::from_secs(24 * 3600), // 24 hours
            alert_thresholds: AlertThresholds::default(),
            enabled_metrics: HashSet::new(),
            platform_configs: HashMap::new(),
            export_settings: ExportSettings::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_gate_error_rate: 0.01,
            max_readout_error_rate: 0.05,
            min_coherence_time: Duration::from_micros(50),
            max_calibration_drift: 0.1,
            max_temperature: 300.0, // mK
            max_queue_depth: 1000,
            max_execution_time: Duration::from_secs(300),
        }
    }
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            enable_export: false,
            export_formats: vec![ExportFormat::JSON],
            export_destinations: vec![],
            export_frequency: Duration::from_secs(3600), // 1 hour
            compression_enabled: true,
        }
    }
}

// Display implementations
impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricType::GateErrorRate => write!(f, "Gate Error Rate"),
            MetricType::QubitCoherenceTime => write!(f, "Qubit Coherence Time"),
            MetricType::SystemUptime => write!(f, "System Uptime"),
            MetricType::Custom(name) => write!(f, "Custom: {}", name),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertLevel::Info => write!(f, "INFO"),
            AlertLevel::Warning => write!(f, "WARNING"),
            AlertLevel::Critical => write!(f, "CRITICAL"),
            AlertLevel::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = RealtimeMonitor::new(config);
        assert!(monitor.is_ok());
    }

    #[test]
    fn test_metric_measurement() {
        let measurement = MetricMeasurement {
            metric_type: MetricType::GateErrorRate,
            value: MetricValue::Float(0.001),
            timestamp: SystemTime::now(),
            qubit: Some(QubitId::new(0)),
            gate_type: Some(NativeGateType::CNOT),
            metadata: HashMap::new(),
            uncertainty: Some(0.0001),
        };

        assert_eq!(measurement.metric_type, MetricType::GateErrorRate);
        assert!(matches!(measurement.value, MetricValue::Float(0.001)));
    }

    #[test]
    fn test_superconducting_collector() {
        let config = PlatformMonitoringConfig {
            platform: HardwarePlatform::Superconducting,
            monitored_metrics: HashSet::new(),
            sampling_rates: HashMap::new(),
            custom_thresholds: HashMap::new(),
            connection_settings: HashMap::new(),
        };

        let mut collector = SuperconductingCollector::new(config);
        assert_eq!(collector.platform(), HardwarePlatform::Superconducting);
        assert!(!collector.is_connected());

        assert!(collector.initialize().is_ok());
        assert!(collector.is_connected());

        let metrics = collector.collect_metrics();
        assert!(metrics.is_ok());
        assert!(!metrics.unwrap().is_empty());
    }

    #[test]
    fn test_data_store() {
        let mut store = RealtimeDataStore::new(Duration::from_secs(3600));

        let measurement = MetricMeasurement {
            metric_type: MetricType::GateErrorRate,
            value: MetricValue::Float(0.001),
            timestamp: SystemTime::now(),
            qubit: None,
            gate_type: None,
            metadata: HashMap::new(),
            uncertainty: None,
        };

        store.add_measurement(measurement);

        assert!(store.time_series.contains_key(&MetricType::GateErrorRate));
        assert!(store
            .aggregated_stats
            .contains_key(&MetricType::GateErrorRate));

        let stats = store
            .aggregated_stats
            .get(&MetricType::GateErrorRate)
            .unwrap();
        assert_eq!(stats.sample_count, 1);
        assert_eq!(stats.mean, 0.001);
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert {
            id: "test_alert".to_string(),
            level: AlertLevel::Warning,
            message: "Test alert message".to_string(),
            affected_metrics: vec![MetricType::GateErrorRate],
            timestamp: SystemTime::now(),
            source: "test".to_string(),
            suggested_actions: vec!["Check calibration".to_string()],
            status: AlertStatus::Active,
        };

        assert_eq!(alert.level, AlertLevel::Warning);
        assert_eq!(alert.status, AlertStatus::Active);
        assert!(alert.affected_metrics.contains(&MetricType::GateErrorRate));
    }

    #[test]
    fn test_monitoring_config() {
        let config = MonitoringConfig::default();
        assert_eq!(config.monitoring_interval, Duration::from_secs(1));
        assert_eq!(config.data_retention_period, Duration::from_secs(24 * 3600));
        assert!(!config.export_settings.enable_export);
    }

    #[test]
    fn test_metric_value_types() {
        let float_value = MetricValue::Float(1.23);
        let int_value = MetricValue::Integer(42);
        let bool_value = MetricValue::Boolean(true);
        let duration_value = MetricValue::Duration(Duration::from_millis(100));

        assert!(matches!(float_value, MetricValue::Float(1.23)));
        assert!(matches!(int_value, MetricValue::Integer(42)));
        assert!(matches!(bool_value, MetricValue::Boolean(true)));
        assert!(matches!(duration_value, MetricValue::Duration(_)));
    }

    #[test]
    fn test_alert_thresholds() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.max_gate_error_rate, 0.01);
        assert_eq!(thresholds.max_readout_error_rate, 0.05);
        assert_eq!(thresholds.min_coherence_time, Duration::from_micros(50));
    }

    #[test]
    fn test_optimization_recommendation() {
        let recommendation = OptimizationRecommendation {
            id: "test_rec".to_string(),
            recommendation_type: RecommendationType::GateOptimization,
            description: "Optimize gate sequence".to_string(),
            affected_components: vec!["qubit_0".to_string()],
            expected_improvement: ExpectedImprovement {
                fidelity_improvement: Some(0.001),
                speed_improvement: Some(0.1),
                error_rate_reduction: Some(0.0005),
                resource_savings: None,
            },
            implementation_difficulty: DifficultyLevel::Medium,
            priority: RecommendationPriority::High,
            timestamp: SystemTime::now(),
        };

        assert_eq!(
            recommendation.recommendation_type,
            RecommendationType::GateOptimization
        );
        assert_eq!(
            recommendation.implementation_difficulty,
            DifficultyLevel::Medium
        );
        assert_eq!(recommendation.priority, RecommendationPriority::High);
    }

    #[test]
    fn test_export_settings() {
        let settings = ExportSettings {
            enable_export: true,
            export_formats: vec![ExportFormat::JSON, ExportFormat::CSV],
            export_destinations: vec![],
            export_frequency: Duration::from_secs(1800),
            compression_enabled: true,
        };

        assert!(settings.enable_export);
        assert_eq!(settings.export_formats.len(), 2);
        assert!(settings.compression_enabled);
    }

    #[test]
    fn test_monitoring_status() {
        let status = MonitoringStatus::new();
        assert_eq!(status.overall_status, SystemStatus::Offline);
        assert_eq!(status.active_collectors, 0);
        assert_eq!(status.total_data_points, 0);
    }
}
