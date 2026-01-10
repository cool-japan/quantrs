//! Type definitions for Real-time Quantum Computing Integration
//!
//! This module provides enums and basic type definitions.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Resource allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    LoadBalanced,
    PriorityBased,
    DeadlineAware,
    Adaptive,
}

/// Device types for quantum hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    SuperconductingQuantumProcessor,
    IonTrapQuantumComputer,
    PhotonicQuantumComputer,
    QuantumAnnealer,
    QuantumSimulator,
    HybridSystem,
}

/// Connectivity types for quantum devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityType {
    AllToAll,
    NearestNeighbor,
    Grid2D,
    Grid3D,
    Tree,
    Custom,
}

/// Measurement basis types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementBasis {
    Computational,
    Pauli(PauliBasis),
    Custom(String),
}

/// Pauli basis types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PauliBasis {
    X,
    Y,
    Z,
}

/// Authentication types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationType {
    ApiKey,
    OAuth2,
    Certificate,
    Token,
}

/// Connection status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Connecting,
    Disconnected,
    Error(String),
    Maintenance,
}

/// Overall device status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverallStatus {
    Online,
    Offline,
    Maintenance,
    Calibration,
    Error,
    Degraded,
}

/// Maintenance types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaintenanceType {
    Scheduled,
    Emergency,
    Calibration,
    Upgrade,
    Repair,
}

/// Component health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentStatus {
    Healthy,
    Warning,
    Critical,
    Failed,
    Unknown,
}

/// Issue types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    Hardware,
    Software,
    Calibration,
    Temperature,
    Network,
    Performance,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Warning types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningType {
    HighTemperature,
    LowFidelity,
    HighErrorRate,
    QueueOverflow,
    MaintenanceRequired,
    CalibrationDrift,
    NetworkIssue,
}

/// Metric collection types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    All,
    Hardware,
    Quantum,
    Performance,
    Environmental,
    Custom(String),
}

/// Alert channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertChannel {
    Email,
    SMS,
    Webhook,
    Dashboard,
    PagerDuty,
    Slack,
}

/// Job priority levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobPriority {
    Critical,
    High,
    Normal,
    Low,
    Background,
}

/// Job status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Scheduled,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout,
}

/// Scheduling algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingAlgorithm {
    FirstComeFirstServe,
    ShortestJobFirst,
    PriorityBased,
    RoundRobin,
    DeadlineAware,
    Fair,
}

/// Preemption policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreemptionPolicy {
    None,
    PriorityBased,
    DeadlineBased,
    ResourceBased,
}

/// Fairness policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FairnessPolicy {
    Strict,
    WeightedFair,
    DominantResourceFairness,
    MaxMinFairness,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    IPHash,
    ResourceBased,
}

/// Aggregation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    Sum,
    Average,
    Min,
    Max,
    Count,
    Percentile(f64),
    StandardDeviation,
    Rate,
}

/// Analytics model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticsModelType {
    LinearRegression,
    TimeSeriesForecasting,
    AnomalyDetection,
    Classification,
    Clustering,
    DeepLearning,
}

/// Widget types for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    LineChart,
    BarChart,
    Gauge,
    Table,
    Heatmap,
    Scatter,
    Pie,
    Text,
}

/// Time range types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeRange {
    Last(Duration),
    Range {
        start: std::time::SystemTime,
        end: std::time::SystemTime,
    },
    RealTime,
}

/// Filter operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    StartsWith,
    EndsWith,
}

/// Color schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Default,
    Dark,
    Light,
    Custom(Vec<String>),
}

/// Axis scale types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AxisScale {
    Linear,
    Logarithmic,
    Auto,
}

/// Legend positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendPosition {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Legend orientations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendOrientation {
    Horizontal,
    Vertical,
}

/// Data source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSourceType {
    Database,
    API,
    File,
    Stream,
    WebSocket,
}

/// Backoff strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Fixed,
    Linear,
    Exponential,
    Random,
}

/// Data formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    JSON,
    CSV,
    XML,
    Binary,
    Custom(String),
}

/// Feature transformation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureTransformation {
    Identity,
    Normalization,
    Scaling,
    Polynomial,
    Fourier,
    Wavelet,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    StatisticalOutlier,
    IsolationForest,
    OneClassSVM,
    LSTM,
    Autoencoder,
    DBSCAN,
}

/// Anomaly types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    PointAnomaly,
    ContextualAnomaly,
    CollectiveAnomaly,
    SeasonalAnomaly,
    TrendChange,
}

/// Fault types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaultType {
    HardwareFailure,
    SoftwareError,
    NetworkFailure,
    PerformanceDegradation,
    SecurityBreach,
    ConfigurationError,
}

/// Recovery action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryActionType {
    Restart,
    Failover,
    Rollback,
    ScaleUp,
    Alert,
    Manual,
}

/// System status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemStatus {
    Healthy,
    Warning,
    Degraded,
    Critical,
    Maintenance,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    System,
    Hardware,
    Performance,
    Security,
    User,
}

/// Resource types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    QuantumProcessor,
    ClassicalProcessor,
    Memory,
    Storage,
    Network,
    Hybrid,
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceConstraint {
    MaxConcurrentJobs(usize),
    RequiredCertification(String),
    GeographicRestriction(String),
    TimeBased {
        start: std::time::SystemTime,
        end: std::time::SystemTime,
    },
    DependsOn(String),
    ExclusiveAccess,
}

/// Pattern types for usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Linear,
    Exponential,
    Seasonal,
    Cyclical,
    Random,
    Hybrid,
}

/// Trend types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Health check types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    Ping,
    HTTP,
    TCP,
    Custom(String),
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckStatus {
    Healthy,
    Unhealthy,
    Unknown,
    Degraded,
}

/// Fault detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaultDetectionMethod {
    ThresholdBased,
    StatisticalAnalysis,
    MachineLearning,
    PatternMatching,
    CorrelationAnalysis,
    RuleEngine,
}

/// Recovery step types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStepType {
    RestartService,
    RecalibrateDevice,
    SwitchToBackup,
    ClearCache,
    ResetConnection,
    NotifyOperator,
    RunDiagnostics,
    Custom(String),
}

/// Metric data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricDataType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Timer,
}

/// Data collection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectionMethod {
    Push,
    Pull,
    Event,
    Streaming,
}

/// Index types for data storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    TagIndex,
    TimeIndex,
    ValueIndex,
    CompositeIndex,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    LZ4,
    Snappy,
    GZIP,
    ZSTD,
}

/// Ensemble methods for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    MajorityVoting,
    WeightedVoting,
    Stacking,
    Averaging,
}
