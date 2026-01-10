//! Platform types for quantum computing platforms.
//!
//! This module contains types for managing quantum computing platforms,
//! credentials, connections, and availability information.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::hardware::{HardwareSpecification, PlatformPerformanceCharacteristics};

/// Quantum platform types
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum QuantumPlatform {
    /// D-Wave Systems
    DWave,
    /// IBM Quantum
    IBM,
    /// IonQ
    IonQ,
    /// Rigetti Computing
    Rigetti,
    /// AWS Braket
    AWSBraket,
    /// Google Quantum AI
    GoogleQuantumAI,
    /// Microsoft Azure Quantum
    AzureQuantum,
    /// Xanadu
    Xanadu,
    /// Quantum Computing Inc.
    QCI,
    /// Local simulator
    LocalSimulator,
    /// Custom platform
    Custom(String),
}

/// Platform registry for managing quantum platforms
pub struct PlatformRegistry {
    /// Registered platforms
    pub platforms: HashMap<QuantumPlatform, PlatformInfo>,
    /// Platform capabilities
    pub capabilities: HashMap<QuantumPlatform, PlatformCapabilities>,
    /// Platform availability
    pub availability: HashMap<QuantumPlatform, AvailabilityInfo>,
    /// Platform performance history
    pub performance_history: HashMap<QuantumPlatform, PerformanceHistory>,
}

impl PlatformRegistry {
    /// Create new platform registry
    pub fn new() -> Self {
        Self {
            platforms: HashMap::new(),
            capabilities: HashMap::new(),
            availability: HashMap::new(),
            performance_history: HashMap::new(),
        }
    }
}

impl Default for PlatformRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Platform information
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    /// Platform name
    pub name: String,
    /// Platform provider
    pub provider: String,
    /// Platform version
    pub version: String,
    /// Access credentials
    pub credentials: PlatformCredentials,
    /// Connection parameters
    pub connection_params: ConnectionParameters,
    /// Platform metadata
    pub metadata: HashMap<String, String>,
}

/// Platform credentials
#[derive(Debug, Clone)]
pub enum PlatformCredentials {
    /// API key
    ApiKey(String),
    /// Token-based
    Token(String),
    /// Certificate-based
    Certificate { cert: String, key: String },
    /// OAuth
    OAuth {
        client_id: String,
        client_secret: String,
    },
    /// Custom credentials
    Custom(HashMap<String, String>),
}

/// Connection parameters
#[derive(Debug, Clone)]
pub struct ConnectionParameters {
    /// Endpoint URL
    pub endpoint: String,
    /// Timeout settings
    pub timeout: Duration,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Connection pooling
    pub connection_pooling: bool,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retries
    pub max_retries: usize,
    /// Base delay
    pub base_delay: Duration,
    /// Maximum delay
    pub max_delay: Duration,
    /// Backoff strategy
    pub backoff_strategy: BackoffStrategy,
}

/// Backoff strategies
#[derive(Debug, Clone, PartialEq)]
pub enum BackoffStrategy {
    /// Fixed delay
    Fixed,
    /// Linear backoff
    Linear,
    /// Exponential backoff
    Exponential,
    /// Jittered exponential
    JitteredExponential,
}

/// Platform capabilities
#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    /// Supported problem types
    pub supported_problem_types: Vec<ProblemType>,
    /// Hardware specifications
    pub hardware_specs: Vec<HardwareSpecification>,
    /// Software capabilities
    pub software_capabilities: SoftwareCapabilities,
    /// Performance characteristics
    pub performance_characteristics: PlatformPerformanceCharacteristics,
    /// Cost structure
    pub cost_structure: CostStructure,
}

/// Problem types supported by platforms
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemType {
    /// Ising model
    Ising,
    /// QUBO
    QUBO,
    /// Gate-based quantum circuits
    GateBased,
    /// Continuous variable
    ContinuousVariable,
    /// Hybrid classical-quantum
    Hybrid,
}

/// Software capabilities
#[derive(Debug, Clone)]
pub struct SoftwareCapabilities {
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Available optimizers
    pub optimizers: Vec<OptimizerType>,
    /// Error mitigation techniques
    pub error_mitigation: Vec<ErrorMitigationType>,
    /// Compilation features
    pub compilation_features: CompilationFeatures,
}

/// Optimizer types
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerType {
    /// Quantum annealing
    QuantumAnnealing,
    /// Variational algorithms
    Variational,
    /// Adiabatic evolution
    Adiabatic,
    /// Hybrid algorithms
    Hybrid,
}

/// Error mitigation types
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorMitigationType {
    /// Zero-noise extrapolation
    ZeroNoiseExtrapolation,
    /// Probabilistic error cancellation
    ProbabilisticErrorCancellation,
    /// Symmetry verification
    SymmetryVerification,
    /// Readout error mitigation
    ReadoutErrorMitigation,
}

/// Compilation features
#[derive(Debug, Clone)]
pub struct CompilationFeatures {
    /// Circuit optimization
    pub circuit_optimization: bool,
    /// Layout optimization
    pub layout_optimization: bool,
    /// Scheduling optimization
    pub scheduling_optimization: bool,
    /// Error-aware compilation
    pub error_aware_compilation: bool,
}

/// Cost structure
#[derive(Debug, Clone)]
pub struct CostStructure {
    /// Pricing model
    pub pricing_model: PricingModel,
    /// Base cost
    pub base_cost: f64,
    /// Variable costs
    pub variable_costs: VariableCosts,
    /// Billing granularity
    pub billing_granularity: BillingGranularity,
}

/// Pricing models
#[derive(Debug, Clone, PartialEq)]
pub enum PricingModel {
    /// Pay per shot
    PerShot,
    /// Pay per circuit
    PerCircuit,
    /// Pay per time
    PerTime,
    /// Subscription
    Subscription,
    /// Credits
    Credits,
}

/// Variable costs
#[derive(Debug, Clone)]
pub struct VariableCosts {
    /// Cost per qubit
    pub per_qubit: f64,
    /// Cost per gate
    pub per_gate: f64,
    /// Cost per second
    pub per_second: f64,
    /// Cost per shot
    pub per_shot: f64,
}

/// Billing granularity
#[derive(Debug, Clone, PartialEq)]
pub enum BillingGranularity {
    /// Per second
    PerSecond,
    /// Per minute
    PerMinute,
    /// Per hour
    PerHour,
    /// Per job
    PerJob,
}

/// Availability information
#[derive(Debug, Clone)]
pub struct AvailabilityInfo {
    /// Current status
    pub status: PlatformStatus,
    /// Uptime percentage
    pub uptime: f64,
    /// Maintenance windows
    pub maintenance_windows: Vec<MaintenanceWindow>,
    /// Queue information
    pub queue_info: QueueInfo,
}

/// Platform status
#[derive(Debug, Clone, PartialEq)]
pub enum PlatformStatus {
    /// Available
    Available,
    /// Busy
    Busy,
    /// Maintenance
    Maintenance,
    /// Unavailable
    Unavailable,
    /// Unknown
    Unknown,
}

/// Maintenance window
#[derive(Debug, Clone)]
pub struct MaintenanceWindow {
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Instant,
    /// Maintenance type
    pub maintenance_type: MaintenanceType,
    /// Description
    pub description: String,
}

/// Maintenance types
#[derive(Debug, Clone, PartialEq)]
pub enum MaintenanceType {
    /// Scheduled maintenance
    Scheduled,
    /// Emergency maintenance
    Emergency,
    /// Calibration
    Calibration,
    /// Upgrade
    Upgrade,
}

/// Queue information
#[derive(Debug, Clone)]
pub struct QueueInfo {
    /// Current queue length
    pub queue_length: usize,
    /// Estimated wait time
    pub estimated_wait_time: Duration,
    /// Queue position
    pub queue_position: Option<usize>,
    /// Priority levels
    pub priority_levels: Vec<PriorityLevel>,
}

/// Priority levels
#[derive(Debug, Clone)]
pub struct PriorityLevel {
    /// Priority name
    pub name: String,
    /// Queue length at this priority
    pub queue_length: usize,
    /// Estimated wait time
    pub estimated_wait_time: Duration,
}

/// Performance history
#[derive(Debug, Clone)]
pub struct PerformanceHistory {
    /// Historical data points
    pub data_points: std::collections::VecDeque<PerformanceDataPoint>,
    /// Performance trends
    pub trends: PerformanceTrends,
    /// Anomaly detection results
    pub anomalies: Vec<PerformanceAnomaly>,
}

/// Performance data point
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Timestamp
    pub timestamp: Instant,
    /// Success rate
    pub success_rate: f64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Average queue time
    pub avg_queue_time: Duration,
    /// Fidelity
    pub fidelity: f64,
}

/// Performance trends
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Success rate trend
    pub success_rate_trend: TrendDirection,
    /// Execution time trend
    pub execution_time_trend: TrendDirection,
    /// Queue time trend
    pub queue_time_trend: TrendDirection,
    /// Fidelity trend
    pub fidelity_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Improving
    Improving,
    /// Stable
    Stable,
    /// Declining
    Declining,
    /// Unknown
    Unknown,
}

/// Performance anomaly
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    /// Detection timestamp
    pub detected_at: Instant,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity
    pub severity: AnomalySeverity,
    /// Description
    pub description: String,
}

/// Anomaly types
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    /// Performance degradation
    PerformanceDegradation,
    /// Availability issue
    AvailabilityIssue,
    /// Error rate spike
    ErrorRateSpike,
    /// Unusual queue behavior
    UnusualQueueBehavior,
}

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalySeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}
