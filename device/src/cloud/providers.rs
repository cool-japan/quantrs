//! Multi-Provider Configuration and Management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported cloud providers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CloudProvider {
    /// IBM Quantum
    IBM,
    /// AWS Braket
    AWS,
    /// Azure Quantum
    Azure,
    /// Google Quantum AI
    Google,
    /// IonQ
    IonQ,
    /// Rigetti
    Rigetti,
    /// Xanadu
    Xanadu,
    /// D-Wave
    DWave,
    /// Custom provider
    Custom(String),
}

/// Multi-provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiProviderConfig {
    /// Enabled cloud providers
    pub enabled_providers: Vec<CloudProvider>,
    /// Provider-specific configurations
    pub provider_configs: HashMap<CloudProvider, ProviderConfig>,
    /// Provider selection strategy
    pub selection_strategy: ProviderSelectionStrategy,
    /// Failover configuration
    pub failover_config: FailoverConfig,
    /// Cross-provider synchronization
    pub sync_config: CrossProviderSyncConfig,
    /// Provider health monitoring
    pub health_monitoring: ProviderHealthConfig,
}

/// Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub api_endpoint: String,
    pub credentials: HashMap<String, String>,
    pub resource_limits: HashMap<String, usize>,
    /// Provider-specific features
    pub features: ProviderFeatures,
    /// Connection settings
    pub connection: ConnectionSettings,
    /// Rate limiting
    pub rate_limits: RateLimits,
}

/// Provider-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderFeatures {
    /// Supported gate sets
    pub gate_sets: Vec<String>,
    /// Maximum qubits
    pub max_qubits: usize,
    /// Coherence times
    pub coherence_times: CoherenceMetrics,
    /// Error rates
    pub error_rates: ErrorMetrics,
    /// Special capabilities
    pub special_capabilities: Vec<SpecialCapability>,
}

/// Coherence time metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceMetrics {
    /// T1 relaxation time
    pub t1_avg: f64,
    /// T2 dephasing time
    pub t2_avg: f64,
    /// Gate time
    pub gate_time_avg: f64,
}

/// Error rate metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Single qubit gate error
    pub single_qubit_error: f64,
    /// Two qubit gate error
    pub two_qubit_error: f64,
    /// Readout error
    pub readout_error: f64,
}

/// Special capabilities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpecialCapability {
    MiddleCircuitMeasurement,
    DynamicCircuits,
    ErrorMitigation,
    PulseControl,
    ParametricGates,
    Transpilation,
    CustomGates,
}

/// Connection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSettings {
    /// Connection timeout
    pub timeout: std::time::Duration,
    /// Retry attempts
    pub retry_attempts: usize,
    /// Keep-alive settings
    pub keep_alive: bool,
    /// SSL/TLS configuration
    pub ssl_config: SSLConfig,
}

/// SSL/TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSLConfig {
    /// Enable SSL/TLS
    pub enabled: bool,
    /// Certificate verification
    pub verify_certificates: bool,
    /// Certificate path
    pub certificate_path: Option<String>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    /// Requests per second
    pub requests_per_second: usize,
    /// Requests per minute
    pub requests_per_minute: usize,
    /// Requests per hour
    pub requests_per_hour: usize,
    /// Burst allowance
    pub burst_allowance: usize,
}

/// Provider selection strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProviderSelectionStrategy {
    /// Round-robin selection
    RoundRobin,
    /// Load-based selection
    LoadBased,
    /// Cost-based selection
    CostBased,
    /// Performance-based selection
    PerformanceBased,
    /// Latency-based selection
    LatencyBased,
    /// Availability-based selection
    AvailabilityBased,
    /// Multi-criteria selection
    MultiCriteria(MultiCriteriaConfig),
    /// Custom selection algorithm
    Custom(String),
}

/// Multi-criteria selection configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiCriteriaConfig {
    /// Criteria weights
    pub weights: HashMap<SelectionCriterion, f64>,
    /// Aggregation method
    pub aggregation_method: AggregationMethod,
    /// Normalization method
    pub normalization: NormalizationMethod,
}

/// Selection criteria
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SelectionCriterion {
    Cost,
    Performance,
    Latency,
    Availability,
    QueueTime,
    ErrorRate,
    ResourceUtilization,
}

/// Aggregation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregationMethod {
    WeightedSum,
    WeightedProduct,
    TOPSIS,
    ELECTRE,
    AHP,
}

/// Normalization methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    Robust,
    None,
}

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    pub enable_failover: bool,
    pub failover_threshold: f64,
    pub failover_providers: Vec<String>,
    /// Failover strategy
    pub strategy: FailoverStrategy,
    /// Detection settings
    pub detection: FailureDetectionConfig,
    /// Recovery settings
    pub recovery: FailoverRecoveryConfig,
}

/// Failover strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FailoverStrategy {
    Immediate,
    Graceful,
    Conditional,
    Hybrid,
}

/// Failure detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureDetectionConfig {
    /// Health check interval
    pub health_check_interval: std::time::Duration,
    /// Failure threshold
    pub failure_threshold: usize,
    /// Detection methods
    pub detection_methods: Vec<FailureDetectionMethod>,
}

/// Failure detection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FailureDetectionMethod {
    HealthEndpoint,
    ResponseTime,
    ErrorRate,
    CustomMetric(String),
}

/// Failover recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverRecoveryConfig {
    /// Recovery timeout
    pub recovery_timeout: std::time::Duration,
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Backoff strategy
    pub backoff: BackoffStrategy,
}

/// Recovery strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    Automatic,
    Manual,
    Hybrid,
}

/// Backoff strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackoffStrategy {
    /// Initial delay
    pub initial_delay: std::time::Duration,
    /// Maximum delay
    pub max_delay: std::time::Duration,
    /// Backoff multiplier
    pub multiplier: f64,
    /// Jitter
    pub jitter: bool,
}

/// Cross-provider sync configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossProviderSyncConfig {
    pub sync_enabled: bool,
    pub sync_interval: u64,
    pub sync_strategies: Vec<String>,
    /// Sync scope
    pub sync_scope: SyncScope,
    /// Conflict resolution
    pub conflict_resolution: ConflictResolution,
    /// Data consistency
    pub consistency: ConsistencyConfig,
}

/// Synchronization scope
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SyncScope {
    JobStatus,
    ResourceUsage,
    Configurations,
    All,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConflictResolution {
    LastWriteWins,
    FirstWriteWins,
    MergeStrategy,
    ManualResolution,
}

/// Consistency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyConfig {
    /// Consistency level
    pub level: ConsistencyLevel,
    /// Timeout
    pub timeout: std::time::Duration,
    /// Retry policy
    pub retry_policy: RetryPolicy,
}

/// Consistency levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    Strong,
    Eventual,
    Session,
    Bounded,
}

/// Retry policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retries
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: std::time::Duration,
    /// Exponential backoff
    pub exponential_backoff: bool,
}

/// Provider health configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealthConfig {
    pub health_check_interval: u64,
    pub health_thresholds: HashMap<String, f64>,
    /// Health metrics
    pub metrics: Vec<HealthMetric>,
    /// Alerting configuration
    pub alerting: HealthAlertingConfig,
}

/// Health metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthMetric {
    ResponseTime,
    ErrorRate,
    Availability,
    QueueLength,
    ResourceUtilization,
    CustomMetric(String),
}

/// Health alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: HashMap<HealthMetric, f64>,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Escalation rules
    pub escalation: EscalationConfig,
}

/// Notification channels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email,
    SMS,
    Slack,
    Webhook,
    Dashboard,
}

/// Escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationConfig {
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
    /// Escalation timeouts
    pub timeouts: HashMap<String, std::time::Duration>,
}

/// Escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level name
    pub name: String,
    /// Threshold
    pub threshold: f64,
    /// Actions
    pub actions: Vec<EscalationAction>,
}

/// Escalation actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EscalationAction {
    Notify,
    Failover,
    Throttle,
    Shutdown,
}

impl Default for MultiProviderConfig {
    fn default() -> Self {
        Self {
            enabled_providers: vec![CloudProvider::IBM, CloudProvider::AWS],
            provider_configs: HashMap::new(),
            selection_strategy: ProviderSelectionStrategy::PerformanceBased,
            failover_config: FailoverConfig::default(),
            sync_config: CrossProviderSyncConfig::default(),
            health_monitoring: ProviderHealthConfig::default(),
        }
    }
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            api_endpoint: String::new(),
            credentials: HashMap::new(),
            resource_limits: HashMap::new(),
            features: ProviderFeatures::default(),
            connection: ConnectionSettings::default(),
            rate_limits: RateLimits::default(),
        }
    }
}

impl Default for ProviderFeatures {
    fn default() -> Self {
        Self {
            gate_sets: vec!["clifford".to_string(), "universal".to_string()],
            max_qubits: 100,
            coherence_times: CoherenceMetrics::default(),
            error_rates: ErrorMetrics::default(),
            special_capabilities: vec![],
        }
    }
}

impl Default for CoherenceMetrics {
    fn default() -> Self {
        Self {
            t1_avg: 100.0,      // microseconds
            t2_avg: 50.0,       // microseconds
            gate_time_avg: 0.1, // microseconds
        }
    }
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            single_qubit_error: 0.001,
            two_qubit_error: 0.01,
            readout_error: 0.02,
        }
    }
}

impl Default for ConnectionSettings {
    fn default() -> Self {
        Self {
            timeout: std::time::Duration::from_secs(30),
            retry_attempts: 3,
            keep_alive: true,
            ssl_config: SSLConfig::default(),
        }
    }
}

impl Default for SSLConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            verify_certificates: true,
            certificate_path: None,
        }
    }
}

impl Default for RateLimits {
    fn default() -> Self {
        Self {
            requests_per_second: 10,
            requests_per_minute: 100,
            requests_per_hour: 1000,
            burst_allowance: 20,
        }
    }
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            enable_failover: true,
            failover_threshold: 0.8,
            failover_providers: vec![],
            strategy: FailoverStrategy::Graceful,
            detection: FailureDetectionConfig::default(),
            recovery: FailoverRecoveryConfig::default(),
        }
    }
}

impl Default for FailureDetectionConfig {
    fn default() -> Self {
        Self {
            health_check_interval: std::time::Duration::from_secs(60),
            failure_threshold: 3,
            detection_methods: vec![
                FailureDetectionMethod::HealthEndpoint,
                FailureDetectionMethod::ResponseTime,
            ],
        }
    }
}

impl Default for FailoverRecoveryConfig {
    fn default() -> Self {
        Self {
            recovery_timeout: std::time::Duration::from_secs(300),
            recovery_strategy: RecoveryStrategy::Automatic,
            backoff: BackoffStrategy::default(),
        }
    }
}

impl Default for BackoffStrategy {
    fn default() -> Self {
        Self {
            initial_delay: std::time::Duration::from_secs(1),
            max_delay: std::time::Duration::from_secs(60),
            multiplier: 2.0,
            jitter: true,
        }
    }
}

impl Default for CrossProviderSyncConfig {
    fn default() -> Self {
        Self {
            sync_enabled: false,
            sync_interval: 300, // 5 minutes
            sync_strategies: vec![],
            sync_scope: SyncScope::JobStatus,
            conflict_resolution: ConflictResolution::LastWriteWins,
            consistency: ConsistencyConfig::default(),
        }
    }
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            level: ConsistencyLevel::Eventual,
            timeout: std::time::Duration::from_secs(30),
            retry_policy: RetryPolicy::default(),
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: std::time::Duration::from_secs(1),
            exponential_backoff: true,
        }
    }
}

impl Default for ProviderHealthConfig {
    fn default() -> Self {
        Self {
            health_check_interval: 60,
            health_thresholds: HashMap::new(),
            metrics: vec![
                HealthMetric::ResponseTime,
                HealthMetric::ErrorRate,
                HealthMetric::Availability,
            ],
            alerting: HealthAlertingConfig::default(),
        }
    }
}

impl Default for HealthAlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: HashMap::new(),
            channels: vec![NotificationChannel::Dashboard],
            escalation: EscalationConfig::default(),
        }
    }
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self {
            levels: vec![],
            timeouts: HashMap::new(),
        }
    }
}
