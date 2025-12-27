//! Configuration types for the comprehensive integration testing framework

use std::collections::HashMap;
use std::time::Duration;

/// Integration testing framework configuration
#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    /// Test execution timeout
    pub execution_timeout: Duration,
    /// Maximum concurrent test executions
    pub max_concurrent_tests: usize,
    /// Test result storage configuration
    pub storage_config: TestStorageConfig,
    /// Performance benchmark settings
    pub benchmark_config: BenchmarkConfig,
    /// Stress testing configuration
    pub stress_config: StressTestConfig,
    /// Fault injection settings
    pub fault_injection_config: FaultInjectionConfig,
    /// Monitoring and reporting settings
    pub monitoring_config: MonitoringConfig,
    /// Test environment configuration
    pub environment_config: TestEnvironmentConfig,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            execution_timeout: Duration::from_secs(300),
            max_concurrent_tests: 4,
            storage_config: TestStorageConfig::default(),
            benchmark_config: BenchmarkConfig::default(),
            stress_config: StressTestConfig::default(),
            fault_injection_config: FaultInjectionConfig::default(),
            monitoring_config: MonitoringConfig::default(),
            environment_config: TestEnvironmentConfig::default(),
        }
    }
}

/// Test result storage configuration
#[derive(Debug, Clone)]
pub struct TestStorageConfig {
    /// Enable persistent storage
    pub enable_persistent_storage: bool,
    /// Storage format
    pub storage_format: StorageFormat,
    /// Retention policy
    pub retention_policy: RetentionPolicy,
    /// Compression settings
    pub compression: CompressionConfig,
}

impl Default for TestStorageConfig {
    fn default() -> Self {
        Self {
            enable_persistent_storage: true,
            storage_format: StorageFormat::JSON,
            retention_policy: RetentionPolicy::KeepLast(1000),
            compression: CompressionConfig::default(),
        }
    }
}

/// Storage formats for test results
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageFormat {
    JSON,
    Binary,
    Database,
    CSV,
}

/// Retention policies for test data
#[derive(Debug, Clone)]
pub enum RetentionPolicy {
    /// Keep last N test results
    KeepLast(usize),
    /// Keep results for duration
    KeepForDuration(Duration),
    /// Keep all results
    KeepAll,
    /// Custom retention logic
    Custom(String),
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: u8,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            algorithm: CompressionAlgorithm::Gzip,
            level: 6,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    None,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Enable performance benchmarking
    pub enable_benchmarking: bool,
    /// Benchmark suite selection
    pub benchmark_suites: Vec<BenchmarkSuite>,
    /// Performance baseline configuration
    pub baseline_config: BaselineConfig,
    /// Statistical analysis settings
    pub statistical_config: StatisticalConfig,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            enable_benchmarking: true,
            benchmark_suites: vec![
                BenchmarkSuite::Performance,
                BenchmarkSuite::Scalability,
                BenchmarkSuite::Accuracy,
            ],
            baseline_config: BaselineConfig::default(),
            statistical_config: StatisticalConfig::default(),
        }
    }
}

/// Benchmark suite types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkSuite {
    /// Performance benchmarks
    Performance,
    /// Scalability tests
    Scalability,
    /// Accuracy validation
    Accuracy,
    /// Resource utilization
    ResourceUtilization,
    /// Integration complexity
    IntegrationComplexity,
    /// Custom benchmark
    Custom(String),
}

/// Baseline configuration for comparisons
#[derive(Debug, Clone)]
pub struct BaselineConfig {
    /// Use historical baselines
    pub use_historical: bool,
    /// Baseline update strategy
    pub update_strategy: BaselineUpdateStrategy,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            use_historical: true,
            update_strategy: BaselineUpdateStrategy::Automatic,
            performance_thresholds: PerformanceThresholds::default(),
        }
    }
}

/// Baseline update strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaselineUpdateStrategy {
    /// Automatic updates based on performance
    Automatic,
    /// Manual baseline updates
    Manual,
    /// Time-based updates
    TimeBased(Duration),
    /// Never update baselines
    Never,
}

/// Performance threshold definitions
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable execution time
    pub max_execution_time: Duration,
    /// Minimum solution quality
    pub min_solution_quality: f64,
    /// Maximum resource usage
    pub max_resource_usage: f64,
    /// Maximum error rate
    pub max_error_rate: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(60),
            min_solution_quality: 0.8,
            max_resource_usage: 0.9,
            max_error_rate: 0.05,
        }
    }
}

/// Statistical analysis configuration
#[derive(Debug, Clone)]
pub struct StatisticalConfig {
    /// Confidence level for analysis
    pub confidence_level: f64,
    /// Number of statistical runs
    pub num_runs: usize,
    /// Statistical tests to perform
    pub statistical_tests: Vec<StatisticalTest>,
    /// Outlier detection method
    pub outlier_detection: OutlierDetection,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            num_runs: 10,
            statistical_tests: vec![
                StatisticalTest::TTest,
                StatisticalTest::KolmogorovSmirnov,
                StatisticalTest::MannWhitney,
            ],
            outlier_detection: OutlierDetection::IQR,
        }
    }
}

/// Statistical tests for analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatisticalTest {
    /// Student's t-test
    TTest,
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Mann-Whitney U test
    MannWhitney,
    /// Wilcoxon signed-rank test
    Wilcoxon,
    /// Chi-squared test
    ChiSquared,
}

/// Outlier detection methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutlierDetection {
    /// Interquartile range method
    IQR,
    /// Z-score method
    ZScore,
    /// Modified Z-score
    ModifiedZScore,
    /// Isolation forest
    IsolationForest,
    /// No outlier detection
    None,
}

/// Stress testing configuration
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Enable stress testing
    pub enable_stress_testing: bool,
    /// Stress test scenarios
    pub stress_scenarios: Vec<StressScenario>,
    /// Maximum stress level
    pub max_stress_level: f64,
    /// Stress ramp-up strategy
    pub ramp_up_strategy: RampUpStrategy,
    /// Failure criteria
    pub failure_criteria: FailureCriteria,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            enable_stress_testing: true,
            stress_scenarios: vec![
                StressScenario::HighLoad,
                StressScenario::ResourceContention,
                StressScenario::NetworkLatency,
            ],
            max_stress_level: 0.95,
            ramp_up_strategy: RampUpStrategy::Linear,
            failure_criteria: FailureCriteria::default(),
        }
    }
}

/// Stress test scenarios
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StressScenario {
    /// High computational load
    HighLoad,
    /// Resource contention
    ResourceContention,
    /// Network latency stress
    NetworkLatency,
    /// Memory pressure
    MemoryPressure,
    /// Concurrent access stress
    ConcurrentAccess,
    /// Custom stress scenario
    Custom(String),
}

/// Stress ramp-up strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RampUpStrategy {
    /// Linear ramp-up
    Linear,
    /// Exponential ramp-up
    Exponential,
    /// Step-wise ramp-up
    StepWise,
    /// Random stress levels
    Random,
}

/// Failure criteria for stress tests
#[derive(Debug, Clone)]
pub struct FailureCriteria {
    /// Maximum acceptable failures
    pub max_failures: usize,
    /// Failure rate threshold
    pub failure_rate_threshold: f64,
    /// Response time threshold
    pub response_time_threshold: Duration,
    /// Resource exhaustion threshold
    pub resource_exhaustion_threshold: f64,
}

impl Default for FailureCriteria {
    fn default() -> Self {
        Self {
            max_failures: 5,
            failure_rate_threshold: 0.1,
            response_time_threshold: Duration::from_secs(10),
            resource_exhaustion_threshold: 0.95,
        }
    }
}

/// Fault injection configuration
#[derive(Debug, Clone)]
pub struct FaultInjectionConfig {
    /// Enable fault injection
    pub enable_fault_injection: bool,
    /// Fault types to inject
    pub fault_types: Vec<FaultType>,
    /// Injection timing strategy
    pub timing_strategy: InjectionTiming,
    /// Fault recovery testing
    pub test_recovery: bool,
    /// Chaos engineering settings
    pub chaos_config: ChaosConfig,
}

impl Default for FaultInjectionConfig {
    fn default() -> Self {
        Self {
            enable_fault_injection: true,
            fault_types: vec![
                FaultType::NetworkFailure,
                FaultType::ComponentFailure,
                FaultType::ResourceExhaustion,
            ],
            timing_strategy: InjectionTiming::Random,
            test_recovery: true,
            chaos_config: ChaosConfig::default(),
        }
    }
}

/// Types of faults to inject
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultType {
    /// Network connectivity failures
    NetworkFailure,
    /// Component/service failures
    ComponentFailure,
    /// Resource exhaustion
    ResourceExhaustion,
    /// Data corruption
    DataCorruption,
    /// Timing issues
    TimingIssues,
    /// Configuration errors
    ConfigurationErrors,
    /// Custom fault type
    Custom(String),
}

/// Fault injection timing strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InjectionTiming {
    /// Random injection times
    Random,
    /// Scheduled injection
    Scheduled(Vec<Duration>),
    /// Trigger-based injection
    TriggerBased(Vec<String>),
    /// Continuous low-level injection
    Continuous,
}

/// Chaos engineering configuration
#[derive(Debug, Clone)]
pub struct ChaosConfig {
    /// Enable chaos engineering
    pub enable_chaos: bool,
    /// Chaos experiments
    pub experiments: Vec<ChaosExperiment>,
    /// Blast radius control
    pub blast_radius: BlastRadius,
    /// Safety measures
    pub safety_measures: SafetyMeasures,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            enable_chaos: false, // Disabled by default for safety
            experiments: vec![],
            blast_radius: BlastRadius::Limited,
            safety_measures: SafetyMeasures::default(),
        }
    }
}

/// Chaos engineering experiments
#[derive(Debug, Clone)]
pub struct ChaosExperiment {
    /// Experiment name
    pub name: String,
    /// Experiment type
    pub experiment_type: ChaosType,
    /// Target components
    pub targets: Vec<String>,
    /// Experiment duration
    pub duration: Duration,
    /// Success criteria
    pub success_criteria: Vec<String>,
}

/// Types of chaos experiments
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChaosType {
    /// Service degradation
    ServiceDegradation,
    /// Resource starvation
    ResourceStarvation,
    /// Network partitioning
    NetworkPartitioning,
    /// Dependency failure
    DependencyFailure,
    /// Custom chaos experiment
    Custom(String),
}

/// Blast radius control for chaos experiments
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlastRadius {
    /// Limited to single components
    Limited,
    /// Controlled multi-component impact
    Controlled,
    /// System-wide impact allowed
    SystemWide,
}

/// Safety measures for chaos engineering
#[derive(Debug, Clone)]
pub struct SafetyMeasures {
    /// Automatic rollback triggers
    pub auto_rollback_triggers: Vec<String>,
    /// Maximum impact duration
    pub max_impact_duration: Duration,
    /// Emergency stop conditions
    pub emergency_stop: Vec<String>,
    /// Health check requirements
    pub health_checks: Vec<String>,
}

impl Default for SafetyMeasures {
    fn default() -> Self {
        Self {
            auto_rollback_triggers: vec![
                "cpu_usage_critical".to_string(),
                "memory_exhausted".to_string(),
            ],
            max_impact_duration: Duration::from_secs(300),
            emergency_stop: vec!["system_failure".to_string(), "data_corruption".to_string()],
            health_checks: vec!["system_health".to_string(), "component_status".to_string()],
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable real-time monitoring
    pub enable_monitoring: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Metrics to monitor
    pub monitored_metrics: Vec<MonitoredMetric>,
    /// Alert configuration
    pub alert_config: AlertConfig,
    /// Reporting configuration
    pub reporting_config: ReportingConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            monitoring_interval: Duration::from_secs(1),
            monitored_metrics: vec![
                MonitoredMetric::CpuUtilization,
                MonitoredMetric::MemoryUsage,
                MonitoredMetric::ErrorRate,
                MonitoredMetric::ResponseTime,
            ],
            alert_config: AlertConfig::default(),
            reporting_config: ReportingConfig::default(),
        }
    }
}

/// Metrics to monitor during testing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MonitoredMetric {
    /// CPU utilization
    CpuUtilization,
    /// Memory usage
    MemoryUsage,
    /// Network I/O
    NetworkIO,
    /// Disk I/O
    DiskIO,
    /// Error rate
    ErrorRate,
    /// Response time
    ResponseTime,
    /// Throughput
    Throughput,
    /// Active connections
    ActiveConnections,
    /// Custom metric
    Custom(String),
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerts
    pub enable_alerts: bool,
    /// Alert thresholds
    pub thresholds: HashMap<MonitoredMetric, f64>,
    /// Alert channels
    pub channels: Vec<AlertChannel>,
    /// Alert frequency limits
    pub frequency_limits: FrequencyLimits,
}

impl Default for AlertConfig {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(MonitoredMetric::ErrorRate, 0.1);
        thresholds.insert(MonitoredMetric::MemoryUsage, 0.9);
        thresholds.insert(MonitoredMetric::CpuUtilization, 0.95);

        Self {
            enable_alerts: true,
            thresholds,
            channels: vec![AlertChannel::Console],
            frequency_limits: FrequencyLimits::default(),
        }
    }
}

/// Alert channels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertChannel {
    /// Console output
    Console,
    /// Log files
    Log,
    /// Email notifications
    Email(String),
    /// Webhook notifications
    Webhook(String),
    /// Custom channel
    Custom(String),
}

/// Alert frequency limits
#[derive(Debug, Clone)]
pub struct FrequencyLimits {
    /// Maximum alerts per minute
    pub max_per_minute: usize,
    /// Cooldown period between similar alerts
    pub cooldown_period: Duration,
    /// Enable alert aggregation
    pub enable_aggregation: bool,
}

impl Default for FrequencyLimits {
    fn default() -> Self {
        Self {
            max_per_minute: 10,
            cooldown_period: Duration::from_secs(60),
            enable_aggregation: true,
        }
    }
}

/// Reporting configuration
#[derive(Debug, Clone)]
pub struct ReportingConfig {
    /// Enable automated reporting
    pub enable_automated_reporting: bool,
    /// Report formats
    pub report_formats: Vec<ReportFormat>,
    /// Report generation frequency
    pub generation_frequency: ReportFrequency,
    /// Report distribution
    pub distribution: ReportDistribution,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            enable_automated_reporting: true,
            report_formats: vec![ReportFormat::HTML, ReportFormat::JSON],
            generation_frequency: ReportFrequency::AfterTestSuite,
            distribution: ReportDistribution::default(),
        }
    }
}

/// Report formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportFormat {
    /// HTML reports
    HTML,
    /// JSON data
    JSON,
    /// PDF reports
    PDF,
    /// CSV data
    CSV,
    /// XML format
    XML,
}

/// Report generation frequency
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportFrequency {
    /// After each test
    AfterEachTest,
    /// After test suite completion
    AfterTestSuite,
    /// Scheduled reporting
    Scheduled(Duration),
    /// Manual reporting only
    Manual,
}

/// Report distribution configuration
#[derive(Debug, Clone)]
pub struct ReportDistribution {
    /// Email recipients
    pub email_recipients: Vec<String>,
    /// File system paths
    pub file_paths: Vec<String>,
    /// HTTP endpoints
    pub http_endpoints: Vec<String>,
    /// Custom distribution targets
    pub custom_targets: Vec<String>,
}

impl Default for ReportDistribution {
    fn default() -> Self {
        Self {
            email_recipients: vec![],
            file_paths: vec!["./test_reports/".to_string()],
            http_endpoints: vec![],
            custom_targets: vec![],
        }
    }
}

/// Test environment configuration
#[derive(Debug, Clone)]
pub struct TestEnvironmentConfig {
    /// Environment name
    pub environment_name: String,
    /// Environment variables
    pub environment_variables: HashMap<String, String>,
    /// Resource allocation
    pub resource_allocation: ResourceAllocationConfig,
    /// Cleanup configuration
    pub cleanup_config: CleanupConfig,
}

impl Default for TestEnvironmentConfig {
    fn default() -> Self {
        Self {
            environment_name: "default".to_string(),
            environment_variables: HashMap::new(),
            resource_allocation: ResourceAllocationConfig::default(),
            cleanup_config: CleanupConfig::default(),
        }
    }
}

/// Resource allocation configuration
#[derive(Debug, Clone)]
pub struct ResourceAllocationConfig {
    /// Maximum CPU cores
    pub max_cpu_cores: usize,
    /// Maximum memory (bytes)
    pub max_memory: usize,
    /// Maximum disk space (bytes)
    pub max_disk_space: usize,
    /// Network bandwidth limit (bytes/sec)
    pub network_bandwidth_limit: Option<usize>,
}

impl Default for ResourceAllocationConfig {
    fn default() -> Self {
        Self {
            max_cpu_cores: num_cpus::get(),
            max_memory: 8 * 1024 * 1024 * 1024,       // 8 GB
            max_disk_space: 100 * 1024 * 1024 * 1024, // 100 GB
            network_bandwidth_limit: None,
        }
    }
}

/// Cleanup configuration
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// Auto cleanup after tests
    pub auto_cleanup: bool,
    /// Cleanup timeout
    pub cleanup_timeout: Duration,
    /// Preserve on failure
    pub preserve_on_failure: bool,
    /// Custom cleanup scripts
    pub custom_scripts: Vec<String>,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            auto_cleanup: true,
            cleanup_timeout: Duration::from_secs(30),
            preserve_on_failure: true,
            custom_scripts: vec![],
        }
    }
}
