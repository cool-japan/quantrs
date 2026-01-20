//! Hardware Adaptation and Device-Specific Configuration

use std::collections::HashMap;
use std::time::Duration;

/// Hardware adaptation configuration
#[derive(Debug, Clone)]
pub struct HardwareAdaptationConfig {
    /// Enable hardware-aware compilation
    pub enable_hardware_aware: bool,
    /// Device characterization integration
    pub device_characterization: DeviceCharacterizationConfig,
    /// Dynamic calibration adjustment
    pub dynamic_calibration: DynamicCalibrationConfig,
    /// Hardware failure handling
    pub failure_handling: FailureHandlingConfig,
    /// Resource optimization
    pub resource_optimization: ResourceOptimizationConfig,
}

/// Device characterization configuration
#[derive(Debug, Clone)]
pub struct DeviceCharacterizationConfig {
    /// Enable device characterization
    pub enable_characterization: bool,
    /// Characterization depth
    pub characterization_depth: CharacterizationDepth,
    /// Characterization frequency
    pub characterization_frequency: Duration,
    /// Calibration integration
    pub calibration_integration: CalibrationIntegrationConfig,
    /// Performance modeling
    pub performance_modeling: PerformanceModelingConfig,
}

/// Levels of device characterization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharacterizationDepth {
    Basic,
    Standard,
    Comprehensive,
    ExhaustiveCharacterization,
}

/// Calibration integration configuration
#[derive(Debug, Clone)]
pub struct CalibrationIntegrationConfig {
    /// Enable calibration integration
    pub enable_integration: bool,
    /// Real-time calibration updates
    pub realtime_updates: bool,
    /// Calibration sources
    pub calibration_sources: Vec<CalibrationSource>,
    /// Update strategy
    pub update_strategy: CalibrationUpdateStrategy,
}

/// Sources of calibration data
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationSource {
    DeviceProvider,
    LocalMeasurement,
    CommunityDatabase,
    MachineLearningPrediction,
    HybridSources,
}

/// Strategies for calibration updates
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationUpdateStrategy {
    Immediate,
    Batch,
    Scheduled,
    EventDriven,
    Adaptive,
}

/// Performance modeling configuration
#[derive(Debug, Clone)]
pub struct PerformanceModelingConfig {
    /// Enable performance modeling
    pub enable_modeling: bool,
    /// Modeling approaches
    pub modeling_approaches: Vec<ModelingApproach>,
    /// Model validation
    pub model_validation: ModelValidationConfig,
    /// Prediction accuracy requirements
    pub accuracy_requirements: ModelAccuracyRequirements,
}

/// Approaches to performance modeling
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelingApproach {
    StatisticalModeling,
    MachineLearningBased,
    PhysicsInformed,
    HybridModeling,
    EmpiricalModeling,
}

/// Model validation configuration
#[derive(Debug, Clone)]
pub struct ModelValidationConfig {
    /// Validation methods
    pub validation_methods: Vec<ValidationMethod>,
    /// Validation frequency
    pub validation_frequency: Duration,
    /// Cross-validation setup
    pub cross_validation: CrossValidationSetup,
}

/// Model validation methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationMethod {
    HoldoutValidation,
    CrossValidation,
    BootstrapValidation,
    TimeSeriesValidation,
    PhysicsBasedValidation,
}

/// Cross-validation setup
#[derive(Debug, Clone)]
pub struct CrossValidationSetup {
    /// Number of folds
    pub folds: usize,
    /// Stratification strategy
    pub stratification: StratificationStrategy,
    /// Temporal considerations
    pub temporal_considerations: TemporalConsiderations,
}

/// Stratification strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StratificationStrategy {
    Random,
    Temporal,
    DeviceBased,
    PerformanceBased,
    Balanced,
}

/// Temporal considerations for validation
#[derive(Debug, Clone)]
pub struct TemporalConsiderations {
    /// Respect temporal order
    pub respect_temporal_order: bool,
    /// Gap between train and test
    pub temporal_gap: Duration,
    /// Seasonal adjustments
    pub seasonal_adjustments: bool,
}

/// Model accuracy requirements
#[derive(Debug, Clone)]
pub struct ModelAccuracyRequirements {
    /// Minimum R-squared
    pub min_r_squared: f64,
    /// Maximum RMSE
    pub max_rmse: f64,
    /// Maximum mean absolute error
    pub max_mae: f64,
    /// Confidence interval requirements
    pub confidence_interval: ConfidenceIntervalRequirements,
}

/// Confidence interval requirements
#[derive(Debug, Clone)]
pub struct ConfidenceIntervalRequirements {
    /// Required confidence level
    pub confidence_level: f64,
    /// Maximum interval width
    pub max_interval_width: f64,
    /// Coverage requirements
    pub coverage_requirements: f64,
}

/// Dynamic calibration configuration
#[derive(Debug, Clone)]
pub struct DynamicCalibrationConfig {
    /// Enable dynamic calibration
    pub enable_dynamic_calibration: bool,
    /// Calibration triggers
    pub calibration_triggers: Vec<CalibrationTrigger>,
    /// Calibration strategy
    pub calibration_strategy: CalibrationStrategy,
    /// Update frequency limits
    pub frequency_limits: CalibrationFrequencyLimits,
    /// Quality assurance
    pub quality_assurance: CalibrationQualityAssurance,
}

/// Triggers for calibration updates
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationTrigger {
    TimeInterval,
    PerformanceDrift,
    ErrorRateIncrease,
    TemperatureChange,
    ManualRequest,
    AutomaticDetection,
}

/// Calibration strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationStrategy {
    FullRecalibration,
    IncrementalUpdate,
    SelectiveCalibration,
    PredictiveCalibration,
    AdaptiveCalibration,
}

/// Frequency limits for calibration
#[derive(Debug, Clone)]
pub struct CalibrationFrequencyLimits {
    /// Minimum interval between calibrations
    pub min_interval: Duration,
    /// Maximum calibrations per day
    pub max_per_day: usize,
    /// Emergency calibration limits
    pub emergency_limits: EmergencyCalibrationLimits,
}

/// Emergency calibration limits
#[derive(Debug, Clone)]
pub struct EmergencyCalibrationLimits {
    /// Allow emergency calibration
    pub allow_emergency: bool,
    /// Maximum emergency calibrations per hour
    pub max_emergency_per_hour: usize,
    /// Emergency threshold
    pub emergency_threshold: f64,
}

/// Calibration quality assurance
#[derive(Debug, Clone)]
pub struct CalibrationQualityAssurance {
    /// Quality metrics
    pub quality_metrics: Vec<CalibrationQualityMetric>,
    /// Validation requirements
    pub validation_requirements: CalibrationValidationRequirements,
    /// Rollback strategy
    pub rollback_strategy: CalibrationRollbackStrategy,
}

/// Quality metrics for calibration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationQualityMetric {
    Repeatability,
    Accuracy,
    Stability,
    Drift,
    Consistency,
}

/// Validation requirements for calibration
#[derive(Debug, Clone)]
pub struct CalibrationValidationRequirements {
    /// Minimum validation samples
    pub min_validation_samples: usize,
    /// Validation accuracy threshold
    pub accuracy_threshold: f64,
    /// Statistical significance level
    pub significance_level: f64,
}

/// Rollback strategies for failed calibration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationRollbackStrategy {
    PreviousCalibration,
    DefaultCalibration,
    InterpolatedCalibration,
    ManualIntervention,
}

/// Failure handling configuration
#[derive(Debug, Clone)]
pub struct FailureHandlingConfig {
    /// Enable failure handling
    pub enable_failure_handling: bool,
    /// Failure detection configuration
    pub failure_detection: FailureDetectionConfig,
    /// Recovery strategies
    pub recovery_strategies: Vec<RecoveryStrategy>,
    /// Escalation procedures
    pub escalation_procedures: EscalationProcedures,
}

/// Failure detection configuration
#[derive(Debug, Clone)]
pub struct FailureDetectionConfig {
    /// Detection methods
    pub detection_methods: Vec<FailureDetectionMethod>,
    /// Detection sensitivity
    pub detection_sensitivity: f64,
    /// Monitoring intervals
    pub monitoring_intervals: MonitoringIntervals,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Methods for detecting hardware failures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureDetectionMethod {
    PerformanceMonitoring,
    ErrorRateTracking,
    PatternRecognition,
    AnomalyDetection,
    PhysicalSensorMonitoring,
    StatisticalAnalysis,
}

/// Monitoring intervals for failure detection
#[derive(Debug, Clone)]
pub struct MonitoringIntervals {
    /// Performance monitoring interval
    pub performance_interval: Duration,
    /// Error rate monitoring interval
    pub error_rate_interval: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
}

/// Alert thresholds for failure detection
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Performance degradation threshold
    pub performance_threshold: f64,
    /// Error rate threshold
    pub error_rate_threshold: f64,
    /// Anomaly score threshold
    pub anomaly_threshold: f64,
}

/// Recovery strategies for hardware failures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStrategy {
    Recalibration,
    ComponentReset,
    AlternativeRouting,
    GracefulDegradation,
    ServiceReplacement,
    ManualIntervention,
}

/// Escalation procedures for failures
#[derive(Debug, Clone, Default)]
pub struct EscalationProcedures {
    /// Escalation levels
    pub escalation_levels: Vec<EscalationLevel>,
    /// Escalation timeouts
    pub escalation_timeouts: HashMap<String, Duration>,
    /// Notification procedures
    pub notification_procedures: NotificationProcedures,
}

/// Escalation levels for failure handling
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    /// Level name
    pub level_name: String,
    /// Severity threshold
    pub severity_threshold: f64,
    /// Required actions
    pub required_actions: Vec<String>,
    /// Responsible parties
    pub responsible_parties: Vec<String>,
}

/// Notification procedures for escalation
#[derive(Debug, Clone)]
pub struct NotificationProcedures {
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Notification content templates
    pub content_templates: HashMap<String, String>,
    /// Delivery preferences
    pub delivery_preferences: DeliveryPreferences,
}

/// Notification delivery channels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotificationChannel {
    Email,
    SMS,
    Slack,
    PagerDuty,
    Dashboard,
    API,
}

/// Notification delivery preferences
#[derive(Debug, Clone)]
pub struct DeliveryPreferences {
    /// Priority-based routing
    pub priority_routing: bool,
    /// Delivery confirmation required
    pub confirmation_required: bool,
    /// Retry attempts
    pub retry_attempts: usize,
    /// Retry intervals
    pub retry_intervals: Vec<Duration>,
}

/// Resource optimization configuration
#[derive(Debug, Clone)]
pub struct ResourceOptimizationConfig {
    /// Enable resource optimization
    pub enable_optimization: bool,
    /// Optimization strategies
    pub optimization_strategies: Vec<ResourceOptimizationStrategy>,
    /// Resource allocation
    pub resource_allocation: ResourceAllocationConfig,
    /// Load balancing
    pub load_balancing: LoadBalancingConfig,
}

/// Resource optimization strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceOptimizationStrategy {
    MinimizeLatency,
    MaximizeThroughput,
    MinimizeEnergyConsumption,
    BalancedOptimization,
    CustomOptimization(String),
}

/// Resource allocation configuration
#[derive(Debug, Clone)]
pub struct ResourceAllocationConfig {
    /// Allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Priority-based allocation
    pub priority_allocation: PriorityAllocationConfig,
    /// Dynamic reallocation
    pub dynamic_reallocation: DynamicReallocationConfig,
}

/// Resource allocation strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    RoundRobin,
    PerformanceBased,
    PredictiveBased,
}

/// Priority-based allocation configuration
#[derive(Debug, Clone)]
pub struct PriorityAllocationConfig {
    /// Enable priority allocation
    pub enable_priority: bool,
    /// Priority levels
    pub priority_levels: Vec<PriorityLevel>,
    /// Preemption policy
    pub preemption_policy: PreemptionPolicy,
}

/// Priority levels for resource allocation
#[derive(Debug, Clone)]
pub struct PriorityLevel {
    /// Priority name
    pub name: String,
    /// Priority value (higher = more important)
    pub value: i32,
    /// Resource guarantees
    pub resource_guarantees: ResourceGuarantees,
}

/// Resource guarantees for priority levels
#[derive(Debug, Clone)]
pub struct ResourceGuarantees {
    /// Minimum CPU allocation
    pub min_cpu: f64,
    /// Minimum memory allocation
    pub min_memory: f64,
    /// Maximum latency guarantee
    pub max_latency: Duration,
}

/// Preemption policies for resource allocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreemptionPolicy {
    NoPreemption,
    PriorityBased,
    FairShare,
    TimeSlicing,
    Cooperative,
}

/// Dynamic reallocation configuration
#[derive(Debug, Clone)]
pub struct DynamicReallocationConfig {
    /// Enable dynamic reallocation
    pub enable_reallocation: bool,
    /// Reallocation triggers
    pub triggers: Vec<ReallocationTrigger>,
    /// Reallocation frequency
    pub frequency: ReallocationFrequency,
    /// Migration strategy
    pub migration_strategy: MigrationStrategy,
}

/// Triggers for dynamic reallocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReallocationTrigger {
    LoadImbalance,
    PerformanceDegradation,
    ResourceContention,
    ScheduledMaintenance,
    ManualRequest,
}

/// Frequency configuration for reallocation
#[derive(Debug, Clone)]
pub struct ReallocationFrequency {
    /// Base interval
    pub base_interval: Duration,
    /// Adaptive frequency
    pub adaptive_frequency: bool,
    /// Maximum frequency
    pub max_frequency: Duration,
    /// Minimum frequency
    pub min_frequency: Duration,
}

/// Migration strategies for reallocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationStrategy {
    LiveMigration,
    CheckpointRestart,
    GracefulShutdown,
    ForceTermination,
}

/// Load balancing configuration
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    /// Enable load balancing
    pub enable_load_balancing: bool,
    /// Load balancing algorithms
    pub algorithms: Vec<LoadBalancingAlgorithm>,
    /// Health checking
    pub health_checking: HealthCheckingConfig,
    /// Traffic distribution
    pub traffic_distribution: TrafficDistributionConfig,
}

/// Load balancing algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    PerformanceBased,
    HashBased,
    AdaptiveBalancing,
}

/// Health checking configuration
#[derive(Debug, Clone)]
pub struct HealthCheckingConfig {
    /// Health check interval
    pub check_interval: Duration,
    /// Health check timeout
    pub check_timeout: Duration,
    /// Failure threshold
    pub failure_threshold: usize,
    /// Recovery threshold
    pub recovery_threshold: usize,
}

/// Traffic distribution configuration
#[derive(Debug, Clone)]
pub struct TrafficDistributionConfig {
    /// Distribution strategy
    pub strategy: DistributionStrategy,
    /// Weight assignment
    pub weight_assignment: WeightAssignmentStrategy,
    /// Sticky sessions
    pub sticky_sessions: bool,
}

/// Traffic distribution strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionStrategy {
    Uniform,
    Weighted,
    PerformanceBased,
    CapacityBased,
    LatencyBased,
}

/// Weight assignment strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightAssignmentStrategy {
    Static,
    Dynamic,
    PerformanceBased,
    CapacityBased,
    Historical,
}

// Default implementations

impl Default for HardwareAdaptationConfig {
    fn default() -> Self {
        Self {
            enable_hardware_aware: true,
            device_characterization: DeviceCharacterizationConfig::default(),
            dynamic_calibration: DynamicCalibrationConfig::default(),
            failure_handling: FailureHandlingConfig::default(),
            resource_optimization: ResourceOptimizationConfig::default(),
        }
    }
}

impl Default for DeviceCharacterizationConfig {
    fn default() -> Self {
        Self {
            enable_characterization: true,
            characterization_depth: CharacterizationDepth::Standard,
            characterization_frequency: Duration::from_secs(3600), // 1 hour
            calibration_integration: CalibrationIntegrationConfig::default(),
            performance_modeling: PerformanceModelingConfig::default(),
        }
    }
}

impl Default for CalibrationIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_integration: true,
            realtime_updates: true,
            calibration_sources: vec![
                CalibrationSource::DeviceProvider,
                CalibrationSource::LocalMeasurement,
            ],
            update_strategy: CalibrationUpdateStrategy::Adaptive,
        }
    }
}

impl Default for PerformanceModelingConfig {
    fn default() -> Self {
        Self {
            enable_modeling: true,
            modeling_approaches: vec![
                ModelingApproach::StatisticalModeling,
                ModelingApproach::MachineLearningBased,
            ],
            model_validation: ModelValidationConfig::default(),
            accuracy_requirements: ModelAccuracyRequirements::default(),
        }
    }
}

impl Default for ModelValidationConfig {
    fn default() -> Self {
        Self {
            validation_methods: vec![ValidationMethod::CrossValidation],
            validation_frequency: Duration::from_secs(86400), // 24 hours
            cross_validation: CrossValidationSetup::default(),
        }
    }
}

impl Default for CrossValidationSetup {
    fn default() -> Self {
        Self {
            folds: 5,
            stratification: StratificationStrategy::Random,
            temporal_considerations: TemporalConsiderations::default(),
        }
    }
}

impl Default for TemporalConsiderations {
    fn default() -> Self {
        Self {
            respect_temporal_order: true,
            temporal_gap: Duration::from_secs(300), // 5 minutes
            seasonal_adjustments: false,
        }
    }
}

impl Default for ModelAccuracyRequirements {
    fn default() -> Self {
        Self {
            min_r_squared: 0.8,
            max_rmse: 0.1,
            max_mae: 0.05,
            confidence_interval: ConfidenceIntervalRequirements::default(),
        }
    }
}

impl Default for ConfidenceIntervalRequirements {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            max_interval_width: 0.2,
            coverage_requirements: 0.95,
        }
    }
}

impl Default for DynamicCalibrationConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_calibration: true,
            calibration_triggers: vec![
                CalibrationTrigger::TimeInterval,
                CalibrationTrigger::PerformanceDrift,
            ],
            calibration_strategy: CalibrationStrategy::AdaptiveCalibration,
            frequency_limits: CalibrationFrequencyLimits::default(),
            quality_assurance: CalibrationQualityAssurance::default(),
        }
    }
}

impl Default for CalibrationFrequencyLimits {
    fn default() -> Self {
        Self {
            min_interval: Duration::from_secs(300), // 5 minutes
            max_per_day: 48,
            emergency_limits: EmergencyCalibrationLimits::default(),
        }
    }
}

impl Default for EmergencyCalibrationLimits {
    fn default() -> Self {
        Self {
            allow_emergency: true,
            max_emergency_per_hour: 3,
            emergency_threshold: 0.01, // 1% performance degradation
        }
    }
}

impl Default for CalibrationQualityAssurance {
    fn default() -> Self {
        Self {
            quality_metrics: vec![
                CalibrationQualityMetric::Accuracy,
                CalibrationQualityMetric::Repeatability,
            ],
            validation_requirements: CalibrationValidationRequirements::default(),
            rollback_strategy: CalibrationRollbackStrategy::PreviousCalibration,
        }
    }
}

impl Default for CalibrationValidationRequirements {
    fn default() -> Self {
        Self {
            min_validation_samples: 10,
            accuracy_threshold: 0.95,
            significance_level: 0.05,
        }
    }
}

impl Default for FailureHandlingConfig {
    fn default() -> Self {
        Self {
            enable_failure_handling: true,
            failure_detection: FailureDetectionConfig::default(),
            recovery_strategies: vec![
                RecoveryStrategy::Recalibration,
                RecoveryStrategy::GracefulDegradation,
            ],
            escalation_procedures: EscalationProcedures::default(),
        }
    }
}

impl Default for FailureDetectionConfig {
    fn default() -> Self {
        Self {
            detection_methods: vec![
                FailureDetectionMethod::PerformanceMonitoring,
                FailureDetectionMethod::ErrorRateTracking,
            ],
            detection_sensitivity: 0.95,
            monitoring_intervals: MonitoringIntervals::default(),
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for MonitoringIntervals {
    fn default() -> Self {
        Self {
            performance_interval: Duration::from_secs(60),
            error_rate_interval: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(300),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            performance_threshold: 0.1, // 10% degradation
            error_rate_threshold: 0.05, // 5% error rate
            anomaly_threshold: 2.0,     // 2 standard deviations
        }
    }
}

impl Default for NotificationProcedures {
    fn default() -> Self {
        Self {
            channels: vec![NotificationChannel::Dashboard],
            content_templates: HashMap::new(),
            delivery_preferences: DeliveryPreferences::default(),
        }
    }
}

impl Default for DeliveryPreferences {
    fn default() -> Self {
        Self {
            priority_routing: true,
            confirmation_required: false,
            retry_attempts: 3,
            retry_intervals: vec![
                Duration::from_secs(60),
                Duration::from_secs(300),
                Duration::from_secs(900),
            ],
        }
    }
}

impl Default for ResourceOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            optimization_strategies: vec![ResourceOptimizationStrategy::BalancedOptimization],
            resource_allocation: ResourceAllocationConfig::default(),
            load_balancing: LoadBalancingConfig::default(),
        }
    }
}

impl Default for ResourceAllocationConfig {
    fn default() -> Self {
        Self {
            allocation_strategy: AllocationStrategy::PerformanceBased,
            priority_allocation: PriorityAllocationConfig::default(),
            dynamic_reallocation: DynamicReallocationConfig::default(),
        }
    }
}

impl Default for PriorityAllocationConfig {
    fn default() -> Self {
        Self {
            enable_priority: true,
            priority_levels: vec![],
            preemption_policy: PreemptionPolicy::PriorityBased,
        }
    }
}

impl Default for DynamicReallocationConfig {
    fn default() -> Self {
        Self {
            enable_reallocation: true,
            triggers: vec![
                ReallocationTrigger::LoadImbalance,
                ReallocationTrigger::PerformanceDegradation,
            ],
            frequency: ReallocationFrequency::default(),
            migration_strategy: MigrationStrategy::LiveMigration,
        }
    }
}

impl Default for ReallocationFrequency {
    fn default() -> Self {
        Self {
            base_interval: Duration::from_secs(300),
            adaptive_frequency: true,
            max_frequency: Duration::from_secs(60),
            min_frequency: Duration::from_secs(3600),
        }
    }
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enable_load_balancing: true,
            algorithms: vec![LoadBalancingAlgorithm::PerformanceBased],
            health_checking: HealthCheckingConfig::default(),
            traffic_distribution: TrafficDistributionConfig::default(),
        }
    }
}

impl Default for HealthCheckingConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 2,
        }
    }
}

impl Default for TrafficDistributionConfig {
    fn default() -> Self {
        Self {
            strategy: DistributionStrategy::PerformanceBased,
            weight_assignment: WeightAssignmentStrategy::Dynamic,
            sticky_sessions: false,
        }
    }
}
