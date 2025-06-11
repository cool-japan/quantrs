//! Resource Allocation and Optimization Configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Resource allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocationConfig {
    /// Allocation algorithms
    pub allocation_algorithms: Vec<AllocationAlgorithm>,
    /// Resource optimization objectives
    pub optimization_objectives: Vec<ResourceOptimizationObjective>,
    /// Allocation constraints
    pub allocation_constraints: AllocationConstraints,
    /// Dynamic reallocation settings
    pub dynamic_reallocation: DynamicReallocationConfig,
    /// Predictive allocation
    pub predictive_allocation: PredictiveAllocationConfig,
    /// Multi-objective optimization
    pub multi_objective_config: MultiObjectiveAllocationConfig,
}

/// Allocation algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AllocationAlgorithm {
    FirstFit,
    BestFit,
    WorstFit,
    NextFit,
    RoundRobin,
    LoadBalanced,
    CostOptimized,
    PerformanceOptimized,
    HybridOptimized,
    MachineLearningBased,
    GeneticAlgorithm,
    SimulatedAnnealing,
}

/// Resource optimization objectives
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceOptimizationObjective {
    MinimizeCost,
    MaximizePerformance,
    MinimizeLatency,
    MaximizeUtilization,
    MinimizeEnergyConsumption,
    BalanceLoadDistribution,
    OptimizeQueueTime,
    CustomObjective(String),
}

/// Allocation constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationConstraints {
    pub max_memory: Option<usize>,
    pub max_cpu: Option<usize>,
    pub max_gpus: Option<usize>,
    pub required_features: Vec<String>,
    /// Geographic constraints
    pub geographic: GeographicConstraints,
    /// Security constraints
    pub security: SecurityConstraints,
    /// Performance constraints
    pub performance: PerformanceConstraints,
    /// Cost constraints
    pub cost: CostConstraints,
}

/// Geographic constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicConstraints {
    /// Allowed regions
    pub allowed_regions: Vec<String>,
    /// Prohibited regions
    pub prohibited_regions: Vec<String>,
    /// Data residency requirements
    pub data_residency: DataResidencyRequirements,
    /// Latency constraints
    pub latency_constraints: LatencyConstraints,
}

/// Data residency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataResidencyRequirements {
    /// Required countries
    pub required_countries: Vec<String>,
    /// Prohibited countries
    pub prohibited_countries: Vec<String>,
    /// Compliance frameworks
    pub compliance_frameworks: Vec<ComplianceFramework>,
}

/// Compliance frameworks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComplianceFramework {
    GDPR,
    HIPAA,
    SOC2,
    ISO27001,
    FedRAMP,
    Custom(String),
}

/// Latency constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyConstraints {
    /// Maximum latency
    pub max_latency: Duration,
    /// Target latency
    pub target_latency: Duration,
    /// Latency percentile requirements
    pub percentile_requirements: HashMap<String, Duration>,
}

/// Security constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConstraints {
    /// Required security features
    pub required_features: Vec<SecurityFeature>,
    /// Encryption requirements
    pub encryption: EncryptionRequirements,
    /// Access control requirements
    pub access_control: AccessControlRequirements,
    /// Audit requirements
    pub audit: AuditRequirements,
}

/// Security features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SecurityFeature {
    EncryptionAtRest,
    EncryptionInTransit,
    MultiFactorAuthentication,
    RoleBasedAccess,
    NetworkIsolation,
    VPN,
    PrivateNetworking,
    Custom(String),
}

/// Encryption requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionRequirements {
    /// Minimum encryption strength
    pub min_key_length: usize,
    /// Allowed algorithms
    pub allowed_algorithms: Vec<String>,
    /// Key management requirements
    pub key_management: KeyManagementRequirements,
}

/// Key management requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementRequirements {
    /// Hardware security modules required
    pub hsm_required: bool,
    /// Key rotation period
    pub rotation_period: Duration,
    /// Backup requirements
    pub backup_requirements: BackupRequirements,
}

/// Backup requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRequirements {
    /// Backup frequency
    pub frequency: Duration,
    /// Retention period
    pub retention_period: Duration,
    /// Geographic distribution
    pub geo_distribution: bool,
}

/// Access control requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlRequirements {
    /// Authentication methods
    pub auth_methods: Vec<AuthenticationMethod>,
    /// Authorization model
    pub authz_model: AuthorizationModel,
    /// Session management
    pub session_management: SessionManagement,
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    Password,
    TwoFactor,
    Biometric,
    Certificate,
    SAML,
    OAuth,
    Custom(String),
}

/// Authorization models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthorizationModel {
    RBAC,
    ABAC,
    DAC,
    MAC,
    Custom(String),
}

/// Session management requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManagement {
    /// Session timeout
    pub timeout: Duration,
    /// Concurrent session limit
    pub concurrent_limit: usize,
    /// Idle timeout
    pub idle_timeout: Duration,
}

/// Audit requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRequirements {
    /// Audit events
    pub events: Vec<AuditEvent>,
    /// Retention period
    pub retention_period: Duration,
    /// Integrity protection
    pub integrity_protection: bool,
}

/// Audit events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditEvent {
    Login,
    Logout,
    DataAccess,
    DataModification,
    ConfigurationChange,
    PrivilegeEscalation,
    Custom(String),
}

/// Performance constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConstraints {
    /// Minimum performance requirements
    pub min_performance: PerformanceRequirements,
    /// Maximum acceptable degradation
    pub max_degradation: f64,
    /// SLA requirements
    pub sla_requirements: SLARequirements,
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Minimum throughput
    pub min_throughput: f64,
    /// Maximum latency
    pub max_latency: Duration,
    /// Minimum availability
    pub min_availability: f64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// SLA requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLARequirements {
    /// Uptime SLA
    pub uptime_sla: f64,
    /// Performance SLA
    pub performance_sla: HashMap<String, f64>,
    /// Support response time
    pub support_response_time: Duration,
}

/// Cost constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConstraints {
    /// Maximum hourly cost
    pub max_hourly_cost: Option<f64>,
    /// Maximum daily cost
    pub max_daily_cost: Option<f64>,
    /// Maximum monthly cost
    pub max_monthly_cost: Option<f64>,
    /// Cost optimization strategy
    pub optimization_strategy: CostOptimizationStrategy,
}

/// Cost optimization strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CostOptimizationStrategy {
    MinimizeTotal,
    OptimizePerformancePerDollar,
    BalanceCostPerformance,
    PreferSpotInstances,
    PreferReservedInstances,
    Custom(String),
}

/// Dynamic reallocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicReallocationConfig {
    pub enable_dynamic_reallocation: bool,
    pub reallocation_threshold: f64,
    pub reallocation_strategies: Vec<String>,
    /// Reallocation triggers
    pub triggers: Vec<ReallocationTrigger>,
    /// Reallocation policies
    pub policies: Vec<ReallocationPolicy>,
    /// Migration settings
    pub migration: MigrationSettings,
}

/// Reallocation triggers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReallocationTrigger {
    ResourceUtilization,
    PerformanceDegradation,
    CostThreshold,
    LoadImbalance,
    MaintenanceWindow,
    Custom(String),
}

/// Reallocation policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReallocationPolicy {
    /// Policy name
    pub name: String,
    /// Conditions
    pub conditions: Vec<PolicyCondition>,
    /// Actions
    pub actions: Vec<PolicyAction>,
    /// Priority
    pub priority: u8,
}

/// Policy conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCondition {
    /// Metric name
    pub metric: String,
    /// Operator
    pub operator: ComparisonOperator,
    /// Threshold value
    pub threshold: f64,
    /// Duration
    pub duration: Duration,
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
    NotEqual,
}

/// Policy actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PolicyAction {
    ScaleUp,
    ScaleDown,
    Migrate,
    Rebalance,
    Alert,
    Custom(String),
}

/// Migration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationSettings {
    /// Migration strategy
    pub strategy: MigrationStrategy,
    /// Downtime tolerance
    pub downtime_tolerance: Duration,
    /// Data transfer settings
    pub data_transfer: DataTransferSettings,
    /// Rollback settings
    pub rollback: RollbackSettings,
}

/// Migration strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MigrationStrategy {
    LiveMigration,
    ColdMigration,
    HybridMigration,
    IncrementalMigration,
}

/// Data transfer settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransferSettings {
    /// Transfer method
    pub method: DataTransferMethod,
    /// Bandwidth limit
    pub bandwidth_limit: Option<u64>,
    /// Compression
    pub compression: bool,
    /// Encryption
    pub encryption: bool,
}

/// Data transfer methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataTransferMethod {
    NetworkCopy,
    PhysicalTransfer,
    SnapshotReplication,
    IncrementalSync,
}

/// Rollback settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackSettings {
    /// Enable automatic rollback
    pub auto_rollback: bool,
    /// Rollback conditions
    pub conditions: Vec<RollbackCondition>,
    /// Rollback timeout
    pub timeout: Duration,
}

/// Rollback conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackCondition {
    /// Condition type
    pub condition_type: RollbackConditionType,
    /// Threshold
    pub threshold: f64,
    /// Duration
    pub duration: Duration,
}

/// Rollback condition types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RollbackConditionType {
    PerformanceDegradation,
    ErrorRateIncrease,
    ServiceFailure,
    UserDefined(String),
}

/// Predictive allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAllocationConfig {
    pub enable_prediction: bool,
    pub prediction_models: Vec<String>,
    pub prediction_window: u64,
    /// Prediction algorithms
    pub algorithms: Vec<PredictionAlgorithm>,
    /// Training configuration
    pub training: PredictionTrainingConfig,
    /// Validation settings
    pub validation: PredictionValidationConfig,
}

/// Prediction algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PredictionAlgorithm {
    LinearRegression,
    ARIMA,
    LSTM,
    RandomForest,
    SVM,
    EnsembleMethod,
    Custom(String),
}

/// Prediction training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionTrainingConfig {
    /// Training data size
    pub data_size: usize,
    /// Update frequency
    pub update_frequency: Duration,
    /// Feature selection
    pub feature_selection: FeatureSelectionConfig,
    /// Model validation
    pub validation: ModelValidationConfig,
}

/// Feature selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    /// Selection method
    pub method: FeatureSelectionMethod,
    /// Number of features
    pub num_features: Option<usize>,
    /// Importance threshold
    pub importance_threshold: Option<f64>,
}

/// Feature selection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    VarianceThreshold,
    UnivariateSelection,
    RecursiveElimination,
    FeatureImportance,
    Custom(String),
}

/// Model validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationConfig {
    /// Validation method
    pub method: ValidationMethod,
    /// Test size
    pub test_size: f64,
    /// Cross-validation folds
    pub cv_folds: usize,
}

/// Validation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationMethod {
    HoldOut,
    CrossValidation,
    TimeSeriesSplit,
    Custom(String),
}

/// Prediction validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionValidationConfig {
    /// Accuracy threshold
    pub accuracy_threshold: f64,
    /// Confidence interval
    pub confidence_interval: f64,
    /// Validation frequency
    pub validation_frequency: Duration,
}

/// Multi-objective allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveAllocationConfig {
    pub objectives: Vec<String>,
    pub objective_weights: Vec<f64>,
    pub optimization_method: String,
    /// Pareto optimization
    pub pareto_optimization: ParetoOptimizationConfig,
    /// Constraint handling
    pub constraint_handling: ConstraintHandlingConfig,
    /// Solution selection
    pub solution_selection: SolutionSelectionConfig,
}

/// Pareto optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoOptimizationConfig {
    /// Population size
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Mutation probability
    pub mutation_prob: f64,
}

/// Constraint handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintHandlingConfig {
    /// Handling method
    pub method: ConstraintHandlingMethod,
    /// Penalty parameters
    pub penalty_params: HashMap<String, f64>,
    /// Repair mechanisms
    pub repair_mechanisms: Vec<RepairMechanism>,
}

/// Constraint handling methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintHandlingMethod {
    PenaltyMethod,
    BarrierMethod,
    AugmentedLagrangian,
    DeathPenalty,
    RepairMethod,
}

/// Repair mechanisms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RepairMechanism {
    RandomRepair,
    GreedyRepair,
    LocalSearchRepair,
    Custom(String),
}

/// Solution selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionSelectionConfig {
    /// Selection method
    pub method: SolutionSelectionMethod,
    /// Selection criteria
    pub criteria: Vec<SelectionCriterion>,
    /// User preferences
    pub preferences: UserPreferences,
}

/// Solution selection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SolutionSelectionMethod {
    WeightedSum,
    TOPSIS,
    ELECTRE,
    PROMETHEE,
    UserInteractive,
}

/// Selection criteria
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectionCriterion {
    MinDistance,
    MaxUtility,
    UserPreference,
    Custom(String),
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Preference weights
    pub weights: HashMap<String, f64>,
    /// Preference constraints
    pub constraints: Vec<PreferenceConstraint>,
    /// Interactive feedback
    pub interactive_feedback: bool,
}

/// Preference constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceConstraint {
    /// Objective name
    pub objective: String,
    /// Minimum acceptable value
    pub min_value: Option<f64>,
    /// Maximum acceptable value
    pub max_value: Option<f64>,
    /// Target value
    pub target_value: Option<f64>,
}

impl Default for ResourceAllocationConfig {
    fn default() -> Self {
        Self {
            allocation_algorithms: vec![
                AllocationAlgorithm::BestFit,
                AllocationAlgorithm::LoadBalanced,
            ],
            optimization_objectives: vec![
                ResourceOptimizationObjective::BalanceLoadDistribution,
                ResourceOptimizationObjective::MinimizeCost,
            ],
            allocation_constraints: AllocationConstraints::default(),
            dynamic_reallocation: DynamicReallocationConfig::default(),
            predictive_allocation: PredictiveAllocationConfig::default(),
            multi_objective_config: MultiObjectiveAllocationConfig::default(),
        }
    }
}

impl Default for AllocationConstraints {
    fn default() -> Self {
        Self {
            max_memory: None,
            max_cpu: None,
            max_gpus: None,
            required_features: vec![],
            geographic: GeographicConstraints::default(),
            security: SecurityConstraints::default(),
            performance: PerformanceConstraints::default(),
            cost: CostConstraints::default(),
        }
    }
}

impl Default for GeographicConstraints {
    fn default() -> Self {
        Self {
            allowed_regions: vec![],
            prohibited_regions: vec![],
            data_residency: DataResidencyRequirements::default(),
            latency_constraints: LatencyConstraints::default(),
        }
    }
}

impl Default for DataResidencyRequirements {
    fn default() -> Self {
        Self {
            required_countries: vec![],
            prohibited_countries: vec![],
            compliance_frameworks: vec![],
        }
    }
}

impl Default for LatencyConstraints {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_secs(5),
            target_latency: Duration::from_secs(1),
            percentile_requirements: HashMap::new(),
        }
    }
}

impl Default for SecurityConstraints {
    fn default() -> Self {
        Self {
            required_features: vec![
                SecurityFeature::EncryptionAtRest,
                SecurityFeature::EncryptionInTransit,
            ],
            encryption: EncryptionRequirements::default(),
            access_control: AccessControlRequirements::default(),
            audit: AuditRequirements::default(),
        }
    }
}

impl Default for EncryptionRequirements {
    fn default() -> Self {
        Self {
            min_key_length: 256,
            allowed_algorithms: vec!["AES-256".to_string()],
            key_management: KeyManagementRequirements::default(),
        }
    }
}

impl Default for KeyManagementRequirements {
    fn default() -> Self {
        Self {
            hsm_required: false,
            rotation_period: Duration::from_secs(86400 * 90), // 90 days
            backup_requirements: BackupRequirements::default(),
        }
    }
}

impl Default for BackupRequirements {
    fn default() -> Self {
        Self {
            frequency: Duration::from_secs(86400),              // daily
            retention_period: Duration::from_secs(86400 * 365), // 1 year
            geo_distribution: false,
        }
    }
}

impl Default for AccessControlRequirements {
    fn default() -> Self {
        Self {
            auth_methods: vec![AuthenticationMethod::Password],
            authz_model: AuthorizationModel::RBAC,
            session_management: SessionManagement::default(),
        }
    }
}

impl Default for SessionManagement {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(3600), // 1 hour
            concurrent_limit: 5,
            idle_timeout: Duration::from_secs(900), // 15 minutes
        }
    }
}

impl Default for AuditRequirements {
    fn default() -> Self {
        Self {
            events: vec![
                AuditEvent::Login,
                AuditEvent::DataAccess,
                AuditEvent::ConfigurationChange,
            ],
            retention_period: Duration::from_secs(86400 * 365), // 1 year
            integrity_protection: true,
        }
    }
}

impl Default for PerformanceConstraints {
    fn default() -> Self {
        Self {
            min_performance: PerformanceRequirements::default(),
            max_degradation: 0.1, // 10%
            sla_requirements: SLARequirements::default(),
        }
    }
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            min_throughput: 1.0,
            max_latency: Duration::from_secs(5),
            min_availability: 0.99,
            custom_metrics: HashMap::new(),
        }
    }
}

impl Default for SLARequirements {
    fn default() -> Self {
        Self {
            uptime_sla: 0.99,
            performance_sla: HashMap::new(),
            support_response_time: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl Default for CostConstraints {
    fn default() -> Self {
        Self {
            max_hourly_cost: None,
            max_daily_cost: None,
            max_monthly_cost: None,
            optimization_strategy: CostOptimizationStrategy::BalanceCostPerformance,
        }
    }
}

impl Default for DynamicReallocationConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_reallocation: true,
            reallocation_threshold: 0.8,
            reallocation_strategies: vec![],
            triggers: vec![
                ReallocationTrigger::ResourceUtilization,
                ReallocationTrigger::PerformanceDegradation,
            ],
            policies: vec![],
            migration: MigrationSettings::default(),
        }
    }
}

impl Default for MigrationSettings {
    fn default() -> Self {
        Self {
            strategy: MigrationStrategy::LiveMigration,
            downtime_tolerance: Duration::from_secs(30),
            data_transfer: DataTransferSettings::default(),
            rollback: RollbackSettings::default(),
        }
    }
}

impl Default for DataTransferSettings {
    fn default() -> Self {
        Self {
            method: DataTransferMethod::NetworkCopy,
            bandwidth_limit: None,
            compression: true,
            encryption: true,
        }
    }
}

impl Default for RollbackSettings {
    fn default() -> Self {
        Self {
            auto_rollback: true,
            conditions: vec![],
            timeout: Duration::from_secs(600), // 10 minutes
        }
    }
}

impl Default for PredictiveAllocationConfig {
    fn default() -> Self {
        Self {
            enable_prediction: false,
            prediction_models: vec![],
            prediction_window: 3600, // 1 hour
            algorithms: vec![PredictionAlgorithm::LinearRegression],
            training: PredictionTrainingConfig::default(),
            validation: PredictionValidationConfig::default(),
        }
    }
}

impl Default for PredictionTrainingConfig {
    fn default() -> Self {
        Self {
            data_size: 1000,
            update_frequency: Duration::from_secs(3600), // 1 hour
            feature_selection: FeatureSelectionConfig::default(),
            validation: ModelValidationConfig::default(),
        }
    }
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            method: FeatureSelectionMethod::VarianceThreshold,
            num_features: None,
            importance_threshold: None,
        }
    }
}

impl Default for ModelValidationConfig {
    fn default() -> Self {
        Self {
            method: ValidationMethod::CrossValidation,
            test_size: 0.2,
            cv_folds: 5,
        }
    }
}

impl Default for PredictionValidationConfig {
    fn default() -> Self {
        Self {
            accuracy_threshold: 0.8,
            confidence_interval: 0.95,
            validation_frequency: Duration::from_secs(86400), // daily
        }
    }
}

impl Default for MultiObjectiveAllocationConfig {
    fn default() -> Self {
        Self {
            objectives: vec!["cost".to_string(), "performance".to_string()],
            objective_weights: vec![0.5, 0.5],
            optimization_method: "NSGA-II".to_string(),
            pareto_optimization: ParetoOptimizationConfig::default(),
            constraint_handling: ConstraintHandlingConfig::default(),
            solution_selection: SolutionSelectionConfig::default(),
        }
    }
}

impl Default for ParetoOptimizationConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 50,
            crossover_prob: 0.9,
            mutation_prob: 0.1,
        }
    }
}

impl Default for ConstraintHandlingConfig {
    fn default() -> Self {
        Self {
            method: ConstraintHandlingMethod::PenaltyMethod,
            penalty_params: HashMap::new(),
            repair_mechanisms: vec![],
        }
    }
}

impl Default for SolutionSelectionConfig {
    fn default() -> Self {
        Self {
            method: SolutionSelectionMethod::TOPSIS,
            criteria: vec![SelectionCriterion::MinDistance],
            preferences: UserPreferences::default(),
        }
    }
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            weights: HashMap::new(),
            constraints: vec![],
            interactive_feedback: false,
        }
    }
}
