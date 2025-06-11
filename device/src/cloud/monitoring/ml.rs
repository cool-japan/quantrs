//! Machine learning and AutoML configuration for monitoring.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// AutoML configuration for monitoring optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLConfig {
    /// Enable AutoML
    pub enabled: bool,
    /// AutoML tasks
    pub tasks: Vec<AutoMLTask>,
    /// Model selection
    pub model_selection: ModelSelectionConfig,
    /// Hyperparameter optimization
    pub hyperparameter_optimization: HyperparameterOptimizationConfig,
    /// Model deployment
    pub deployment: ModelDeploymentConfig,
}

/// AutoML tasks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AutoMLTask {
    AnomalyDetection,
    ForecastingOptimization,
    AlertClassification,
    ResourcePrediction,
    CostOptimization,
    PerformanceTuning,
    Custom(String),
}

/// Model selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionConfig {
    /// Selection criteria
    pub criteria: Vec<SelectionCriterion>,
    /// Evaluation metrics
    pub metrics: Vec<EvaluationMetric>,
    /// Cross-validation
    pub cross_validation: AutoMLCrossValidationConfig,
    /// Model families to consider
    pub model_families: Vec<ModelFamily>,
}

/// Selection criteria
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectionCriterion {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC,
    Speed,
    MemoryUsage,
    Interpretability,
    Custom(String),
}

/// Evaluation metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvaluationMetric {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    RocAuc,
    MAE,
    MSE,
    RMSE,
    Custom(String),
}

/// AutoML cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLCrossValidationConfig {
    /// CV method
    pub method: CrossValidationMethod,
    /// Number of folds
    pub folds: u32,
    /// Stratified sampling
    pub stratified: bool,
    /// Random seed
    pub seed: Option<u64>,
}

/// Cross-validation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CrossValidationMethod {
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    Custom(String),
}

/// Model families
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFamily {
    LinearModels,
    TreeModels,
    EnsembleMethods,
    NeuralNetworks,
    SupportVectorMachines,
    NaiveBayes,
    KNearestNeighbors,
    Custom(String),
}

/// Hyperparameter optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterOptimizationConfig {
    /// Optimization method
    pub method: OptimizationMethod,
    /// Search space
    pub search_space: SearchSpaceConfig,
    /// Optimization budget
    pub budget: OptimizationBudget,
    /// Early stopping
    pub early_stopping: EarlyStoppingConfig,
}

/// Optimization methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationMethod {
    RandomSearch,
    GridSearch,
    BayesianOptimization,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    Custom(String),
}

/// Search space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpaceConfig {
    /// Parameter definitions
    pub parameters: Vec<ParameterDefinition>,
    /// Constraints
    pub constraints: Vec<ParameterConstraint>,
}

/// Parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Default value
    pub default: Option<String>,
    /// Description
    pub description: String,
}

/// Parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Integer { min: i64, max: i64 },
    Float { min: f64, max: f64 },
    Categorical { values: Vec<String> },
    Boolean,
    Custom(String),
}

/// Parameter constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Parameters involved
    pub parameters: Vec<String>,
    /// Constraint expression
    pub expression: String,
}

/// Constraint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintType {
    LinearEquality,
    LinearInequality,
    NonLinear,
    Conditional,
    Custom(String),
}

/// Optimization budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationBudget {
    /// Maximum evaluations
    pub max_evaluations: u32,
    /// Maximum time
    pub max_time: Duration,
    /// Maximum parallel trials
    pub max_parallel: u32,
    /// Resource limits
    pub resources: ResourceLimits,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// CPU cores
    pub cpu_cores: Option<u32>,
    /// Memory (GB)
    pub memory_gb: Option<u32>,
    /// GPU count
    pub gpu_count: Option<u32>,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,
    /// Patience (evaluations without improvement)
    pub patience: u32,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Evaluation frequency
    pub evaluation_frequency: u32,
}

/// Model deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDeploymentConfig {
    /// Deployment strategy
    pub strategy: DeploymentStrategy,
    /// Model versioning
    pub versioning: VersioningConfig,
    /// Model monitoring
    pub monitoring: ModelMonitoringConfig,
    /// Rollback configuration
    pub rollback: RollbackConfig,
}

/// Deployment strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    BlueGreen,
    Canary,
    RollingUpdate,
    Immediate,
    Custom(String),
}

/// Versioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningConfig {
    /// Version scheme
    pub scheme: VersionScheme,
    /// Model registry
    pub registry: ModelRegistryConfig,
    /// Artifact storage
    pub artifacts: ArtifactStorageConfig,
}

/// Version schemes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VersionScheme {
    Semantic,
    Sequential,
    Timestamp,
    Custom(String),
}

/// Model registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryConfig {
    /// Registry type
    pub registry_type: RegistryType,
    /// Connection settings
    pub connection: RegistryConnection,
    /// Metadata tracking
    pub metadata: MetadataConfig,
}

/// Registry types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegistryType {
    MLflow,
    ModelDB,
    KubeflowPipelines,
    Custom(String),
}

/// Registry connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConnection {
    /// Connection URL
    pub url: String,
    /// Authentication
    pub auth: AuthConfig,
    /// Connection timeout
    pub timeout: Duration,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Auth method
    pub method: AuthMethod,
    /// Credentials
    pub credentials: std::collections::HashMap<String, String>,
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthMethod {
    Token,
    BasicAuth,
    OAuth2,
    Certificate,
    Custom(String),
}

/// Metadata configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataConfig {
    /// Track hyperparameters
    pub track_hyperparameters: bool,
    /// Track metrics
    pub track_metrics: bool,
    /// Track artifacts
    pub track_artifacts: bool,
    /// Custom metadata
    pub custom_fields: Vec<MetadataField>,
}

/// Metadata field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: MetadataFieldType,
    /// Required field
    pub required: bool,
}

/// Metadata field types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataFieldType {
    String,
    Number,
    Boolean,
    Date,
    JSON,
    Custom(String),
}

/// Artifact storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactStorageConfig {
    /// Storage backend
    pub backend: StorageBackend,
    /// Storage path
    pub path: String,
    /// Compression settings
    pub compression: CompressionConfig,
}

/// Storage backends
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StorageBackend {
    Local,
    S3,
    GCS,
    Azure,
    HDFS,
    Custom(String),
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression enabled
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: u8,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Lz4,
    Zstd,
    Bzip2,
    Custom(String),
}

/// Model monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMonitoringConfig {
    /// Performance monitoring
    pub performance: PerformanceMonitoringConfig,
    /// Data drift detection
    pub data_drift: DataDriftConfig,
    /// Model drift detection
    pub model_drift: ModelDriftConfig,
    /// Alert configuration
    pub alerts: ModelAlertConfig,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    /// Metrics to track
    pub metrics: Vec<PerformanceMetric>,
    /// Monitoring frequency
    pub frequency: Duration,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
}

/// Performance metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PerformanceMetric {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    Latency,
    Throughput,
    MemoryUsage,
    CPUUsage,
    Custom(String),
}

/// Performance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Accuracy threshold
    pub accuracy: Option<f64>,
    /// Latency threshold (ms)
    pub latency_ms: Option<f64>,
    /// Throughput threshold (requests/sec)
    pub throughput: Option<f64>,
    /// Memory threshold (MB)
    pub memory_mb: Option<f64>,
}

/// Data drift configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDriftConfig {
    /// Enable drift detection
    pub enabled: bool,
    /// Detection methods
    pub methods: Vec<DriftDetectionMethod>,
    /// Detection frequency
    pub frequency: Duration,
    /// Reference window
    pub reference_window: Duration,
}

/// Drift detection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DriftDetectionMethod {
    KolmogorovSmirnov,
    ChiSquare,
    PopulationStabilityIndex,
    JensenShannonDivergence,
    Custom(String),
}

/// Model drift configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDriftConfig {
    /// Enable model drift detection
    pub enabled: bool,
    /// Drift threshold
    pub threshold: f64,
    /// Detection window
    pub window: Duration,
    /// Retraining trigger
    pub retrain_trigger: RetrainTrigger,
}

/// Retrain triggers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RetrainTrigger {
    PerformanceDegradation,
    DataDrift,
    TimeBased,
    Manual,
    Custom(String),
}

/// Model alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAlertConfig {
    /// Alert types
    pub types: Vec<ModelAlertType>,
    /// Notification channels
    pub channels: Vec<String>,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
}

/// Model alert types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelAlertType {
    PerformanceDegradation,
    DataDrift,
    ModelDrift,
    PredictionBias,
    ServiceUnavailable,
    Custom(String),
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Performance degradation threshold
    pub performance_degradation: f64,
    /// Data drift threshold
    pub data_drift: f64,
    /// Model drift threshold
    pub model_drift: f64,
    /// Bias threshold
    pub bias: f64,
}

/// Rollback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackConfig {
    /// Enable automatic rollback
    pub auto_rollback: bool,
    /// Rollback triggers
    pub triggers: Vec<RollbackTrigger>,
    /// Rollback strategy
    pub strategy: RollbackStrategy,
    /// Rollback timeout
    pub timeout: Duration,
}

/// Rollback triggers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RollbackTrigger {
    PerformanceDegradation,
    ErrorRateIncrease,
    LatencyIncrease,
    Manual,
    Custom(String),
}

/// Rollback strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RollbackStrategy {
    PreviousVersion,
    StableVersion,
    SpecificVersion,
    Custom(String),
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tasks: vec![],
            model_selection: ModelSelectionConfig::default(),
            hyperparameter_optimization: HyperparameterOptimizationConfig::default(),
            deployment: ModelDeploymentConfig::default(),
        }
    }
}

impl Default for ModelSelectionConfig {
    fn default() -> Self {
        Self {
            criteria: vec![SelectionCriterion::Accuracy],
            metrics: vec![EvaluationMetric::Accuracy],
            cross_validation: AutoMLCrossValidationConfig::default(),
            model_families: vec![ModelFamily::LinearModels, ModelFamily::TreeModels],
        }
    }
}

impl Default for AutoMLCrossValidationConfig {
    fn default() -> Self {
        Self {
            method: CrossValidationMethod::KFold,
            folds: 5,
            stratified: true,
            seed: Some(42),
        }
    }
}

impl Default for HyperparameterOptimizationConfig {
    fn default() -> Self {
        Self {
            method: OptimizationMethod::BayesianOptimization,
            search_space: SearchSpaceConfig::default(),
            budget: OptimizationBudget::default(),
            early_stopping: EarlyStoppingConfig::default(),
        }
    }
}

impl Default for SearchSpaceConfig {
    fn default() -> Self {
        Self {
            parameters: vec![],
            constraints: vec![],
        }
    }
}

impl Default for OptimizationBudget {
    fn default() -> Self {
        Self {
            max_evaluations: 100,
            max_time: Duration::from_secs(3600), // 1 hour
            max_parallel: 4,
            resources: ResourceLimits::default(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            cpu_cores: Some(4),
            memory_gb: Some(8),
            gpu_count: None,
        }
    }
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 10,
            min_improvement: 0.001,
            evaluation_frequency: 5,
        }
    }
}

impl Default for ModelDeploymentConfig {
    fn default() -> Self {
        Self {
            strategy: DeploymentStrategy::RollingUpdate,
            versioning: VersioningConfig::default(),
            monitoring: ModelMonitoringConfig::default(),
            rollback: RollbackConfig::default(),
        }
    }
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            scheme: VersionScheme::Semantic,
            registry: ModelRegistryConfig::default(),
            artifacts: ArtifactStorageConfig::default(),
        }
    }
}

impl Default for ModelRegistryConfig {
    fn default() -> Self {
        Self {
            registry_type: RegistryType::MLflow,
            connection: RegistryConnection::default(),
            metadata: MetadataConfig::default(),
        }
    }
}

impl Default for RegistryConnection {
    fn default() -> Self {
        Self {
            url: "http://localhost:5000".to_string(),
            auth: AuthConfig::default(),
            timeout: Duration::from_secs(30),
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            method: AuthMethod::Token,
            credentials: std::collections::HashMap::new(),
        }
    }
}

impl Default for MetadataConfig {
    fn default() -> Self {
        Self {
            track_hyperparameters: true,
            track_metrics: true,
            track_artifacts: true,
            custom_fields: vec![],
        }
    }
}

impl Default for ArtifactStorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Local,
            path: "./models".to_string(),
            compression: CompressionConfig::default(),
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Gzip,
            level: 6,
        }
    }
}

impl Default for ModelMonitoringConfig {
    fn default() -> Self {
        Self {
            performance: PerformanceMonitoringConfig::default(),
            data_drift: DataDriftConfig::default(),
            model_drift: ModelDriftConfig::default(),
            alerts: ModelAlertConfig::default(),
        }
    }
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            metrics: vec![PerformanceMetric::Accuracy, PerformanceMetric::Latency],
            frequency: Duration::from_secs(300), // 5 minutes
            thresholds: PerformanceThresholds::default(),
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            accuracy: Some(0.85),
            latency_ms: Some(1000.0),
            throughput: Some(100.0),
            memory_mb: Some(1024.0),
        }
    }
}

impl Default for DataDriftConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            methods: vec![DriftDetectionMethod::KolmogorovSmirnov],
            frequency: Duration::from_secs(3600), // hourly
            reference_window: Duration::from_secs(86400 * 7), // 1 week
        }
    }
}

impl Default for ModelDriftConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 0.1,
            window: Duration::from_secs(86400), // daily
            retrain_trigger: RetrainTrigger::PerformanceDegradation,
        }
    }
}

impl Default for ModelAlertConfig {
    fn default() -> Self {
        Self {
            types: vec![ModelAlertType::PerformanceDegradation],
            channels: vec!["email".to_string()],
            thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            performance_degradation: 0.05,
            data_drift: 0.1,
            model_drift: 0.1,
            bias: 0.05,
        }
    }
}

impl Default for RollbackConfig {
    fn default() -> Self {
        Self {
            auto_rollback: false,
            triggers: vec![RollbackTrigger::PerformanceDegradation],
            strategy: RollbackStrategy::PreviousVersion,
            timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}
