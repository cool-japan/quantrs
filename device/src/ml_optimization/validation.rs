//! ML Validation Configuration Types

use serde::{Deserialize, Serialize};

/// ML validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLValidationConfig {
    /// Validation methods
    pub validation_methods: Vec<ValidationMethod>,
    /// Performance metrics
    pub performance_metrics: Vec<PerformanceMetric>,
    /// Statistical significance testing
    pub statistical_testing: bool,
    /// Robustness testing
    pub robustness_testing: RobustnessTestingConfig,
    /// Fairness evaluation
    pub fairness_evaluation: bool,
}

/// Validation methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationMethod {
    CrossValidation,
    HoldoutValidation,
    BootstrapValidation,
    TimeSeriesValidation,
    WalkForwardValidation,
}

/// Performance metrics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceMetric {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC,
    MAE,
    MSE,
    RMSE,
    R2Score,
    LogLoss,
}

/// Robustness testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessTestingConfig {
    /// Enable robustness testing
    pub enable_testing: bool,
    /// Adversarial testing
    pub adversarial_testing: bool,
    /// Distribution shift testing
    pub distribution_shift_testing: bool,
    /// Noise sensitivity testing
    pub noise_sensitivity_testing: bool,
    /// Fairness testing
    pub fairness_testing: bool,
}

// Additional config types for QEC compatibility

/// Validation configuration (alias for compatibility)
pub type ValidationConfig = MLValidationConfig;

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Inference batch size
    pub batch_size: usize,
    /// Inference timeout
    pub timeout: std::time::Duration,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// Inference precision
    pub precision: InferencePrecision,
    /// Caching configuration
    pub caching: CachingConfig,
}

/// Model management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManagementConfig {
    /// Model versioning
    pub versioning: bool,
    /// Model storage path
    pub storage_path: String,
    /// Model lifecycle policy
    pub lifecycle_policy: ModelLifecyclePolicy,
    /// Model monitoring
    pub monitoring: ModelMonitoringConfig,
    /// Model deployment strategy
    pub deployment_strategy: DeploymentStrategy,
}

/// Inference precision options
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferencePrecision {
    Float32,
    Float64,
    Mixed,
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable caching
    pub enable: bool,
    /// Cache size limit (MB)
    pub size_limit_mb: usize,
    /// Cache expiration time
    pub expiration: std::time::Duration,
}

/// Model lifecycle policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLifecyclePolicy {
    /// Maximum model age
    pub max_age: std::time::Duration,
    /// Auto-retirement threshold
    pub retirement_threshold: f64,
    /// Backup strategy
    pub backup_strategy: BackupStrategy,
}

/// Model monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMonitoringConfig {
    /// Enable performance monitoring
    pub performance_monitoring: bool,
    /// Enable drift detection
    pub drift_detection: bool,
    /// Monitoring frequency
    pub frequency: std::time::Duration,
}

/// Deployment strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    BlueGreen,
    Canary,
    Rolling,
    Immediate,
}

/// Backup strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupStrategy {
    Daily,
    Weekly,
    OnDemand,
    Never,
}

// Default implementations

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            validation_methods: vec![ValidationMethod::CrossValidation],
            performance_metrics: vec![PerformanceMetric::Accuracy],
            statistical_testing: true,
            robustness_testing: RobustnessTestingConfig {
                enable_testing: true,
                adversarial_testing: false,
                distribution_shift_testing: true,
                noise_sensitivity_testing: true,
                fairness_testing: false,
            },
            fairness_evaluation: false,
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            timeout: std::time::Duration::from_secs(30),
            use_gpu: false,
            precision: InferencePrecision::Float32,
            caching: CachingConfig {
                enable: true,
                size_limit_mb: 1024,
                expiration: std::time::Duration::from_secs(3600),
            },
        }
    }
}

impl Default for ModelManagementConfig {
    fn default() -> Self {
        Self {
            versioning: true,
            storage_path: "/tmp/models".to_string(),
            lifecycle_policy: ModelLifecyclePolicy {
                max_age: std::time::Duration::from_secs(30 * 24 * 3600), // 30 days
                retirement_threshold: 0.8,
                backup_strategy: BackupStrategy::Daily,
            },
            monitoring: ModelMonitoringConfig {
                performance_monitoring: true,
                drift_detection: true,
                frequency: std::time::Duration::from_secs(3600),
            },
            deployment_strategy: DeploymentStrategy::Rolling,
        }
    }
}
