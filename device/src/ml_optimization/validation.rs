//! ML Validation Configuration Types

/// ML validation configuration
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationMethod {
    CrossValidation,
    HoldoutValidation,
    BootstrapValidation,
    TimeSeriesValidation,
    WalkForwardValidation,
}

/// Performance metrics
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone)]
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
