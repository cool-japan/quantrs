//! Evaluation Configuration
//!
//! This module defines evaluation strategies, metrics, and cross-validation configurations.

/// Evaluation configuration
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Cross-validation strategy
    pub cv_strategy: CrossValidationStrategy,

    /// Evaluation metrics
    pub metrics: Vec<EvaluationMetric>,

    /// Test set size
    pub test_size: f64,

    /// Validation set size
    pub validation_size: f64,

    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Cross-validation strategies
#[derive(Debug, Clone)]
pub enum CrossValidationStrategy {
    KFold { k: usize },
    StratifiedKFold { k: usize },
    TimeSeriesSplit { n_splits: usize },
    LeaveOneOut,
    Bootstrap { n_bootstrap: usize },
    HoldOut { test_size: f64 },
}

/// Evaluation metrics
#[derive(Debug, Clone)]
pub enum EvaluationMetric {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC,
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    QuantumAdvantage,
    ResourceEfficiency,
    InferenceTime,
    TrainingTime,
    ModelComplexity,
    Robustness,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            cv_strategy: CrossValidationStrategy::KFold { k: 5 },
            metrics: vec![EvaluationMetric::Accuracy, EvaluationMetric::F1Score],
            test_size: 0.2,
            validation_size: 0.2,
            random_seed: Some(42),
        }
    }
}

impl EvaluationConfig {
    /// Quick evaluation configuration
    pub fn quick() -> Self {
        Self {
            cv_strategy: CrossValidationStrategy::HoldOut { test_size: 0.3 },
            metrics: vec![EvaluationMetric::Accuracy],
            test_size: 0.3,
            validation_size: 0.2,
            random_seed: Some(42),
        }
    }

    /// Rigorous evaluation configuration
    pub fn rigorous() -> Self {
        Self {
            cv_strategy: CrossValidationStrategy::StratifiedKFold { k: 10 },
            metrics: vec![
                EvaluationMetric::Accuracy,
                EvaluationMetric::Precision,
                EvaluationMetric::Recall,
                EvaluationMetric::F1Score,
                EvaluationMetric::AUC,
                EvaluationMetric::QuantumAdvantage,
                EvaluationMetric::ResourceEfficiency,
            ],
            test_size: 0.15,
            validation_size: 0.15,
            random_seed: Some(42),
        }
    }

    /// Production evaluation configuration
    pub fn production() -> Self {
        Self {
            cv_strategy: CrossValidationStrategy::StratifiedKFold { k: 5 },
            metrics: vec![
                EvaluationMetric::Accuracy,
                EvaluationMetric::F1Score,
                EvaluationMetric::InferenceTime,
                EvaluationMetric::Robustness,
            ],
            test_size: 0.2,
            validation_size: 0.2,
            random_seed: Some(42),
        }
    }

    /// Time series evaluation configuration
    pub fn time_series() -> Self {
        Self {
            cv_strategy: CrossValidationStrategy::TimeSeriesSplit { n_splits: 5 },
            metrics: vec![
                EvaluationMetric::MeanAbsoluteError,
                EvaluationMetric::MeanSquaredError,
                EvaluationMetric::R2Score,
            ],
            test_size: 0.2,
            validation_size: 0.2,
            random_seed: Some(42),
        }
    }

    /// Regression evaluation configuration
    pub fn regression() -> Self {
        Self {
            cv_strategy: CrossValidationStrategy::KFold { k: 5 },
            metrics: vec![
                EvaluationMetric::MeanSquaredError,
                EvaluationMetric::MeanAbsoluteError,
                EvaluationMetric::R2Score,
            ],
            test_size: 0.2,
            validation_size: 0.2,
            random_seed: Some(42),
        }
    }

    /// Quantum-focused evaluation configuration
    pub fn quantum_focused() -> Self {
        Self {
            cv_strategy: CrossValidationStrategy::KFold { k: 5 },
            metrics: vec![
                EvaluationMetric::Accuracy,
                EvaluationMetric::QuantumAdvantage,
                EvaluationMetric::ResourceEfficiency,
                EvaluationMetric::ModelComplexity,
            ],
            test_size: 0.2,
            validation_size: 0.2,
            random_seed: Some(42),
        }
    }
}

impl EvaluationMetric {
    /// Check if this metric should be maximized
    pub fn is_maximization(&self) -> bool {
        matches!(
            self,
            EvaluationMetric::Accuracy
                | EvaluationMetric::Precision
                | EvaluationMetric::Recall
                | EvaluationMetric::F1Score
                | EvaluationMetric::AUC
                | EvaluationMetric::R2Score
                | EvaluationMetric::QuantumAdvantage
                | EvaluationMetric::ResourceEfficiency
                | EvaluationMetric::Robustness
        )
    }

    /// Get the metric name as a string
    pub fn name(&self) -> &'static str {
        match self {
            EvaluationMetric::Accuracy => "accuracy",
            EvaluationMetric::Precision => "precision",
            EvaluationMetric::Recall => "recall",
            EvaluationMetric::F1Score => "f1_score",
            EvaluationMetric::AUC => "auc",
            EvaluationMetric::MeanSquaredError => "mse",
            EvaluationMetric::MeanAbsoluteError => "mae",
            EvaluationMetric::R2Score => "r2_score",
            EvaluationMetric::QuantumAdvantage => "quantum_advantage",
            EvaluationMetric::ResourceEfficiency => "resource_efficiency",
            EvaluationMetric::InferenceTime => "inference_time",
            EvaluationMetric::TrainingTime => "training_time",
            EvaluationMetric::ModelComplexity => "model_complexity",
            EvaluationMetric::Robustness => "robustness",
        }
    }
}

impl CrossValidationStrategy {
    /// Get the number of folds/splits for this strategy
    pub fn n_splits(&self) -> usize {
        match self {
            CrossValidationStrategy::KFold { k } => *k,
            CrossValidationStrategy::StratifiedKFold { k } => *k,
            CrossValidationStrategy::TimeSeriesSplit { n_splits } => *n_splits,
            CrossValidationStrategy::LeaveOneOut => 1, // Special case
            CrossValidationStrategy::Bootstrap { n_bootstrap } => *n_bootstrap,
            CrossValidationStrategy::HoldOut { .. } => 1,
        }
    }

    /// Check if this strategy preserves class distribution
    pub fn is_stratified(&self) -> bool {
        matches!(self, CrossValidationStrategy::StratifiedKFold { .. })
    }

    /// Check if this strategy is suitable for time series data
    pub fn is_time_series_safe(&self) -> bool {
        matches!(
            self,
            CrossValidationStrategy::TimeSeriesSplit { .. }
                | CrossValidationStrategy::HoldOut { .. }
        )
    }
}
