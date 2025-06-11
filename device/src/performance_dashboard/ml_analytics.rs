//! ML Analytics Configuration Types

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Machine learning analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLAnalyticsConfig {
    /// Enable ML analytics
    pub enable_ml_analytics: bool,
    /// Prediction models to use
    pub prediction_models: Vec<PredictionModel>,
    /// Feature engineering configuration
    pub feature_config: FeatureConfig,
    /// Model training configuration
    pub training_config: TrainingConfig,
    /// Model evaluation configuration
    pub evaluation_config: EvaluationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PredictionModel {
    LinearRegression,
    PolynomialRegression,
    RandomForest,
    NeuralNetwork,
    TimeSeriesARIMA,
    SupportVectorRegression,
    GradientBoosting,
    Ensemble,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub feature_selection_methods: Vec<FeatureSelectionMethod>,
    pub feature_engineering_rules: Vec<FeatureEngineeringRule>,
    pub dimensionality_reduction: Option<DimensionalityReduction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub training_data_size: usize,
    pub validation_split: f64,
    pub cross_validation_folds: usize,
    pub hyperparameter_tuning: bool,
    pub model_selection_criteria: ModelSelectionCriteria,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub evaluation_metrics: Vec<EvaluationMetric>,
    pub test_data_size: usize,
    pub evaluation_frequency: Duration,
    pub performance_tracking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FeatureSelectionMethod {
    VarianceThreshold,
    UnivariateSelection,
    RecursiveFeatureElimination,
    LassoRegularization,
    MutualInformation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FeatureEngineeringRule {
    PolynomialFeatures,
    InteractionTerms,
    MovingAverage,
    Differencing,
    LogTransform,
    Normalization,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DimensionalityReduction {
    PCA,
    ICA,
    LDA,
    TSNE,
    UMAP,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelSelectionCriteria {
    RMSE,
    MAE,
    R2Score,
    AIC,
    BIC,
    CrossValidationScore,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvaluationMetric {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    RMSE,
    MAE,
    R2Score,
}
