//! ML Optimization Configuration Types

use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for ML-driven optimization with SciRS2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLOptimizationConfig {
    /// Enable ML optimization
    pub enable_optimization: bool,
    /// ML model configuration
    pub model_config: MLModelConfig,
    /// Circuit feature extraction settings
    pub feature_extraction: FeatureExtractionConfig,
    /// Hardware prediction settings
    pub hardware_prediction: HardwarePredictionConfig,
    /// Online learning configuration
    pub online_learning: OnlineLearningConfig,
    /// Transfer learning settings
    pub transfer_learning: TransferLearningConfig,
    /// Ensemble methods configuration
    pub ensemble_config: EnsembleConfig,
    /// Optimization strategy settings
    pub optimization_strategy: OptimizationStrategyConfig,
    /// Validation and testing configuration
    pub validation_config: MLValidationConfig,
    /// Performance monitoring settings
    pub monitoring_config: MLMonitoringConfig,
}

/// ML model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelConfig {
    /// Primary ML algorithms to use
    pub primary_algorithms: Vec<MLAlgorithm>,
    /// Fallback algorithms
    pub fallback_algorithms: Vec<MLAlgorithm>,
    /// Model hyperparameters
    pub hyperparameters: HashMap<String, MLHyperparameter>,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Model selection strategy
    pub model_selection: ModelSelectionStrategy,
    /// Regularization settings
    pub regularization: RegularizationConfig,
}

/// ML algorithms available
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MLAlgorithm {
    /// Deep Neural Network
    DeepNeuralNetwork,
    /// Random Forest
    RandomForest,
    /// Gradient Boosting
    GradientBoosting,
    /// Support Vector Machine
    SupportVectorMachine,
    /// Gaussian Process
    GaussianProcess,
    /// Ensemble Methods
    EnsembleMethods,
    /// Reinforcement Learning
    ReinforcementLearning,
    /// Quantum Neural Network
    QuantumNeuralNetwork,
    /// Graph Neural Network
    GraphNeuralNetwork,
    /// Transformer Networks
    TransformerNetwork,
    /// Bayesian Networks
    BayesianNetwork,
    /// Custom algorithm
    Custom(String),
}

/// ML hyperparameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLHyperparameter {
    pub parameter_type: HyperparameterType,
    pub value: HyperparameterValue,
    pub search_space: Option<HyperparameterSearchSpace>,
    pub importance: f64,
}

/// Hyperparameter types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HyperparameterType {
    Integer,
    Float,
    Categorical,
    Boolean,
    Array,
}

/// Hyperparameter values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperparameterValue {
    Integer(i64),
    Float(f64),
    Categorical(String),
    Boolean(bool),
    Array(Array1<f64>),
}

/// Hyperparameter search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperparameterSearchSpace {
    IntegerRange(i64, i64),
    FloatRange(f64, f64),
    CategoricalOptions(Vec<String>),
    BooleanOptions,
    ArrayBounds(Vec<(f64, f64)>),
}

/// Model selection strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    CrossValidation,
    HoldoutValidation,
    BootstrapValidation,
    TimeSeriesValidation,
    BayesianModelSelection,
    EnsembleSelection,
}

// Forward declarations for types that will be defined in other modules
use super::{
    ensemble::EnsembleConfig,
    features::FeatureExtractionConfig,
    hardware::HardwarePredictionConfig,
    monitoring::MLMonitoringConfig,
    online_learning::OnlineLearningConfig,
    optimization::OptimizationStrategyConfig,
    training::{RegularizationConfig, TrainingConfig},
    transfer_learning::TransferLearningConfig,
    validation::MLValidationConfig,
};
