//! Machine learning integration for advanced mapping

use super::*;

/// Machine learning configuration for mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLMappingConfig {
    /// Enable ML-enhanced mapping
    pub enable_ml: bool,
    /// Model types to use
    pub model_types: Vec<MLModelType>,
    /// Feature engineering configuration
    pub feature_config: FeatureConfig,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Prediction configuration
    pub prediction_config: PredictionConfig,
    /// Transfer learning configuration
    pub transfer_learning: TransferLearningConfig,
}

/// Feature configuration for ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Enable graph structural features
    pub enable_structural: bool,
    /// Enable temporal features
    pub enable_temporal: bool,
    /// Enable hardware-specific features
    pub enable_hardware: bool,
    /// Enable circuit-specific features
    pub enable_circuit: bool,
    /// Feature selection method
    pub selection_method: FeatureSelectionMethod,
    /// Maximum number of features
    pub max_features: usize,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Validation split
    pub validation_split: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Regularization parameters
    pub regularization: RegularizationParams,
}

/// Regularization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationParams {
    /// L1 regularization
    pub l1_lambda: f64,
    /// L2 regularization
    pub l2_lambda: f64,
    /// Dropout rate
    pub dropout: f64,
    /// Batch normalization
    pub batch_norm: bool,
}

/// Prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Ensemble size for prediction
    pub ensemble_size: usize,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Use uncertainty estimation
    pub use_uncertainty_estimation: bool,
    /// Monte Carlo samples for uncertainty
    pub monte_carlo_samples: usize,
    /// Temperature scaling for calibration
    pub temperature_scaling: bool,
    /// Calibration method
    pub calibration_method: CalibrationMethod,
}

/// Transfer learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningConfig {
    /// Enable transfer learning
    pub enable_transfer: bool,
    /// Source domains for transfer
    pub source_domains: Vec<String>,
    /// Domain adaptation method
    pub adaptation_method: DomainAdaptationMethod,
    /// Fine-tuning configuration
    pub fine_tuning: FineTuningConfig,
}

/// Fine-tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningConfig {
    /// Layers to freeze during initial training
    pub freeze_layers: Vec<usize>,
    /// Epochs after which to unfreeze layers
    pub unfreeze_after_epochs: usize,
    /// Reduced learning rate for fine-tuning
    pub reduced_learning_rate: f64,
}

/// Performance predictions from ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePredictions {
    /// Predicted swap count
    pub predicted_swaps: f64,
    /// Predicted execution time
    pub predicted_time: f64,
    /// Predicted fidelity
    pub predicted_fidelity: f64,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Model uncertainty estimates
    pub uncertainty_estimates: HashMap<String, f64>,
}

/// ML model performance results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPerformanceResult {
    /// Model accuracy metrics
    pub model_accuracy: HashMap<String, f64>,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f64>,
    /// Prediction reliability
    pub prediction_reliability: f64,
    /// Model training history
    pub training_history: Vec<TrainingEpoch>,
}

/// Training epoch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEpoch {
    /// Epoch number
    pub epoch: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Learning rate
    pub learning_rate: f64,
}

/// Adaptive mapping insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveMappingInsights {
    /// Learning progress metrics
    pub learning_progress: HashMap<String, f64>,
    /// Adaptation effectiveness
    pub adaptation_effectiveness: HashMap<String, f64>,
    /// Performance trend analysis
    pub performance_trends: HashMap<String, Vec<f64>>,
    /// Recommended parameter adjustments
    pub recommended_adjustments: Vec<ParameterAdjustment>,
}

/// Parameter adjustment recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterAdjustment {
    /// Parameter name
    pub parameter: String,
    /// Current value
    pub current_value: f64,
    /// Recommended value
    pub recommended_value: f64,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Confidence in recommendation
    pub confidence: f64,
}