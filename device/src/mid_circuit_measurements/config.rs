//! Configuration types for mid-circuit measurements

use serde::{Deserialize, Serialize};

/// Comprehensive configuration for mid-circuit measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidCircuitConfig {
    /// Maximum allowed measurement latency (microseconds)
    pub max_measurement_latency: f64,
    /// Enable real-time classical processing
    pub enable_realtime_processing: bool,
    /// Buffer size for measurement results
    pub measurement_buffer_size: usize,
    /// Timeout for classical condition evaluation (microseconds)
    pub classical_timeout: f64,
    /// Enable measurement error mitigation
    pub enable_measurement_mitigation: bool,
    /// Parallel measurement execution
    pub enable_parallel_measurements: bool,
    /// Hardware-specific optimizations
    pub hardware_optimizations: HardwareOptimizations,
    /// Validation settings
    pub validation_config: ValidationConfig,
    /// Advanced SciRS2 analytics configuration
    pub analytics_config: AdvancedAnalyticsConfig,
    /// Adaptive measurement configuration
    pub adaptive_config: AdaptiveConfig,
    /// Machine learning optimization configuration
    pub ml_optimization_config: MLOptimizationConfig,
    /// Real-time prediction configuration
    pub prediction_config: PredictionConfig,
    /// Enable adaptive protocols
    pub enable_adaptive_protocols: bool,
}

impl Default for MidCircuitConfig {
    fn default() -> Self {
        Self {
            max_measurement_latency: 100.0, // 100 microseconds
            enable_realtime_processing: true,
            measurement_buffer_size: 1024,
            classical_timeout: 50.0, // 50 microseconds
            enable_measurement_mitigation: true,
            enable_parallel_measurements: true,
            hardware_optimizations: HardwareOptimizations::default(),
            validation_config: ValidationConfig::default(),
            analytics_config: AdvancedAnalyticsConfig::default(),
            adaptive_config: AdaptiveConfig::default(),
            ml_optimization_config: MLOptimizationConfig::default(),
            prediction_config: PredictionConfig::default(),
            enable_adaptive_protocols: true,
        }
    }
}

/// Hardware-specific optimizations for mid-circuit measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimizations {
    /// Batch measurement operations when possible
    pub batch_measurements: bool,
    /// Optimize measurement scheduling
    pub optimize_scheduling: bool,
    /// Use hardware-native measurement protocols
    pub use_native_protocols: bool,
    /// Enable measurement compression
    pub measurement_compression: bool,
    /// Pre-compile classical conditions
    pub precompile_conditions: bool,
}

impl Default for HardwareOptimizations {
    fn default() -> Self {
        Self {
            batch_measurements: true,
            optimize_scheduling: true,
            use_native_protocols: true,
            measurement_compression: false,
            precompile_conditions: true,
        }
    }
}

/// Validation configuration for mid-circuit measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validate backend measurement capabilities
    pub validate_capabilities: bool,
    /// Check measurement timing constraints
    pub check_timing_constraints: bool,
    /// Validate classical register sizes
    pub validate_register_sizes: bool,
    /// Check for measurement conflicts
    pub check_measurement_conflicts: bool,
    /// Validate feed-forward operations
    pub validate_feedforward: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            validate_capabilities: true,
            check_timing_constraints: true,
            validate_register_sizes: true,
            check_measurement_conflicts: true,
            validate_feedforward: true,
        }
    }
}

/// Advanced analytics configuration for mid-circuit measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedAnalyticsConfig {
    /// Enable real-time statistical analysis
    pub enable_realtime_stats: bool,
    /// Enable correlation analysis between measurements
    pub enable_correlation_analysis: bool,
    /// Enable time series analysis
    pub enable_time_series: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Statistical significance threshold
    pub significance_threshold: f64,
    /// Rolling window size for analysis
    pub analysis_window_size: usize,
    /// Enable distribution fitting
    pub enable_distribution_fitting: bool,
    /// Enable causal inference
    pub enable_causal_inference: bool,
}

impl Default for AdvancedAnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_realtime_stats: true,
            enable_correlation_analysis: true,
            enable_time_series: false,
            enable_anomaly_detection: true,
            significance_threshold: 0.05,
            analysis_window_size: 100,
            enable_distribution_fitting: false,
            enable_causal_inference: false,
        }
    }
}

/// Adaptive measurement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable adaptive measurement scheduling
    pub enable_adaptive_scheduling: bool,
    /// Enable dynamic threshold adjustment
    pub enable_dynamic_thresholds: bool,
    /// Enable measurement protocol adaptation
    pub enable_protocol_adaptation: bool,
    /// Adaptation learning rate
    pub learning_rate: f64,
    /// Baseline update rate for adaptive learning
    pub baseline_update_rate: f64,
    /// Drift threshold for detecting changes
    pub drift_threshold: f64,
    /// Adaptation window size
    pub adaptation_window: usize,
    /// Enable feedback-based optimization
    pub enable_feedback_optimization: bool,
    /// Performance improvement threshold for adaptation
    pub improvement_threshold: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_scheduling: false,
            enable_dynamic_thresholds: false,
            enable_protocol_adaptation: false,
            learning_rate: 0.01,
            baseline_update_rate: 0.1,
            drift_threshold: 0.1,
            adaptation_window: 50,
            enable_feedback_optimization: false,
            improvement_threshold: 0.05,
        }
    }
}

/// Machine learning optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLOptimizationConfig {
    /// Enable ML-driven measurement optimization
    pub enable_ml_optimization: bool,
    /// Model types to use
    pub model_types: Vec<MLModelType>,
    /// Training configuration
    pub training_config: MLTrainingConfig,
    /// Feature engineering settings
    pub feature_engineering: FeatureEngineeringConfig,
    /// Enable transfer learning
    pub enable_transfer_learning: bool,
    /// Online learning configuration
    pub online_learning: OnlineLearningConfig,
}

impl Default for MLOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_ml_optimization: false,
            model_types: vec![MLModelType::LinearRegression],
            training_config: MLTrainingConfig::default(),
            feature_engineering: FeatureEngineeringConfig::default(),
            enable_transfer_learning: false,
            online_learning: OnlineLearningConfig::default(),
        }
    }
}

/// ML model types for measurement optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MLModelType {
    LinearRegression,
    RandomForest {
        n_estimators: usize,
    },
    GradientBoosting {
        n_estimators: usize,
        learning_rate: f64,
    },
    NeuralNetwork {
        hidden_layers: Vec<usize>,
    },
    SupportVectorMachine {
        kernel: String,
    },
    GaussianProcess,
    ReinforcementLearning {
        algorithm: String,
    },
}

/// ML training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLTrainingConfig {
    /// Training data size
    pub training_size: usize,
    /// Validation split ratio
    pub validation_split: f64,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Learning rate schedule
    pub learning_rate_schedule: LearningRateSchedule,
    /// Regularization parameters
    pub regularization: RegularizationConfig,
}

impl Default for MLTrainingConfig {
    fn default() -> Self {
        Self {
            training_size: 1000,
            validation_split: 0.2,
            cv_folds: 5,
            early_stopping_patience: 10,
            learning_rate_schedule: LearningRateSchedule::Constant { rate: 0.001 },
            regularization: RegularizationConfig::default(),
        }
    }
}

/// Learning rate schedule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant { rate: f64 },
    Exponential { initial_rate: f64, decay_rate: f64 },
    StepWise { rates: Vec<(usize, f64)> },
    Adaptive { patience: usize, factor: f64 },
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_alpha: f64,
    /// L2 regularization strength
    pub l2_alpha: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Enable batch normalization
    pub batch_normalization: bool,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_alpha: 0.01,
            l2_alpha: 0.01,
            dropout_rate: 0.1,
            batch_normalization: false,
        }
    }
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Enable temporal features
    pub enable_temporal_features: bool,
    /// Enable statistical features
    pub enable_statistical_features: bool,
    /// Enable frequency domain features
    pub enable_frequency_features: bool,
    /// Enable interaction features
    pub enable_interaction_features: bool,
    /// Feature selection method
    pub selection_method: FeatureSelectionMethod,
    /// Maximum number of features
    pub max_features: usize,
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            enable_temporal_features: true,
            enable_statistical_features: true,
            enable_frequency_features: false,
            enable_interaction_features: false,
            selection_method: FeatureSelectionMethod::VarianceThreshold { threshold: 0.01 },
            max_features: 20,
        }
    }
}

/// Feature selection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    VarianceThreshold { threshold: f64 },
    UnivariateSelection { k_best: usize },
    RecursiveFeatureElimination { n_features: usize },
    LassoRegularization { alpha: f64 },
    MutualInformation { k_best: usize },
}

/// Online learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearningConfig {
    /// Enable online learning
    pub enable_online_learning: bool,
    /// Update frequency (number of samples)
    pub update_frequency: usize,
    /// Memory window size
    pub memory_window: usize,
    /// Concept drift detection
    pub drift_detection: DriftDetectionConfig,
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            enable_online_learning: false,
            update_frequency: 100,
            memory_window: 1000,
            drift_detection: DriftDetectionConfig::default(),
        }
    }
}

/// Concept drift detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionConfig {
    /// Enable drift detection
    pub enable_drift_detection: bool,
    /// Detection method
    pub detection_method: DriftDetectionMethod,
    /// Detection threshold
    pub detection_threshold: f64,
    /// Minimum samples for detection
    pub min_samples: usize,
}

impl Default for DriftDetectionConfig {
    fn default() -> Self {
        Self {
            enable_drift_detection: false,
            detection_method: DriftDetectionMethod::KolmogorovSmirnov,
            detection_threshold: 0.05,
            min_samples: 30,
        }
    }
}

/// Drift detection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DriftDetectionMethod {
    KolmogorovSmirnov,
    PageHinkley {
        delta: f64,
        lambda: f64,
    },
    ADWIN {
        delta: f64,
    },
    DDM {
        alpha_warning: f64,
        alpha_drift: f64,
    },
}

/// Real-time prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Enable predictive modeling
    pub enable_prediction: bool,
    /// Prediction horizon (number of measurements)
    pub prediction_horizon: usize,
    /// Minimum training samples required
    pub min_training_samples: usize,
    /// Sequence length for time series predictions
    pub sequence_length: usize,
    /// Time series analysis configuration
    pub time_series_config: TimeSeriesConfig,
    /// Uncertainty quantification
    pub uncertainty_config: UncertaintyConfig,
    /// Enable ensemble predictions
    pub enable_ensemble: bool,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            enable_prediction: false,
            prediction_horizon: 5,
            min_training_samples: 100,
            sequence_length: 10,
            time_series_config: TimeSeriesConfig::default(),
            uncertainty_config: UncertaintyConfig::default(),
            enable_ensemble: false,
        }
    }
}

/// Time series analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfig {
    /// Enable trend analysis
    pub enable_trend: bool,
    /// Enable seasonality detection
    pub enable_seasonality: bool,
    /// Seasonality period
    pub seasonality_period: usize,
    /// Enable autocorrelation analysis
    pub enable_autocorrelation: bool,
    /// Forecasting method
    pub forecasting_method: ForecastingMethod,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            enable_trend: true,
            enable_seasonality: false,
            seasonality_period: 24,
            enable_autocorrelation: true,
            forecasting_method: ForecastingMethod::ExponentialSmoothing {
                alpha: 0.3,
                beta: 0.1,
                gamma: 0.1,
            },
        }
    }
}

/// Forecasting methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ForecastingMethod {
    ARIMA {
        p: usize,
        d: usize,
        q: usize,
    },
    ExponentialSmoothing {
        alpha: f64,
        beta: f64,
        gamma: f64,
    },
    Prophet,
    LSTM {
        hidden_size: usize,
        num_layers: usize,
    },
    Transformer {
        d_model: usize,
        n_heads: usize,
    },
}

/// Uncertainty quantification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyConfig {
    /// Enable uncertainty quantification
    pub enable_uncertainty: bool,
    /// Confidence level
    pub confidence_level: f64,
    /// Uncertainty method
    pub uncertainty_method: UncertaintyMethod,
    /// Bootstrap samples for uncertainty estimation
    pub bootstrap_samples: usize,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            enable_uncertainty: false,
            confidence_level: 0.95,
            uncertainty_method: UncertaintyMethod::Bootstrap,
            bootstrap_samples: 1000,
        }
    }
}

/// Uncertainty quantification methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintyMethod {
    Bootstrap,
    BayesianInference,
    ConformalPrediction,
    GaussianProcess,
    EnsembleVariance,
}
