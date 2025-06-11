//! Configuration structures and enums for SciRS2 noise modeling

use serde::{Deserialize, Serialize};

/// Advanced noise modeling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciRS2NoiseConfig {
    /// Enable machine learning-based modeling
    pub enable_ml_modeling: bool,
    /// Enable spectral noise analysis
    pub enable_spectral_analysis: bool,
    /// Enable temporal correlation modeling
    pub enable_temporal_modeling: bool,
    /// Enable spatial correlation modeling
    pub enable_spatial_modeling: bool,
    /// Enable multi-level noise decomposition
    pub enable_multi_level_decomposition: bool,
    /// Statistical confidence level
    pub confidence_level: f64,
    /// Number of noise realizations for statistical analysis
    pub num_realizations: usize,
    /// Sampling frequency for temporal analysis (Hz)
    pub sampling_frequency: f64,
    /// Spatial correlation range (number of qubits)
    pub spatial_range: usize,
    /// Enable adaptive noise modeling
    pub enable_adaptive_modeling: bool,
    /// Model validation configuration
    pub validation_config: ValidationConfig,
}

/// Model validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Train/test split ratio
    pub test_ratio: f64,
    /// Enable bootstrap validation
    pub enable_bootstrap: bool,
    /// Number of bootstrap samples
    pub bootstrap_samples: usize,
    /// Validation metrics to compute
    pub metrics: Vec<ValidationMetric>,
}

/// Validation metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationMetric {
    RMSE,
    MAE,
    R2,
    LogLikelihood,
    AIC,
    BIC,
    KLDivergence,
    WassersteinDistance,
}

/// Distribution types for noise characterization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributionType {
    Normal,
    LogNormal,
    Gamma,
    Exponential,
    Poisson,
    Mixture,
    Empirical,
    Custom(String),
}

/// Noise color classifications
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NoiseColor {
    White,
    Pink,
    Brown,
    Blue,
    Violet,
    Custom { exponent: f64 },
}

/// AR model types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ARModelType {
    AR,
    MA,
    ARMA,
    ARIMA,
    GARCH,
    Custom(String),
}

/// Machine learning model types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MLModelType {
    GaussianProcess,
    NeuralNetwork,
    RandomForest,
    SupportVector,
    Ensemble,
    Custom(String),
}

/// Spatial interpolation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpatialInterpolation {
    Kriging,
    RadialBasisFunction,
    InverseDistance,
    NearestNeighbor,
    Spline,
    Custom(String),
}

/// Decomposition methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecompositionMethod {
    PCA,
    ICA,
    NMF,
    FastICA,
    Wavelet,
    Custom(String),
}

impl Default for SciRS2NoiseConfig {
    fn default() -> Self {
        Self {
            enable_ml_modeling: true,
            enable_spectral_analysis: true,
            enable_temporal_modeling: true,
            enable_spatial_modeling: true,
            enable_multi_level_decomposition: true,
            confidence_level: 0.95,
            num_realizations: 1000,
            sampling_frequency: 1e9, // 1 GHz
            spatial_range: 5,
            enable_adaptive_modeling: true,
            validation_config: ValidationConfig {
                cv_folds: 5,
                test_ratio: 0.2,
                enable_bootstrap: true,
                bootstrap_samples: 100,
                metrics: vec![
                    ValidationMetric::RMSE,
                    ValidationMetric::R2,
                    ValidationMetric::LogLikelihood,
                ],
            },
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            cv_folds: 5,
            test_ratio: 0.2,
            enable_bootstrap: true,
            bootstrap_samples: 100,
            metrics: vec![ValidationMetric::RMSE, ValidationMetric::R2],
        }
    }
}
