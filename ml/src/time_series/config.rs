//! Configuration structures for quantum time series forecasting

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantum time series forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTimeSeriesConfig {
    /// Number of qubits for quantum processing
    pub num_qubits: usize,

    /// Forecasting model type
    pub model_type: TimeSeriesModel,

    /// Input window size
    pub window_size: usize,

    /// Forecast horizon
    pub forecast_horizon: usize,

    /// Feature engineering configuration
    pub feature_config: FeatureEngineeringConfig,

    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,

    /// Seasonality configuration
    pub seasonality_config: SeasonalityConfig,

    /// Ensemble configuration
    pub ensemble_config: Option<EnsembleConfig>,
}

/// Time series forecasting models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeSeriesModel {
    /// Quantum ARIMA model
    QuantumARIMA {
        p: usize,                                       // autoregressive order
        d: usize,                                       // differencing order
        q: usize,                                       // moving average order
        seasonal: Option<(usize, usize, usize, usize)>, // (P, D, Q, period)
    },

    /// Quantum LSTM for time series
    QuantumLSTM {
        hidden_size: usize,
        num_layers: usize,
        dropout: f64,
    },

    /// Quantum Transformer for time series
    QuantumTransformerTS {
        model_dim: usize,
        num_heads: usize,
        num_layers: usize,
    },

    /// Quantum State Space Model
    QuantumStateSpace {
        state_dim: usize,
        emission_dim: usize,
        transition_type: TransitionType,
    },

    /// Quantum Prophet (inspired by Facebook Prophet)
    QuantumProphet {
        growth_type: GrowthType,
        changepoint_prior_scale: f64,
        seasonality_prior_scale: f64,
    },

    /// Quantum Neural Prophet
    QuantumNeuralProphet {
        hidden_layers: Vec<usize>,
        ar_order: usize,
        ma_order: usize,
    },

    /// Quantum Temporal Fusion Transformer
    QuantumTFT {
        state_size: usize,
        attention_heads: usize,
        num_layers: usize,
    },
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Use quantum Fourier features
    pub quantum_fourier_features: bool,

    /// Lag features
    pub lag_features: Vec<usize>,

    /// Rolling statistics window sizes
    pub rolling_windows: Vec<usize>,

    /// Wavelet decomposition
    pub wavelet_decomposition: bool,

    /// Quantum feature extraction
    pub quantum_features: bool,

    /// Interaction features
    pub interaction_features: bool,
}

/// Seasonality configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityConfig {
    /// Daily seasonality
    pub daily: Option<usize>,

    /// Weekly seasonality
    pub weekly: Option<usize>,

    /// Monthly seasonality
    pub monthly: Option<usize>,

    /// Yearly seasonality
    pub yearly: Option<usize>,

    /// Custom seasonality periods
    pub custom_periods: Vec<usize>,

    /// Quantum seasonal decomposition
    pub quantum_decomposition: bool,
}

/// Quantum enhancement levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumEnhancementLevel {
    /// Minimal quantum processing
    Low,

    /// Balanced quantum-classical
    Medium,

    /// Maximum quantum advantage
    High,

    /// Custom quantum configuration
    Custom {
        quantum_layers: Vec<usize>,
        entanglement_strength: f64,
        measurement_strategy: MeasurementStrategy,
    },
}

/// Measurement strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementStrategy {
    /// Standard computational basis
    Computational,

    /// Hadamard basis
    Hadamard,

    /// Custom basis rotation
    Custom(Array2<f64>),

    /// Adaptive measurement
    Adaptive,
}

/// State transition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    /// Linear transition
    Linear,

    /// Nonlinear quantum transition
    NonlinearQuantum,

    /// Recurrent transition
    Recurrent,

    /// Attention-based transition
    Attention,
}

/// Growth types for trend modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrowthType {
    /// Linear growth
    Linear,

    /// Logistic growth with capacity
    Logistic(f64),

    /// Flat (no growth)
    Flat,

    /// Quantum superposition of growth modes
    QuantumSuperposition,
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Ensemble method
    pub method: EnsembleMethod,

    /// Number of models in ensemble
    pub num_models: usize,

    /// Model diversity strategy
    pub diversity_strategy: DiversityStrategy,

    /// Quantum voting mechanism
    pub quantum_voting: bool,
}

/// Ensemble methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    /// Simple averaging
    Average,

    /// Weighted average
    Weighted(Vec<f64>),

    /// Quantum superposition ensemble
    QuantumSuperposition,

    /// Stacking with meta-learner
    Stacking,

    /// Bayesian model averaging
    BayesianAverage,
}

/// Diversity strategies for ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityStrategy {
    /// Random initialization
    RandomInit,

    /// Bootstrap sampling
    Bootstrap,

    /// Feature bagging
    FeatureBagging,

    /// Quantum diversity
    QuantumDiversity,
}

/// Wavelet types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WaveletType {
    Haar,
    Daubechies(usize),
    Morlet,
    Mexican,
    Quantum,
}

/// Anomaly types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Point anomaly
    Point,

    /// Contextual anomaly
    Contextual,

    /// Collective anomaly
    Collective,

    /// Quantum uncertainty anomaly
    QuantumUncertainty,

    /// Changepoint
    Changepoint,
}

// Configuration implementations
impl Default for QuantumTimeSeriesConfig {
    fn default() -> Self {
        Self {
            num_qubits: 10,
            model_type: TimeSeriesModel::QuantumLSTM {
                hidden_size: 64,
                num_layers: 2,
                dropout: 0.1,
            },
            window_size: 30,
            forecast_horizon: 7,
            feature_config: FeatureEngineeringConfig::default(),
            quantum_enhancement: QuantumEnhancementLevel::Medium,
            seasonality_config: SeasonalityConfig::default(),
            ensemble_config: None,
        }
    }
}

impl QuantumTimeSeriesConfig {
    /// Configuration for financial time series
    pub fn financial(forecast_horizon: usize) -> Self {
        Self {
            num_qubits: 12,
            model_type: TimeSeriesModel::QuantumTFT {
                state_size: 128,
                attention_heads: 8,
                num_layers: 4,
            },
            window_size: 60,
            forecast_horizon,
            feature_config: FeatureEngineeringConfig::financial(),
            quantum_enhancement: QuantumEnhancementLevel::High,
            seasonality_config: SeasonalityConfig::financial(),
            ensemble_config: Some(EnsembleConfig::default()),
        }
    }

    /// Configuration for IoT/sensor data
    pub fn iot_sensor(sampling_rate: usize) -> Self {
        Self {
            num_qubits: 14,
            model_type: TimeSeriesModel::QuantumStateSpace {
                state_dim: 32,
                emission_dim: 16,
                transition_type: TransitionType::NonlinearQuantum,
            },
            window_size: sampling_rate * 60,      // 1 hour window
            forecast_horizon: sampling_rate * 10, // 10 minute forecast
            feature_config: FeatureEngineeringConfig::iot(),
            quantum_enhancement: QuantumEnhancementLevel::High,
            seasonality_config: SeasonalityConfig::hourly(),
            ensemble_config: None,
        }
    }

    /// Configuration for demand forecasting
    pub fn demand_forecasting() -> Self {
        Self {
            num_qubits: 12,
            model_type: TimeSeriesModel::QuantumProphet {
                growth_type: GrowthType::Linear,
                changepoint_prior_scale: 0.05,
                seasonality_prior_scale: 10.0,
            },
            window_size: 365,     // 1 year of daily data
            forecast_horizon: 30, // 1 month forecast
            feature_config: FeatureEngineeringConfig::retail(),
            quantum_enhancement: QuantumEnhancementLevel::Medium,
            seasonality_config: SeasonalityConfig::retail(),
            ensemble_config: Some(EnsembleConfig::stacking()),
        }
    }
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            quantum_fourier_features: true,
            lag_features: vec![1, 7, 14, 30],
            rolling_windows: vec![7, 14, 30],
            wavelet_decomposition: false,
            quantum_features: true,
            interaction_features: false,
        }
    }
}

impl FeatureEngineeringConfig {
    /// Financial data configuration
    pub fn financial() -> Self {
        Self {
            quantum_fourier_features: true,
            lag_features: vec![1, 5, 10, 20, 60], // Various trading periods
            rolling_windows: vec![5, 10, 20, 60], // Moving averages
            wavelet_decomposition: true,
            quantum_features: true,
            interaction_features: true,
        }
    }

    /// IoT sensor configuration
    pub fn iot() -> Self {
        Self {
            quantum_fourier_features: true,
            lag_features: vec![1, 6, 12, 24], // Hourly patterns
            rolling_windows: vec![6, 12, 24, 48],
            wavelet_decomposition: true,
            quantum_features: true,
            interaction_features: false,
        }
    }

    /// Retail/demand configuration
    pub fn retail() -> Self {
        Self {
            quantum_fourier_features: false,
            lag_features: vec![1, 7, 14, 28, 365], // Daily, weekly, monthly, yearly
            rolling_windows: vec![7, 14, 28],
            wavelet_decomposition: false,
            quantum_features: true,
            interaction_features: true,
        }
    }
}

impl Default for SeasonalityConfig {
    fn default() -> Self {
        Self {
            daily: None,
            weekly: Some(7),
            monthly: None,
            yearly: None,
            custom_periods: Vec::new(),
            quantum_decomposition: true,
        }
    }
}

impl SeasonalityConfig {
    /// Financial seasonality
    pub fn financial() -> Self {
        Self {
            daily: Some(1),
            weekly: Some(5),          // Trading days
            monthly: Some(21),        // Trading month
            yearly: Some(252),        // Trading year
            custom_periods: vec![63], // Quarterly
            quantum_decomposition: true,
        }
    }

    /// Hourly seasonality for IoT
    pub fn hourly() -> Self {
        Self {
            daily: Some(24),
            weekly: Some(168), // 24 * 7
            monthly: None,
            yearly: None,
            custom_periods: Vec::new(),
            quantum_decomposition: true,
        }
    }

    /// Retail seasonality
    pub fn retail() -> Self {
        Self {
            daily: None,
            weekly: Some(7),
            monthly: Some(30),
            yearly: Some(365),
            custom_periods: vec![90, 180], // Quarterly, semi-annual
            quantum_decomposition: true,
        }
    }

    /// Check if any seasonality is configured
    pub fn has_seasonality(&self) -> bool {
        self.daily.is_some()
            || self.weekly.is_some()
            || self.monthly.is_some()
            || self.yearly.is_some()
            || !self.custom_periods.is_empty()
    }
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            method: EnsembleMethod::Average,
            num_models: 3,
            diversity_strategy: DiversityStrategy::RandomInit,
            quantum_voting: true,
        }
    }
}

impl EnsembleConfig {
    /// Stacking ensemble
    pub fn stacking() -> Self {
        Self {
            method: EnsembleMethod::Stacking,
            num_models: 5,
            diversity_strategy: DiversityStrategy::FeatureBagging,
            quantum_voting: true,
        }
    }
}
