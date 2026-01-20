//! Performance prediction configurations

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Performance prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePredictionConfig {
    /// Enable prediction
    pub enabled: bool,
    /// Prediction algorithms
    pub algorithms: Vec<PerformancePredictionAlgorithm>,
    /// Prediction window
    pub window: Duration,
    /// Update frequency
    pub update_frequency: Duration,
}

/// Performance prediction algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformancePredictionAlgorithm {
    LinearRegression,
    ARIMA,
    LSTM,
    RandomForest,
    EnsembleMethod,
    Custom(String),
}
