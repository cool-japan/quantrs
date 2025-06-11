//! Quantum Time Series Forecasting - Modular Implementation
//!
//! This module provides a comprehensive framework for quantum-enhanced time series forecasting
//! that leverages quantum computing principles for improved prediction accuracy,
//! pattern recognition, and temporal modeling in sequential data.
//!
//! The module is organized into focused submodules for maintainability and clarity:
//! - `config`: Configuration structures and enums for all time series models
//! - `models`: Time series model implementations (ARIMA, LSTM, Transformer, etc.)
//! - `features`: Quantum feature extraction and engineering
//! - `ensemble`: Ensemble methods and quantum voting mechanisms
//! - `decomposition`: Seasonal and trend decomposition with quantum enhancement
//! - `forecaster`: Main forecasting coordinator and execution logic
//! - `metrics`: Performance metrics and evaluation tools
//! - `utils`: Utility functions and synthetic data generation

pub mod config;
pub mod decomposition;
pub mod ensemble;
pub mod features;
pub mod forecaster;
pub mod metrics;
pub mod models;
pub mod utils;

// Re-export main types for backward compatibility
pub use config::*;
pub use decomposition::{
    ChangePoint, ChangeType, ChangepointAlgorithm, QuantumChangepointDetector,
    QuantumResidualAnalyzer, QuantumSeasonalDecomposer, QuantumTrendExtractor, ResidualStatistics,
    TrendParameters,
};
pub use ensemble::*;
pub use features::*;
pub use forecaster::*;
pub use metrics::*; // This will provide AnomalyPoint
pub use models::*;
pub use utils::*;

// Convenient type aliases
pub type Result<T> = crate::error::Result<T>;
pub type MLError = crate::error::MLError;

/// Main quantum time series forecasting entry point
pub fn create_forecaster(config: QuantumTimeSeriesConfig) -> Result<QuantumTimeSeriesForecaster> {
    QuantumTimeSeriesForecaster::new(config)
}

/// Create financial forecasting configuration
pub fn financial_config(forecast_horizon: usize) -> QuantumTimeSeriesConfig {
    QuantumTimeSeriesConfig::financial(forecast_horizon)
}

/// Create IoT sensor forecasting configuration  
pub fn iot_config(sampling_rate: usize) -> QuantumTimeSeriesConfig {
    QuantumTimeSeriesConfig::iot_sensor(sampling_rate)
}

/// Create demand forecasting configuration
pub fn demand_config() -> QuantumTimeSeriesConfig {
    QuantumTimeSeriesConfig::demand_forecasting()
}
