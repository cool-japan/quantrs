//! Time Series Modeling for Quantum Reservoir Computing
//!
//! This module provides time series prediction models including ARIMA and NAR.

use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::config::TimeSeriesConfig;
use super::types::ActivationFunction;

/// Time series prediction models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPredictor {
    /// ARIMA model parameters
    pub arima_params: ARIMAParams,
    /// NAR model state
    pub nar_state: NARState,
    /// Memory kernel weights
    pub kernel_weights: Array1<f64>,
    /// Trend model
    pub trend_model: TrendModel,
}

impl TimeSeriesPredictor {
    /// Create new time series predictor
    #[must_use]
    pub fn new(config: &TimeSeriesConfig) -> Self {
        Self {
            arima_params: ARIMAParams {
                ar_coeffs: Array1::zeros(config.ar_order),
                ma_coeffs: Array1::zeros(config.ma_order),
                diff_order: config.diff_order,
                residuals: VecDeque::with_capacity(config.ma_order),
                variance: 1.0,
            },
            nar_state: NARState {
                order: config.nar_order,
                coeffs: Array2::zeros((config.nar_order, config.nar_order)),
                history: VecDeque::with_capacity(config.nar_order),
                activation: ActivationFunction::Tanh,
            },
            kernel_weights: Array1::from_vec(config.kernel_params.clone()),
            trend_model: TrendModel {
                params: vec![0.0, 0.0], // Linear trend: intercept, slope
                strength: 0.0,
                direction: 0.0,
            },
        }
    }
}

/// ARIMA model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARIMAParams {
    /// AR coefficients
    pub ar_coeffs: Array1<f64>,
    /// MA coefficients
    pub ma_coeffs: Array1<f64>,
    /// Differencing order
    pub diff_order: usize,
    /// Model residuals
    pub residuals: VecDeque<f64>,
    /// Model variance
    pub variance: f64,
}

/// Nonlinear autoregressive model state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NARState {
    /// Model order
    pub order: usize,
    /// Nonlinear coefficients
    pub coeffs: Array2<f64>,
    /// Past values buffer
    pub history: VecDeque<f64>,
    /// Activation function
    pub activation: ActivationFunction,
}

/// Trend model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendModel {
    /// Model parameters
    pub params: Vec<f64>,
    /// Trend strength
    pub strength: f64,
    /// Trend direction
    pub direction: f64,
}
