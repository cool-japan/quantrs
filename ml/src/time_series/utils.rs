//! Utility functions and synthetic data generation for time series

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Generate synthetic time series data for testing and benchmarking
pub fn generate_synthetic_time_series(
    length: usize,
    seasonality: Option<usize>,
    trend: f64,
    noise: f64,
) -> Array2<f64> {
    let mut data = Array2::zeros((length, 1));

    for i in 0..length {
        let t = i as f64;

        // Trend component
        let trend_val = trend * t;

        // Seasonal component
        let seasonal_val = if let Some(period) = seasonality {
            10.0 * (2.0 * PI * t / period as f64).sin()
        } else {
            0.0
        };

        // Noise component
        let noise_val = noise * (fastrand::f64() - 0.5);

        data[[i, 0]] = 50.0 + trend_val + seasonal_val + noise_val;
    }

    data
}

/// Generate complex synthetic time series with multiple components
pub fn generate_complex_time_series(
    length: usize,
    config: SyntheticDataConfig,
) -> Result<Array2<f64>> {
    let mut data = Array2::zeros((length, config.num_features));

    for feature_idx in 0..config.num_features {
        for i in 0..length {
            let t = i as f64;
            let mut value = config.base_level;

            // Add trend components
            for trend in &config.trends {
                value += trend.apply(t);
            }

            // Add seasonal components
            for seasonal in &config.seasonalities {
                value += seasonal.apply(t);
            }

            // Add noise
            value += config.noise_level * (fastrand::f64() - 0.5);

            // Add feature-specific variation
            let feature_factor = 1.0 + 0.1 * feature_idx as f64;
            value *= feature_factor;

            data[[i, feature_idx]] = value;
        }
    }

    // Add correlations between features if specified
    if config.feature_correlations {
        data = add_feature_correlations(data, 0.3)?;
    }

    // Add outliers if specified
    if config.outlier_probability > 0.0 {
        data = add_outliers(data, config.outlier_probability, config.outlier_magnitude)?;
    }

    Ok(data)
}

/// Configuration for synthetic data generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticDataConfig {
    /// Number of features/variables
    pub num_features: usize,

    /// Base level of the time series
    pub base_level: f64,

    /// Trend components
    pub trends: Vec<TrendComponent>,

    /// Seasonal components
    pub seasonalities: Vec<SeasonalComponent>,

    /// Noise level
    pub noise_level: f64,

    /// Add correlations between features
    pub feature_correlations: bool,

    /// Probability of outliers
    pub outlier_probability: f64,

    /// Magnitude of outliers
    pub outlier_magnitude: f64,
}

/// Trend component specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendComponent {
    /// Type of trend
    pub trend_type: TrendType,

    /// Magnitude/coefficient
    pub magnitude: f64,

    /// Additional parameters
    pub parameters: Vec<f64>,
}

/// Types of trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendType {
    Linear,
    Quadratic,
    Exponential,
    Logarithmic,
    Sinusoidal,
    Custom(String),
}

/// Seasonal component specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalComponent {
    /// Seasonal period
    pub period: usize,

    /// Amplitude
    pub amplitude: f64,

    /// Phase shift
    pub phase: f64,

    /// Seasonal pattern type
    pub pattern_type: SeasonalPattern,
}

/// Types of seasonal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalPattern {
    Sinusoidal,
    Triangular,
    Square,
    Sawtooth,
    Custom(Vec<f64>),
}

impl TrendComponent {
    /// Apply trend component at time t
    pub fn apply(&self, t: f64) -> f64 {
        match &self.trend_type {
            TrendType::Linear => self.magnitude * t,
            TrendType::Quadratic => self.magnitude * t * t,
            TrendType::Exponential => {
                let rate = self.parameters.get(0).copied().unwrap_or(0.01);
                self.magnitude * (rate * t).exp()
            }
            TrendType::Logarithmic => {
                if t > 0.0 {
                    self.magnitude * t.ln()
                } else {
                    0.0
                }
            }
            TrendType::Sinusoidal => {
                let frequency = self.parameters.get(0).copied().unwrap_or(0.1);
                self.magnitude * (2.0 * PI * frequency * t).sin()
            }
            TrendType::Custom(_) => {
                // Placeholder for custom trend functions
                self.magnitude * t
            }
        }
    }
}

impl SeasonalComponent {
    /// Apply seasonal component at time t
    pub fn apply(&self, t: f64) -> f64 {
        let normalized_time = (t % self.period as f64) / self.period as f64;
        let phase_adjusted_time = (normalized_time + self.phase / (2.0 * PI)) % 1.0;

        match &self.pattern_type {
            SeasonalPattern::Sinusoidal => self.amplitude * (2.0 * PI * phase_adjusted_time).sin(),
            SeasonalPattern::Triangular => {
                let triangle_val = if phase_adjusted_time < 0.5 {
                    4.0 * phase_adjusted_time - 1.0
                } else {
                    3.0 - 4.0 * phase_adjusted_time
                };
                self.amplitude * triangle_val
            }
            SeasonalPattern::Square => {
                let square_val = if phase_adjusted_time < 0.5 { 1.0 } else { -1.0 };
                self.amplitude * square_val
            }
            SeasonalPattern::Sawtooth => self.amplitude * (2.0 * phase_adjusted_time - 1.0),
            SeasonalPattern::Custom(pattern) => {
                if pattern.is_empty() {
                    0.0
                } else {
                    let index = (phase_adjusted_time * pattern.len() as f64) as usize;
                    let index = index.min(pattern.len() - 1);
                    self.amplitude * pattern[index]
                }
            }
        }
    }
}

/// Add correlations between features
fn add_feature_correlations(
    mut data: Array2<f64>,
    correlation_strength: f64,
) -> Result<Array2<f64>> {
    let (n_samples, n_features) = data.dim();

    if n_features < 2 {
        return Ok(data);
    }

    // Add correlation by blending features
    for i in 1..n_features {
        for j in 0..n_samples {
            let original_value = data[[j, i]];
            let correlated_value = data[[j, i - 1]];

            data[[j, i]] = original_value * (1.0 - correlation_strength)
                + correlated_value * correlation_strength;
        }
    }

    Ok(data)
}

/// Add outliers to the data
fn add_outliers(mut data: Array2<f64>, probability: f64, magnitude: f64) -> Result<Array2<f64>> {
    let (n_samples, n_features) = data.dim();

    for i in 0..n_samples {
        for j in 0..n_features {
            if fastrand::f64() < probability {
                let outlier_direction = if fastrand::bool() { 1.0 } else { -1.0 };
                data[[i, j]] += outlier_direction * magnitude;
            }
        }
    }

    Ok(data)
}

/// Data preprocessing utilities
pub struct DataPreprocessor;

impl DataPreprocessor {
    /// Normalize data to zero mean and unit variance
    pub fn standardize(data: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        let (n_samples, n_features) = data.dim();
        let mut normalized = data.clone();
        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::zeros(n_features);

        for j in 0..n_features {
            let column = data.column(j);
            let mean = column.mean().unwrap_or(0.0);
            let std = column.std(1.0).max(1e-8); // Avoid division by zero

            means[j] = mean;
            stds[j] = std;

            for i in 0..n_samples {
                normalized[[i, j]] = (data[[i, j]] - mean) / std;
            }
        }

        Ok((normalized, means, stds))
    }

    /// Normalize data to [0, 1] range
    pub fn min_max_scale(data: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        let (n_samples, n_features) = data.dim();
        let mut scaled = data.clone();
        let mut mins = Array1::zeros(n_features);
        let mut ranges = Array1::zeros(n_features);

        for j in 0..n_features {
            let column = data.column(j);
            let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = (max_val - min_val).max(1e-8); // Avoid division by zero

            mins[j] = min_val;
            ranges[j] = range;

            for i in 0..n_samples {
                scaled[[i, j]] = (data[[i, j]] - min_val) / range;
            }
        }

        Ok((scaled, mins, ranges))
    }

    /// Remove outliers using IQR method
    pub fn remove_outliers(data: &Array2<f64>, iqr_multiplier: f64) -> Result<Array2<f64>> {
        let (n_samples, n_features) = data.dim();
        let mut cleaned_data = Vec::new();

        for i in 0..n_samples {
            let mut is_outlier = false;

            for j in 0..n_features {
                let column = data.column(j);
                let mut sorted_values: Vec<f64> = column.iter().cloned().collect();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let q1_idx = sorted_values.len() / 4;
                let q3_idx = 3 * sorted_values.len() / 4;
                let q1 = sorted_values[q1_idx];
                let q3 = sorted_values[q3_idx];
                let iqr = q3 - q1;

                let lower_bound = q1 - iqr_multiplier * iqr;
                let upper_bound = q3 + iqr_multiplier * iqr;

                if data[[i, j]] < lower_bound || data[[i, j]] > upper_bound {
                    is_outlier = true;
                    break;
                }
            }

            if !is_outlier {
                cleaned_data.push(data.row(i).to_owned());
            }
        }

        if cleaned_data.is_empty() {
            return Err(MLError::DataError(
                "All data points were classified as outliers".to_string(),
            ));
        }

        let cleaned_samples = cleaned_data.len();
        let mut cleaned_array = Array2::zeros((cleaned_samples, n_features));

        for (i, row) in cleaned_data.iter().enumerate() {
            cleaned_array.row_mut(i).assign(row);
        }

        Ok(cleaned_array)
    }

    /// Interpolate missing values
    pub fn interpolate_missing(data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut interpolated = data.clone();
        let (n_samples, n_features) = data.dim();

        for j in 0..n_features {
            for i in 0..n_samples {
                if !data[[i, j]].is_finite() {
                    // Linear interpolation between neighboring valid points
                    let mut left_val = None;
                    let mut right_val = None;

                    // Find left neighbor
                    for k in (0..i).rev() {
                        if data[[k, j]].is_finite() {
                            left_val = Some((k, data[[k, j]]));
                            break;
                        }
                    }

                    // Find right neighbor
                    for k in (i + 1)..n_samples {
                        if data[[k, j]].is_finite() {
                            right_val = Some((k, data[[k, j]]));
                            break;
                        }
                    }

                    // Interpolate
                    let interpolated_value = match (left_val, right_val) {
                        (Some((left_idx, left_val)), Some((right_idx, right_val))) => {
                            let weight = (i - left_idx) as f64 / (right_idx - left_idx) as f64;
                            left_val + weight * (right_val - left_val)
                        }
                        (Some((_, left_val)), None) => left_val,
                        (None, Some((_, right_val))) => right_val,
                        (None, None) => 0.0, // No valid neighbors found
                    };

                    interpolated[[i, j]] = interpolated_value;
                }
            }
        }

        Ok(interpolated)
    }
}

/// Time series analysis utilities
pub struct TimeSeriesAnalyzer;

impl TimeSeriesAnalyzer {
    /// Detect stationarity using Augmented Dickey-Fuller test (simplified)
    pub fn is_stationary(data: &Array1<f64>) -> bool {
        // Simplified stationarity test based on variance of differences
        if data.len() < 10 {
            return false;
        }

        let mut differences = Array1::zeros(data.len() - 1);
        for i in 1..data.len() {
            differences[i - 1] = data[i] - data[i - 1];
        }

        let original_var = data.var(1.0);
        let diff_var = differences.var(1.0);

        // If variance of differences is much smaller, likely stationary
        diff_var < 0.8 * original_var
    }

    /// Compute autocorrelation function
    pub fn autocorrelation(data: &Array1<f64>, max_lag: usize) -> Array1<f64> {
        let n = data.len();
        let max_lag = max_lag.min(n / 2);
        let mut autocorr = Array1::zeros(max_lag + 1);

        let mean = data.mean().unwrap_or(0.0);
        let variance = data.var(1.0);

        for lag in 0..=max_lag {
            let mut sum = 0.0;
            let mut count = 0;

            for t in lag..n {
                sum += (data[t] - mean) * (data[t - lag] - mean);
                count += 1;
            }

            if count > 0 && variance > 1e-10 {
                autocorr[lag] = sum / (count as f64 * variance);
            }
        }

        autocorr
    }

    /// Detect seasonal periods using autocorrelation
    pub fn detect_seasonality(data: &Array1<f64>, max_period: usize) -> Vec<usize> {
        let autocorr = Self::autocorrelation(data, max_period);
        let mut seasonal_periods = Vec::new();

        // Find peaks in autocorrelation function
        for lag in 2..autocorr.len() {
            if lag < autocorr.len() - 1 {
                let current = autocorr[lag];
                let prev = autocorr[lag - 1];
                let next = autocorr[lag + 1];

                // Check if current value is a local maximum above threshold
                if current > prev && current > next && current > 0.3 {
                    seasonal_periods.push(lag);
                }
            }
        }

        seasonal_periods
    }

    /// Estimate trend using simple linear regression
    pub fn estimate_trend(data: &Array1<f64>) -> (f64, f64) {
        let n = data.len() as f64;

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let denominator = n * sum_x2 - sum_x * sum_x;

        if denominator.abs() > 1e-10 {
            let slope = (n * sum_xy - sum_x * sum_y) / denominator;
            let intercept = (sum_y - slope * sum_x) / n;
            (slope, intercept)
        } else {
            (0.0, sum_y / n)
        }
    }
}

/// Default configurations for common scenarios
impl Default for SyntheticDataConfig {
    fn default() -> Self {
        Self {
            num_features: 1,
            base_level: 100.0,
            trends: vec![TrendComponent {
                trend_type: TrendType::Linear,
                magnitude: 0.1,
                parameters: Vec::new(),
            }],
            seasonalities: vec![SeasonalComponent {
                period: 12,
                amplitude: 10.0,
                phase: 0.0,
                pattern_type: SeasonalPattern::Sinusoidal,
            }],
            noise_level: 5.0,
            feature_correlations: false,
            outlier_probability: 0.02,
            outlier_magnitude: 20.0,
        }
    }
}

impl SyntheticDataConfig {
    /// Configuration for financial time series
    pub fn financial() -> Self {
        Self {
            num_features: 5, // OHLCV
            base_level: 100.0,
            trends: vec![
                TrendComponent {
                    trend_type: TrendType::Linear,
                    magnitude: 0.05,
                    parameters: Vec::new(),
                },
                TrendComponent {
                    trend_type: TrendType::Sinusoidal,
                    magnitude: 2.0,
                    parameters: vec![0.02], // Low frequency trend
                },
            ],
            seasonalities: vec![
                SeasonalComponent {
                    period: 5, // Weekly seasonality
                    amplitude: 3.0,
                    phase: 0.0,
                    pattern_type: SeasonalPattern::Sinusoidal,
                },
                SeasonalComponent {
                    period: 21, // Monthly seasonality
                    amplitude: 1.5,
                    phase: PI / 4.0,
                    pattern_type: SeasonalPattern::Sinusoidal,
                },
            ],
            noise_level: 2.0,
            feature_correlations: true,
            outlier_probability: 0.05,
            outlier_magnitude: 10.0,
        }
    }

    /// Configuration for IoT sensor data
    pub fn iot_sensor() -> Self {
        Self {
            num_features: 3, // Temperature, humidity, pressure
            base_level: 25.0,
            trends: vec![TrendComponent {
                trend_type: TrendType::Sinusoidal,
                magnitude: 5.0,
                parameters: vec![1.0 / 24.0], // Daily cycle
            }],
            seasonalities: vec![
                SeasonalComponent {
                    period: 24, // Hourly readings, daily pattern
                    amplitude: 8.0,
                    phase: 0.0,
                    pattern_type: SeasonalPattern::Sinusoidal,
                },
                SeasonalComponent {
                    period: 168, // Weekly pattern
                    amplitude: 3.0,
                    phase: PI / 3.0,
                    pattern_type: SeasonalPattern::Sinusoidal,
                },
            ],
            noise_level: 1.0,
            feature_correlations: true,
            outlier_probability: 0.01,
            outlier_magnitude: 15.0,
        }
    }

    /// Configuration for demand forecasting
    pub fn demand() -> Self {
        Self {
            num_features: 1,
            base_level: 1000.0,
            trends: vec![TrendComponent {
                trend_type: TrendType::Linear,
                magnitude: 2.0,
                parameters: Vec::new(),
            }],
            seasonalities: vec![
                SeasonalComponent {
                    period: 7, // Weekly seasonality
                    amplitude: 200.0,
                    phase: 0.0,
                    pattern_type: SeasonalPattern::Sinusoidal,
                },
                SeasonalComponent {
                    period: 365, // Yearly seasonality
                    amplitude: 500.0,
                    phase: PI / 2.0,
                    pattern_type: SeasonalPattern::Sinusoidal,
                },
            ],
            noise_level: 50.0,
            feature_correlations: false,
            outlier_probability: 0.03,
            outlier_magnitude: 300.0,
        }
    }
}
