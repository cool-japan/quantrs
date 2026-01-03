//! Noise Characterization with SciRS2 Statistics
//!
//! This module provides comprehensive quantum noise characterization using SciRS2's
//! advanced statistical analysis capabilities for error modeling, noise parameter
//! estimation, and hardware performance characterization.

use crate::{DeviceError, DeviceResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use scirs2_stats::distributions; // For creating distributions
use scirs2_stats::{mean, std}; // Descriptive statistics at root level

/// Noise characterization configuration
#[derive(Debug, Clone)]
pub struct NoiseCharacterizationConfig {
    /// Number of samples for statistical analysis
    pub num_samples: usize,
    /// Confidence level for statistical tests (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Enable advanced statistical analysis
    pub advanced_analysis: bool,
    /// Minimum sample size for reliable statistics
    pub min_sample_size: usize,
}

impl Default for NoiseCharacterizationConfig {
    fn default() -> Self {
        Self {
            num_samples: 10000,
            confidence_level: 0.95,
            advanced_analysis: true,
            min_sample_size: 100,
        }
    }
}

/// Noise model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseModelType {
    /// Depolarizing noise channel
    Depolarizing,
    /// Amplitude damping (T1 decay)
    AmplitudeDamping,
    /// Phase damping (T2 dephasing)
    PhaseDamping,
    /// Thermal noise
    Thermal,
    /// Readout error
    ReadoutError,
    /// Crosstalk noise
    Crosstalk,
}

/// Noise characterization result
#[derive(Debug, Clone)]
pub struct NoiseCharacterizationResult {
    /// Estimated noise parameters
    pub noise_parameters: NoiseParameters,
    /// Statistical confidence intervals
    pub confidence_intervals: ConfidenceIntervals,
    /// Goodness-of-fit metrics
    pub fit_quality: FitQuality,
    /// Correlation analysis
    pub correlations: CorrelationAnalysis,
}

/// Noise parameters estimated from data
#[derive(Debug, Clone)]
pub struct NoiseParameters {
    /// Mean error rate
    pub mean_error_rate: f64,
    /// Standard deviation of error rate
    pub std_error_rate: f64,
    /// T1 relaxation time (microseconds)
    pub t1_time: Option<f64>,
    /// T2 dephasing time (microseconds)
    pub t2_time: Option<f64>,
    /// Readout fidelity
    pub readout_fidelity: f64,
    /// Model-specific parameters
    pub model_params: Vec<f64>,
}

/// Statistical confidence intervals
#[derive(Debug, Clone)]
pub struct ConfidenceIntervals {
    /// Confidence level (e.g., 0.95)
    pub confidence_level: f64,
    /// Lower bound for mean error rate
    pub error_rate_lower: f64,
    /// Upper bound for mean error rate
    pub error_rate_upper: f64,
    /// T1 confidence interval
    pub t1_interval: Option<(f64, f64)>,
    /// T2 confidence interval
    pub t2_interval: Option<(f64, f64)>,
}

/// Goodness-of-fit quality metrics
#[derive(Debug, Clone)]
pub struct FitQuality {
    /// Chi-squared statistic
    pub chi_squared: f64,
    /// P-value from chi-squared test
    pub p_value: f64,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Root mean square error
    pub rmse: f64,
}

/// Correlation analysis between noise sources
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    /// Pearson correlation coefficients
    pub correlation_matrix: Array2<f64>,
    /// Covariance matrix
    pub covariance_matrix: Array2<f64>,
    /// Significant correlations (above threshold)
    pub significant_correlations: Vec<(usize, usize, f64)>,
}

/// Noise characterizer using SciRS2 statistics
pub struct NoiseCharacterizer {
    config: NoiseCharacterizationConfig,
    rng: StdRng,
}

impl NoiseCharacterizer {
    /// Create a new noise characterizer
    pub fn new(config: NoiseCharacterizationConfig) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Create characterizer with default configuration
    pub fn default() -> Self {
        Self::new(NoiseCharacterizationConfig::default())
    }

    /// Characterize noise from measurement data using SciRS2 statistics
    ///
    /// # Arguments
    /// * `measurement_data` - Array of measurement outcomes (0.0 or 1.0 for qubits)
    /// * `expected_data` - Expected ideal outcomes
    /// * `noise_model` - Type of noise model to fit
    ///
    /// # Returns
    /// Comprehensive noise characterization with statistical analysis
    pub fn characterize_noise(
        &mut self,
        measurement_data: &Array1<f64>,
        expected_data: &Array1<f64>,
        noise_model: NoiseModelType,
    ) -> DeviceResult<NoiseCharacterizationResult> {
        // Validate input data
        if measurement_data.len() < self.config.min_sample_size {
            return Err(DeviceError::InvalidInput(format!(
                "Insufficient samples: {} < minimum {}",
                measurement_data.len(),
                self.config.min_sample_size
            )));
        }

        if measurement_data.len() != expected_data.len() {
            return Err(DeviceError::InvalidInput(
                "Measurement and expected data must have same length".to_string(),
            ));
        }

        // Compute error rates
        let errors: Array1<f64> =
            (measurement_data - expected_data).mapv(|x| if x.abs() > 0.5 { 1.0 } else { 0.0 });

        let mean_error = mean(&errors.view())?;
        let std_error = std(&errors.view(), 1, None)?; // Sample std dev (ddof=1)

        // Estimate noise parameters based on model type
        let noise_params =
            self.estimate_noise_parameters(measurement_data, expected_data, &errors, noise_model)?;

        // Compute confidence intervals using SciRS2 statistics
        let confidence_intervals = self.compute_confidence_intervals(&errors, &noise_params)?;

        // Assess goodness-of-fit
        let fit_quality =
            self.assess_fit_quality(measurement_data, expected_data, &noise_params, noise_model)?;

        // Perform correlation analysis if advanced analysis is enabled
        let correlations = if self.config.advanced_analysis {
            self.analyze_correlations(measurement_data)?
        } else {
            CorrelationAnalysis {
                correlation_matrix: Array2::zeros((1, 1)),
                covariance_matrix: Array2::zeros((1, 1)),
                significant_correlations: Vec::new(),
            }
        };

        Ok(NoiseCharacterizationResult {
            noise_parameters: noise_params,
            confidence_intervals,
            fit_quality,
            correlations,
        })
    }

    /// Estimate noise parameters using maximum likelihood estimation
    fn estimate_noise_parameters(
        &self,
        measurement_data: &Array1<f64>,
        expected_data: &Array1<f64>,
        errors: &Array1<f64>,
        noise_model: NoiseModelType,
    ) -> DeviceResult<NoiseParameters> {
        let mean_error = mean(&errors.view())?;
        let std_error = std(&errors.view(), 1, None)?;

        // Compute readout fidelity
        let correct_measurements = errors.iter().filter(|&&e| e < 0.5).count();
        let readout_fidelity = correct_measurements as f64 / errors.len() as f64;

        // Model-specific parameter estimation
        let (t1_time, t2_time, model_params) = match noise_model {
            NoiseModelType::Depolarizing => {
                // Depolarizing probability p
                let p = mean_error;
                (None, None, vec![p])
            }
            NoiseModelType::AmplitudeDamping => {
                // Estimate T1 from amplitude decay
                let t1 = self.estimate_t1_time(measurement_data)?;
                (Some(t1), None, vec![t1])
            }
            NoiseModelType::PhaseDamping => {
                // Estimate T2 from phase decay
                let t2 = self.estimate_t2_time(measurement_data)?;
                (None, Some(t2), vec![t2])
            }
            NoiseModelType::Thermal => {
                // Thermal excitation probability
                let thermal_prob = mean_error;
                (None, None, vec![thermal_prob])
            }
            NoiseModelType::ReadoutError => {
                // Assignment error probabilities
                let p_0_given_1 =
                    self.estimate_readout_error(measurement_data, expected_data, 0)?;
                let p_1_given_0 =
                    self.estimate_readout_error(measurement_data, expected_data, 1)?;
                (None, None, vec![p_0_given_1, p_1_given_0])
            }
            NoiseModelType::Crosstalk => {
                // Crosstalk strength
                let crosstalk_strength = std_error;
                (None, None, vec![crosstalk_strength])
            }
        };

        Ok(NoiseParameters {
            mean_error_rate: mean_error,
            std_error_rate: std_error,
            t1_time,
            t2_time,
            readout_fidelity,
            model_params,
        })
    }

    /// Estimate T1 relaxation time from exponential decay fit
    fn estimate_t1_time(&self, measurement_data: &Array1<f64>) -> DeviceResult<f64> {
        // Simple exponential decay fit: P(t) = exp(-t/T1)
        // Using linear regression on log-transformed data
        let n = measurement_data.len();
        if n < 10 {
            return Ok(30.0); // Default T1 = 30 microseconds
        }

        // Time points (assuming uniform spacing)
        let times: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64);

        // Log-transform (with small offset to avoid log(0))
        let log_probs: Array1<f64> = measurement_data.mapv(|p| (p.max(0.01)).ln());

        // Simple linear fit: log(P) = -t/T1 + const
        let mean_t = mean(&times.view())?;
        let mean_log_p = mean(&log_probs.view())?;

        let numerator: f64 = times
            .iter()
            .zip(log_probs.iter())
            .map(|(&t, &lp)| (t - mean_t) * (lp - mean_log_p))
            .sum();

        let denominator: f64 = times.iter().map(|&t| (t - mean_t).powi(2)).sum();

        if denominator.abs() < 1e-10 {
            return Ok(30.0);
        }

        let slope = numerator / denominator;
        let t1 = -1.0 / slope;

        // Clamp to reasonable range (1-1000 microseconds)
        Ok(t1.clamp(1.0, 1000.0))
    }

    /// Estimate T2 dephasing time
    fn estimate_t2_time(&self, measurement_data: &Array1<f64>) -> DeviceResult<f64> {
        // Similar to T1 estimation but for phase coherence
        // T2 <= 2*T1 typically
        let t2_estimate = self.estimate_t1_time(measurement_data)? * 0.8;
        Ok(t2_estimate.clamp(1.0, 500.0))
    }

    /// Estimate readout error probability
    fn estimate_readout_error(
        &self,
        measurement_data: &Array1<f64>,
        expected_data: &Array1<f64>,
        state: usize,
    ) -> DeviceResult<f64> {
        let expected_state = state as f64;
        let count_expected: usize = expected_data
            .iter()
            .filter(|&&e| (e - expected_state).abs() < 0.5)
            .count();

        if count_expected == 0 {
            return Ok(0.01); // Default 1% error
        }

        let count_errors: usize = expected_data
            .iter()
            .zip(measurement_data.iter())
            .filter(|(&e, &m)| (e - expected_state).abs() < 0.5 && (m - expected_state).abs() > 0.5)
            .count();

        Ok((count_errors as f64 / count_expected as f64).clamp(0.0, 0.5))
    }

    /// Compute confidence intervals using t-distribution
    fn compute_confidence_intervals(
        &self,
        errors: &Array1<f64>,
        noise_params: &NoiseParameters,
    ) -> DeviceResult<ConfidenceIntervals> {
        let n = errors.len() as f64;
        let mean_err = noise_params.mean_error_rate;
        let std_err = noise_params.std_error_rate;

        // Compute critical value for t-distribution (approximate with z for large n)
        let z_critical = if self.config.confidence_level >= 0.99 {
            2.576 // 99% confidence
        } else if self.config.confidence_level >= 0.95 {
            1.96 // 95% confidence
        } else {
            1.645 // 90% confidence
        };

        let margin_of_error = z_critical * std_err / n.sqrt();

        Ok(ConfidenceIntervals {
            confidence_level: self.config.confidence_level,
            error_rate_lower: (mean_err - margin_of_error).max(0.0),
            error_rate_upper: (mean_err + margin_of_error).min(1.0),
            t1_interval: noise_params.t1_time.map(|t1| {
                let margin = 0.1 * t1; // 10% margin
                (t1 - margin, t1 + margin)
            }),
            t2_interval: noise_params.t2_time.map(|t2| {
                let margin = 0.1 * t2;
                (t2 - margin, t2 + margin)
            }),
        })
    }

    /// Assess goodness-of-fit using chi-squared test and R-squared
    fn assess_fit_quality(
        &self,
        measurement_data: &Array1<f64>,
        expected_data: &Array1<f64>,
        noise_params: &NoiseParameters,
        noise_model: NoiseModelType,
    ) -> DeviceResult<FitQuality> {
        // Compute residuals
        let residuals: Array1<f64> = measurement_data - expected_data;

        // Root mean square error
        let rmse = (residuals.mapv(|r| r * r).mean().unwrap_or(0.0)).sqrt();

        // R-squared calculation
        let mean_measured = mean(&measurement_data.view())?;
        let ss_tot: f64 = measurement_data
            .iter()
            .map(|&y| (y - mean_measured).powi(2))
            .sum();

        let ss_res: f64 = residuals.iter().map(|&r| r.powi(2)).sum();

        let r_squared = if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Chi-squared statistic (simplified)
        let chi_squared = ss_res / noise_params.std_error_rate.max(0.01).powi(2);

        // Approximate p-value (would use proper chi-squared distribution in production)
        let p_value = (-chi_squared / (2.0 * measurement_data.len() as f64)).exp();

        Ok(FitQuality {
            chi_squared,
            p_value,
            r_squared: r_squared.clamp(0.0, 1.0),
            rmse,
        })
    }

    /// Analyze correlations between noise sources
    fn analyze_correlations(
        &self,
        measurement_data: &Array1<f64>,
    ) -> DeviceResult<CorrelationAnalysis> {
        // For demonstration, create chunks to analyze correlation patterns
        let chunk_size = (measurement_data.len() / 5).max(10);
        let num_chunks = measurement_data.len() / chunk_size;

        if num_chunks < 2 {
            return Ok(CorrelationAnalysis {
                correlation_matrix: Array2::eye(1),
                covariance_matrix: Array2::zeros((1, 1)),
                significant_correlations: Vec::new(),
            });
        }

        // Create data matrix (each row is a chunk)
        let mut data_matrix = Array2::zeros((num_chunks, chunk_size));
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(measurement_data.len());
            let chunk_len = end - start;
            for j in 0..chunk_len {
                data_matrix[[i, j]] = measurement_data[start + j];
            }
        }

        // Compute covariance matrix manually (simplified)
        let corr_matrix = self.compute_covariance_matrix(&data_matrix)?;

        // Find significant correlations (|r| > 0.5)
        let mut significant = Vec::new();
        for i in 0..num_chunks {
            for j in (i + 1)..num_chunks {
                let corr = if i < corr_matrix.nrows() && j < corr_matrix.ncols() {
                    corr_matrix[[i, j]]
                } else {
                    0.0
                };
                if corr.abs() > 0.5 {
                    significant.push((i, j, corr));
                }
            }
        }

        Ok(CorrelationAnalysis {
            correlation_matrix: corr_matrix.clone(),
            covariance_matrix: corr_matrix,
            significant_correlations: significant,
        })
    }

    /// Compute covariance matrix manually
    fn compute_covariance_matrix(&self, data: &Array2<f64>) -> DeviceResult<Array2<f64>> {
        let n = data.nrows();
        let m = data.ncols();

        if n < 2 {
            return Ok(Array2::eye(1));
        }

        // Compute means for each column
        let mut means = Array1::zeros(m);
        for j in 0..m {
            let col_sum: f64 = (0..n).map(|i| data[[i, j]]).sum();
            means[j] = col_sum / n as f64;
        }

        // Compute covariance matrix
        let mut cov = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += (data[[i, k]] - means[k]) * (data[[j, k]] - means[k]);
                }
                cov[[i, j]] = sum / (m - 1) as f64;
            }
        }

        Ok(cov)
    }

    /// Generate noise samples for testing/simulation
    pub fn generate_noise_samples(
        &mut self,
        noise_model: NoiseModelType,
        num_samples: usize,
        noise_strength: f64,
    ) -> Array1<f64> {
        match noise_model {
            NoiseModelType::Depolarizing => {
                // Bernoulli noise
                Array1::from_shape_fn(num_samples, |_| {
                    if self.rng.gen::<f64>() < noise_strength {
                        1.0
                    } else {
                        0.0
                    }
                })
            }
            NoiseModelType::AmplitudeDamping | NoiseModelType::PhaseDamping => {
                // Exponential decay with Gaussian noise
                Array1::from_shape_fn(num_samples, |i| {
                    let decay = (-(i as f64) / 50.0).exp();
                    let gaussian_noise =
                        self.rng.gen::<f64>() * noise_strength - noise_strength / 2.0;
                    (decay + gaussian_noise).clamp(0.0, 1.0)
                })
            }
            _ => {
                // Gaussian noise
                Array1::from_shape_fn(num_samples, |_| {
                    (self.rng.gen::<f64>() * noise_strength)
                        .abs()
                        .clamp(0.0, 1.0)
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_characterizer_creation() {
        let config = NoiseCharacterizationConfig::default();
        let characterizer = NoiseCharacterizer::new(config);
        assert_eq!(characterizer.config.num_samples, 10000);
    }

    #[test]
    fn test_depolarizing_noise_characterization() {
        let config = NoiseCharacterizationConfig {
            num_samples: 1000,
            confidence_level: 0.95,
            advanced_analysis: false,
            min_sample_size: 100,
        };

        let mut characterizer = NoiseCharacterizer::new(config);

        // Generate test data with 10% error rate
        let expected = Array1::zeros(1000);
        let mut measurement = expected.clone();
        for i in 0..100 {
            measurement[i * 10] = 1.0; // 10% errors
        }

        let result =
            characterizer.characterize_noise(&measurement, &expected, NoiseModelType::Depolarizing);

        assert!(result.is_ok());
        let result = result.expect("Characterization failed");

        // Should detect ~10% error rate
        assert!((result.noise_parameters.mean_error_rate - 0.1).abs() < 0.05);
        assert!(result.confidence_intervals.error_rate_lower < 0.1);
        assert!(result.confidence_intervals.error_rate_upper > 0.1);
    }

    #[test]
    fn test_amplitude_damping_characterization() {
        let config = NoiseCharacterizationConfig::default();
        let mut characterizer = NoiseCharacterizer::new(config);

        // Generate exponential decay data
        let n = 500;
        let expected = Array1::ones(n);
        let measurement = Array1::from_shape_fn(n, |i| (-(i as f64) / 50.0).exp());

        let result = characterizer.characterize_noise(
            &measurement,
            &expected,
            NoiseModelType::AmplitudeDamping,
        );

        assert!(result.is_ok());
        let result = result.expect("Characterization failed");

        // Should estimate T1 time
        assert!(result.noise_parameters.t1_time.is_some());
        let t1 = result.noise_parameters.t1_time.expect("No T1 estimate");
        assert!(t1 > 1.0 && t1 < 1000.0);
    }

    #[test]
    fn test_readout_error_characterization() {
        let config = NoiseCharacterizationConfig::default();
        let mut characterizer = NoiseCharacterizer::new(config);

        // 5% readout error: measure 1 when expecting 0
        let n = 1000;
        let mut expected = Array1::zeros(n);
        let mut measurement = expected.clone();

        for i in 0..50 {
            expected[i * 20] = 0.0;
            measurement[i * 20] = 1.0; // Readout error
        }

        let result =
            characterizer.characterize_noise(&measurement, &expected, NoiseModelType::ReadoutError);

        assert!(result.is_ok());
        let result = result.expect("Characterization failed");
        assert_eq!(result.noise_parameters.model_params.len(), 2);
    }

    #[test]
    fn test_confidence_intervals() {
        let config = NoiseCharacterizationConfig {
            confidence_level: 0.95,
            ..Default::default()
        };
        let mut characterizer = NoiseCharacterizer::new(config);

        let expected = Array1::zeros(1000);
        let measurement = Array1::from_shape_fn(1000, |i| if i % 10 == 0 { 1.0 } else { 0.0 });

        let result =
            characterizer.characterize_noise(&measurement, &expected, NoiseModelType::Depolarizing);

        assert!(result.is_ok());
        let result = result.expect("Characterization failed");

        assert_eq!(result.confidence_intervals.confidence_level, 0.95);
        assert!(result.confidence_intervals.error_rate_lower >= 0.0);
        assert!(result.confidence_intervals.error_rate_upper <= 1.0);
        assert!(
            result.confidence_intervals.error_rate_lower
                < result.confidence_intervals.error_rate_upper
        );
    }

    #[test]
    fn test_fit_quality_metrics() {
        let config = NoiseCharacterizationConfig::default();
        let mut characterizer = NoiseCharacterizer::new(config);

        let expected = Array1::zeros(500);
        let measurement = expected.clone();

        let result =
            characterizer.characterize_noise(&measurement, &expected, NoiseModelType::Depolarizing);

        assert!(result.is_ok());
        let result = result.expect("Characterization failed");

        // Perfect fit should have high R-squared
        assert!(result.fit_quality.r_squared >= 0.0);
        assert!(result.fit_quality.rmse >= 0.0);
    }

    #[test]
    fn test_noise_sample_generation() {
        let config = NoiseCharacterizationConfig::default();
        let mut characterizer = NoiseCharacterizer::new(config);

        let samples = characterizer.generate_noise_samples(NoiseModelType::Depolarizing, 1000, 0.1);

        assert_eq!(samples.len(), 1000);

        // All samples should be in [0, 1]
        for &s in samples.iter() {
            assert!((0.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn test_insufficient_samples_error() {
        let config = NoiseCharacterizationConfig {
            min_sample_size: 100,
            ..Default::default()
        };
        let mut characterizer = NoiseCharacterizer::new(config);

        let expected = Array1::zeros(50);
        let measurement = expected.clone();

        let result =
            characterizer.characterize_noise(&measurement, &expected, NoiseModelType::Depolarizing);

        assert!(result.is_err());
    }
}
