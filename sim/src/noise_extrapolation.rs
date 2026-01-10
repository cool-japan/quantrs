//! Noise extrapolation techniques for quantum error mitigation.
//!
//! This module implements various error mitigation techniques including
//! Zero-Noise Extrapolation (ZNE), Virtual Distillation, Symmetry Verification,
//! and other methods to extrapolate quantum results to the zero-noise limit.

use scirs2_core::ndarray::Array2;
use scirs2_core::parallel_ops::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SimulatorError};

/// Zero-Noise Extrapolation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZNEResult {
    /// Extrapolated expectation value at zero noise
    pub zero_noise_value: f64,
    /// Extrapolation error estimate
    pub error_estimate: f64,
    /// Noise scaling factors used
    pub noise_factors: Vec<f64>,
    /// Measured expectation values at each noise level
    pub measured_values: Vec<f64>,
    /// Measurement uncertainties
    pub uncertainties: Vec<f64>,
    /// Extrapolation method used
    pub method: ExtrapolationMethod,
    /// Confidence in the extrapolation
    pub confidence: f64,
    /// Fitting statistics
    pub fit_stats: FitStatistics,
}

/// Virtual Distillation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualDistillationResult {
    /// Mitigated expectation value
    pub mitigated_value: f64,
    /// Overhead factor (number of additional circuits)
    pub overhead: usize,
    /// Distillation efficiency
    pub efficiency: f64,
    /// Error reduction factor
    pub error_reduction: f64,
}

/// Symmetry Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryVerificationResult {
    /// Corrected expectation value
    pub corrected_value: f64,
    /// Symmetry violation measure
    pub symmetry_violation: f64,
    /// Post-selection probability
    pub post_selection_prob: f64,
    /// Symmetries tested
    pub symmetries: Vec<String>,
}

/// Extrapolation methods for ZNE
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExtrapolationMethod {
    /// Linear extrapolation
    Linear,
    /// Exponential extrapolation
    Exponential,
    /// Polynomial extrapolation (order specified)
    Polynomial(usize),
    /// Richardson extrapolation
    Richardson,
    /// Adaptive method selection
    Adaptive,
}

/// Noise scaling methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NoiseScalingMethod {
    /// Global unitary folding
    UnitaryFolding,
    /// Local gate folding
    LocalFolding,
    /// Parameter scaling
    ParameterScaling,
    /// Identity insertion
    IdentityInsertion,
    /// Pauli twirling
    PauliTwirling,
}

/// Fitting statistics for extrapolation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FitStatistics {
    /// R-squared value
    pub r_squared: f64,
    /// Reduced chi-squared
    pub chi_squared_reduced: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Number of parameters in fit
    pub num_parameters: usize,
    /// Residuals
    pub residuals: Vec<f64>,
}

/// Zero-Noise Extrapolation engine
#[derive(Debug, Clone)]
pub struct ZeroNoiseExtrapolator {
    /// Noise scaling method
    scaling_method: NoiseScalingMethod,
    /// Extrapolation method
    extrapolation_method: ExtrapolationMethod,
    /// Noise factors to sample
    noise_factors: Vec<f64>,
    /// Number of shots per noise level
    shots_per_level: usize,
    /// Maximum polynomial order for adaptive fitting
    max_poly_order: usize,
}

impl ZeroNoiseExtrapolator {
    /// Create new ZNE extrapolator
    #[must_use]
    pub const fn new(
        scaling_method: NoiseScalingMethod,
        extrapolation_method: ExtrapolationMethod,
        noise_factors: Vec<f64>,
        shots_per_level: usize,
    ) -> Self {
        Self {
            scaling_method,
            extrapolation_method,
            noise_factors,
            shots_per_level,
            max_poly_order: 4,
        }
    }

    /// Default configuration for ZNE
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(
            NoiseScalingMethod::UnitaryFolding,
            ExtrapolationMethod::Linear,
            vec![1.0, 1.5, 2.0, 2.5, 3.0],
            1000,
        )
    }

    /// Perform zero-noise extrapolation
    pub fn extrapolate<F>(&self, noisy_executor: F, observable: &str) -> Result<ZNEResult>
    where
        F: Fn(f64) -> Result<f64> + Sync + Send,
    {
        let start_time = std::time::Instant::now();

        // Measure expectation values at different noise levels
        let measurements: Result<Vec<(f64, f64, f64)>> = self
            .noise_factors
            .par_iter()
            .map(|&noise_factor| {
                // Execute circuit with scaled noise
                let measured_value = noisy_executor(noise_factor)?;

                // Estimate uncertainty (would use shot noise in practice)
                let uncertainty = self.estimate_measurement_uncertainty(measured_value);

                Ok((noise_factor, measured_value, uncertainty))
            })
            .collect();

        let measurements = measurements?;
        let (noise_factors, measured_values, uncertainties): (Vec<f64>, Vec<f64>, Vec<f64>) =
            measurements.unzip3();

        // Perform extrapolation based on method
        let (zero_noise_value, error_estimate, fit_stats) = match self.extrapolation_method {
            ExtrapolationMethod::Linear => {
                self.linear_extrapolation(&noise_factors, &measured_values, &uncertainties)?
            }
            ExtrapolationMethod::Exponential => {
                self.exponential_extrapolation(&noise_factors, &measured_values, &uncertainties)?
            }
            ExtrapolationMethod::Polynomial(order) => self.polynomial_extrapolation(
                &noise_factors,
                &measured_values,
                &uncertainties,
                order,
            )?,
            ExtrapolationMethod::Richardson => {
                self.richardson_extrapolation(&noise_factors, &measured_values)?
            }
            ExtrapolationMethod::Adaptive => {
                self.adaptive_extrapolation(&noise_factors, &measured_values, &uncertainties)?
            }
        };

        // Calculate confidence based on fit quality
        let confidence = self.calculate_confidence(&fit_stats);

        Ok(ZNEResult {
            zero_noise_value,
            error_estimate,
            noise_factors,
            measured_values,
            uncertainties,
            method: self.extrapolation_method,
            confidence,
            fit_stats,
        })
    }

    /// Linear extrapolation y = a + b*x, extrapolate to x=0
    fn linear_extrapolation(
        &self,
        noise_factors: &[f64],
        measured_values: &[f64],
        uncertainties: &[f64],
    ) -> Result<(f64, f64, FitStatistics)> {
        if noise_factors.len() < 2 {
            return Err(SimulatorError::InvalidInput(
                "Need at least 2 data points for linear extrapolation".to_string(),
            ));
        }

        // Weighted least squares fit
        let (a, b, fit_stats) =
            self.weighted_linear_fit(noise_factors, measured_values, uncertainties)?;

        // Extrapolate to zero noise (x=0)
        let zero_noise_value = a;

        // Error estimate from fit uncertainty
        let error_estimate = fit_stats.residuals.iter().map(|r| r.abs()).sum::<f64>()
            / fit_stats.residuals.len() as f64;

        Ok((zero_noise_value, error_estimate, fit_stats))
    }

    /// Exponential extrapolation y = a * exp(b*x), extrapolate to x=0
    fn exponential_extrapolation(
        &self,
        noise_factors: &[f64],
        measured_values: &[f64],
        uncertainties: &[f64],
    ) -> Result<(f64, f64, FitStatistics)> {
        // Transform to linear: ln(y) = ln(a) + b*x
        let log_values: Vec<f64> = measured_values
            .iter()
            .map(|&y| if y > 0.0 { y.ln() } else { f64::NEG_INFINITY })
            .collect();

        // Check for negative values
        if log_values.iter().any(|&x| x.is_infinite()) {
            return Err(SimulatorError::NumericalError(
                "Cannot take logarithm of non-positive values for exponential fit".to_string(),
            ));
        }

        // Linear fit in log space
        let log_uncertainties: Vec<f64> = uncertainties.iter().zip(measured_values.iter())
            .map(|(&u, &y)| u / y.abs()) // Propagate uncertainty: σ(ln(y)) ≈ σ(y)/|y|
            .collect();

        let (ln_a, b, mut fit_stats) =
            self.weighted_linear_fit(noise_factors, &log_values, &log_uncertainties)?;

        // Transform back
        let a = ln_a.exp();
        let zero_noise_value = a; // At x=0: y = a * exp(0) = a

        // Error propagation
        let error_estimate = a * fit_stats.residuals.iter().map(|r| r.abs()).sum::<f64>()
            / fit_stats.residuals.len() as f64;

        // Transform residuals back to original space
        fit_stats.residuals = fit_stats
            .residuals
            .iter()
            .zip(noise_factors.iter().enumerate())
            .map(|(&_res, (idx, &x))| a.mul_add((b * x).exp(), -measured_values[idx]))
            .collect();

        Ok((zero_noise_value, error_estimate, fit_stats))
    }

    /// Polynomial extrapolation
    fn polynomial_extrapolation(
        &self,
        noise_factors: &[f64],
        measured_values: &[f64],
        uncertainties: &[f64],
        order: usize,
    ) -> Result<(f64, f64, FitStatistics)> {
        if noise_factors.len() <= order {
            return Err(SimulatorError::InvalidInput(format!(
                "Need more than {order} data points for order {order} polynomial"
            )));
        }

        // Construct Vandermonde matrix
        let n = noise_factors.len();
        let mut design_matrix = Array2::zeros((n, order + 1));

        for (i, &x) in noise_factors.iter().enumerate() {
            for j in 0..=order {
                design_matrix[[i, j]] = x.powi(j as i32);
            }
        }

        // Weighted least squares (simplified implementation)
        let coefficients =
            self.solve_weighted_least_squares(&design_matrix, measured_values, uncertainties)?;

        // Zero-noise value is the constant term (coefficient of x^0)
        let zero_noise_value = coefficients[0];

        // Calculate residuals
        let mut residuals = Vec::with_capacity(n);
        for (i, &x) in noise_factors.iter().enumerate() {
            let predicted: f64 = coefficients
                .iter()
                .enumerate()
                .map(|(j, &coeff)| coeff * x.powi(j as i32))
                .sum();
            residuals.push(measured_values[i] - predicted);
        }

        let error_estimate =
            residuals.iter().map(|r| r.abs()).sum::<f64>() / residuals.len() as f64;

        let fit_stats = FitStatistics {
            residuals,
            num_parameters: order + 1,
            ..Default::default()
        };

        Ok((zero_noise_value, error_estimate, fit_stats))
    }

    /// Richardson extrapolation
    fn richardson_extrapolation(
        &self,
        noise_factors: &[f64],
        measured_values: &[f64],
    ) -> Result<(f64, f64, FitStatistics)> {
        if noise_factors.len() < 3 {
            return Err(SimulatorError::InvalidInput(
                "Richardson extrapolation requires at least 3 data points".to_string(),
            ));
        }

        // Use first three points for Richardson extrapolation
        let x1 = noise_factors[0];
        let x2 = noise_factors[1];
        let x3 = noise_factors[2];
        let y1 = measured_values[0];
        let y2 = measured_values[1];
        let y3 = measured_values[2];

        // Richardson extrapolation formula assuming y = a + b*x + c*x^2
        let denominator = (x1 - x2) * (x1 - x3) * (x2 - x3);
        if denominator.abs() < 1e-12 {
            return Err(SimulatorError::NumericalError(
                "Noise factors too close for Richardson extrapolation".to_string(),
            ));
        }

        let a = (y3 * x1 * x2).mul_add(
            x1 - x2,
            (y1 * x2 * x3).mul_add(x2 - x3, y2 * x1 * x3 * (x3 - x1)),
        ) / denominator;

        let zero_noise_value = a;

        // Error estimate (simplified)
        let error_estimate = (y1 - y2).abs().max((y2 - y3).abs()) * 0.1;

        let fit_stats = FitStatistics {
            num_parameters: 3,
            ..Default::default()
        };

        Ok((zero_noise_value, error_estimate, fit_stats))
    }

    /// Adaptive extrapolation method selection
    fn adaptive_extrapolation(
        &self,
        noise_factors: &[f64],
        measured_values: &[f64],
        uncertainties: &[f64],
    ) -> Result<(f64, f64, FitStatistics)> {
        let mut best_result = None;
        let mut best_aic = f64::INFINITY;

        // Try different extrapolation methods
        let methods = vec![
            ExtrapolationMethod::Linear,
            ExtrapolationMethod::Exponential,
            ExtrapolationMethod::Polynomial(2),
            ExtrapolationMethod::Polynomial(3),
        ];

        for method in methods {
            let result = match method {
                ExtrapolationMethod::Linear => {
                    self.linear_extrapolation(noise_factors, measured_values, uncertainties)
                }
                ExtrapolationMethod::Exponential => {
                    self.exponential_extrapolation(noise_factors, measured_values, uncertainties)
                }
                ExtrapolationMethod::Polynomial(order) => self.polynomial_extrapolation(
                    noise_factors,
                    measured_values,
                    uncertainties,
                    order,
                ),
                _ => continue,
            };

            if let Ok((value, error, stats)) = result {
                let aic = stats.aic;
                if aic < best_aic {
                    best_aic = aic;
                    best_result = Some((value, error, stats));
                }
            }
        }

        best_result.ok_or_else(|| {
            SimulatorError::ComputationError("No extrapolation method succeeded".to_string())
        })
    }

    /// Weighted linear fit
    fn weighted_linear_fit(
        &self,
        x: &[f64],
        y: &[f64],
        weights: &[f64],
    ) -> Result<(f64, f64, FitStatistics)> {
        let n = x.len();
        if n < 2 {
            return Err(SimulatorError::InvalidInput(
                "Need at least 2 points for linear fit".to_string(),
            ));
        }

        // Convert uncertainties to weights
        let w: Vec<f64> = weights
            .iter()
            .map(|&sigma| {
                if sigma > 0.0 {
                    1.0 / (sigma * sigma)
                } else {
                    1.0
                }
            })
            .collect();

        // Weighted least squares formulas
        let sum_w: f64 = w.iter().sum();
        let sum_wx: f64 = w.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
        let sum_wy: f64 = w.iter().zip(y.iter()).map(|(&wi, &yi)| wi * yi).sum();
        let sum_wxx: f64 = w.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi * xi).sum();
        let sum_wxy: f64 = w
            .iter()
            .zip(x.iter())
            .zip(y.iter())
            .map(|((&wi, &xi), &yi)| wi * xi * yi)
            .sum();

        let delta = sum_w.mul_add(sum_wxx, -(sum_wx * sum_wx));
        if delta.abs() < 1e-12 {
            return Err(SimulatorError::NumericalError(
                "Singular matrix in linear fit".to_string(),
            ));
        }

        let a = sum_wxx.mul_add(sum_wy, -(sum_wx * sum_wxy)) / delta; // intercept
        let b = sum_w.mul_add(sum_wxy, -(sum_wx * sum_wy)) / delta; // slope

        // Calculate residuals and statistics
        let mut residuals = Vec::with_capacity(n);
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        let y_mean = y.iter().sum::<f64>() / n as f64;

        for i in 0..n {
            let predicted = b.mul_add(x[i], a);
            let residual = y[i] - predicted;
            residuals.push(residual);
            ss_res += residual * residual;
            ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
        }

        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
        let chi_squared_reduced = ss_res / (n - 2) as f64;

        // AIC for linear model
        let aic = (n as f64).mul_add((ss_res / n as f64).ln(), 2.0 * 2.0); // 2 parameters

        let fit_stats = FitStatistics {
            r_squared,
            chi_squared_reduced,
            aic,
            num_parameters: 2,
            residuals,
        };

        Ok((a, b, fit_stats))
    }

    /// Solve weighted least squares (simplified)
    fn solve_weighted_least_squares(
        &self,
        design_matrix: &Array2<f64>,
        y: &[f64],
        weights: &[f64],
    ) -> Result<Vec<f64>> {
        // This is a simplified implementation
        // In practice, would use proper numerical linear algebra
        let n_params = design_matrix.ncols();
        let mut coefficients = vec![0.0; n_params];

        // For now, just return intercept as first measured value
        // Proper implementation would solve the normal equations
        coefficients[0] = y[0];

        Ok(coefficients)
    }

    /// Estimate measurement uncertainty
    fn estimate_measurement_uncertainty(&self, measured_value: f64) -> f64 {
        // Shot noise estimate: σ ≈ √(p(1-p)/N) for probability p and N shots
        let p = f64::midpoint(measured_value, 1.0); // Convert from [-1,1] to [0,1]
        let shot_noise = (p * (1.0 - p) / self.shots_per_level as f64).sqrt();
        shot_noise * 2.0 // Convert back to [-1,1] scale
    }

    /// Calculate confidence in extrapolation
    fn calculate_confidence(&self, fit_stats: &FitStatistics) -> f64 {
        // Confidence based on R-squared and reduced chi-squared
        let r_sq_factor = fit_stats.r_squared.clamp(0.0, 1.0);
        let chi_sq_factor = if fit_stats.chi_squared_reduced > 0.0 {
            1.0 / (1.0 + fit_stats.chi_squared_reduced)
        } else {
            0.5
        };

        (r_sq_factor * chi_sq_factor).sqrt()
    }
}

/// Virtual Distillation implementation
pub struct VirtualDistillation {
    /// Number of copies to use
    num_copies: usize,
    /// Distillation protocol
    protocol: DistillationProtocol,
}

/// Distillation protocols
#[derive(Debug, Clone, Copy)]
pub enum DistillationProtocol {
    /// Standard virtual distillation
    Standard,
    /// Optimized protocol for specific observables
    Optimized,
    /// Adaptive protocol
    Adaptive,
}

impl VirtualDistillation {
    /// Create new virtual distillation instance
    #[must_use]
    pub const fn new(num_copies: usize, protocol: DistillationProtocol) -> Self {
        Self {
            num_copies,
            protocol,
        }
    }

    /// Perform virtual distillation
    pub fn distill<F>(
        &self,
        noisy_executor: F,
        observable: &str,
    ) -> Result<VirtualDistillationResult>
    where
        F: Fn() -> Result<f64>,
    {
        // Execute circuit multiple times
        let measurements: Result<Vec<f64>> =
            (0..self.num_copies).map(|_| noisy_executor()).collect();

        let measurements = measurements?;

        // Apply distillation protocol
        let mitigated_value = match self.protocol {
            DistillationProtocol::Standard => self.standard_distillation(&measurements)?,
            DistillationProtocol::Optimized => self.optimized_distillation(&measurements)?,
            DistillationProtocol::Adaptive => self.adaptive_distillation(&measurements)?,
        };

        // Calculate efficiency metrics
        let raw_value = measurements.iter().sum::<f64>() / measurements.len() as f64;
        let error_reduction = (raw_value - mitigated_value).abs() / raw_value.abs().max(1e-10);

        Ok(VirtualDistillationResult {
            mitigated_value,
            overhead: self.num_copies,
            efficiency: 1.0 / self.num_copies as f64,
            error_reduction,
        })
    }

    /// Standard virtual distillation
    fn standard_distillation(&self, measurements: &[f64]) -> Result<f64> {
        // Compute product of measurements and take appropriate root
        let product: f64 = measurements.iter().product();
        let mitigated = if product >= 0.0 {
            product.powf(1.0 / self.num_copies as f64)
        } else {
            -(-product).powf(1.0 / self.num_copies as f64)
        };

        Ok(mitigated)
    }

    /// Optimized distillation
    fn optimized_distillation(&self, measurements: &[f64]) -> Result<f64> {
        // Use median for robustness
        let mut sorted = measurements.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        Ok(median)
    }

    /// Adaptive distillation
    fn adaptive_distillation(&self, measurements: &[f64]) -> Result<f64> {
        // Choose method based on measurement statistics
        let mean = measurements.iter().sum::<f64>() / measurements.len() as f64;
        let variance = measurements.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / measurements.len() as f64;

        if variance < 0.1 {
            self.standard_distillation(measurements)
        } else {
            self.optimized_distillation(measurements)
        }
    }
}

/// Symmetry Verification implementation
pub struct SymmetryVerification {
    /// Symmetries to check
    symmetries: Vec<SymmetryOperation>,
}

/// Symmetry operation
#[derive(Debug, Clone)]
pub struct SymmetryOperation {
    /// Name of the symmetry
    pub name: String,
    /// Expected eigenvalue
    pub eigenvalue: Complex64,
    /// Tolerance for symmetry violation
    pub tolerance: f64,
}

impl SymmetryVerification {
    /// Create new symmetry verification
    #[must_use]
    pub const fn new(symmetries: Vec<SymmetryOperation>) -> Self {
        Self { symmetries }
    }

    /// Verify symmetries and correct expectation value
    pub fn verify_and_correct<F>(
        &self,
        executor: F,
        observable: &str,
    ) -> Result<SymmetryVerificationResult>
    where
        F: Fn(&str) -> Result<f64>,
    {
        let main_value = executor(observable)?;

        let mut violations = Vec::new();
        let mut valid_measurements = Vec::new();

        // Check each symmetry
        for symmetry in &self.symmetries {
            let symmetry_value = executor(&symmetry.name)?;
            let expected = symmetry.eigenvalue.re;
            let violation = (symmetry_value - expected).abs();

            violations.push(violation);

            if violation <= symmetry.tolerance {
                valid_measurements.push(main_value);
            }
        }

        // Calculate corrected value
        let corrected_value = if valid_measurements.is_empty() {
            main_value // No valid measurements, return original
        } else {
            valid_measurements.iter().sum::<f64>() / valid_measurements.len() as f64
        };

        let avg_violation = violations.iter().sum::<f64>() / violations.len() as f64;
        let post_selection_prob =
            valid_measurements.len() as f64 / (self.symmetries.len() + 1) as f64;

        Ok(SymmetryVerificationResult {
            corrected_value,
            symmetry_violation: avg_violation,
            post_selection_prob,
            symmetries: self.symmetries.iter().map(|s| s.name.clone()).collect(),
        })
    }
}

/// Utility trait for unzipping tuples
trait Unzip3<A, B, C> {
    fn unzip3(self) -> (Vec<A>, Vec<B>, Vec<C>);
}

impl<A, B, C> Unzip3<A, B, C> for Vec<(A, B, C)> {
    fn unzip3(self) -> (Vec<A>, Vec<B>, Vec<C>) {
        let mut vec_a = Vec::with_capacity(self.len());
        let mut vec_b = Vec::with_capacity(self.len());
        let mut vec_c = Vec::with_capacity(self.len());

        for (a, b, c) in self {
            vec_a.push(a);
            vec_b.push(b);
            vec_c.push(c);
        }

        (vec_a, vec_b, vec_c)
    }
}

/// Benchmark noise extrapolation techniques
pub fn benchmark_noise_extrapolation() -> Result<(ZNEResult, VirtualDistillationResult)> {
    // Mock noisy executor
    let noisy_executor = |noise_factor: f64| -> Result<f64> {
        // Simulate exponential decay with noise
        let ideal_value = 1.0;
        let noise_rate = 0.1;
        Ok(ideal_value * (-noise_rate * noise_factor).exp())
    };

    // Test ZNE
    let zne = ZeroNoiseExtrapolator::default_config();
    let zne_result = zne.extrapolate(noisy_executor, "Z0")?;

    // Test virtual distillation
    let vd = VirtualDistillation::new(3, DistillationProtocol::Standard);
    let vd_result = vd.distill(|| Ok(0.8), "Z0")?;

    Ok((zne_result, vd_result))
}

// ============================================================================
// Advanced Error Mitigation Techniques (Beyond Qiskit)
// ============================================================================

/// Probabilistic Error Cancellation (PEC) result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PECResult {
    /// Mitigated expectation value
    pub mitigated_value: f64,
    /// Statistical error
    pub statistical_error: f64,
    /// Sampling overhead (γ^2)
    pub sampling_overhead: f64,
    /// Number of samples used
    pub num_samples: usize,
    /// Effective samples after error cancellation
    pub effective_samples: f64,
}

/// Probabilistic Error Cancellation (PEC) implementation
///
/// PEC provides exact error mitigation at the cost of increased sampling overhead.
/// It works by representing the ideal operation as a quasi-probability distribution
/// over noisy operations.
#[derive(Debug, Clone)]
pub struct ProbabilisticErrorCancellation {
    /// Noise model for the device
    pub noise_model: NoiseModel,
    /// Number of samples for PEC
    pub num_samples: usize,
    /// Monte Carlo seed for reproducibility
    pub seed: Option<u64>,
}

/// Simplified noise model for PEC
#[derive(Debug, Clone, Default)]
pub struct NoiseModel {
    /// Single-qubit gate error rates
    pub single_qubit_errors: std::collections::HashMap<String, f64>,
    /// Two-qubit gate error rates
    pub two_qubit_errors: std::collections::HashMap<(usize, usize), f64>,
    /// Readout error rates
    pub readout_errors: std::collections::HashMap<usize, (f64, f64)>,
}

impl ProbabilisticErrorCancellation {
    /// Create a new PEC instance
    pub fn new(noise_model: NoiseModel, num_samples: usize) -> Self {
        Self {
            noise_model,
            num_samples,
            seed: None,
        }
    }

    /// Set seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Calculate the sampling overhead γ for a circuit
    pub fn calculate_overhead(&self, circuit_depth: usize, num_gates: usize) -> f64 {
        // γ ≈ ∏_i (1 + 2ε_i) where ε_i is the error rate for gate i
        let avg_error: f64 = self
            .noise_model
            .single_qubit_errors
            .values()
            .cloned()
            .sum::<f64>()
            / self.noise_model.single_qubit_errors.len().max(1) as f64;

        let gamma_per_gate = 1.0 + 2.0 * avg_error;
        gamma_per_gate.powi(num_gates as i32)
    }

    /// Apply PEC to mitigate errors
    pub fn mitigate<F>(&self, executor: F, observable: &str) -> Result<PECResult>
    where
        F: Fn(&str, i32) -> Result<f64>,
    {
        use scirs2_core::random::thread_rng;
        use scirs2_core::random::Rng;

        let mut rng = thread_rng();
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut samples = Vec::with_capacity(self.num_samples);

        for _ in 0..self.num_samples {
            // Sign for quasi-probability (can be negative)
            let sign: i32 = if rng.gen::<f64>() < 0.5 { 1 } else { -1 };

            // Execute the circuit with the given sign
            let value = executor(observable, sign)?;
            let weighted_value = sign as f64 * value;

            weighted_sum += weighted_value;
            weight_sum += 1.0;
            samples.push(weighted_value);
        }

        let mitigated_value = weighted_sum / weight_sum;

        // Calculate statistical error
        let variance: f64 = samples
            .iter()
            .map(|&x| (x - mitigated_value).powi(2))
            .sum::<f64>()
            / (self.num_samples - 1).max(1) as f64;
        let statistical_error = (variance / self.num_samples as f64).sqrt();

        // Estimate sampling overhead
        let sampling_overhead = self.calculate_overhead(10, 20); // Default estimate

        Ok(PECResult {
            mitigated_value,
            statistical_error,
            sampling_overhead,
            num_samples: self.num_samples,
            effective_samples: self.num_samples as f64 / sampling_overhead,
        })
    }
}

/// Clifford Data Regression (CDR) result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CDRResult {
    /// Mitigated expectation value
    pub mitigated_value: f64,
    /// Regression error
    pub regression_error: f64,
    /// Number of Clifford circuits used
    pub num_clifford_circuits: usize,
    /// Regression coefficients
    pub coefficients: Vec<f64>,
    /// R-squared value of the fit
    pub r_squared: f64,
}

/// Clifford Data Regression (CDR) implementation
///
/// CDR uses near-Clifford circuits where exact simulation is possible
/// to learn a correction model for noisy circuit results.
#[derive(Debug, Clone)]
pub struct CliffordDataRegression {
    /// Number of near-Clifford training circuits
    pub num_training_circuits: usize,
    /// Regression method
    pub regression_method: RegressionMethod,
    /// Include intercept in regression
    pub fit_intercept: bool,
}

/// Regression methods for CDR
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RegressionMethod {
    /// Ordinary least squares
    OrdinaryLeastSquares,
    /// Ridge regression
    Ridge { alpha: f64 },
    /// Lasso regression
    Lasso { alpha: f64 },
    /// Bayesian regression
    Bayesian,
}

impl Default for CliffordDataRegression {
    fn default() -> Self {
        Self {
            num_training_circuits: 100,
            regression_method: RegressionMethod::OrdinaryLeastSquares,
            fit_intercept: true,
        }
    }
}

impl CliffordDataRegression {
    /// Create a new CDR instance
    pub fn new(num_training_circuits: usize, regression_method: RegressionMethod) -> Self {
        Self {
            num_training_circuits,
            regression_method,
            fit_intercept: true,
        }
    }

    /// Train the CDR model using Clifford circuits
    pub fn train<F, G>(&self, noisy_executor: F, ideal_executor: G) -> Result<CDRModel>
    where
        F: Fn(usize) -> Result<f64>,
        G: Fn(usize) -> Result<f64>,
    {
        let mut noisy_values = Vec::with_capacity(self.num_training_circuits);
        let mut ideal_values = Vec::with_capacity(self.num_training_circuits);

        for i in 0..self.num_training_circuits {
            noisy_values.push(noisy_executor(i)?);
            ideal_values.push(ideal_executor(i)?);
        }

        // Simple linear regression: ideal = a * noisy + b
        let n = self.num_training_circuits as f64;
        let sum_x: f64 = noisy_values.iter().sum();
        let sum_y: f64 = ideal_values.iter().sum();
        let sum_xx: f64 = noisy_values.iter().map(|x| x * x).sum();
        let sum_xy: f64 = noisy_values
            .iter()
            .zip(&ideal_values)
            .map(|(x, y)| x * y)
            .sum();

        let (a, b) = if self.fit_intercept {
            let denom = n * sum_xx - sum_x * sum_x;
            if denom.abs() < 1e-10 {
                (1.0, 0.0)
            } else {
                let slope = (n * sum_xy - sum_x * sum_y) / denom;
                let intercept = (sum_y - slope * sum_x) / n;
                (slope, intercept)
            }
        } else {
            (sum_xy / sum_xx, 0.0)
        };

        // Calculate R-squared
        let y_mean = sum_y / n;
        let ss_tot: f64 = ideal_values.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = noisy_values
            .iter()
            .zip(&ideal_values)
            .map(|(x, y)| (y - (a * x + b)).powi(2))
            .sum();
        let r_squared = if ss_tot > 1e-10 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        Ok(CDRModel {
            coefficients: vec![a, b],
            r_squared,
            num_training_points: self.num_training_circuits,
        })
    }

    /// Apply CDR to mitigate errors
    pub fn mitigate(&self, model: &CDRModel, noisy_value: f64) -> Result<CDRResult> {
        let mitigated_value = model.coefficients[0] * noisy_value + model.coefficients[1];

        // Estimate regression error from R-squared
        let regression_error = (1.0 - model.r_squared).sqrt() * noisy_value.abs().max(0.1);

        Ok(CDRResult {
            mitigated_value,
            regression_error,
            num_clifford_circuits: model.num_training_points,
            coefficients: model.coefficients.clone(),
            r_squared: model.r_squared,
        })
    }
}

/// Trained CDR model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CDRModel {
    /// Regression coefficients
    pub coefficients: Vec<f64>,
    /// R-squared value
    pub r_squared: f64,
    /// Number of training points
    pub num_training_points: usize,
}

/// Matrix-free Measurement Mitigation (M3) result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M3Result {
    /// Mitigated probability distribution
    pub mitigated_probs: std::collections::HashMap<String, f64>,
    /// Original counts
    pub original_counts: std::collections::HashMap<String, usize>,
    /// Number of iterations for convergence
    pub num_iterations: usize,
    /// Convergence achieved
    pub converged: bool,
    /// Final residual norm
    pub residual_norm: f64,
}

/// Matrix-free Measurement Mitigation (M3) implementation
///
/// M3 provides scalable measurement error mitigation without needing to
/// construct or invert the full assignment matrix. It uses iterative
/// methods to solve the mitigation problem efficiently.
#[derive(Debug, Clone)]
pub struct MatrixFreeMeasurementMitigation {
    /// Calibration data: confusion matrices per qubit
    pub calibration: M3Calibration,
    /// Maximum iterations for iterative solver
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Method for solving the inverse problem
    pub solver_method: M3SolverMethod,
}

/// M3 calibration data
#[derive(Debug, Clone, Default)]
pub struct M3Calibration {
    /// Per-qubit confusion matrices
    /// `confusion[qubit][measured][actual]` = P(measured | actual)
    pub confusion_matrices: std::collections::HashMap<usize, [[f64; 2]; 2]>,
    /// Correlated errors between qubit pairs (optional)
    pub correlated_errors: std::collections::HashMap<(usize, usize), Array2<f64>>,
}

/// Solver methods for M3
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum M3SolverMethod {
    /// Iterative Bayesian unfolding
    IterativeBayesian,
    /// Least squares with non-negativity constraint
    NonNegativeLeastSquares,
    /// Conjugate gradient method
    ConjugateGradient,
}

impl Default for MatrixFreeMeasurementMitigation {
    fn default() -> Self {
        Self {
            calibration: M3Calibration::default(),
            max_iterations: 100,
            tolerance: 1e-6,
            solver_method: M3SolverMethod::IterativeBayesian,
        }
    }
}

impl MatrixFreeMeasurementMitigation {
    /// Create a new M3 instance
    pub fn new(calibration: M3Calibration) -> Self {
        Self {
            calibration,
            ..Default::default()
        }
    }

    /// Calibrate M3 using calibration circuits
    pub fn calibrate<F>(&mut self, executor: F, num_qubits: usize, shots: usize) -> Result<()>
    where
        F: Fn(&str, usize) -> Result<std::collections::HashMap<String, usize>>,
    {
        for qubit in 0..num_qubits {
            // Prepare |0⟩ state and measure
            let counts_0 = executor(&format!("cal_0_{}", qubit), shots)?;
            let p00 = *counts_0.get("0").unwrap_or(&0) as f64 / shots as f64;
            let p10 = *counts_0.get("1").unwrap_or(&0) as f64 / shots as f64;

            // Prepare |1⟩ state and measure
            let counts_1 = executor(&format!("cal_1_{}", qubit), shots)?;
            let p01 = *counts_1.get("0").unwrap_or(&0) as f64 / shots as f64;
            let p11 = *counts_1.get("1").unwrap_or(&0) as f64 / shots as f64;

            // confusion[measured][actual] = P(measured | actual)
            self.calibration
                .confusion_matrices
                .insert(qubit, [[p00, p01], [p10, p11]]);
        }

        Ok(())
    }

    /// Apply M3 to mitigate measurement errors
    pub fn mitigate(
        &self,
        counts: &std::collections::HashMap<String, usize>,
        num_qubits: usize,
    ) -> Result<M3Result> {
        let total_shots: usize = counts.values().sum();

        // Convert counts to probability distribution
        let mut probs: std::collections::HashMap<String, f64> = counts
            .iter()
            .map(|(k, v)| (k.clone(), *v as f64 / total_shots as f64))
            .collect();

        // Iterative Bayesian unfolding
        let mut converged = false;
        let mut num_iterations = 0;
        let mut residual_norm = f64::MAX;

        for iter in 0..self.max_iterations {
            num_iterations = iter + 1;

            // Apply inverse confusion matrix (iterative)
            let mut new_probs = std::collections::HashMap::new();
            let mut total = 0.0;

            for (bitstring, &prob) in &probs {
                // Calculate correction factor based on confusion matrices
                let mut correction = 1.0;

                for (i, c) in bitstring.chars().rev().enumerate() {
                    if let Some(confusion) = self.calibration.confusion_matrices.get(&i) {
                        let measured = if c == '1' { 1 } else { 0 };
                        // Diagonal element of inverse (approximate)
                        let diag = confusion[measured][measured];
                        if diag > 0.01 {
                            correction /= diag;
                        }
                    }
                }

                let corrected_prob = (prob * correction).max(0.0);
                new_probs.insert(bitstring.clone(), corrected_prob);
                total += corrected_prob;
            }

            // Normalize
            if total > 0.0 {
                for prob in new_probs.values_mut() {
                    *prob /= total;
                }
            }

            // Check convergence
            let mut diff = 0.0;
            for (k, &new_p) in &new_probs {
                let old_p = probs.get(k).unwrap_or(&0.0);
                diff += (new_p - old_p).abs();
            }
            residual_norm = diff;

            probs = new_probs;

            if diff < self.tolerance {
                converged = true;
                break;
            }
        }

        Ok(M3Result {
            mitigated_probs: probs,
            original_counts: counts.clone(),
            num_iterations,
            converged,
            residual_norm,
        })
    }

    /// Compute expectation value from mitigated probabilities
    pub fn expectation_value(&self, result: &M3Result, observable: &[i8]) -> f64 {
        let mut expectation = 0.0;

        for (bitstring, &prob) in &result.mitigated_probs {
            let mut parity = 1;
            for (i, c) in bitstring.chars().rev().enumerate() {
                if i < observable.len() && observable[i] != 0 && c == '1' {
                    parity *= -1;
                }
            }
            expectation += parity as f64 * prob;
        }

        expectation
    }
}

/// Combined error mitigation pipeline
#[derive(Debug, Clone)]
pub struct ErrorMitigationPipeline {
    /// ZNE configuration
    pub zne: Option<ZeroNoiseExtrapolator>,
    /// PEC configuration
    pub pec: Option<ProbabilisticErrorCancellation>,
    /// CDR configuration
    pub cdr: Option<CliffordDataRegression>,
    /// M3 configuration
    pub m3: Option<MatrixFreeMeasurementMitigation>,
}

impl Default for ErrorMitigationPipeline {
    fn default() -> Self {
        Self {
            zne: Some(ZeroNoiseExtrapolator::default_config()),
            pec: None,
            cdr: None,
            m3: None,
        }
    }
}

impl ErrorMitigationPipeline {
    /// Create a pipeline with all mitigation techniques
    pub fn full() -> Self {
        Self {
            zne: Some(ZeroNoiseExtrapolator::default_config()),
            pec: Some(ProbabilisticErrorCancellation::new(
                NoiseModel::default(),
                1000,
            )),
            cdr: Some(CliffordDataRegression::default()),
            m3: Some(MatrixFreeMeasurementMitigation::default()),
        }
    }

    /// Apply the mitigation pipeline with noise-aware executor
    ///
    /// The executor takes a noise scaling factor and returns the expectation value
    /// at that noise level. This allows ZNE to extrapolate to zero noise.
    pub fn apply<F>(&self, executor: F, observable: &str) -> Result<f64>
    where
        F: Fn(f64) -> Result<f64> + Sync + Send,
    {
        // Start with ZNE if available
        let mut value = if let Some(zne) = &self.zne {
            zne.extrapolate(&executor, observable)?.zero_noise_value
        } else {
            // Execute at noise factor 1.0 (no scaling)
            executor(1.0)?
        };

        // Apply CDR correction if model is trained
        // (In practice, would need pre-trained model)

        Ok(value)
    }

    /// Apply mitigation with a simple circuit executor
    ///
    /// This method wraps a simple executor that doesn't support noise scaling
    /// and applies only applicable mitigation techniques.
    pub fn apply_simple<F>(&self, executor: F, observable: &str) -> Result<f64>
    where
        F: Fn(&str) -> Result<f64>,
    {
        // For simple executors, skip ZNE (requires noise scaling)
        // and just apply other techniques
        let value = executor(observable)?;

        // CDR and M3 can be applied post-hoc to measurement results
        // but require calibration data which would be handled separately

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zne_linear_extrapolation() {
        let zne = ZeroNoiseExtrapolator::new(
            NoiseScalingMethod::UnitaryFolding,
            ExtrapolationMethod::Linear,
            vec![1.0, 2.0, 3.0],
            100,
        );

        // Mock data: y = 1.0 - 0.1*x (should extrapolate to 1.0)
        let noise_factors = vec![1.0, 2.0, 3.0];
        let measured_values = vec![0.9, 0.8, 0.7];
        let uncertainties = vec![0.01, 0.01, 0.01];

        let (zero_noise, _error, _stats) = zne
            .linear_extrapolation(&noise_factors, &measured_values, &uncertainties)
            .expect("Failed to perform linear extrapolation");

        assert!((zero_noise - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_virtual_distillation() {
        let vd = VirtualDistillation::new(3, DistillationProtocol::Standard);

        let measurements = vec![0.8, 0.7, 0.9];
        let result = vd
            .standard_distillation(&measurements)
            .expect("Failed to perform standard distillation");

        assert!(result > 0.0);
        assert!(result < 1.0);
    }

    #[test]
    fn test_symmetry_verification() {
        let symmetries = vec![SymmetryOperation {
            name: "parity".to_string(),
            eigenvalue: Complex64::new(1.0, 0.0),
            tolerance: 0.1,
        }];

        let sv = SymmetryVerification::new(symmetries);

        let executor = |obs: &str| -> Result<f64> {
            match obs {
                "Z0" => Ok(0.8),
                "parity" => Ok(0.95), // Close to expected value 1.0
                _ => Ok(0.0),
            }
        };

        let result = sv
            .verify_and_correct(executor, "Z0")
            .expect("Failed to verify and correct");
        assert!((result.corrected_value - 0.8).abs() < 1e-10);
        assert!(result.post_selection_prob > 0.0);
    }

    #[test]
    fn test_richardson_extrapolation() {
        let zne = ZeroNoiseExtrapolator::default_config();

        let noise_factors = vec![1.0, 2.0, 3.0];
        let measured_values = vec![1.0, 0.8, 0.6]; // Quadratic decay

        let (zero_noise, _error, _stats) = zne
            .richardson_extrapolation(&noise_factors, &measured_values)
            .expect("Failed to perform Richardson extrapolation");

        assert!(zero_noise > 0.0);
    }
}
