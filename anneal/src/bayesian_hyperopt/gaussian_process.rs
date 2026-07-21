//! Gaussian Process Surrogate Models
//!
//! This module provides a real Gaussian-process regression surrogate used by the
//! Bayesian hyperparameter optimizer. Predictions follow the standard exact-GP
//! equations (Rasmussen & Williams, *Gaussian Processes for Machine Learning*,
//! Algorithm 2.1): the training kernel matrix `K + σ²I` is Cholesky-factorized
//! once per fit, and the predictive mean and variance are obtained from triangular
//! solves against the resulting lower-triangular factor.

use super::config::{BayesianOptError, BayesianOptResult};
use std::f64::consts::PI;

/// Gaussian process configuration (alias for backward compatibility)
pub type GaussianProcessConfig = GaussianProcessSurrogate;

/// Gaussian process configuration holder.
///
/// This type stores only the *configuration* of a Gaussian process (kernel, noise
/// level, and prior mean function). It intentionally carries no training data and
/// therefore cannot produce predictions on its own — construct a
/// [`GaussianProcessModel`] from observed data for a fitted, predictive process.
#[derive(Debug, Clone)]
pub struct GaussianProcessSurrogate {
    pub kernel: KernelFunction,
    pub noise_variance: f64,
    pub mean_function: MeanFunction,
}

impl Default for GaussianProcessSurrogate {
    fn default() -> Self {
        Self {
            kernel: KernelFunction::RBF,
            noise_variance: 1e-6,
            mean_function: MeanFunction::Zero,
        }
    }
}

impl GaussianProcessSurrogate {
    /// Predictions are not available on a bare configuration holder.
    ///
    /// `GaussianProcessSurrogate` describes *how* a GP should behave but holds no
    /// observations, so it has nothing to condition on. Build a
    /// [`GaussianProcessModel`] from training data (`GaussianProcessModel::new`)
    /// to obtain real posterior mean/variance predictions. This method returns an
    /// honest error rather than a fabricated `(0.0, 1.0)` so that a misconfigured
    /// call site fails loudly instead of silently degrading to a constant.
    pub fn predict(&self, _x: &[f64]) -> BayesianOptResult<(f64, f64)> {
        Err(BayesianOptError::GaussianProcessError(
            "GaussianProcessSurrogate stores configuration only and cannot predict; \
             construct a GaussianProcessModel from training data instead"
                .to_string(),
        ))
    }
}

/// Kernel functions for Gaussian processes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelFunction {
    /// Radial Basis Function (RBF) kernel
    RBF,
    /// Matern kernel
    Matern,
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial,
    /// Spectral mixture kernel
    SpectralMixture,
}

/// Mean functions for Gaussian processes
#[derive(Debug, Clone, PartialEq)]
pub enum MeanFunction {
    /// Zero mean function
    Zero,
    /// Constant mean function
    Constant(f64),
    /// Linear mean function
    Linear,
    /// Polynomial mean function
    Polynomial { degree: usize },
}

/// Gaussian process hyperparameters
#[derive(Debug, Clone)]
pub struct GPHyperparameters {
    pub length_scales: Vec<f64>,
    pub signal_variance: f64,
    pub noise_variance: f64,
    pub mean_parameters: Vec<f64>,
}

impl Default for GPHyperparameters {
    fn default() -> Self {
        Self {
            length_scales: vec![1.0],
            signal_variance: 1.0,
            noise_variance: 1e-6,
            mean_parameters: vec![0.0],
        }
    }
}

/// Gaussian Process regression model.
///
/// Conditioned on training data, this model provides posterior mean and variance
/// predictions via an exact Cholesky-based solve of `K + σ²I`.
#[derive(Debug, Clone)]
pub struct GaussianProcessModel {
    /// Training input data
    pub x_train: Vec<Vec<f64>>,
    /// Training output data
    pub y_train: Vec<f64>,
    /// GP configuration
    pub config: GaussianProcessConfig,
    /// Learned hyperparameters
    pub hyperparameters: GPHyperparameters,
    /// Lower-triangular Cholesky factor `L` of `K + σ²I` (row-major), where
    /// `L * Lᵀ = K + σ²I`. Populated by [`GaussianProcessModel::fit`].
    l_factor: Option<Vec<Vec<f64>>>,
    /// Precomputed `α = (K + σ²I)⁻¹ (y − m)`, used for the predictive mean.
    alpha: Option<Vec<f64>>,
}

impl GaussianProcessModel {
    /// Create new Gaussian Process model
    pub fn new(
        x_train: Vec<Vec<f64>>,
        y_train: Vec<f64>,
        config: GaussianProcessConfig,
    ) -> BayesianOptResult<Self> {
        if x_train.len() != y_train.len() {
            return Err(BayesianOptError::GaussianProcessError(
                "Training inputs and outputs must have same length".to_string(),
            ));
        }

        if x_train.is_empty() {
            return Err(BayesianOptError::GaussianProcessError(
                "Training data cannot be empty".to_string(),
            ));
        }

        let input_dim = x_train[0].len();
        let hyperparameters = GPHyperparameters {
            length_scales: vec![1.0; input_dim.max(1)],
            signal_variance: 1.0,
            noise_variance: config.noise_variance,
            mean_parameters: vec![0.0],
        };

        let mut model = Self {
            x_train,
            y_train,
            config,
            hyperparameters,
            l_factor: None,
            alpha: None,
        };

        // Fit the model (heuristic hyperparameters + Cholesky factorization).
        model.fit()?;

        Ok(model)
    }

    /// Fit the Gaussian Process model.
    ///
    /// Sets heuristic hyperparameters, then Cholesky-factorizes `K + σ²I` and
    /// precomputes `α` for fast predictive-mean evaluation.
    pub fn fit(&mut self) -> BayesianOptResult<()> {
        self.optimize_hyperparameters()?;
        self.factorize()?;
        Ok(())
    }

    /// Heuristic hyperparameter selection.
    ///
    /// Length scales are set to half the per-dimension data range and the signal
    /// variance to the empirical output variance. (A full type-II maximum-likelihood
    /// optimization could refine these, but these data-driven heuristics keep the
    /// kernel well-conditioned for the small designs seen during Bayesian
    /// optimization.)
    fn optimize_hyperparameters(&mut self) -> BayesianOptResult<()> {
        let n = self.x_train.len();
        if n == 0 {
            return Ok(());
        }

        let input_dim = self.x_train[0].len();

        // Set length scales based on the spread of the observed inputs.
        for dim in 0..input_dim {
            let values: Vec<f64> = self.x_train.iter().map(|x| x[dim]).collect();
            let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = (max_val - min_val).max(1e-6);

            self.hyperparameters.length_scales[dim] = range / 2.0;
        }

        // Set signal variance based on the empirical output variance.
        let mean_y = self.y_train.iter().sum::<f64>() / n as f64;
        let var_y = self
            .y_train
            .iter()
            .map(|&y| (y - mean_y).powi(2))
            .sum::<f64>()
            / n as f64;

        self.hyperparameters.signal_variance = var_y.max(1e-6);

        Ok(())
    }

    /// Cholesky-factorize `K + σ²I` and precompute `α`.
    ///
    /// If the (nominally positive-definite) matrix is numerically indefinite —
    /// e.g. because two training inputs nearly coincide — an increasing jitter is
    /// added to the diagonal until the factorization succeeds, a standard
    /// regularization used by production GP libraries.
    fn factorize(&mut self) -> BayesianOptResult<()> {
        let n = self.x_train.len();

        // Build the (noise-free) kernel Gram matrix.
        let mut k_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let value = self.kernel(&self.x_train[i], &self.x_train[j]);
                k_matrix[i][j] = value;
                k_matrix[j][i] = value;
            }
        }

        let signal_scale = self.hyperparameters.signal_variance.max(1e-12);
        let mut jitter = self.hyperparameters.noise_variance.max(0.0);

        let mut factor = None;
        for _attempt in 0..8 {
            let mut regularized = k_matrix.clone();
            for d in 0..n {
                regularized[d][d] += jitter;
            }
            if let Some(l) = cholesky_lower(&regularized) {
                factor = Some(l);
                break;
            }
            // Grow the jitter geometrically (seeded relative to the signal scale
            // when the noise term is zero) and retry.
            jitter = if jitter <= 0.0 {
                1e-10 * signal_scale
            } else {
                jitter * 10.0
            };
        }

        let l = factor.ok_or_else(|| {
            BayesianOptError::GaussianProcessError(
                "Kernel matrix is not positive definite even after jitter regularization"
                    .to_string(),
            )
        })?;

        // Center targets by the prior mean, then solve (K + σ²I) α = (y − m) via
        // two triangular solves: L z = (y − m), then Lᵀ α = z.
        let prior_mean = self.prior_mean_vector();
        let centered: Vec<f64> = self
            .y_train
            .iter()
            .zip(prior_mean.iter())
            .map(|(&y, &m)| y - m)
            .collect();

        let z = forward_substitution(&l, &centered);
        let alpha = back_substitution_transpose(&l, &z);

        self.l_factor = Some(l);
        self.alpha = Some(alpha);

        Ok(())
    }

    /// Prior-mean values evaluated at every training input.
    fn prior_mean_vector(&self) -> Vec<f64> {
        self.x_train
            .iter()
            .map(|x| self.mean_function_value(x))
            .collect()
    }

    /// Compute kernel function between two points
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        match self.config.kernel {
            KernelFunction::RBF => self.rbf_kernel(x1, x2),
            KernelFunction::Matern => self.matern_kernel(x1, x2),
            KernelFunction::Linear => self.linear_kernel(x1, x2),
            KernelFunction::Polynomial => self.polynomial_kernel(x1, x2),
            KernelFunction::SpectralMixture => self.rbf_kernel(x1, x2), // Fallback to RBF
        }
    }

    /// RBF (Gaussian) kernel with per-dimension length scales.
    fn rbf_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let mut distance_sq = 0.0;
        for (i, (&xi, &xj)) in x1.iter().zip(x2.iter()).enumerate() {
            let length_scale = self.hyperparameters.length_scales.get(i).unwrap_or(&1.0);
            distance_sq += ((xi - xj) / length_scale).powi(2);
        }

        self.hyperparameters.signal_variance * (-0.5 * distance_sq).exp()
    }

    /// Matern kernel (Matern 3/2).
    fn matern_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let mut distance = 0.0;
        for (i, (&xi, &xj)) in x1.iter().zip(x2.iter()).enumerate() {
            let length_scale = self.hyperparameters.length_scales.get(i).unwrap_or(&1.0);
            distance += ((xi - xj) / length_scale).powi(2);
        }
        distance = distance.sqrt();

        let sqrt3_r = 3.0_f64.sqrt() * distance;
        self.hyperparameters.signal_variance * (1.0 + sqrt3_r) * (-sqrt3_r).exp()
    }

    /// Linear kernel
    fn linear_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(&xi, &xj)| xi * xj).sum();
        self.hyperparameters.signal_variance * dot_product
    }

    /// Polynomial kernel
    fn polynomial_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(&xi, &xj)| xi * xj).sum();
        self.hyperparameters.signal_variance * (1.0 + dot_product).powi(2)
    }

    /// Predict the posterior mean and variance at a new point.
    ///
    /// Implements the exact-GP predictive equations:
    /// `μ(x*) = m(x*) + k*ᵀ α` and `σ²(x*) = k(x*, x*) − vᵀ v` with `v = L⁻¹ k*`.
    pub fn predict(&self, x: &[f64]) -> BayesianOptResult<(f64, f64)> {
        let l = self.l_factor.as_ref().ok_or_else(|| {
            BayesianOptError::GaussianProcessError("Model not fitted".to_string())
        })?;
        let alpha = self.alpha.as_ref().ok_or_else(|| {
            BayesianOptError::GaussianProcessError("Model not fitted".to_string())
        })?;

        // Cross-covariance k* between x and every training input.
        let k_star: Vec<f64> = self
            .x_train
            .iter()
            .map(|x_train| self.kernel(x, x_train))
            .collect();

        // Predictive mean: prior mean plus k*ᵀ α.
        let mut mean = self.mean_function_value(x);
        for (ks, a) in k_star.iter().zip(alpha.iter()) {
            mean += ks * a;
        }

        // Predictive variance: k(x, x) − ||L⁻¹ k*||².
        let v = forward_substitution(l, &k_star);
        let mut variance = self.kernel(x, x);
        for vi in &v {
            variance -= vi * vi;
        }

        // Numerical guard: posterior variance must stay non-negative.
        variance = variance.max(1e-12);

        Ok((mean, variance))
    }

    /// Evaluate mean function
    fn mean_function_value(&self, x: &[f64]) -> f64 {
        match self.config.mean_function {
            MeanFunction::Zero => 0.0,
            MeanFunction::Constant(c) => c,
            MeanFunction::Linear => {
                // Simple linear mean: sum of coordinates
                x.iter().sum::<f64>() * self.hyperparameters.mean_parameters.first().unwrap_or(&0.0)
            }
            MeanFunction::Polynomial { degree: _ } => {
                // Simplified polynomial mean
                let x_sum = x.iter().sum::<f64>();
                x_sum * self.hyperparameters.mean_parameters.first().unwrap_or(&0.0)
            }
        }
    }

    /// Exact log marginal likelihood of the training data under the fitted GP.
    ///
    /// `log p(y) = −½ (y − m)ᵀ α − Σ ln L_ii − ½ n ln(2π)`, where the middle term
    /// equals `½ ln|K + σ²I|` because `L` is the Cholesky factor.
    pub fn log_marginal_likelihood(&self) -> BayesianOptResult<f64> {
        let l = self.l_factor.as_ref().ok_or_else(|| {
            BayesianOptError::GaussianProcessError("Model not fitted".to_string())
        })?;
        let alpha = self.alpha.as_ref().ok_or_else(|| {
            BayesianOptError::GaussianProcessError("Model not fitted".to_string())
        })?;

        let n = self.y_train.len();
        let prior_mean = self.prior_mean_vector();

        // Data-fit term: (y − m)ᵀ α.
        let mut data_fit = 0.0;
        for i in 0..n {
            data_fit += (self.y_train[i] - prior_mean[i]) * alpha[i];
        }

        // Complexity term: ½ ln|K| = Σ ln L_ii.
        let mut half_log_det = 0.0;
        for i in 0..n {
            half_log_det += l[i][i].ln();
        }

        let log_likelihood = (-0.5 * data_fit) - half_log_det - (0.5 * n as f64) * (2.0 * PI).ln();

        Ok(log_likelihood)
    }
}

/// Compute the lower-triangular Cholesky factor `L` of a symmetric matrix `a`,
/// such that `L * Lᵀ = a`.
///
/// Returns `None` if `a` is not positive definite (a non-positive pivot appears),
/// which the caller uses to trigger jitter regularization.
fn cholesky_lower(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i][i] = sum.sqrt();
            } else {
                let pivot = l[j][j];
                if pivot.abs() < 1e-300 {
                    return None;
                }
                l[i][j] = sum / pivot;
            }
        }
    }

    Some(l)
}

/// Solve the lower-triangular system `L y = b` for `y` by forward substitution.
fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];

    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }

    y
}

/// Solve the upper-triangular system `Lᵀ x = z` for `x` by back substitution,
/// using the lower-triangular factor `L` transposed implicitly.
fn back_substitution_transpose(l: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
    let n = z.len();
    let mut x = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: fit an exact GP to a sampled 1-D quadratic and verify the
    /// posterior interpolates the training data with near-zero variance there,
    /// while variance grows far from the data.
    #[test]
    fn test_gp_cholesky_quadratic_regression() {
        // f(x) = (x - 2)^2 sampled on an integer grid.
        let grid = [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let x_train: Vec<Vec<f64>> = grid.iter().map(|&x| vec![x]).collect();
        let y_train: Vec<f64> = grid.iter().map(|&x| (x - 2.0).powi(2)).collect();

        let config = GaussianProcessSurrogate {
            kernel: KernelFunction::RBF,
            noise_variance: 1e-8,
            mean_function: MeanFunction::Zero,
        };

        let model = GaussianProcessModel::new(x_train, y_train.clone(), config)
            .expect("GP should fit on well-separated quadratic samples");

        // At each training input the posterior mean matches the observation and the
        // posterior variance is tiny.
        let mut train_var_max = 0.0f64;
        for (&x, &y) in grid.iter().zip(y_train.iter()) {
            let (mean, variance) = model.predict(&[x]).expect("prediction should succeed");
            assert!(
                (mean - y).abs() < 1e-3,
                "at x={x}, predicted mean {mean} should match target {y}"
            );
            assert!(
                variance < 1e-2,
                "posterior variance {variance} at training point x={x} should be near zero"
            );
            train_var_max = train_var_max.max(variance);
        }

        // Interpolation between samples: the mean should track the underlying
        // quadratic and the variance stays finite and positive.
        let (mean_mid, var_mid) = model.predict(&[2.5]).expect("interpolation should succeed");
        let true_mid = (2.5f64 - 2.0).powi(2);
        assert!(
            (mean_mid - true_mid).abs() < 0.75,
            "interpolated mean {mean_mid} should be near the true value {true_mid}"
        );
        assert!(var_mid > 0.0, "interpolation variance should be positive");

        // Extrapolation far from the data has much larger posterior variance than
        // at a training point.
        let (_mean_far, var_far) = model
            .predict(&[12.0])
            .expect("extrapolation should succeed");
        assert!(
            var_far > 10.0 * train_var_max,
            "extrapolation variance {var_far} should exceed training-point variance {train_var_max}"
        );
    }

    /// The Cholesky factor must satisfy L Lᵀ = A for a known SPD matrix.
    #[test]
    fn test_cholesky_lower_reconstructs_matrix() {
        let a = vec![
            vec![4.0, 2.0, 2.0],
            vec![2.0, 5.0, 3.0],
            vec![2.0, 3.0, 6.0],
        ];
        let l = cholesky_lower(&a).expect("SPD matrix should factorize");

        for i in 0..3 {
            for j in 0..3 {
                let mut reconstructed = 0.0;
                for k in 0..3 {
                    reconstructed += l[i][k] * l[j][k];
                }
                assert!(
                    (reconstructed - a[i][j]).abs() < 1e-9,
                    "L Lᵀ mismatch at ({i},{j})"
                );
            }
        }
    }

    /// A non-positive-definite matrix must be rejected (so callers can add jitter).
    #[test]
    fn test_cholesky_rejects_non_pd() {
        // Negative eigenvalue -> not positive definite.
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(cholesky_lower(&a).is_none());
    }

    /// Triangular solves must invert the factorization: L Lᵀ x = b.
    #[test]
    fn test_triangular_solves_roundtrip() {
        let a = vec![
            vec![4.0, 2.0, 2.0],
            vec![2.0, 5.0, 3.0],
            vec![2.0, 3.0, 6.0],
        ];
        let l = cholesky_lower(&a).expect("SPD matrix should factorize");
        let b = vec![1.0, -2.0, 3.0];

        // Solve A x = b via L z = b, Lᵀ x = z.
        let z = forward_substitution(&l, &b);
        let x = back_substitution_transpose(&l, &z);

        // Verify A x = b.
        for i in 0..3 {
            let mut ax = 0.0;
            for j in 0..3 {
                ax += a[i][j] * x[j];
            }
            assert!((ax - b[i]).abs() < 1e-9, "A x != b at row {i}");
        }
    }

    /// The bare configuration holder must fail loudly rather than fabricate a value.
    #[test]
    fn test_surrogate_predict_is_honest_error() {
        let surrogate = GaussianProcessSurrogate::default();
        assert!(surrogate.predict(&[0.0]).is_err());
    }

    /// Log marginal likelihood is finite for a well-conditioned fit.
    #[test]
    fn test_log_marginal_likelihood_finite() {
        let x_train = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y_train = vec![0.0, 1.0, 4.0, 9.0];
        let config = GaussianProcessSurrogate {
            kernel: KernelFunction::RBF,
            noise_variance: 1e-6,
            mean_function: MeanFunction::Zero,
        };
        let model = GaussianProcessModel::new(x_train, y_train, config).expect("fit");
        let lml = model
            .log_marginal_likelihood()
            .expect("log marginal likelihood should be computable");
        assert!(lml.is_finite());
    }
}
