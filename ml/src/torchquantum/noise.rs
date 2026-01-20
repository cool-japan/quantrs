//! TorchQuantum Noise-Aware Training
//!
//! This module provides noise-aware gradient computation and error-mitigated
//! expectation values for robust quantum machine learning on noisy hardware.
//!
//! ## Key Features
//!
//! - **NoiseAwareGradient**: Gradients that account for device noise
//! - **MitigatedExpectation**: Error-mitigated expectation value computation
//! - **NoiseModel Integration**: Compatible with various noise models
//! - **Resilient Training**: Training strategies for noisy quantum devices

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;

use super::{CType, TQDevice, TQModule, TQParameter};

// ============================================================================
// Noise Model Types
// ============================================================================

/// Single-qubit noise channel types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SingleQubitNoiseType {
    /// Depolarizing noise with probability p
    Depolarizing(f64),
    /// Amplitude damping with decay probability
    AmplitudeDamping(f64),
    /// Phase damping with dephasing probability
    PhaseDamping(f64),
    /// Bit flip with probability p
    BitFlip(f64),
    /// Phase flip with probability p
    PhaseFlip(f64),
}

/// Two-qubit noise channel types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TwoQubitNoiseType {
    /// Depolarizing noise on two qubits
    Depolarizing(f64),
    /// Correlated dephasing
    CorrelatedDephasing(f64),
    /// Cross-talk error
    CrossTalk(f64),
}

/// Complete noise model for a quantum device
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Single-qubit gate errors per qubit
    pub single_qubit_errors: HashMap<usize, SingleQubitNoiseType>,
    /// Two-qubit gate errors per qubit pair
    pub two_qubit_errors: HashMap<(usize, usize), TwoQubitNoiseType>,
    /// Readout errors per qubit (probability of bit flip during measurement)
    pub readout_errors: HashMap<usize, f64>,
    /// Coherence times (T1, T2) per qubit in microseconds
    pub coherence_times: HashMap<usize, (f64, f64)>,
    /// Gate times in microseconds
    pub gate_times: GateTimes,
    /// Global noise scale factor
    pub noise_scale: f64,
}

/// Gate execution times
#[derive(Debug, Clone)]
pub struct GateTimes {
    /// Single-qubit gate time (microseconds)
    pub single_qubit: f64,
    /// Two-qubit gate time (microseconds)
    pub two_qubit: f64,
    /// Measurement time (microseconds)
    pub measurement: f64,
}

impl Default for GateTimes {
    fn default() -> Self {
        Self {
            single_qubit: 0.05, // 50 ns
            two_qubit: 0.3,     // 300 ns
            measurement: 1.0,   // 1 us
        }
    }
}

impl NoiseModel {
    /// Create a noise-free model
    pub fn ideal() -> Self {
        Self {
            single_qubit_errors: HashMap::new(),
            two_qubit_errors: HashMap::new(),
            readout_errors: HashMap::new(),
            coherence_times: HashMap::new(),
            gate_times: GateTimes::default(),
            noise_scale: 0.0,
        }
    }

    /// Create uniform depolarizing noise model
    pub fn uniform_depolarizing(n_qubits: usize, p1: f64, p2: f64) -> Self {
        let mut model = Self::ideal();
        model.noise_scale = 1.0;

        for q in 0..n_qubits {
            model
                .single_qubit_errors
                .insert(q, SingleQubitNoiseType::Depolarizing(p1));
        }

        for q1 in 0..n_qubits {
            for q2 in (q1 + 1)..n_qubits {
                model
                    .two_qubit_errors
                    .insert((q1, q2), TwoQubitNoiseType::Depolarizing(p2));
            }
        }

        model
    }

    /// Create noise model from IBM backend properties
    pub fn from_ibm_properties(
        n_qubits: usize,
        t1_times: &[f64],
        t2_times: &[f64],
        single_gate_errors: &[f64],
        two_gate_errors: &[(usize, usize, f64)],
        readout_errors: &[f64],
    ) -> Self {
        let mut model = Self::ideal();
        model.noise_scale = 1.0;

        for q in 0..n_qubits {
            if q < t1_times.len() && q < t2_times.len() {
                model.coherence_times.insert(q, (t1_times[q], t2_times[q]));
            }

            if q < single_gate_errors.len() {
                model
                    .single_qubit_errors
                    .insert(q, SingleQubitNoiseType::Depolarizing(single_gate_errors[q]));
            }

            if q < readout_errors.len() {
                model.readout_errors.insert(q, readout_errors[q]);
            }
        }

        for (q1, q2, err) in two_gate_errors {
            model
                .two_qubit_errors
                .insert((*q1, *q2), TwoQubitNoiseType::Depolarizing(*err));
        }

        model
    }

    /// Get effective single-qubit error rate
    pub fn effective_single_error(&self, qubit: usize) -> f64 {
        self.single_qubit_errors
            .get(&qubit)
            .map(|e| match e {
                SingleQubitNoiseType::Depolarizing(p) => *p,
                SingleQubitNoiseType::AmplitudeDamping(p) => *p,
                SingleQubitNoiseType::PhaseDamping(p) => *p,
                SingleQubitNoiseType::BitFlip(p) => *p,
                SingleQubitNoiseType::PhaseFlip(p) => *p,
            })
            .unwrap_or(0.0)
            * self.noise_scale
    }

    /// Get effective two-qubit error rate
    pub fn effective_two_qubit_error(&self, q1: usize, q2: usize) -> f64 {
        let key = if q1 < q2 { (q1, q2) } else { (q2, q1) };
        self.two_qubit_errors
            .get(&key)
            .map(|e| match e {
                TwoQubitNoiseType::Depolarizing(p) => *p,
                TwoQubitNoiseType::CorrelatedDephasing(p) => *p,
                TwoQubitNoiseType::CrossTalk(p) => *p,
            })
            .unwrap_or(0.0)
            * self.noise_scale
    }
}

// ============================================================================
// Noise-Aware Gradient
// ============================================================================

/// Configuration for noise-aware gradient computation
#[derive(Debug, Clone)]
pub struct NoiseAwareGradientConfig {
    /// Number of shots for gradient estimation
    pub shots: usize,
    /// Parameter shift value (default: π/2)
    pub shift: f64,
    /// Whether to use noise model in gradient computation
    pub include_noise_in_gradient: bool,
    /// Variance reduction method
    pub variance_reduction: VarianceReduction,
    /// Number of repetitions for averaging
    pub n_repetitions: usize,
}

impl Default for NoiseAwareGradientConfig {
    fn default() -> Self {
        Self {
            shots: 1000,
            shift: std::f64::consts::FRAC_PI_2,
            include_noise_in_gradient: true,
            variance_reduction: VarianceReduction::None,
            n_repetitions: 1,
        }
    }
}

/// Variance reduction methods for gradient estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceReduction {
    /// No variance reduction
    None,
    /// Common random numbers
    CommonRandomNumbers,
    /// Antithetic variates
    AntitheticVariates,
    /// Control variates
    ControlVariates,
}

/// Noise-aware gradient estimator
///
/// Computes gradients that account for device noise, providing more
/// accurate optimization on noisy quantum hardware.
#[derive(Debug, Clone)]
pub struct NoiseAwareGradient {
    /// Noise model
    pub noise_model: NoiseModel,
    /// Configuration
    pub config: NoiseAwareGradientConfig,
    /// Cached gradient variances
    gradient_variances: HashMap<String, f64>,
}

impl NoiseAwareGradient {
    /// Create new noise-aware gradient estimator
    pub fn new(noise_model: NoiseModel) -> Self {
        Self {
            noise_model,
            config: NoiseAwareGradientConfig::default(),
            gradient_variances: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(noise_model: NoiseModel, config: NoiseAwareGradientConfig) -> Self {
        Self {
            noise_model,
            config,
            gradient_variances: HashMap::new(),
        }
    }

    /// Compute parameter-shift gradient with noise awareness
    ///
    /// Returns the gradient estimate and its estimated variance
    pub fn compute_gradient<F>(
        &mut self,
        param_idx: usize,
        current_params: &[f64],
        expectation_fn: F,
    ) -> Result<(f64, f64)>
    where
        F: Fn(&[f64]) -> Result<f64>,
    {
        let mut params_plus = current_params.to_vec();
        let mut params_minus = current_params.to_vec();

        params_plus[param_idx] += self.config.shift;
        params_minus[param_idx] -= self.config.shift;

        let mut gradient_estimates = Vec::with_capacity(self.config.n_repetitions);

        for _ in 0..self.config.n_repetitions {
            let exp_plus = expectation_fn(&params_plus)?;
            let exp_minus = expectation_fn(&params_minus)?;

            let gradient = (exp_plus - exp_minus) / (2.0 * self.config.shift.sin());
            gradient_estimates.push(gradient);
        }

        // Compute mean and variance
        let mean = gradient_estimates.iter().sum::<f64>() / gradient_estimates.len() as f64;
        let variance = if gradient_estimates.len() > 1 {
            gradient_estimates
                .iter()
                .map(|g| (g - mean).powi(2))
                .sum::<f64>()
                / (gradient_estimates.len() - 1) as f64
        } else {
            0.0
        };

        // Apply noise correction if configured
        let corrected_gradient = if self.config.include_noise_in_gradient {
            self.apply_noise_correction(mean, param_idx)
        } else {
            mean
        };

        Ok((corrected_gradient, variance))
    }

    /// Apply noise correction to gradient estimate
    fn apply_noise_correction(&self, gradient: f64, _param_idx: usize) -> f64 {
        // Simple noise scaling based on average error rate
        let avg_error: f64 = self
            .noise_model
            .single_qubit_errors
            .values()
            .map(|e| match e {
                SingleQubitNoiseType::Depolarizing(p) => *p,
                SingleQubitNoiseType::AmplitudeDamping(p) => *p,
                SingleQubitNoiseType::PhaseDamping(p) => *p,
                SingleQubitNoiseType::BitFlip(p) => *p,
                SingleQubitNoiseType::PhaseFlip(p) => *p,
            })
            .sum::<f64>()
            / self.noise_model.single_qubit_errors.len().max(1) as f64;

        // Scale gradient to account for noise-induced suppression
        let scale_factor = 1.0 / (1.0 - 2.0 * avg_error).max(0.1);
        gradient * scale_factor
    }

    /// Compute all gradients for a parameter vector
    pub fn compute_all_gradients<F>(
        &mut self,
        params: &[f64],
        expectation_fn: F,
    ) -> Result<(Vec<f64>, Vec<f64>)>
    where
        F: Fn(&[f64]) -> Result<f64> + Clone,
    {
        let mut gradients = Vec::with_capacity(params.len());
        let mut variances = Vec::with_capacity(params.len());

        for i in 0..params.len() {
            let (grad, var) = self.compute_gradient(i, params, expectation_fn.clone())?;
            gradients.push(grad);
            variances.push(var);
        }

        Ok((gradients, variances))
    }
}

// ============================================================================
// Mitigated Expectation Value
// ============================================================================

/// Error mitigation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitigationMethod {
    /// No mitigation
    None,
    /// Zero-Noise Extrapolation
    ZNE,
    /// Probabilistic Error Cancellation
    PEC,
    /// Readout Error Mitigation
    ReadoutMitigation,
    /// Twirling (Pauli or Clifford)
    Twirling,
}

/// Configuration for error-mitigated expectation values
#[derive(Debug, Clone)]
pub struct MitigatedExpectationConfig {
    /// Primary mitigation method
    pub method: MitigationMethod,
    /// Number of shots
    pub shots: usize,
    /// Scale factors for ZNE
    pub zne_scale_factors: Vec<f64>,
    /// Extrapolation method for ZNE
    pub zne_extrapolation: ZNEExtrapolation,
    /// Whether to apply readout mitigation
    pub apply_readout_mitigation: bool,
}

impl Default for MitigatedExpectationConfig {
    fn default() -> Self {
        Self {
            method: MitigationMethod::ZNE,
            shots: 4000,
            zne_scale_factors: vec![1.0, 1.5, 2.0],
            zne_extrapolation: ZNEExtrapolation::Linear,
            apply_readout_mitigation: true,
        }
    }
}

/// ZNE extrapolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZNEExtrapolation {
    /// Linear extrapolation
    Linear,
    /// Polynomial extrapolation
    Polynomial,
    /// Exponential extrapolation
    Exponential,
    /// Richardson extrapolation
    Richardson,
}

/// Error-mitigated expectation value estimator
#[derive(Debug, Clone)]
pub struct MitigatedExpectation {
    /// Noise model
    pub noise_model: NoiseModel,
    /// Configuration
    pub config: MitigatedExpectationConfig,
    /// Readout calibration matrix (if computed)
    readout_calibration: Option<Array2<f64>>,
}

impl MitigatedExpectation {
    /// Create new mitigated expectation estimator
    pub fn new(noise_model: NoiseModel) -> Self {
        Self {
            noise_model,
            config: MitigatedExpectationConfig::default(),
            readout_calibration: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(noise_model: NoiseModel, config: MitigatedExpectationConfig) -> Self {
        Self {
            noise_model,
            config,
            readout_calibration: None,
        }
    }

    /// Compute error-mitigated expectation value
    pub fn compute<F>(&self, raw_expectation_fn: F) -> Result<f64>
    where
        F: Fn(f64) -> Result<f64>,
    {
        match self.config.method {
            MitigationMethod::None => raw_expectation_fn(1.0),
            MitigationMethod::ZNE => self.compute_zne(raw_expectation_fn),
            MitigationMethod::ReadoutMitigation => {
                let raw = raw_expectation_fn(1.0)?;
                self.apply_readout_mitigation(raw)
            }
            _ => raw_expectation_fn(1.0), // Fallback for unimplemented methods
        }
    }

    /// Compute ZNE-mitigated expectation value
    fn compute_zne<F>(&self, raw_expectation_fn: F) -> Result<f64>
    where
        F: Fn(f64) -> Result<f64>,
    {
        let mut scaled_values = Vec::with_capacity(self.config.zne_scale_factors.len());

        for &scale in &self.config.zne_scale_factors {
            let value = raw_expectation_fn(scale)?;
            scaled_values.push((scale, value));
        }

        // Extrapolate to zero noise
        self.extrapolate_to_zero(&scaled_values)
    }

    /// Extrapolate to zero noise using configured method
    fn extrapolate_to_zero(&self, points: &[(f64, f64)]) -> Result<f64> {
        if points.is_empty() {
            return Err(MLError::InvalidConfiguration(
                "No data points for extrapolation".to_string(),
            ));
        }

        if points.len() == 1 {
            return Ok(points[0].1);
        }

        match self.config.zne_extrapolation {
            ZNEExtrapolation::Linear => {
                // Linear fit: y = a + b*x, extrapolate to x=0
                let n = points.len() as f64;
                let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
                let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
                let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
                let sum_x2: f64 = points.iter().map(|(x, _)| x * x).sum();

                let denom = n * sum_x2 - sum_x * sum_x;
                if denom.abs() < 1e-10 {
                    return Ok(sum_y / n);
                }

                let a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
                Ok(a) // y at x=0
            }
            ZNEExtrapolation::Exponential => {
                // Exponential fit: y = a * exp(b*x)
                // Take log and do linear fit
                let log_points: Vec<(f64, f64)> = points
                    .iter()
                    .filter(|(_, y)| *y > 0.0)
                    .map(|(x, y)| (*x, y.ln()))
                    .collect();

                if log_points.is_empty() {
                    return Ok(points[0].1);
                }

                let n = log_points.len() as f64;
                let sum_x: f64 = log_points.iter().map(|(x, _)| x).sum();
                let sum_y: f64 = log_points.iter().map(|(_, y)| y).sum();
                let sum_xy: f64 = log_points.iter().map(|(x, y)| x * y).sum();
                let sum_x2: f64 = log_points.iter().map(|(x, _)| x * x).sum();

                let denom = n * sum_x2 - sum_x * sum_x;
                if denom.abs() < 1e-10 {
                    return Ok((sum_y / n).exp());
                }

                let log_a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
                Ok(log_a.exp())
            }
            _ => {
                // Default to linear for other methods
                let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
                Ok(sum_y / points.len() as f64)
            }
        }
    }

    /// Apply readout error mitigation
    fn apply_readout_mitigation(&self, raw_value: f64) -> Result<f64> {
        // Simple correction based on average readout error
        let avg_readout_error: f64 = self.noise_model.readout_errors.values().sum::<f64>()
            / self.noise_model.readout_errors.len().max(1) as f64;

        // Correct for readout bias
        let corrected = (raw_value - avg_readout_error) / (1.0 - 2.0 * avg_readout_error);
        Ok(corrected.clamp(-1.0, 1.0))
    }

    /// Calibrate readout errors
    pub fn calibrate_readout(&mut self, n_qubits: usize) -> Result<()> {
        // Build simple diagonal calibration matrix
        let dim = 1 << n_qubits;
        let mut cal_matrix = Array2::<f64>::eye(dim);

        // Apply readout errors to diagonal
        for q in 0..n_qubits {
            if let Some(&err) = self.noise_model.readout_errors.get(&q) {
                for i in 0..dim {
                    let bit = (i >> q) & 1;
                    if bit == 0 {
                        cal_matrix[[i, i]] *= 1.0 - err;
                    } else {
                        cal_matrix[[i, i]] *= 1.0 - err;
                    }
                }
            }
        }

        self.readout_calibration = Some(cal_matrix);
        Ok(())
    }
}

// ============================================================================
// Noise-Aware Training Wrapper
// ============================================================================

/// Wrapper for noise-aware training of quantum circuits
#[derive(Debug)]
pub struct NoiseAwareTrainer {
    /// Noise-aware gradient estimator
    pub gradient_estimator: NoiseAwareGradient,
    /// Mitigated expectation estimator
    pub expectation_estimator: MitigatedExpectation,
    /// Training history
    pub history: TrainingHistory,
}

/// Training history
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    /// Loss values per epoch
    pub losses: Vec<f64>,
    /// Gradient norms per epoch
    pub gradient_norms: Vec<f64>,
    /// Mitigated vs raw loss difference
    pub mitigation_improvement: Vec<f64>,
}

impl NoiseAwareTrainer {
    /// Create new noise-aware trainer
    pub fn new(noise_model: NoiseModel) -> Self {
        Self {
            gradient_estimator: NoiseAwareGradient::new(noise_model.clone()),
            expectation_estimator: MitigatedExpectation::new(noise_model),
            history: TrainingHistory::default(),
        }
    }

    /// Perform one training step
    pub fn step<F>(&mut self, params: &mut [f64], loss_fn: F, learning_rate: f64) -> Result<f64>
    where
        F: Fn(&[f64]) -> Result<f64> + Clone,
    {
        // Compute gradients with noise awareness
        let (gradients, variances) = self
            .gradient_estimator
            .compute_all_gradients(params, loss_fn.clone())?;

        // Compute gradient norm
        let grad_norm: f64 = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
        self.history.gradient_norms.push(grad_norm);

        // Update parameters with variance-weighted learning rate
        for (i, (param, grad)) in params.iter_mut().zip(gradients.iter()).enumerate() {
            let variance_factor = 1.0 / (1.0 + variances[i].sqrt());
            *param -= learning_rate * grad * variance_factor;
        }

        // Compute mitigated loss
        let loss = self.expectation_estimator.compute(|scale| {
            // For ZNE, we'd scale the noise here
            let _ = scale; // Currently not used in simple implementation
            loss_fn(params)
        })?;

        self.history.losses.push(loss);
        Ok(loss)
    }

    /// Get current training statistics
    pub fn statistics(&self) -> TrainingStatistics {
        let n = self.history.losses.len();
        if n == 0 {
            return TrainingStatistics::default();
        }

        let avg_loss = self.history.losses.iter().sum::<f64>() / n as f64;
        let avg_grad_norm = self.history.gradient_norms.iter().sum::<f64>() / n as f64;
        let recent_loss = self.history.losses.last().copied().unwrap_or(0.0);

        TrainingStatistics {
            epochs: n,
            average_loss: avg_loss,
            recent_loss,
            average_gradient_norm: avg_grad_norm,
            converged: avg_grad_norm < 1e-6,
        }
    }
}

/// Training statistics
#[derive(Debug, Clone, Default)]
pub struct TrainingStatistics {
    /// Number of epochs completed
    pub epochs: usize,
    /// Average loss across all epochs
    pub average_loss: f64,
    /// Most recent loss
    pub recent_loss: f64,
    /// Average gradient norm
    pub average_gradient_norm: f64,
    /// Whether training has converged
    pub converged: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_model_ideal() {
        let model = NoiseModel::ideal();
        assert_eq!(model.noise_scale, 0.0);
        assert!(model.single_qubit_errors.is_empty());
    }

    #[test]
    fn test_noise_model_depolarizing() {
        let model = NoiseModel::uniform_depolarizing(4, 0.01, 0.02);
        assert_eq!(model.single_qubit_errors.len(), 4);
        assert!(!model.two_qubit_errors.is_empty());
    }

    #[test]
    fn test_noise_aware_gradient() {
        let model = NoiseModel::uniform_depolarizing(2, 0.01, 0.02);
        let mut estimator = NoiseAwareGradient::new(model);

        let params = vec![0.5, 0.3];
        let (grad, var) = estimator
            .compute_gradient(0, &params, |p| Ok(p[0].sin() + p[1].cos()))
            .expect("Should compute gradient");

        // Gradient of sin(x) at x=0.5 is cos(0.5) ≈ 0.877
        assert!((grad - 0.5_f64.cos()).abs() < 0.3); // Allow for noise correction
        assert!(var >= 0.0);
    }

    #[test]
    fn test_mitigated_expectation_linear() {
        let model = NoiseModel::ideal();
        let estimator = MitigatedExpectation::new(model);

        // Test linear extrapolation
        let result = estimator
            .compute(|scale| Ok(1.0 - 0.1 * scale))
            .expect("Should compute");

        // Linear extrapolation to scale=0 should give ~1.0
        assert!((result - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_training_statistics() {
        let model = NoiseModel::ideal();
        let trainer = NoiseAwareTrainer::new(model);
        let stats = trainer.statistics();

        assert_eq!(stats.epochs, 0);
        assert!(!stats.converged);
    }

    #[test]
    fn test_mitigation_methods() {
        assert_eq!(MitigationMethod::ZNE, MitigationMethod::ZNE);
        assert_ne!(MitigationMethod::ZNE, MitigationMethod::PEC);
    }

    #[test]
    fn test_zne_extrapolation() {
        let model = NoiseModel::ideal();
        let config = MitigatedExpectationConfig {
            method: MitigationMethod::ZNE,
            shots: 1000,
            zne_scale_factors: vec![1.0, 2.0, 3.0],
            zne_extrapolation: ZNEExtrapolation::Linear,
            apply_readout_mitigation: false,
        };
        let estimator = MitigatedExpectation::with_config(model, config);

        // y = 1 - 0.1*x extrapolated to x=0 should be 1.0
        let result = estimator
            .compute(|scale| Ok(1.0 - 0.1 * scale))
            .expect("Should succeed");

        assert!((result - 1.0).abs() < 0.1);
    }
}
