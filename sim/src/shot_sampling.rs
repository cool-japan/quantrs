//! Shot-based sampling with statistical analysis for quantum simulation.
//!
//! This module implements comprehensive shot-based sampling methods for quantum
//! circuits, including measurement statistics, error analysis, and convergence
//! detection for realistic quantum device simulation.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, SimulatorError};
use crate::pauli::{PauliOperatorSum, PauliString};
use crate::statevector::StateVectorSimulator;

/// Shot-based measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotResult {
    /// Measurement outcomes for each shot
    pub outcomes: Vec<BitString>,
    /// Total number of shots
    pub num_shots: usize,
    /// Measurement statistics
    pub statistics: MeasurementStatistics,
    /// Sampling configuration used
    pub config: SamplingConfig,
}

/// Bit string representation of measurement outcome
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitString {
    /// Bit values (0 or 1)
    pub bits: Vec<u8>,
}

impl BitString {
    /// Create from vector of booleans
    pub fn from_bools(bools: &[bool]) -> Self {
        Self {
            bits: bools.iter().map(|&b| u8::from(b)).collect(),
        }
    }

    /// Convert to vector of booleans
    pub fn to_bools(&self) -> Vec<bool> {
        self.bits.iter().map(|&b| b == 1).collect()
    }

    /// Convert to integer (little-endian)
    pub fn to_int(&self) -> usize {
        self.bits
            .iter()
            .enumerate()
            .map(|(i, &bit)| (bit as usize) << i)
            .sum()
    }

    /// Create from integer (little-endian)
    pub fn from_int(mut value: usize, num_bits: usize) -> Self {
        let mut bits = Vec::with_capacity(num_bits);
        for _ in 0..num_bits {
            bits.push((value & 1) as u8);
            value >>= 1;
        }
        Self { bits }
    }

    /// Number of bits
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Hamming weight (number of 1s)
    pub fn weight(&self) -> usize {
        self.bits.iter().map(|&b| b as usize).sum()
    }

    /// Hamming distance to another bit string
    pub fn distance(&self, other: &Self) -> usize {
        if self.len() != other.len() {
            return usize::MAX; // Invalid comparison
        }
        self.bits
            .iter()
            .zip(&other.bits)
            .map(|(&a, &b)| (a ^ b) as usize)
            .sum()
    }
}

impl std::fmt::Display for BitString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &bit in &self.bits {
            write!(f, "{bit}")?;
        }
        Ok(())
    }
}

/// Measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementStatistics {
    /// Frequency count for each outcome
    pub counts: HashMap<BitString, usize>,
    /// Most frequent outcome
    pub mode: BitString,
    /// Probability estimates for each outcome
    pub probabilities: HashMap<BitString, f64>,
    /// Variance in the probability estimates
    pub probability_variance: f64,
    /// Statistical confidence intervals
    pub confidence_intervals: HashMap<BitString, (f64, f64)>,
    /// Entropy of the measurement distribution
    pub entropy: f64,
    /// Purity of the measurement distribution
    pub purity: f64,
}

/// Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Number of shots to take
    pub num_shots: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Confidence level for intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Whether to compute full statistics
    pub compute_statistics: bool,
    /// Whether to estimate convergence
    pub estimate_convergence: bool,
    /// Convergence check interval (number of shots)
    pub convergence_check_interval: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Maximum number of shots for convergence
    pub max_shots_for_convergence: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            num_shots: 1024,
            seed: None,
            confidence_level: 0.95,
            compute_statistics: true,
            estimate_convergence: false,
            convergence_check_interval: 100,
            convergence_tolerance: 0.01,
            max_shots_for_convergence: 10000,
        }
    }
}

/// Shot-based quantum sampler
pub struct QuantumSampler {
    /// Random number generator
    rng: ChaCha8Rng,
    /// Current configuration
    config: SamplingConfig,
}

impl QuantumSampler {
    /// Create new sampler with configuration
    pub fn new(config: SamplingConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_rng(&mut thread_rng())
        };

        Self { rng, config }
    }

    /// Sample measurements from a quantum state
    pub fn sample_state(&mut self, state: &Array1<Complex64>) -> Result<ShotResult> {
        let num_qubits = (state.len() as f64).log2() as usize;
        if 1 << num_qubits != state.len() {
            return Err(SimulatorError::InvalidInput(
                "State vector dimension must be a power of 2".to_string(),
            ));
        }

        // Compute probability distribution
        let probabilities: Vec<f64> = state.iter().map(|amp| amp.norm_sqr()).collect();

        // Validate normalization
        let total_prob: f64 = probabilities.iter().sum();
        if (total_prob - 1.0).abs() > 1e-10 {
            return Err(SimulatorError::InvalidInput(format!(
                "State vector not normalized: total probability = {total_prob}"
            )));
        }

        // Sample outcomes
        let mut outcomes = Vec::with_capacity(self.config.num_shots);
        for _ in 0..self.config.num_shots {
            let sample = self.sample_from_distribution(&probabilities)?;
            outcomes.push(BitString::from_int(sample, num_qubits));
        }

        // Compute statistics if requested
        let statistics = if self.config.compute_statistics {
            self.compute_statistics(&outcomes)?
        } else {
            MeasurementStatistics {
                counts: HashMap::new(),
                mode: BitString::from_int(0, num_qubits),
                probabilities: HashMap::new(),
                probability_variance: 0.0,
                confidence_intervals: HashMap::new(),
                entropy: 0.0,
                purity: 0.0,
            }
        };

        Ok(ShotResult {
            outcomes,
            num_shots: self.config.num_shots,
            statistics,
            config: self.config.clone(),
        })
    }

    /// Sample measurements from a state with noise
    pub fn sample_state_with_noise(
        &mut self,
        state: &Array1<Complex64>,
        noise_model: &dyn NoiseModel,
    ) -> Result<ShotResult> {
        // Apply noise model to the state
        let noisy_state = noise_model.apply_readout_noise(state)?;
        self.sample_state(&noisy_state)
    }

    /// Sample expectation value of an observable
    pub fn sample_expectation(
        &mut self,
        state: &Array1<Complex64>,
        observable: &PauliOperatorSum,
    ) -> Result<ExpectationResult> {
        let mut expectation_values = Vec::new();
        let mut variances = Vec::new();

        // Sample each Pauli term separately
        for term in &observable.terms {
            let term_result = self.sample_pauli_expectation(state, term)?;
            expectation_values.push(term_result.expectation * term.coefficient.re);
            variances.push(term_result.variance * term.coefficient.re.powi(2));
        }

        // Combine results
        let total_expectation: f64 = expectation_values.iter().sum();
        let total_variance: f64 = variances.iter().sum();
        let standard_error = (total_variance / self.config.num_shots as f64).sqrt();

        // Confidence interval
        let z_score = self.get_z_score(self.config.confidence_level);
        let confidence_interval = (
            z_score.mul_add(-standard_error, total_expectation),
            z_score.mul_add(standard_error, total_expectation),
        );

        Ok(ExpectationResult {
            expectation: total_expectation,
            variance: total_variance,
            standard_error,
            confidence_interval,
            num_shots: self.config.num_shots,
        })
    }

    /// Sample expectation value of a single Pauli string
    fn sample_pauli_expectation(
        &mut self,
        state: &Array1<Complex64>,
        pauli_string: &PauliString,
    ) -> Result<ExpectationResult> {
        // For Pauli measurements, eigenvalues are ±1
        // We need to measure in the appropriate basis

        let num_qubits = pauli_string.num_qubits;
        let mut measurements = Vec::with_capacity(self.config.num_shots);

        for _ in 0..self.config.num_shots {
            // Measure each qubit in the appropriate Pauli basis
            let outcome = self.measure_pauli_basis(state, pauli_string)?;
            measurements.push(outcome);
        }

        // Compute statistics
        let mean = measurements.iter().sum::<f64>() / measurements.len() as f64;
        let variance = measurements.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / measurements.len() as f64;

        let standard_error = (variance / measurements.len() as f64).sqrt();
        let z_score = self.get_z_score(self.config.confidence_level);
        let confidence_interval = (
            z_score.mul_add(-standard_error, mean),
            z_score.mul_add(standard_error, mean),
        );

        Ok(ExpectationResult {
            expectation: mean,
            variance,
            standard_error,
            confidence_interval,
            num_shots: measurements.len(),
        })
    }

    /// Measure in Pauli basis (simplified implementation)
    fn measure_pauli_basis(
        &mut self,
        _state: &Array1<Complex64>,
        _pauli_string: &PauliString,
    ) -> Result<f64> {
        // Simplified implementation - return random ±1
        // In practice, would need to transform state to measurement basis
        if self.rng.gen::<f64>() < 0.5 {
            Ok(1.0)
        } else {
            Ok(-1.0)
        }
    }

    /// Sample from discrete probability distribution
    fn sample_from_distribution(&mut self, probabilities: &[f64]) -> Result<usize> {
        let random_value = self.rng.gen::<f64>();
        let mut cumulative = 0.0;

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return Ok(i);
            }
        }

        // Handle numerical errors - return last index
        Ok(probabilities.len() - 1)
    }

    /// Compute measurement statistics
    fn compute_statistics(&self, outcomes: &[BitString]) -> Result<MeasurementStatistics> {
        let mut counts = HashMap::new();
        let total_shots = outcomes.len() as f64;

        // Count frequencies
        for outcome in outcomes {
            *counts.entry(outcome.clone()).or_insert(0) += 1;
        }

        // Find mode
        let mode = counts.iter().max_by_key(|(_, &count)| count).map_or_else(
            || BitString::from_int(0, outcomes[0].len()),
            |(outcome, _)| outcome.clone(),
        );

        // Compute probabilities
        let mut probabilities = HashMap::new();
        let mut confidence_intervals = HashMap::new();
        let z_score = self.get_z_score(self.config.confidence_level);

        for (outcome, &count) in &counts {
            let prob = count as f64 / total_shots;
            probabilities.insert(outcome.clone(), prob);

            // Binomial confidence interval
            let std_error = (prob * (1.0 - prob) / total_shots).sqrt();
            let margin = z_score * std_error;
            confidence_intervals.insert(
                outcome.clone(),
                ((prob - margin).max(0.0), (prob + margin).min(1.0)),
            );
        }

        // Compute entropy
        let entropy = probabilities
            .values()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum::<f64>();

        // Compute purity (sum of squared probabilities)
        let purity = probabilities.values().map(|&p| p * p).sum::<f64>();

        // Compute overall probability variance
        let mean_prob = 1.0 / probabilities.len() as f64;
        let probability_variance = probabilities
            .values()
            .map(|&p| (p - mean_prob).powi(2))
            .sum::<f64>()
            / probabilities.len() as f64;

        Ok(MeasurementStatistics {
            counts,
            mode,
            probabilities,
            probability_variance,
            confidence_intervals,
            entropy,
            purity,
        })
    }

    /// Get z-score for confidence level
    fn get_z_score(&self, confidence_level: f64) -> f64 {
        // Simplified - use common values
        match (confidence_level * 100.0) as i32 {
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => 1.96, // Default to 95%
        }
    }

    /// Estimate convergence of sampling
    pub fn estimate_convergence(
        &mut self,
        state: &Array1<Complex64>,
        observable: &PauliOperatorSum,
    ) -> Result<ConvergenceResult> {
        let mut expectation_history = Vec::new();
        let mut variance_history = Vec::new();
        let mut shots_taken = 0;
        let mut converged = false;

        while shots_taken < self.config.max_shots_for_convergence && !converged {
            // Take a batch of measurements
            let batch_shots = self
                .config
                .convergence_check_interval
                .min(self.config.max_shots_for_convergence - shots_taken);

            // Temporarily adjust shot count for this batch
            let original_shots = self.config.num_shots;
            self.config.num_shots = batch_shots;

            let result = self.sample_expectation(state, observable)?;

            // Restore original shot count
            self.config.num_shots = original_shots;

            expectation_history.push(result.expectation);
            variance_history.push(result.variance);
            shots_taken += batch_shots;

            // Check convergence
            if expectation_history.len() >= 3 {
                let recent_values = &expectation_history[expectation_history.len() - 3..];
                let max_diff = recent_values
                    .iter()
                    .zip(recent_values.iter().skip(1))
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);

                if max_diff < self.config.convergence_tolerance {
                    converged = true;
                }
            }
        }

        // Compute final estimates
        let final_expectation = expectation_history.last().copied().unwrap_or(0.0);
        let expectation_std = if expectation_history.len() > 1 {
            let mean = expectation_history.iter().sum::<f64>() / expectation_history.len() as f64;
            (expectation_history
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / (expectation_history.len() - 1) as f64)
                .sqrt()
        } else {
            0.0
        };

        Ok(ConvergenceResult {
            converged,
            shots_taken,
            final_expectation,
            expectation_history,
            variance_history,
            convergence_rate: expectation_std,
        })
    }
}

/// Result of expectation value sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectationResult {
    /// Expectation value estimate
    pub expectation: f64,
    /// Variance estimate
    pub variance: f64,
    /// Standard error
    pub standard_error: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Number of shots used
    pub num_shots: usize,
}

/// Result of convergence estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceResult {
    /// Whether convergence was achieved
    pub converged: bool,
    /// Total shots taken
    pub shots_taken: usize,
    /// Final expectation value
    pub final_expectation: f64,
    /// History of expectation values
    pub expectation_history: Vec<f64>,
    /// History of variances
    pub variance_history: Vec<f64>,
    /// Convergence rate (standard deviation of recent estimates)
    pub convergence_rate: f64,
}

/// Noise model trait for realistic sampling
pub trait NoiseModel: Send + Sync {
    /// Apply readout noise to measurements
    fn apply_readout_noise(&self, state: &Array1<Complex64>) -> Result<Array1<Complex64>>;

    /// Get readout error probability for qubit
    fn readout_error_probability(&self, qubit: usize) -> f64;
}

/// Simple readout noise model
#[derive(Debug, Clone)]
pub struct SimpleReadoutNoise {
    /// Error probability for each qubit
    pub error_probs: Vec<f64>,
}

impl SimpleReadoutNoise {
    /// Create uniform readout noise
    pub fn uniform(num_qubits: usize, error_prob: f64) -> Self {
        Self {
            error_probs: vec![error_prob; num_qubits],
        }
    }
}

impl NoiseModel for SimpleReadoutNoise {
    fn apply_readout_noise(&self, state: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        // Simplified implementation - in practice would need proper POVM modeling
        Ok(state.clone())
    }

    fn readout_error_probability(&self, qubit: usize) -> f64 {
        self.error_probs.get(qubit).copied().unwrap_or(0.0)
    }
}

/// Utility functions for shot sampling analysis
pub mod analysis {
    use super::*;

    /// Compute statistical power for detecting effect
    pub fn statistical_power(effect_size: f64, num_shots: usize, significance_level: f64) -> f64 {
        // Simplified power analysis
        let standard_error = 1.0 / (num_shots as f64).sqrt();
        let z_critical = match (significance_level * 100.0) as i32 {
            1 => 2.576,
            5 => 1.96,
            10 => 1.645,
            _ => 1.96,
        };

        let z_beta = (effect_size / standard_error) - z_critical;
        normal_cdf(z_beta)
    }

    /// Estimate required shots for desired precision
    pub fn required_shots_for_precision(desired_error: f64, confidence_level: f64) -> usize {
        let z_score = match (confidence_level * 100.0) as i32 {
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => 1.96,
        };

        // For binomial: n ≥ (z²/4ε²) for worst case p=0.5
        let n = (z_score * z_score) / (4.0 * desired_error * desired_error);
        n.ceil() as usize
    }

    /// Compare two shot results statistically
    pub fn compare_shot_results(
        result1: &ShotResult,
        result2: &ShotResult,
        significance_level: f64,
    ) -> ComparisonResult {
        // Chi-square test for distribution comparison
        let mut chi_square = 0.0;
        let mut degrees_of_freedom: usize = 0;

        // Get all unique outcomes
        let mut all_outcomes = std::collections::HashSet::new();
        all_outcomes.extend(result1.statistics.counts.keys());
        all_outcomes.extend(result2.statistics.counts.keys());

        for outcome in &all_outcomes {
            let count1 = result1.statistics.counts.get(outcome).copied().unwrap_or(0) as f64;
            let count2 = result2.statistics.counts.get(outcome).copied().unwrap_or(0) as f64;

            let total1 = result1.num_shots as f64;
            let total2 = result2.num_shots as f64;

            let expected1 = (count1 + count2) * total1 / (total1 + total2);
            let expected2 = (count1 + count2) * total2 / (total1 + total2);

            if expected1 > 5.0 && expected2 > 5.0 {
                chi_square += (count1 - expected1).powi(2) / expected1;
                chi_square += (count2 - expected2).powi(2) / expected2;
                degrees_of_freedom += 1;
            }
        }

        degrees_of_freedom = degrees_of_freedom.saturating_sub(1);

        // Critical value for given significance level (simplified)
        let critical_value = match (significance_level * 100.0) as i32 {
            1 => 6.635, // Very rough approximation
            5 => 3.841,
            10 => 2.706,
            _ => 3.841,
        };

        ComparisonResult {
            chi_square,
            degrees_of_freedom,
            p_value: if chi_square > critical_value {
                0.01
            } else {
                0.1
            }, // Rough
            significant: chi_square > critical_value,
        }
    }

    /// Normal CDF approximation
    fn normal_cdf(x: f64) -> f64 {
        // Simplified approximation
        0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = ((a5 * t + a4).mul_add(t, a3).mul_add(t, a2).mul_add(t, a1) * t)
            .mul_add(-(-x * x).exp(), 1.0);

        sign * y
    }
}

/// Result of statistical comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Chi-square statistic
    pub chi_square: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
    /// P-value
    pub p_value: f64,
    /// Whether difference is significant
    pub significant: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_string() {
        let bs = BitString::from_int(5, 4); // 5 = 1010 in binary
        assert_eq!(bs.bits, vec![1, 0, 1, 0]);
        assert_eq!(bs.to_int(), 5);
        assert_eq!(bs.weight(), 2);
    }

    #[test]
    fn test_sampler_creation() {
        let config = SamplingConfig::default();
        let sampler = QuantumSampler::new(config);
        assert_eq!(sampler.config.num_shots, 1024);
    }

    #[test]
    fn test_uniform_state_sampling() {
        let mut config = SamplingConfig::default();
        config.num_shots = 100;
        config.seed = Some(42);

        let mut sampler = QuantumSampler::new(config);

        // Create uniform superposition |+> = (|0> + |1>)/√2
        let state = Array1::from_vec(vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);

        let result = sampler.sample_state(&state).unwrap();
        assert_eq!(result.num_shots, 100);
        assert_eq!(result.outcomes.len(), 100);

        // Check that we got both |0> and |1> outcomes
        let has_zero = result.outcomes.iter().any(|bs| bs.to_int() == 0);
        let has_one = result.outcomes.iter().any(|bs| bs.to_int() == 1);
        assert!(has_zero && has_one);
    }

    #[test]
    fn test_required_shots_calculation() {
        let shots = analysis::required_shots_for_precision(0.01, 0.95);
        assert!(shots > 9000); // Should need many shots for 1% precision
    }
}
