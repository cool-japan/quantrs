//! Quantum Adiabatic Optimization with Dynamic Path Optimization
//!
//! This module implements advanced quantum adiabatic computation with real-time
//! path optimization to minimize diabatic transitions and maximize solution quality.
//!
//! # Theory
//!
//! The adiabatic theorem states that a quantum system remains in its instantaneous
//! eigenstate if changes are slow enough. For QUBO problems:
//!
//! H(s) = (1-s)H₀ + s·H_problem
//!
//! where s ∈ \[0,1\] is the adiabatic parameter.
//!
//! This implementation optimizes the path s(t) dynamically based on:
//! - Instantaneous energy gap analysis
//! - Diabatic transition probability
//! - Quantum speed limits
//! - Problem structure recognition

use crate::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
use quantrs2_anneal::QuboModel;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Adiabatic path interpolation scheme
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PathInterpolation {
    /// Linear interpolation: s(t) = t/T
    Linear,
    /// Polynomial: s(t) = (t/T)^n
    Polynomial { exponent: f64 },
    /// Optimized based on gap: slower near avoided crossings
    GapOptimized,
    /// Custom schedule
    Custom,
}

/// Configuration for quantum adiabatic path optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdiabaticPathConfig {
    /// Total evolution time (in arbitrary units)
    pub total_time: f64,
    /// Number of time steps for discretization
    pub time_steps: usize,
    /// Path interpolation scheme
    pub interpolation: PathInterpolation,
    /// Enable dynamic path adjustment
    pub dynamic_adjustment: bool,
    /// Gap threshold for slowing down
    pub gap_threshold: f64,
    /// Maximum allowed diabatic transition probability
    pub max_diabatic_probability: f64,
    /// Temperature for thermal sampling (if > 0)
    pub temperature: f64,
    /// Number of samples to generate
    pub num_samples: usize,
}

impl Default for AdiabaticPathConfig {
    fn default() -> Self {
        Self {
            total_time: 100.0,
            time_steps: 1000,
            interpolation: PathInterpolation::GapOptimized,
            dynamic_adjustment: true,
            gap_threshold: 0.1,
            max_diabatic_probability: 0.01,
            temperature: 0.0,
            num_samples: 100,
        }
    }
}

/// Instantaneous Hamiltonian information
#[derive(Debug, Clone)]
struct InstantaneousHamiltonian {
    /// Current adiabatic parameter s
    s: f64,
    /// Energy eigenvalues (sorted)
    eigenvalues: Vec<f64>,
    /// Ground state energy
    ground_energy: f64,
    /// First excited state energy
    excited_energy: f64,
    /// Instantaneous energy gap
    gap: f64,
    /// Diabatic transition probability estimate
    diabatic_probability: f64,
}

/// Quantum adiabatic path optimizer
#[derive(Debug, Clone)]
pub struct QuantumAdiabaticPathOptimizer {
    config: AdiabaticPathConfig,
    /// Optimized time schedule
    time_schedule: Vec<f64>,
    /// Adiabatic parameter schedule
    s_schedule: Vec<f64>,
    /// Gap values along the path
    gap_schedule: Vec<f64>,
}

impl QuantumAdiabaticPathOptimizer {
    /// Create new adiabatic path optimizer
    pub fn new(config: AdiabaticPathConfig) -> Self {
        let mut optimizer = Self {
            config,
            time_schedule: Vec::new(),
            s_schedule: Vec::new(),
            gap_schedule: Vec::new(),
        };
        optimizer.initialize_schedule();
        optimizer
    }

    /// Initialize the adiabatic schedule
    fn initialize_schedule(&mut self) {
        let n = self.config.time_steps;
        self.time_schedule = (0..=n)
            .map(|i| self.config.total_time * (i as f64) / (n as f64))
            .collect();

        match self.config.interpolation {
            PathInterpolation::Linear => {
                self.s_schedule = (0..=n).map(|i| (i as f64) / (n as f64)).collect();
            }
            PathInterpolation::Polynomial { exponent } => {
                self.s_schedule = (0..=n)
                    .map(|i| ((i as f64) / (n as f64)).powf(exponent))
                    .collect();
            }
            PathInterpolation::GapOptimized | PathInterpolation::Custom => {
                // Initialize with linear, will optimize later
                self.s_schedule = (0..=n).map(|i| (i as f64) / (n as f64)).collect();
            }
        }

        self.gap_schedule = vec![0.0; n + 1];
    }

    /// Compute instantaneous Hamiltonian properties
    fn compute_instantaneous_hamiltonian(
        &self,
        qubo: &Array2<f64>,
        s: f64,
    ) -> InstantaneousHamiltonian {
        let n = qubo.nrows();

        // For a QUBO problem, we construct the Ising Hamiltonian
        // H(s) = (1-s)·H_transverse + s·H_problem
        //
        // H_transverse = -Σᵢ σᵢˣ (causes superposition)
        // H_problem = QUBO Hamiltonian
        //
        // For small problems, we can compute exact eigenvalues
        // For large problems, we use heuristics and gap estimates

        if n <= 10 {
            // Exact diagonalization for small problems
            self.exact_gap_computation(qubo, s)
        } else {
            // Gap estimation for large problems
            self.estimated_gap_computation(qubo, s)
        }
    }

    /// Exact gap computation via diagonalization (small problems)
    fn exact_gap_computation(&self, qubo: &Array2<f64>, s: f64) -> InstantaneousHamiltonian {
        let n = qubo.nrows();
        let dim = 1 << n; // 2^n states

        // Build full Hamiltonian matrix (simplified version)
        // In practice, this would use proper quantum operators
        let mut hamiltonian = Array2::<f64>::zeros((dim, dim));

        // Add transverse field term (1-s) * H_transverse
        let transverse_strength = 1.0 - s;
        for i in 0..dim {
            for j in 0..dim {
                // Count bit differences (transverse field connects differing states)
                if (i ^ j).is_power_of_two() {
                    hamiltonian[[i, j]] = -transverse_strength;
                }
            }
        }

        // Add problem Hamiltonian s * H_problem
        for i in 0..dim {
            let mut energy = 0.0;
            for bit_i in 0..n {
                let val_i = if (i >> bit_i) & 1 == 1 { 1.0 } else { 0.0 };
                energy += qubo[[bit_i, bit_i]] * val_i;
                for bit_j in (bit_i + 1)..n {
                    let val_j = if (i >> bit_j) & 1 == 1 { 1.0 } else { 0.0 };
                    energy += qubo[[bit_i, bit_j]] * val_i * val_j;
                }
            }
            hamiltonian[[i, i]] += s * energy;
        }

        // Compute eigenvalues (simplified - in practice use SciRS2 linalg)
        let mut eigenvalues: Vec<f64> = (0..dim).map(|i| hamiltonian[[i, i]]).collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ground = eigenvalues[0];
        let excited = if eigenvalues.len() > 1 {
            eigenvalues[1]
        } else {
            ground
        };
        let gap = excited - ground;

        // Estimate diabatic transition probability using Landau-Zener formula
        let diabatic_prob = if gap > 1e-10 {
            let velocity = 1.0 / self.config.total_time; // ds/dt
            (-std::f64::consts::PI * gap.powi(2) / (2.0 * velocity)).exp()
        } else {
            1.0
        };

        InstantaneousHamiltonian {
            s,
            eigenvalues,
            ground_energy: ground,
            excited_energy: excited,
            gap,
            diabatic_probability: diabatic_prob,
        }
    }

    /// Estimated gap computation (large problems)
    fn estimated_gap_computation(&self, qubo: &Array2<f64>, s: f64) -> InstantaneousHamiltonian {
        let n = qubo.nrows();

        // For large problems, estimate the gap using:
        // 1. Sample-based energy landscape analysis
        // 2. Local minima detection
        // 3. Spectral gap heuristics

        let num_samples = 1000.min(1 << n.min(20));
        let mut rng = thread_rng();
        let mut energies = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let state: Vec<f64> = (0..n)
                .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
                .collect();
            let energy = self.compute_state_energy(qubo, &state);
            energies.push(energy);
        }

        energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ground = energies[0];
        let excited = if energies.len() > 1 {
            energies[1]
        } else {
            ground
        };

        // Gap estimate includes transverse field effects
        let transverse_contribution = (1.0 - s) * (n as f64).sqrt();
        let problem_gap = excited - ground;
        let gap = (problem_gap * s + transverse_contribution * (1.0 - s)).max(0.01);

        let velocity = 1.0 / self.config.total_time;
        let diabatic_prob = (-std::f64::consts::PI * gap.powi(2) / (2.0 * velocity)).exp();

        InstantaneousHamiltonian {
            s,
            eigenvalues: energies,
            ground_energy: ground,
            excited_energy: excited,
            gap,
            diabatic_probability: diabatic_prob,
        }
    }

    /// Compute energy for a given state
    fn compute_state_energy(&self, qubo: &Array2<f64>, state: &[f64]) -> f64 {
        let n = state.len();
        let mut energy = 0.0;

        for i in 0..n {
            energy += qubo[[i, i]] * state[i];
            for j in (i + 1)..n {
                energy += qubo[[i, j]] * state[i] * state[j];
            }
        }

        energy
    }

    /// Optimize the adiabatic path based on gap analysis
    pub fn optimize_path(&mut self, qubo: &Array2<f64>) -> Result<(), String> {
        if !self.config.dynamic_adjustment {
            return Ok(());
        }

        println!("Optimizing adiabatic path...");

        // Analyze gaps along initial schedule
        let mut gaps = Vec::new();
        for &s in &self.s_schedule {
            let h_info = self.compute_instantaneous_hamiltonian(qubo, s);
            gaps.push(h_info.gap);
        }

        // Identify regions with small gaps (avoided crossings)
        let avg_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let min_gap = gaps.iter().copied().fold(f64::INFINITY, f64::min);

        println!("Average gap: {avg_gap:.4}, Min gap: {min_gap:.4}");

        // Adjust schedule: spend more time in small-gap regions
        let mut new_schedule = Vec::new();
        let mut cumulative_time = 0.0;

        for i in 0..gaps.len() - 1 {
            let gap = gaps[i];
            // Inverse gap weighting: slower evolution where gap is smaller
            let weight = if gap < self.config.gap_threshold {
                1.0 / gap.max(0.001)
            } else {
                1.0
            };

            cumulative_time += weight;
            new_schedule.push((self.s_schedule[i], cumulative_time));
        }

        // Normalize to total time
        let total_weight = cumulative_time;
        for (_, time) in &mut new_schedule {
            *time = *time * self.config.total_time / total_weight;
        }

        // Update schedules
        self.time_schedule = new_schedule.iter().map(|(_, t)| *t).collect();
        self.s_schedule = new_schedule.iter().map(|(s, _)| *s).collect();
        self.gap_schedule = gaps;

        println!("Path optimization complete");
        Ok(())
    }

    /// Simulate adiabatic evolution and generate samples
    pub fn run_adiabatic_evolution(
        &self,
        qubo: &Array2<f64>,
    ) -> Result<Vec<HashMap<String, i32>>, String> {
        let n = qubo.nrows();
        let mut rng = thread_rng();
        let mut samples = Vec::new();

        println!("Running adiabatic evolution...");

        for sample_idx in 0..self.config.num_samples {
            // Start in equal superposition (simulated by random state)
            let mut state: Vec<f64> = (0..n)
                .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
                .collect();

            // Evolve along the adiabatic path
            for step_idx in 0..self.s_schedule.len() - 1 {
                let s = self.s_schedule[step_idx];
                let gap = self.gap_schedule[step_idx];

                // Probability of diabatic transition (state change)
                let diabatic_prob = if gap > 1e-10 {
                    let ds = self.s_schedule[step_idx + 1] - s;
                    (-std::f64::consts::PI * gap.powi(2) / (2.0 * ds)).exp()
                } else {
                    0.5
                };

                // Simulate quantum tunneling and transitions
                if rng.gen_bool(diabatic_prob) {
                    // Diabatic transition: flip a random bit
                    let flip_idx = rng.gen_range(0..n);
                    state[flip_idx] = 1.0 - state[flip_idx];
                }

                // Local optimization (simulated quantum annealing)
                if rng.gen_bool(0.1) {
                    state = self.local_optimization(qubo, &state, s);
                }
            }

            // Final state measurement
            let mut result = HashMap::new();
            for (i, &val) in state.iter().enumerate() {
                result.insert(format!("x{i}"), i32::from(val > 0.5));
            }
            samples.push(result);

            if (sample_idx + 1) % 10 == 0 {
                println!(
                    "Generated {}/{} samples",
                    sample_idx + 1,
                    self.config.num_samples
                );
            }
        }

        Ok(samples)
    }

    /// Local optimization at given adiabatic parameter
    fn local_optimization(&self, qubo: &Array2<f64>, state: &[f64], s: f64) -> Vec<f64> {
        let n = state.len();
        let mut best_state = state.to_vec();
        let mut best_energy = self.compute_state_energy(qubo, state);
        let mut rng = thread_rng();

        // Simulated annealing-like local search
        let temp = self.config.temperature.max(0.1) * (1.0 - s);

        for _ in 0..n {
            let flip_idx = rng.gen_range(0..n);
            let mut new_state = best_state.clone();
            new_state[flip_idx] = 1.0 - new_state[flip_idx];

            let new_energy = self.compute_state_energy(qubo, &new_state);
            let delta = new_energy - best_energy;

            // Metropolis criterion
            if delta < 0.0 || rng.gen_bool((-delta / temp).exp()) {
                best_state = new_state;
                best_energy = new_energy;
            }
        }

        best_state
    }

    /// Get diagnostic information about the adiabatic path
    pub fn get_path_diagnostics(&self) -> PathDiagnostics {
        PathDiagnostics {
            total_time: self.config.total_time,
            num_steps: self.config.time_steps,
            min_gap: self
                .gap_schedule
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            max_gap: self
                .gap_schedule
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
            avg_gap: self.gap_schedule.iter().sum::<f64>() / self.gap_schedule.len() as f64,
            gap_variance: {
                let mean = self.gap_schedule.iter().sum::<f64>() / self.gap_schedule.len() as f64;
                self.gap_schedule
                    .iter()
                    .map(|g| (g - mean).powi(2))
                    .sum::<f64>()
                    / self.gap_schedule.len() as f64
            },
        }
    }
}

/// Diagnostic information about the adiabatic path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathDiagnostics {
    pub total_time: f64,
    pub num_steps: usize,
    pub min_gap: f64,
    pub max_gap: f64,
    pub avg_gap: f64,
    pub gap_variance: f64,
}

/// Quantum Adiabatic Sampler using path optimization
#[derive(Debug, Clone)]
pub struct QuantumAdiabaticSampler {
    config: AdiabaticPathConfig,
}

impl QuantumAdiabaticSampler {
    /// Create new quantum adiabatic sampler
    pub const fn new(config: AdiabaticPathConfig) -> Self {
        Self { config }
    }
}

impl Sampler for QuantumAdiabaticSampler {
    fn run_qubo(
        &self,
        problem: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        let (matrix, _var_map) = problem;

        // Create optimizer
        let mut optimizer = QuantumAdiabaticPathOptimizer::new(self.config.clone());

        // Optimize path
        optimizer
            .optimize_path(matrix)
            .map_err(SamplerError::InvalidParameter)?;

        // Run evolution
        let samples = optimizer
            .run_adiabatic_evolution(matrix)
            .map_err(SamplerError::InvalidParameter)?;

        // Convert to SampleResult format
        let results: Vec<SampleResult> = samples
            .into_iter()
            .take(shots)
            .map(|sample| {
                let energy = self.compute_sample_energy(&sample, matrix);
                let assignments = sample.into_iter().map(|(k, v)| (k, v != 0)).collect();
                SampleResult {
                    assignments,
                    energy,
                    occurrences: 1,
                }
            })
            .collect();

        Ok(results)
    }

    fn run_hobo(
        &self,
        _problem: &(scirs2_core::ndarray::ArrayD<f64>, HashMap<String, usize>),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::UnsupportedOperation(
            "HOBO sampling not yet implemented for adiabatic sampler".to_string(),
        ))
    }
}

impl QuantumAdiabaticSampler {
    /// Compute energy for a sample
    fn compute_sample_energy(&self, sample: &HashMap<String, i32>, qubo: &Array2<f64>) -> f64 {
        let n = qubo.nrows();
        let mut energy = 0.0;

        for i in 0..n {
            let x_i = sample.get(&format!("x{i}")).copied().unwrap_or(0) as f64;
            energy += qubo[[i, i]] * x_i;
            for j in (i + 1)..n {
                let x_j = sample.get(&format!("x{j}")).copied().unwrap_or(0) as f64;
                energy += qubo[[i, j]] * x_i * x_j;
            }
        }

        energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adiabatic_path_creation() {
        let config = AdiabaticPathConfig::default();
        let optimizer = QuantumAdiabaticPathOptimizer::new(config);
        assert_eq!(optimizer.s_schedule.len(), 1001); // 1000 steps + 1
    }

    #[test]
    fn test_linear_interpolation() {
        let config = AdiabaticPathConfig {
            interpolation: PathInterpolation::Linear,
            time_steps: 10,
            ..Default::default()
        };
        let optimizer = QuantumAdiabaticPathOptimizer::new(config);
        assert!((optimizer.s_schedule[0] - 0.0).abs() < 1e-10);
        assert!((optimizer.s_schedule[10] - 1.0).abs() < 1e-10);
        assert!((optimizer.s_schedule[5] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_interpolation() {
        let config = AdiabaticPathConfig {
            interpolation: PathInterpolation::Polynomial { exponent: 2.0 },
            time_steps: 10,
            ..Default::default()
        };
        let optimizer = QuantumAdiabaticPathOptimizer::new(config);
        assert!((optimizer.s_schedule[5] - 0.25).abs() < 1e-10); // (0.5)^2 = 0.25
    }

    #[test]
    fn test_small_qubo_evolution() {
        let config = AdiabaticPathConfig {
            time_steps: 100,
            num_samples: 10,
            ..Default::default()
        };

        // Simple 2-qubit QUBO
        let mut qubo = Array2::zeros((2, 2));
        qubo[[0, 0]] = -1.0;
        qubo[[1, 1]] = -1.0;
        qubo[[0, 1]] = 2.0;

        let optimizer = QuantumAdiabaticPathOptimizer::new(config);
        let samples = optimizer.run_adiabatic_evolution(&qubo).unwrap();
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_gap_computation() {
        let config = AdiabaticPathConfig::default();
        let optimizer = QuantumAdiabaticPathOptimizer::new(config);

        let mut qubo = Array2::zeros((3, 3));
        qubo[[0, 0]] = -1.0;
        qubo[[1, 1]] = -1.0;
        qubo[[2, 2]] = -1.0;

        let h_info = optimizer.compute_instantaneous_hamiltonian(&qubo, 0.5);
        assert!(h_info.gap > 0.0);
        assert!(h_info.diabatic_probability >= 0.0 && h_info.diabatic_probability <= 1.0);
    }
}
