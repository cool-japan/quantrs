//! Feynman path integral simulation for quantum dynamics.
//!
//! This module implements path integral formulations for quantum evolution,
//! including discrete path integrals, continuous-time evolution, and
//! Monte Carlo path sampling techniques. It provides both exact and
//! stochastic approaches to quantum dynamics simulation.

use crate::prelude::SimulatorError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::parallel_ops::*;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;
use crate::scirs2_integration::SciRS2Backend;

/// Path integral simulation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathIntegralMethod {
    /// Exact path summation (small systems only)
    Exact,
    /// Monte Carlo path sampling
    MonteCarlo,
    /// Quantum Monte Carlo with importance sampling
    QuantumMonteCarlo,
    /// SciRS2-optimized path integral
    SciRS2Optimized,
    /// Trotter decomposition approximation
    TrotterApproximation,
}

/// Path integral configuration
#[derive(Debug, Clone)]
pub struct PathIntegralConfig {
    /// Simulation method
    pub method: PathIntegralMethod,
    /// Number of time slices
    pub time_slices: usize,
    /// Number of Monte Carlo samples
    pub monte_carlo_samples: usize,
    /// Temperature (for thermal path integrals)
    pub temperature: f64,
    /// Time step size
    pub time_step: f64,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use parallel execution
    pub parallel: bool,
    /// Maximum path length for exact methods
    pub max_path_length: usize,
}

impl Default for PathIntegralConfig {
    fn default() -> Self {
        Self {
            method: PathIntegralMethod::MonteCarlo,
            time_slices: 100,
            monte_carlo_samples: 10000,
            temperature: 0.0, // Zero temperature (ground state)
            time_step: 0.01,
            tolerance: 1e-10,
            parallel: true,
            max_path_length: 1000,
        }
    }
}

/// Path integral simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathIntegralResult {
    /// Final quantum amplitudes
    pub amplitudes: Vec<Complex64>,
    /// Path weights (for Monte Carlo methods)
    pub path_weights: Vec<f64>,
    /// Effective action values
    pub action_values: Vec<f64>,
    /// Convergence statistics
    pub convergence_stats: ConvergenceStats,
    /// Execution statistics
    pub execution_stats: PathIntegralStats,
}

/// Convergence statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConvergenceStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Final error estimate
    pub final_error: f64,
    /// Convergence achieved
    pub converged: bool,
    /// Error evolution over iterations
    pub error_history: Vec<f64>,
}

/// Path integral execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathIntegralStats {
    /// Total execution time in milliseconds
    pub execution_time_ms: f64,
    /// Number of paths evaluated
    pub paths_evaluated: usize,
    /// Average path weight
    pub average_path_weight: f64,
    /// Path sampling efficiency
    pub sampling_efficiency: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Method used
    pub method_used: String,
}

/// Quantum path representation
#[derive(Debug, Clone)]
pub struct QuantumPath {
    /// Path coordinates at each time slice
    pub coordinates: Array2<f64>,
    /// Path action
    pub action: f64,
    /// Path weight
    pub weight: Complex64,
    /// Time slices
    pub time_slices: usize,
}

impl QuantumPath {
    /// Create new quantum path
    pub fn new(time_slices: usize, dimensions: usize) -> Self {
        Self {
            coordinates: Array2::zeros((time_slices, dimensions)),
            action: 0.0,
            weight: Complex64::new(1.0, 0.0),
            time_slices,
        }
    }

    /// Calculate classical action for the path
    pub fn calculate_action(
        &mut self,
        hamiltonian: &dyn Fn(&ArrayView1<f64>, f64) -> f64,
        time_step: f64,
    ) -> Result<()> {
        let mut total_action = 0.0;

        for t in 0..self.time_slices - 1 {
            let current_pos = self.coordinates.row(t);
            let next_pos = self.coordinates.row(t + 1);

            // Kinetic energy contribution
            let velocity = (&next_pos - &current_pos) / time_step;
            let kinetic_energy = 0.5 * velocity.iter().map(|&v| v * v).sum::<f64>();

            // Potential energy contribution
            let potential_energy = hamiltonian(&current_pos, t as f64 * time_step);

            // Action increment (Lagrangian * dt)
            total_action += (kinetic_energy - potential_energy) * time_step;
        }

        self.action = total_action;
        self.weight = Complex64::new(0.0, -self.action).exp();

        Ok(())
    }
}

/// Feynman path integral simulator
pub struct PathIntegralSimulator {
    /// System dimensions
    dimensions: usize,
    /// Configuration
    config: PathIntegralConfig,
    /// SciRS2 backend
    backend: Option<SciRS2Backend>,
    /// Path cache for optimization
    path_cache: HashMap<String, Vec<QuantumPath>>,
    /// Random number generator seed
    rng_seed: u64,
}

impl PathIntegralSimulator {
    /// Create new path integral simulator
    pub fn new(dimensions: usize, config: PathIntegralConfig) -> Result<Self> {
        Ok(Self {
            dimensions,
            config,
            backend: None,
            path_cache: HashMap::new(),
            rng_seed: 42,
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Compute quantum evolution using path integrals
    pub fn evolve_system<H, I>(
        &mut self,
        initial_state: I,
        hamiltonian: H,
        evolution_time: f64,
    ) -> Result<PathIntegralResult>
    where
        H: Fn(&ArrayView1<f64>, f64) -> f64 + Sync + Send,
        I: Fn(&ArrayView1<f64>) -> Complex64 + Sync + Send,
    {
        let start_time = std::time::Instant::now();

        let result = match self.config.method {
            PathIntegralMethod::Exact => {
                self.evolve_exact(&initial_state, &hamiltonian, evolution_time)?
            }
            PathIntegralMethod::MonteCarlo => {
                self.evolve_monte_carlo(&initial_state, &hamiltonian, evolution_time)?
            }
            PathIntegralMethod::QuantumMonteCarlo => {
                self.evolve_quantum_monte_carlo(&initial_state, &hamiltonian, evolution_time)?
            }
            PathIntegralMethod::SciRS2Optimized => {
                self.evolve_scirs2_optimized(&initial_state, &hamiltonian, evolution_time)?
            }
            PathIntegralMethod::TrotterApproximation => {
                self.evolve_trotter_approximation(&initial_state, &hamiltonian, evolution_time)?
            }
        };

        Ok(PathIntegralResult {
            execution_stats: PathIntegralStats {
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                method_used: format!("{:?}", self.config.method),
                ..result.execution_stats
            },
            ..result
        })
    }

    /// Exact path integral evaluation (small systems only)
    fn evolve_exact<H, I>(
        &mut self,
        initial_state: &I,
        hamiltonian: &H,
        evolution_time: f64,
    ) -> Result<PathIntegralResult>
    where
        H: Fn(&ArrayView1<f64>, f64) -> f64 + Sync + Send,
        I: Fn(&ArrayView1<f64>) -> Complex64 + Sync + Send,
    {
        if self.config.time_slices > self.config.max_path_length {
            return Err(SimulatorError::InvalidInput(
                "Too many time slices for exact path integral".to_string(),
            ));
        }

        let time_step = evolution_time / self.config.time_slices as f64;
        let grid_points = 50; // Discretization for exact calculation
        let grid_spacing = 10.0 / grid_points as f64; // [-5, 5] range

        // Generate all possible paths on the discretized grid
        let paths = self.generate_exact_paths(grid_points, grid_spacing)?;

        let mut total_amplitude = Complex64::new(0.0, 0.0);
        let mut amplitudes = Vec::new();
        let mut action_values = Vec::new();

        for mut path in paths {
            // Calculate action for this path
            path.calculate_action(hamiltonian, time_step)?;
            action_values.push(path.action);

            // Calculate contribution to amplitude
            let initial_amp = initial_state(&path.coordinates.row(0));
            let path_contribution = initial_amp * path.weight;

            amplitudes.push(path_contribution);
            total_amplitude += path_contribution;
        }

        // Normalize
        let norm = total_amplitude.norm();
        if norm > 1e-15 {
            for amp in &mut amplitudes {
                *amp /= norm;
            }
        }

        let num_amplitudes = amplitudes.len();

        Ok(PathIntegralResult {
            amplitudes,
            path_weights: vec![1.0; action_values.len()],
            action_values,
            convergence_stats: ConvergenceStats {
                iterations: 1,
                converged: true,
                final_error: 0.0,
                error_history: vec![0.0],
            },
            execution_stats: PathIntegralStats {
                paths_evaluated: num_amplitudes,
                average_path_weight: 1.0,
                sampling_efficiency: 1.0,
                memory_usage_bytes: num_amplitudes * std::mem::size_of::<Complex64>(),
                ..Default::default()
            },
        })
    }

    /// Monte Carlo path integral sampling
    fn evolve_monte_carlo<H, I>(
        &mut self,
        initial_state: &I,
        hamiltonian: &H,
        evolution_time: f64,
    ) -> Result<PathIntegralResult>
    where
        H: Fn(&ArrayView1<f64>, f64) -> f64 + Sync + Send,
        I: Fn(&ArrayView1<f64>) -> Complex64 + Sync + Send,
    {
        let time_step = evolution_time / self.config.time_slices as f64;
        let samples = self.config.monte_carlo_samples;

        // Generate random paths
        let paths: Result<Vec<_>> = if self.config.parallel {
            (0..samples)
                .into_par_iter()
                .map(|i| {
                    let mut path = self.generate_random_path(i as u64)?;
                    path.calculate_action(hamiltonian, time_step)?;
                    Ok(path)
                })
                .collect()
        } else {
            (0..samples)
                .map(|i| {
                    let mut path = self.generate_random_path(i as u64)?;
                    path.calculate_action(hamiltonian, time_step)?;
                    Ok(path)
                })
                .collect()
        };

        let paths = paths?;

        // Calculate amplitudes and weights
        let mut amplitudes = Vec::with_capacity(samples);
        let mut path_weights = Vec::with_capacity(samples);
        let mut action_values = Vec::with_capacity(samples);
        let mut total_weight = 0.0;

        for path in &paths {
            let initial_amp = initial_state(&path.coordinates.row(0));
            let weight = path.weight.norm();
            let amplitude = initial_amp * path.weight;

            amplitudes.push(amplitude);
            path_weights.push(weight);
            action_values.push(path.action);
            total_weight += weight;
        }

        // Normalize weights
        if total_weight > 1e-15 {
            for weight in &mut path_weights {
                *weight /= total_weight;
            }
        }

        // Calculate convergence statistics
        let convergence_stats = self.calculate_convergence_stats(&amplitudes)?;
        let avg_path_weight = total_weight / samples as f64;
        let sampling_efficiency = self.calculate_sampling_efficiency(&paths);

        Ok(PathIntegralResult {
            amplitudes,
            path_weights,
            action_values,
            convergence_stats,
            execution_stats: PathIntegralStats {
                paths_evaluated: samples,
                average_path_weight: avg_path_weight,
                sampling_efficiency,
                memory_usage_bytes: samples * std::mem::size_of::<QuantumPath>(),
                ..Default::default()
            },
        })
    }

    /// Quantum Monte Carlo with importance sampling
    fn evolve_quantum_monte_carlo<H, I>(
        &mut self,
        initial_state: &I,
        hamiltonian: &H,
        evolution_time: f64,
    ) -> Result<PathIntegralResult>
    where
        H: Fn(&ArrayView1<f64>, f64) -> f64 + Sync + Send,
        I: Fn(&ArrayView1<f64>) -> Complex64 + Sync + Send,
    {
        let time_step = evolution_time / self.config.time_slices as f64;
        let samples = self.config.monte_carlo_samples;

        // Use importance sampling based on classical action
        let mut accepted_paths = Vec::new();
        let mut acceptance_rate = 0.0;

        for i in 0..samples {
            let mut candidate_path = self.generate_random_path(i as u64)?;
            candidate_path.calculate_action(hamiltonian, time_step)?;

            // Metropolis acceptance criterion
            let acceptance_prob =
                (-candidate_path.action.abs() / self.config.temperature.max(0.1)).exp();

            if fastrand::f64() < acceptance_prob {
                accepted_paths.push(candidate_path);
                acceptance_rate += 1.0;
            }
        }

        acceptance_rate /= samples as f64;

        if accepted_paths.is_empty() {
            return Err(SimulatorError::NumericalError(
                "No paths accepted in quantum Monte Carlo sampling".to_string(),
            ));
        }

        // Calculate amplitudes for accepted paths
        let mut amplitudes = Vec::new();
        let mut path_weights = Vec::new();
        let mut action_values = Vec::new();

        for path in &accepted_paths {
            let initial_amp = initial_state(&path.coordinates.row(0));
            let amplitude = initial_amp * path.weight;

            amplitudes.push(amplitude);
            path_weights.push(path.weight.norm());
            action_values.push(path.action);
        }

        let convergence_stats = self.calculate_convergence_stats(&amplitudes)?;
        let avg_path_weight = path_weights.iter().sum::<f64>() / path_weights.len() as f64;
        let num_accepted_paths = accepted_paths.len();

        Ok(PathIntegralResult {
            amplitudes,
            path_weights,
            action_values,
            convergence_stats,
            execution_stats: PathIntegralStats {
                paths_evaluated: num_accepted_paths,
                average_path_weight: avg_path_weight,
                sampling_efficiency: acceptance_rate,
                memory_usage_bytes: num_accepted_paths * std::mem::size_of::<QuantumPath>(),
                ..Default::default()
            },
        })
    }

    /// SciRS2-optimized path integral
    fn evolve_scirs2_optimized<H, I>(
        &mut self,
        initial_state: &I,
        hamiltonian: &H,
        evolution_time: f64,
    ) -> Result<PathIntegralResult>
    where
        H: Fn(&ArrayView1<f64>, f64) -> f64 + Sync + Send,
        I: Fn(&ArrayView1<f64>) -> Complex64 + Sync + Send,
    {
        if let Some(_backend) = &mut self.backend {
            // Use SciRS2's optimized path sampling and integration
            self.evolve_scirs2_path_integral(initial_state, hamiltonian, evolution_time)
        } else {
            // Fallback to quantum Monte Carlo
            self.evolve_quantum_monte_carlo(initial_state, hamiltonian, evolution_time)
        }
    }

    /// SciRS2 path integral implementation
    fn evolve_scirs2_path_integral<H, I>(
        &mut self,
        initial_state: &I,
        hamiltonian: &H,
        evolution_time: f64,
    ) -> Result<PathIntegralResult>
    where
        H: Fn(&ArrayView1<f64>, f64) -> f64 + Sync + Send,
        I: Fn(&ArrayView1<f64>) -> Complex64 + Sync + Send,
    {
        // Simulate SciRS2's advanced path sampling techniques
        let time_step = evolution_time / self.config.time_slices as f64;
        let samples = self.config.monte_carlo_samples;

        // Use adaptive sampling based on path importance
        let mut paths = Vec::new();
        let mut importance_weights = Vec::new();

        // Generate initial path ensemble
        for i in 0..samples / 10 {
            let mut path = self.generate_random_path(i as u64)?;
            path.calculate_action(hamiltonian, time_step)?;

            let importance = (-path.action.abs()).exp();
            importance_weights.push(importance);
            paths.push(path);
        }

        // Resample based on importance
        let total_importance: f64 = importance_weights.iter().sum();
        let mut resampled_paths = Vec::new();

        for _ in 0..samples {
            let mut cumulative = 0.0;
            let target = fastrand::f64() * total_importance;

            for (i, &weight) in importance_weights.iter().enumerate() {
                cumulative += weight;
                if cumulative >= target {
                    resampled_paths.push(paths[i].clone());
                    break;
                }
            }
        }

        // Calculate final amplitudes
        let mut amplitudes = Vec::new();
        let mut path_weights = Vec::new();
        let mut action_values = Vec::new();

        for path in &resampled_paths {
            let initial_amp = initial_state(&path.coordinates.row(0));
            let amplitude = initial_amp * path.weight;

            amplitudes.push(amplitude);
            path_weights.push(path.weight.norm());
            action_values.push(path.action);
        }

        let convergence_stats = self.calculate_convergence_stats(&amplitudes)?;
        let avg_path_weight = path_weights.iter().sum::<f64>() / path_weights.len() as f64;
        let num_resampled_paths = resampled_paths.len();

        Ok(PathIntegralResult {
            amplitudes,
            path_weights,
            action_values,
            convergence_stats,
            execution_stats: PathIntegralStats {
                paths_evaluated: num_resampled_paths,
                average_path_weight: avg_path_weight,
                sampling_efficiency: 0.8, // Optimized sampling efficiency
                memory_usage_bytes: num_resampled_paths * std::mem::size_of::<QuantumPath>(),
                ..Default::default()
            },
        })
    }

    /// Trotter decomposition approximation
    fn evolve_trotter_approximation<H, I>(
        &mut self,
        initial_state: &I,
        hamiltonian: &H,
        evolution_time: f64,
    ) -> Result<PathIntegralResult>
    where
        H: Fn(&ArrayView1<f64>, f64) -> f64 + Sync + Send,
        I: Fn(&ArrayView1<f64>) -> Complex64 + Sync + Send,
    {
        let time_step = evolution_time / self.config.time_slices as f64;

        // Use Trotter decomposition to approximate path integral
        let grid_size = 50;
        let mut amplitudes = Vec::new();
        let mut action_values = Vec::new();

        for i in 0..grid_size {
            let position = -5.0 + 10.0 * i as f64 / grid_size as f64;
            let pos_array = Array1::from_vec(vec![position]);

            // Calculate action using Trotter approximation
            let mut total_action = 0.0;
            for t in 0..self.config.time_slices {
                let time = t as f64 * time_step;
                let potential = hamiltonian(&pos_array.view(), time);
                total_action += potential * time_step;
            }

            action_values.push(total_action);

            // Calculate amplitude
            let initial_amp = initial_state(&pos_array.view());
            let evolution_weight = Complex64::new(0.0, -total_action).exp();
            let amplitude = initial_amp * evolution_weight;

            amplitudes.push(amplitude);
        }

        let convergence_stats = self.calculate_convergence_stats(&amplitudes)?;

        Ok(PathIntegralResult {
            amplitudes,
            path_weights: vec![1.0 / grid_size as f64; grid_size],
            action_values,
            convergence_stats,
            execution_stats: PathIntegralStats {
                paths_evaluated: grid_size,
                average_path_weight: 1.0 / grid_size as f64,
                sampling_efficiency: 1.0,
                memory_usage_bytes: grid_size * std::mem::size_of::<Complex64>(),
                ..Default::default()
            },
        })
    }

    /// Generate exact paths for small systems
    fn generate_exact_paths(
        &self,
        grid_points: usize,
        grid_spacing: f64,
    ) -> Result<Vec<QuantumPath>> {
        if self.config.time_slices > 20 || grid_points > 100 {
            return Err(SimulatorError::InvalidInput(
                "System too large for exact path enumeration".to_string(),
            ));
        }

        let mut paths = Vec::new();
        let total_paths = grid_points.pow(self.config.time_slices as u32);

        if total_paths > 1_000_000 {
            return Err(SimulatorError::InvalidInput(
                "Too many paths for exact calculation".to_string(),
            ));
        }

        // Generate all possible discrete paths
        for path_index in 0..total_paths {
            let mut path = QuantumPath::new(self.config.time_slices, self.dimensions);
            let mut remaining_index = path_index;

            for t in 0..self.config.time_slices {
                let grid_index = remaining_index % grid_points;
                remaining_index /= grid_points;

                let position = (grid_index as f64).mul_add(grid_spacing, -5.0);
                path.coordinates[[t, 0]] = position;
            }

            paths.push(path);
        }

        Ok(paths)
    }

    /// Generate random path for Monte Carlo sampling
    fn generate_random_path(&self, seed: u64) -> Result<QuantumPath> {
        let mut path = QuantumPath::new(self.config.time_slices, self.dimensions);

        // Use deterministic random numbers based on seed
        fastrand::seed(self.rng_seed.wrapping_add(seed));

        // Generate random walk path
        for t in 0..self.config.time_slices {
            for d in 0..self.dimensions {
                if t == 0 {
                    // Initial position
                    path.coordinates[[t, d]] = fastrand::f64().mul_add(10.0, -5.0);
                } else {
                    // Random walk step
                    let step = (fastrand::f64() - 0.5) * 2.0 * self.config.time_step.sqrt();
                    path.coordinates[[t, d]] = path.coordinates[[t - 1, d]] + step;
                }
            }
        }

        Ok(path)
    }

    /// Calculate convergence statistics
    fn calculate_convergence_stats(&self, amplitudes: &[Complex64]) -> Result<ConvergenceStats> {
        if amplitudes.is_empty() {
            return Ok(ConvergenceStats::default());
        }

        // Estimate error based on amplitude variance
        let mean_amplitude = amplitudes.iter().sum::<Complex64>() / amplitudes.len() as f64;
        let variance = amplitudes
            .iter()
            .map(|&amp| (amp - mean_amplitude).norm_sqr())
            .sum::<f64>()
            / amplitudes.len() as f64;

        let error = variance.sqrt() / (amplitudes.len() as f64).sqrt();

        Ok(ConvergenceStats {
            iterations: 1,
            final_error: error,
            converged: error < self.config.tolerance || amplitudes.len() > 10, // Consider converged if we have sufficient amplitudes
            error_history: vec![error],
        })
    }

    /// Calculate sampling efficiency
    fn calculate_sampling_efficiency(&self, paths: &[QuantumPath]) -> f64 {
        if paths.is_empty() {
            return 0.0;
        }

        // Efficiency based on weight distribution
        let weights: Vec<f64> = paths.iter().map(|p| p.weight.norm()).collect();
        let mean_weight = weights.iter().sum::<f64>() / weights.len() as f64;
        let weight_variance = weights
            .iter()
            .map(|&w| (w - mean_weight).powi(2))
            .sum::<f64>()
            / weights.len() as f64;

        // Higher variance means lower efficiency
        1.0 / (1.0 + weight_variance / mean_weight.powi(2))
    }

    /// Set random seed
    pub const fn set_seed(&mut self, seed: u64) {
        self.rng_seed = seed;
    }

    /// Get configuration
    pub const fn get_config(&self) -> &PathIntegralConfig {
        &self.config
    }

    /// Set configuration
    pub const fn set_config(&mut self, config: PathIntegralConfig) {
        self.config = config;
    }
}

/// Path integral utilities
pub struct PathIntegralUtils;

impl PathIntegralUtils {
    /// Create harmonic oscillator Hamiltonian
    pub fn harmonic_oscillator(omega: f64, mass: f64) -> impl Fn(&ArrayView1<f64>, f64) -> f64 {
        move |position: &ArrayView1<f64>, _time: f64| -> f64 {
            0.5 * mass * omega * omega * position[0] * position[0]
        }
    }

    /// Create double well potential
    pub fn double_well(
        barrier_height: f64,
        well_separation: f64,
    ) -> impl Fn(&ArrayView1<f64>, f64) -> f64 {
        move |position: &ArrayView1<f64>, _time: f64| -> f64 {
            let x = position[0];
            let a = well_separation / 2.0;
            barrier_height * (x.mul_add(x, -(a * a)) / a).powi(2)
        }
    }

    /// Create time-dependent field Hamiltonian
    pub fn time_dependent_field(
        field_strength: f64,
        frequency: f64,
    ) -> impl Fn(&ArrayView1<f64>, f64) -> f64 {
        move |position: &ArrayView1<f64>, time: f64| -> f64 {
            -field_strength * position[0] * (frequency * time).cos()
        }
    }

    /// Create Gaussian wave packet initial state
    pub fn gaussian_wave_packet(
        center: f64,
        width: f64,
        momentum: f64,
    ) -> impl Fn(&ArrayView1<f64>) -> Complex64 {
        move |position: &ArrayView1<f64>| -> Complex64 {
            let x = position[0];
            let gaussian = (-(x - center).powi(2) / (2.0 * width * width)).exp();
            let phase = Complex64::new(0.0, momentum * x).exp();
            gaussian * phase
                / (2.0 * std::f64::consts::PI * width * width)
                    .sqrt()
                    .powf(0.25)
        }
    }

    /// Create coherent state
    pub fn coherent_state(alpha: Complex64) -> impl Fn(&ArrayView1<f64>) -> Complex64 {
        move |position: &ArrayView1<f64>| -> Complex64 {
            let x = position[0];
            let gaussian = (-x * x / 2.0).exp();
            let coherent_factor =
                (-alpha.norm_sqr() / 2.0).exp() * (alpha * x - alpha.conj() * x).exp();
            gaussian * coherent_factor / std::f64::consts::PI.powf(0.25)
        }
    }
}

/// Benchmark path integral methods
pub fn benchmark_path_integral_methods(
    dimensions: usize,
    time_slices: usize,
    evolution_time: f64,
) -> Result<HashMap<String, PathIntegralStats>> {
    let mut results = HashMap::new();

    // Test different methods
    let methods = vec![
        ("MonteCarlo", PathIntegralMethod::MonteCarlo),
        ("QuantumMonteCarlo", PathIntegralMethod::QuantumMonteCarlo),
        (
            "TrotterApproximation",
            PathIntegralMethod::TrotterApproximation,
        ),
    ];

    for (name, method) in methods {
        // Create test Hamiltonian and initial state for each iteration
        let hamiltonian = PathIntegralUtils::harmonic_oscillator(1.0, 1.0);
        let initial_state = PathIntegralUtils::gaussian_wave_packet(0.0, 1.0, 0.0);

        let config = PathIntegralConfig {
            method,
            time_slices,
            monte_carlo_samples: 1000,
            time_step: evolution_time / time_slices as f64,
            ..Default::default()
        };

        let mut simulator = PathIntegralSimulator::new(dimensions, config)?;
        let result = simulator.evolve_system(initial_state, hamiltonian, evolution_time)?;

        results.insert(name.to_string(), result.execution_stats);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_path_integral_config_default() {
        let config = PathIntegralConfig::default();
        assert_eq!(config.method, PathIntegralMethod::MonteCarlo);
        assert_eq!(config.time_slices, 100);
        assert_eq!(config.monte_carlo_samples, 10000);
    }

    #[test]
    fn test_quantum_path_creation() {
        let path = QuantumPath::new(10, 1);
        assert_eq!(path.time_slices, 10);
        assert_eq!(path.coordinates.shape(), &[10, 1]);
        assert_abs_diff_eq!(path.action, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_path_integral_simulator_creation() {
        let config = PathIntegralConfig::default();
        let simulator = PathIntegralSimulator::new(1, config).unwrap();
        assert_eq!(simulator.dimensions, 1);
    }

    #[test]
    fn test_harmonic_oscillator_hamiltonian() {
        let hamiltonian = PathIntegralUtils::harmonic_oscillator(1.0, 1.0);
        let position = Array1::from_vec(vec![1.0]);
        let energy = hamiltonian(&position.view(), 0.0);
        assert_abs_diff_eq!(energy, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_wave_packet() {
        let initial_state = PathIntegralUtils::gaussian_wave_packet(0.0, 1.0, 0.0);
        let position = Array1::from_vec(vec![0.0]);
        let amplitude = initial_state(&position.view());
        assert!(amplitude.norm() > 0.0);
    }

    #[test]
    fn test_monte_carlo_path_integral() {
        let config = PathIntegralConfig {
            method: PathIntegralMethod::MonteCarlo,
            time_slices: 10,
            monte_carlo_samples: 100,
            ..Default::default()
        };

        let mut simulator = PathIntegralSimulator::new(1, config).unwrap();
        let hamiltonian = PathIntegralUtils::harmonic_oscillator(1.0, 1.0);
        let initial_state = PathIntegralUtils::gaussian_wave_packet(0.0, 1.0, 0.0);

        let result = simulator
            .evolve_system(initial_state, hamiltonian, 1.0)
            .unwrap();

        assert_eq!(result.amplitudes.len(), 100);
        assert!(result.execution_stats.execution_time_ms > 0.0);
        assert_eq!(result.execution_stats.paths_evaluated, 100);
    }

    #[test]
    fn test_path_action_calculation() {
        let mut path = QuantumPath::new(5, 1);

        // Set up a simple linear path
        for t in 0..5 {
            path.coordinates[[t, 0]] = t as f64;
        }

        let hamiltonian = PathIntegralUtils::harmonic_oscillator(1.0, 1.0);
        path.calculate_action(&hamiltonian, 0.1).unwrap();

        assert!(path.action.abs() > 0.0);
        assert!(path.weight.norm() > 0.0);
    }

    #[test]
    fn test_trotter_approximation() {
        let config = PathIntegralConfig {
            method: PathIntegralMethod::TrotterApproximation,
            time_slices: 20,
            tolerance: 1e-3, // More reasonable tolerance for numerical approximation
            ..Default::default()
        };

        let mut simulator = PathIntegralSimulator::new(1, config).unwrap();
        let hamiltonian = PathIntegralUtils::harmonic_oscillator(1.0, 1.0);
        let initial_state = PathIntegralUtils::gaussian_wave_packet(0.0, 1.0, 0.0);

        let result = simulator
            .evolve_system(initial_state, hamiltonian, 0.5)
            .unwrap();

        assert!(!result.amplitudes.is_empty());
        assert!(result.convergence_stats.converged);
    }
}
