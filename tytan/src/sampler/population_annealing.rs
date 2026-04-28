//! # Population Annealing (PA) Sampler
//!
//! Population Annealing (PA) maintains a population of R replicas that are
//! annealed simultaneously through a sequence of inverse temperatures β₁ < β₂ < … < β_K.
//! After each temperature step the replicas are resampled by importance weights,
//! concentrating computational effort on the lowest-energy region of the search space
//! while preserving the correct thermodynamic ensemble.
//!
//! ## Algorithm
//!
//! 1. **Initialise**: create R replicas with independent random binary assignments.
//! 2. **For each β_k in the schedule**, run `sweeps_per_step` Metropolis sweeps on
//!    every replica, then:
//!    - Compute importance weights `w_r = exp(-(β_{k+1} - β_k) * E_r)`.
//!    - Compute ESS = `(Σ w_r)² / Σ w_r²`.
//!    - If `ESS / R < resample_threshold`, multinomial-resample R replicas by weight.
//! 3. Return `shots` samples drawn from the final population.
//!
//! ## Mathematical Formulation
//!
//! Given a QUBO matrix Q, the objective is to minimise:
//!
//! ```text
//! E(x) = sum_{i,j} Q[i,j] * x[i] * x[j],   x[i] in {0, 1}
//! ```
//!
//! The Metropolis acceptance probability at inverse temperature β for a move
//! with energy change ΔE:
//!
//! ```text
//! P(accept) = min(1, exp(-β * ΔE))
//! ```
//!
//! ## Citation
//!
//! Hukushima, K., & Iba, Y. (2003).
//! Population Annealing and Its Application to a Spin Glass.
//! *AIP Conference Proceedings*, 690(1), 200–206.
//! <https://doi.org/10.1063/1.1632130>
//!
//! ## Parameters
//!
//! See [`PAParams`] for all tunable parameters (`population`, `beta_schedule`,
//! `sweeps_per_step`, `resample_threshold`).
//!
//! ## When to Use
//!
//! - **Best for**: problems where you need an ensemble of near-ground-state solutions
//!   or accurate estimates of low-temperature thermodynamic observables.
//! - **Strengths**: provides diversity across the low-energy landscape; ESS-based
//!   resampling avoids weight collapse.
//! - **Limitations**: higher memory usage (R replicas stored simultaneously);
//!   slower per-shot than SA for single-solution queries.
//!
//! ## Usage
//!
//! ```
//! use quantrs2_tytan::sampler::{PopulationAnnealingSampler, Sampler};
//! use scirs2_core::ndarray::Array;
//! use std::collections::HashMap;
//!
//! // K3 Max-Cut QUBO
//! let mut q = Array::<f64, _>::zeros((3, 3));
//! q[[0, 0]] = -1.0;
//! q[[1, 1]] = -1.0;
//! q[[2, 2]] = -1.0;
//! q[[0, 1]] = 2.0;
//! q[[0, 2]] = 2.0;
//! q[[1, 2]] = 2.0;
//!
//! let mut var_map = HashMap::new();
//! var_map.insert("x0".to_string(), 0);
//! var_map.insert("x1".to_string(), 1);
//! var_map.insert("x2".to_string(), 2);
//!
//! // Small schedule for a fast doc-test
//! let betas: Vec<f64> = (0..10).map(|i| 0.1 + 2.9 * i as f64 / 9.0).collect();
//! let sampler = PopulationAnnealingSampler::new()
//!     .with_seed(42)
//!     .with_population(20)
//!     .with_beta_schedule(betas);
//!
//! let results = sampler.run_qubo(&(q, var_map), 5).expect("PA sampler failed");
//! assert!(!results.is_empty());
//! println!("Best energy: {}", results[0].energy);
//! ```

use scirs2_core::ndarray::{Array, ArrayD, Ix2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use std::collections::HashMap;

use super::energy::{
    compute_influence_simd, energy_full_simd, update_influence_simd,
};
use super::{SampleResult, Sampler, SamplerError, SamplerResult};

/// Parameters for the Population Annealing algorithm
#[derive(Debug, Clone)]
pub struct PAParams {
    /// Number of replicas in the population
    pub population: usize,
    /// Inverse temperature schedule (ascending from low to high β)
    pub beta_schedule: Vec<f64>,
    /// Number of Metropolis sweeps per temperature step
    pub sweeps_per_step: usize,
    /// ESS/population threshold below which to resample
    pub resample_threshold: f64,
}

impl Default for PAParams {
    fn default() -> Self {
        // Linspace(0.1, 3.0, 50)
        let betas: Vec<f64> = (0..50)
            .map(|i| 0.1 + (3.0 - 0.1) * i as f64 / 49.0)
            .collect();
        Self {
            population: 100,
            beta_schedule: betas,
            sweeps_per_step: 5,
            resample_threshold: 0.5,
        }
    }
}

/// Population Annealing Sampler
///
/// Uses the Population Annealing algorithm to solve QUBO/HOBO problems.
/// Maintains a population of replicas that evolve through a temperature
/// schedule, with resampling to concentrate on low-energy regions.
///
/// # Example
///
/// ```
/// use quantrs2_tytan::sampler::{PopulationAnnealingSampler, Sampler};
/// use std::collections::HashMap;
/// use scirs2_core::ndarray::Array;
///
/// let mut q = Array::<f64, _>::zeros((3, 3));
/// q[[0, 0]] = -1.0;
/// q[[1, 1]] = -1.0;
/// q[[2, 2]] = -1.0;
/// q[[0, 1]] = 2.0;
/// q[[0, 2]] = 2.0;
/// q[[1, 2]] = 2.0;
///
/// let mut var_map = HashMap::new();
/// var_map.insert("x0".to_string(), 0);
/// var_map.insert("x1".to_string(), 1);
/// var_map.insert("x2".to_string(), 2);
///
/// // Small schedule for a fast doc-test
/// let betas: Vec<f64> = (0..10).map(|i| 0.1 + 2.9 * i as f64 / 9.0).collect();
/// let sampler = PopulationAnnealingSampler::new()
///     .with_seed(42)
///     .with_population(20)
///     .with_beta_schedule(betas);
///
/// let results = sampler.run_qubo(&(q, var_map), 5).expect("Population annealing failed");
/// assert!(!results.is_empty());
/// println!("Best energy: {}", results[0].energy);
/// ```
#[derive(Debug, Clone)]
pub struct PopulationAnnealingSampler {
    /// Random number generator seed
    pub seed: Option<u64>,
    /// PA algorithm parameters
    pub params: PAParams,
}

impl PopulationAnnealingSampler {
    /// Create a new Population Annealing sampler with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self {
            seed: None,
            params: PAParams::default(),
        }
    }

    /// Set the random seed for reproducibility
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the population size (number of replicas)
    #[must_use]
    pub fn with_population(mut self, population: usize) -> Self {
        self.params.population = population;
        self
    }

    /// Set the number of sweeps per temperature step
    #[must_use]
    pub fn with_sweeps_per_step(mut self, sweeps: usize) -> Self {
        self.params.sweeps_per_step = sweeps;
        self
    }

    /// Set the ESS/population resampling threshold
    #[must_use]
    pub fn with_resample_threshold(mut self, threshold: f64) -> Self {
        self.params.resample_threshold = threshold;
        self
    }

    /// Set a custom beta (inverse temperature) schedule
    #[must_use]
    pub fn with_beta_schedule(mut self, schedule: Vec<f64>) -> Self {
        self.params.beta_schedule = schedule;
        self
    }

    /// Compute QUBO energy: E(x) = sum_{i,j} Q[i,j] * x[i] * x[j]
    ///
    /// Delegates to [`energy_full_simd`] which uses 4-wide f64 SIMD for n >= 32.
    #[inline]
    fn compute_qubo_energy_flat(q_matrix: &[f64], state: &[bool], n: usize) -> f64 {
        energy_full_simd(state, q_matrix, n)
    }

    /// Compute influence vector g[i] = Q[i,i] + sum_{j!=i} (Q[i,j] + Q[j,i]) * x[j]
    ///
    /// ΔE from flipping bit i = (1 - 2*x[i]) * g[i]
    ///
    /// Delegates to [`compute_influence_simd`] which uses SIMD for n >= 32.
    #[inline]
    fn compute_influence_flat(q_matrix: &[f64], state: &[bool], n: usize) -> Vec<f64> {
        compute_influence_simd(state, q_matrix, n)
    }

    /// Update influence vector after flipping bit k.
    ///
    /// Delegates to [`update_influence_simd`] which uses SIMD for n >= 32.
    #[inline]
    fn update_influence_flat(g: &mut [f64], q_matrix: &[f64], k: usize, new_val: bool, n: usize) {
        update_influence_simd(g, q_matrix, n, k, new_val);
    }

    /// Evaluate HOBO energy for a generic tensor
    fn evaluate_hobo_energy<D>(tensor: &Array<f64, D>, state: &[bool], n_vars: usize) -> f64
    where
        D: scirs2_core::ndarray::Dimension + 'static,
    {
        // Convert to dynamic dimensionality to allow slice-based indexing
        let dyn_tensor: ArrayD<f64> = tensor.to_owned().into_dyn();
        Self::evaluate_hobo_energy_dyn(&dyn_tensor, state, n_vars)
    }

    /// Evaluate HOBO energy for a dynamic-dimension tensor
    fn evaluate_hobo_energy_dyn(tensor: &ArrayD<f64>, state: &[bool], n_vars: usize) -> f64 {
        let mut energy = 0.0;
        let shape = tensor.shape();
        let ndim = shape.len();

        if ndim == 2 {
            let n0 = shape[0].min(n_vars);
            let n1 = shape[1].min(n_vars);
            for i in 0..n0 {
                if !state[i] {
                    continue;
                }
                for j in 0..n1 {
                    if state[j] {
                        if let Some(&v) = tensor.get([i, j].as_slice()) {
                            energy += v;
                        }
                    }
                }
            }
        } else if ndim == 3 {
            let n0 = shape[0].min(n_vars);
            let n1 = shape[1].min(n_vars);
            let n2 = shape[2].min(n_vars);
            for i in 0..n0 {
                if !state[i] {
                    continue;
                }
                for j in 0..n1 {
                    if !state[j] {
                        continue;
                    }
                    for k in 0..n2 {
                        if state[k] {
                            if let Some(&v) = tensor.get([i, j, k].as_slice()) {
                                energy += v;
                            }
                        }
                    }
                }
            }
        }
        energy
    }

    /// Perform multinomial resampling of the population
    ///
    /// Given a population and weights, draw `n` samples with replacement.
    fn multinomial_resample(
        population: &[Vec<bool>],
        weights: &[f64],
        n: usize,
        rng: &mut StdRng,
    ) -> Vec<Vec<bool>> {
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 || population.is_empty() {
            // Fallback: uniform resampling
            return (0..n)
                .map(|_| {
                    let idx = rng.random_range(0..population.len());
                    population[idx].clone()
                })
                .collect();
        }

        // Build cumulative weight distribution
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut running = 0.0;
        for &w in weights {
            running += w / total_weight;
            cumulative.push(running);
        }
        // Ensure last entry is exactly 1.0
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }

        // Sample n indices
        (0..n)
            .map(|_| {
                let u: f64 = rng.random_range(0.0f64..1.0f64);
                let idx = cumulative
                    .iter()
                    .position(|&c| u <= c)
                    .unwrap_or(cumulative.len().saturating_sub(1));
                population[idx.min(population.len() - 1)].clone()
            })
            .collect()
    }

    /// Run population annealing on a QUBO problem (flat matrix)
    fn run_pa_qubo(
        &self,
        q_flat: &[f64],
        n: usize,
        shots: usize,
        seed: u64,
    ) -> Vec<(Vec<bool>, f64)> {
        let pop_size = self.params.population;
        let beta_schedule = &self.params.beta_schedule;
        let sweeps_per_step = self.params.sweeps_per_step;
        let resample_threshold = self.params.resample_threshold;

        let mut rng = StdRng::seed_from_u64(seed);

        if beta_schedule.is_empty() || pop_size == 0 || n == 0 {
            return vec![];
        }

        // Initialize population: each replica is a random binary state
        let mut population: Vec<Vec<bool>> = (0..pop_size)
            .map(|_| (0..n).map(|_| rng.random_bool(0.5)).collect())
            .collect();

        // Initialize energies for each replica
        let mut energies: Vec<f64> = population
            .iter()
            .map(|s| Self::compute_qubo_energy_flat(q_flat, s, n))
            .collect();

        // Track current beta (initialized at 0 before the first step)
        let mut beta_prev = 0.0;

        for &beta_new in beta_schedule {
            let delta_beta = beta_new - beta_prev;

            // MC sweeps on each replica
            for r in 0..pop_size {
                let state = &mut population[r];
                let mut energy = energies[r];
                let mut g = Self::compute_influence_flat(q_flat, state, n);

                for _ in 0..sweeps_per_step {
                    for i in 0..n {
                        let delta_e = (1.0 - 2.0 * if state[i] { 1.0 } else { 0.0 }) * g[i];
                        // Metropolis acceptance: flip with probability 1 / (1 + exp(beta * delta_e))
                        let accept = if delta_e <= 0.0 {
                            true
                        } else {
                            let threshold = 1.0 / (1.0 + (beta_new * delta_e).exp());
                            rng.random_range(0.0f64..1.0f64) < threshold
                        };

                        if accept {
                            let new_val = !state[i];
                            Self::update_influence_flat(&mut g, q_flat, i, new_val, n);
                            state[i] = new_val;
                            energy += delta_e;
                        }
                    }
                }
                energies[r] = energy;
            }

            // Compute importance weights w_r = exp(-delta_beta * E_r)
            // Use log-sum-exp trick for numerical stability
            let log_weights: Vec<f64> = energies.iter().map(|&e| -delta_beta * e).collect();
            let max_log_w = log_weights
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            let weights: Vec<f64> = log_weights
                .iter()
                .map(|&lw| (lw - max_log_w).exp())
                .collect();

            // Compute ESS = (Σw_r)² / Σw_r²
            let sum_w: f64 = weights.iter().sum();
            let sum_w2: f64 = weights.iter().map(|&w| w * w).sum();

            let ess = if sum_w2 > 0.0 {
                sum_w * sum_w / sum_w2
            } else {
                0.0
            };

            // Resample if ESS/N < threshold
            if (ess / pop_size as f64) < resample_threshold {
                let new_population =
                    Self::multinomial_resample(&population, &weights, pop_size, &mut rng);
                // Recompute energies after resampling
                let new_energies: Vec<f64> = new_population
                    .iter()
                    .map(|s| Self::compute_qubo_energy_flat(q_flat, s, n))
                    .collect();
                population = new_population;
                energies = new_energies;
            }

            beta_prev = beta_new;
        }

        // Return shots samples from final population
        let pop_size_actual = population.len();
        (0..shots)
            .map(|i| {
                let idx = i % pop_size_actual;
                (population[idx].clone(), energies[idx])
            })
            .collect()
    }

    /// Run population annealing on a HOBO tensor
    fn run_pa_hobo<D>(
        &self,
        tensor: &Array<f64, D>,
        n_vars: usize,
        shots: usize,
        seed: u64,
    ) -> Vec<(Vec<bool>, f64)>
    where
        D: scirs2_core::ndarray::Dimension + 'static,
    {
        // Convert once to dynamic dimensionality for repeated indexing
        let dyn_tensor: ArrayD<f64> = tensor.to_owned().into_dyn();
        self.run_pa_hobo_dyn(&dyn_tensor, n_vars, shots, seed)
    }

    /// Run population annealing on a dynamic-dimension HOBO tensor
    fn run_pa_hobo_dyn(
        &self,
        tensor: &ArrayD<f64>,
        n_vars: usize,
        shots: usize,
        seed: u64,
    ) -> Vec<(Vec<bool>, f64)> {
        let pop_size = self.params.population;
        let beta_schedule = &self.params.beta_schedule;
        let sweeps_per_step = self.params.sweeps_per_step;
        let resample_threshold = self.params.resample_threshold;

        let mut rng = StdRng::seed_from_u64(seed);

        if beta_schedule.is_empty() || pop_size == 0 || n_vars == 0 {
            return vec![];
        }

        let mut population: Vec<Vec<bool>> = (0..pop_size)
            .map(|_| (0..n_vars).map(|_| rng.random_bool(0.5)).collect())
            .collect();

        let mut energies: Vec<f64> = population
            .iter()
            .map(|s| Self::evaluate_hobo_energy_dyn(tensor, s, n_vars))
            .collect();

        let mut beta_prev = 0.0;

        for &beta_new in beta_schedule {
            let delta_beta = beta_new - beta_prev;

            for r in 0..pop_size {
                let state = &mut population[r];
                let mut energy = energies[r];

                for _ in 0..sweeps_per_step {
                    for i in 0..n_vars {
                        state[i] = !state[i];
                        let new_energy = Self::evaluate_hobo_energy_dyn(tensor, state, n_vars);
                        let delta_e = new_energy - energy;
                        state[i] = !state[i]; // Restore for now

                        let accept = if delta_e <= 0.0 {
                            true
                        } else {
                            let threshold = 1.0 / (1.0 + (beta_new * delta_e).exp());
                            rng.random_range(0.0f64..1.0f64) < threshold
                        };

                        if accept {
                            state[i] = !state[i];
                            energy = new_energy;
                        }
                    }
                }
                energies[r] = energy;
            }

            let log_weights: Vec<f64> = energies.iter().map(|&e| -delta_beta * e).collect();
            let max_log_w = log_weights
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            let weights: Vec<f64> = log_weights
                .iter()
                .map(|&lw| (lw - max_log_w).exp())
                .collect();

            let sum_w: f64 = weights.iter().sum();
            let sum_w2: f64 = weights.iter().map(|&w| w * w).sum();

            let ess = if sum_w2 > 0.0 {
                sum_w * sum_w / sum_w2
            } else {
                0.0
            };

            if (ess / pop_size as f64) < resample_threshold {
                let new_population =
                    Self::multinomial_resample(&population, &weights, pop_size, &mut rng);
                let new_energies: Vec<f64> = new_population
                    .iter()
                    .map(|s| Self::evaluate_hobo_energy_dyn(tensor, s, n_vars))
                    .collect();
                population = new_population;
                energies = new_energies;
            }

            beta_prev = beta_new;
        }

        let pop_size_actual = population.len();
        (0..shots)
            .map(|i| {
                let idx = i % pop_size_actual;
                (population[idx].clone(), energies[idx])
            })
            .collect()
    }

    /// Run generic sampler
    fn run_generic<D>(
        &self,
        matrix_or_tensor: &Array<f64, D>,
        var_map: &HashMap<String, usize>,
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>>
    where
        D: scirs2_core::ndarray::Dimension + 'static,
    {
        let shots = shots.max(1);
        let n_vars = var_map.len();
        if n_vars == 0 {
            return Err(SamplerError::InvalidParameter(
                "Variable map is empty".to_string(),
            ));
        }

        if self.params.beta_schedule.is_empty() {
            return Err(SamplerError::InvalidParameter(
                "Beta schedule is empty".to_string(),
            ));
        }

        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Determine seed for this run
        let run_seed = match self.seed {
            Some(s) => s,
            None => {
                let mut rng_tmp = thread_rng();
                rng_tmp.random()
            }
        };

        // Run population annealing
        let raw_results = if matrix_or_tensor.ndim() == 2 {
            let q2 = matrix_or_tensor
                .to_owned()
                .into_dimensionality::<Ix2>()
                .map_err(|e| SamplerError::InvalidParameter(format!("Array cast error: {e}")))?;

            let n = q2.dim().0;
            if n != q2.dim().1 {
                return Err(SamplerError::InvalidParameter(
                    "QUBO matrix must be square".to_string(),
                ));
            }

            let q_flat: Vec<f64> = q2
                .as_slice()
                .ok_or_else(|| {
                    SamplerError::InvalidParameter("Non-contiguous QUBO matrix".to_string())
                })?
                .to_vec();

            self.run_pa_qubo(&q_flat, n, shots, run_seed)
        } else {
            self.run_pa_hobo(matrix_or_tensor, n_vars, shots, run_seed)
        };

        if raw_results.is_empty() {
            return Ok(vec![]);
        }

        // Aggregate results: count occurrences of identical states
        let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();
        for (state, energy) in raw_results {
            let entry = solution_counts.entry(state).or_insert((energy, 0));
            entry.1 += 1;
        }

        // Sort by (energy, state) for deterministic ordering when energies are equal
        let mut pairs: Vec<(Vec<bool>, SampleResult)> = solution_counts
            .into_iter()
            .map(|(state, (energy, count))| {
                let assignments: HashMap<String, bool> = state
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &value)| {
                        idx_to_var.get(&idx).map(|name| (name.clone(), value))
                    })
                    .collect();
                let result = SampleResult {
                    assignments,
                    energy,
                    occurrences: count,
                };
                (state, result)
            })
            .collect();

        pairs.sort_by(|(state_a, a), (state_b, b)| {
            a.energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| state_a.cmp(state_b))
        });

        let results: Vec<SampleResult> = pairs.into_iter().map(|(_, r)| r).collect();
        Ok(results)
    }
}

impl Sampler for PopulationAnnealingSampler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        self.run_generic(&qubo.0, &qubo.1, shots)
    }

    fn run_hobo(
        &self,
        hobo: &(ArrayD<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        self.run_generic(&hobo.0, &hobo.1, shots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build K3 Max-Cut QUBO matrix.
    /// Optimal energy: -2.0
    fn build_k3_maxcut_qubo() -> (Array2<f64>, HashMap<String, usize>) {
        let mut q = Array2::<f64>::zeros((3, 3));
        q[[0, 0]] = -2.0;
        q[[1, 1]] = -2.0;
        q[[2, 2]] = -2.0;
        q[[0, 1]] = 2.0;
        q[[0, 2]] = 2.0;
        q[[1, 2]] = 2.0;

        let mut var_map = HashMap::new();
        var_map.insert("x0".to_string(), 0);
        var_map.insert("x1".to_string(), 1);
        var_map.insert("x2".to_string(), 2);

        (q, var_map)
    }

    #[test]
    fn test_pa_3var_maxcut() {
        let (q, var_map) = build_k3_maxcut_qubo();
        let sampler = PopulationAnnealingSampler::new()
            .with_seed(42)
            .with_population(50)
            .with_sweeps_per_step(3);

        let results = sampler
            .run_qubo(&(q, var_map), 50)
            .expect("PA run_qubo failed");

        assert!(!results.is_empty(), "Expected non-empty results");
        let best_energy = results[0].energy;
        assert!(
            best_energy <= -2.0 + 1e-9,
            "Expected optimal energy <= -2.0, got {best_energy}"
        );
    }

    #[test]
    fn test_pa_determinism() {
        let (q, var_map) = build_k3_maxcut_qubo();

        let s1 = PopulationAnnealingSampler::new()
            .with_seed(42)
            .with_population(20);
        let s2 = PopulationAnnealingSampler::new()
            .with_seed(42)
            .with_population(20);

        let r1 = s1
            .run_qubo(&(q.clone(), var_map.clone()), 10)
            .expect("Run 1 failed");
        let r2 = s2.run_qubo(&(q, var_map), 10).expect("Run 2 failed");

        assert_eq!(r1.len(), r2.len(), "Result lengths differ");
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!(
                (a.energy - b.energy).abs() < 1e-12,
                "Energies differ: {} vs {}",
                a.energy,
                b.energy
            );
            assert_eq!(
                a.assignments, b.assignments,
                "Assignments differ for same seed"
            );
        }
    }

    #[test]
    fn test_pa_hobo_smoke() {
        // Simple 2D HOBO smoke test
        let mut q = Array2::<f64>::zeros((2, 2));
        q[[0, 0]] = -1.0;
        q[[1, 1]] = -1.0;

        let mut var_map = HashMap::new();
        var_map.insert("a".to_string(), 0);
        var_map.insert("b".to_string(), 1);

        let sampler = PopulationAnnealingSampler::new()
            .with_seed(7)
            .with_population(20);
        let q_dyn = q.into_dyn();
        let results = sampler
            .run_hobo(&(q_dyn, var_map), 10)
            .expect("HOBO PA run failed");

        assert!(!results.is_empty());
        assert!(results[0].energy <= -2.0 + 1e-9);
    }

    #[test]
    fn test_pa_results_sorted_ascending() {
        let (q, var_map) = build_k3_maxcut_qubo();
        let sampler = PopulationAnnealingSampler::new()
            .with_seed(5)
            .with_population(30);

        let results = sampler.run_qubo(&(q, var_map), 30).expect("PA run failed");

        // Verify ascending energy order
        for window in results.windows(2) {
            assert!(
                window[0].energy <= window[1].energy + 1e-12,
                "Results not sorted: {} > {}",
                window[0].energy,
                window[1].energy
            );
        }
    }

    #[test]
    fn test_pa_custom_schedule() {
        let (q, var_map) = build_k3_maxcut_qubo();
        // Coarser beta schedule
        let betas: Vec<f64> = (0..10).map(|i| 0.2 + 2.8 * i as f64 / 9.0).collect();
        let sampler = PopulationAnnealingSampler::new()
            .with_seed(42)
            .with_population(30)
            .with_beta_schedule(betas);

        let results = sampler
            .run_qubo(&(q, var_map), 20)
            .expect("PA custom schedule failed");

        assert!(!results.is_empty());
        // Should still find a reasonable solution
        assert!(results[0].energy <= 0.0 + 1e-9);
    }
}
