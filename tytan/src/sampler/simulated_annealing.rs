//! # Simulated Annealing (SA) Sampler
//!
//! Simulated Annealing (SA) is a probabilistic optimization algorithm inspired by
//! the annealing process in metallurgy, where controlled cooling reduces crystal
//! defects. In combinatorial optimization, SA accepts worse solutions with a
//! decreasing probability as a "temperature" parameter T decreases, allowing
//! the search to escape local optima early in the run.
//!
//! ## Algorithm
//!
//! At each step:
//!
//! 1. Select a random bit flip (neighbour).
//! 2. Compute ΔE = energy(flipped) − energy(current).
//! 3. Accept the flip if ΔE < 0 (improvement), or with probability
//!    `exp(-ΔE / T)` otherwise (Metropolis criterion).
//! 4. Decrease T according to the cooling schedule.
//!
//! Geometric cooling: `T_k = T₀ · α^k`, where α ∈ (0, 1) is the cooling rate.
//!
//! ## Mathematical Formulation
//!
//! Given a QUBO matrix Q, the objective is to minimise:
//!
//! ```text
//! E(x) = Σ_{i,j} Q[i,j] · x[i] · x[j],   x[i] ∈ {0, 1}
//! ```
//!
//! The acceptance probability at temperature T for a move with energy change ΔE:
//!
//! ```text
//! P(accept) = min(1, exp(-ΔE / T))
//! ```
//!
//! ## Citation
//!
//! Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983).
//! Optimization by Simulated Annealing.
//! *Science*, 220(4598), 671–680.
//! <https://doi.org/10.1126/science.220.4598.671>
//!
//! ## Parameters
//!
//! See [`quantrs2_anneal::simulator::AnnealingParams`] for all tunable parameters
//! (initial temperature, cooling schedule, number of sweeps).
//!
//! ## When to Use
//!
//! - **Best for**: general-purpose QUBO/HOBO problems of any size.
//! - **Strengths**: robust, easy to tune, works out of the box.
//! - **Limitations**: slow on very large dense matrices compared to [`crate::sampler::SBSampler`].
//!
//! ## Usage
//!
//! ```
//! use quantrs2_tytan::sampler::{SASampler, Sampler};
//! use scirs2_core::ndarray::Array;
//! use std::collections::HashMap;
//!
//! // Minimise: -x0 - x1 + 2*x0*x1  (mutual exclusion problem)
//! let mut q = Array::<f64, _>::zeros((2, 2));
//! q[[0, 0]] = -1.0;
//! q[[1, 1]] = -1.0;
//! q[[0, 1]] = 2.0;
//!
//! let mut var_map = HashMap::new();
//! var_map.insert("x0".to_string(), 0);
//! var_map.insert("x1".to_string(), 1);
//!
//! let sampler = SASampler::new(Some(42));
//! let results = sampler.run_qubo(&(q, var_map), 5).expect("SA sampler failed");
//! assert!(!results.is_empty());
//! println!("Best energy: {}", results[0].energy);
//! ```

use scirs2_core::ndarray::{Array, Ix2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use std::collections::HashMap;

use quantrs2_anneal::{
    simulator::{AnnealingParams, ClassicalAnnealingSimulator, TemperatureSchedule},
    QuboModel,
};

use super::{SampleResult, Sampler, SamplerResult};

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

/// Simulated Annealing Sampler
///
/// Probabilistic metaheuristic for QUBO/HOBO optimisation that uses a
/// temperature schedule to control the probability of accepting worse
/// solutions, allowing escape from local optima.
///
/// # Example
///
/// ```
/// use quantrs2_tytan::sampler::{SASampler, Sampler};
/// use scirs2_core::ndarray::Array;
/// use std::collections::HashMap;
///
/// // Minimise: -x0 - x1 + 2*x0*x1  (mutual exclusion)
/// let mut q = Array::<f64, _>::zeros((2, 2));
/// q[[0, 0]] = -1.0;
/// q[[1, 1]] = -1.0;
/// q[[0, 1]] = 2.0;
///
/// let mut var_map = HashMap::new();
/// var_map.insert("x0".to_string(), 0);
/// var_map.insert("x1".to_string(), 1);
///
/// let sampler = SASampler::new(Some(42));
/// let results = sampler.run_qubo(&(q, var_map), 5).expect("SA sampler failed");
/// assert!(!results.is_empty());
/// println!("Best energy: {}", results[0].energy);
/// ```
#[derive(Clone)]
pub struct SASampler {
    /// Random number generator seed
    seed: Option<u64>,
    /// Annealing parameters
    params: AnnealingParams,
}

impl SASampler {
    /// Create a new Simulated Annealing sampler
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    #[must_use]
    pub fn new(seed: Option<u64>) -> Self {
        // Create default annealing parameters
        let mut params = AnnealingParams::default();

        // Customize based on seed
        if let Some(seed) = seed {
            params.seed = Some(seed);
        }

        Self { seed, params }
    }

    /// Create a new Simulated Annealing sampler with custom parameters
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    /// * `params` - Custom annealing parameters
    #[must_use]
    pub const fn with_params(seed: Option<u64>, params: AnnealingParams) -> Self {
        let mut params = params;

        // Override seed if provided
        if let Some(seed) = seed {
            params.seed = Some(seed);
        }

        Self { seed, params }
    }

    /// Set beta range for simulated annealing
    pub fn with_beta_range(mut self, beta_min: f64, beta_max: f64) -> Self {
        // Convert beta (inverse temperature) to temperature
        self.params.initial_temperature = 1.0 / beta_max;
        // Use exponential temperature schedule to approximate final beta
        self.params.temperature_schedule = TemperatureSchedule::Exponential(beta_min / beta_max);
        self
    }

    /// Set number of sweeps
    pub const fn with_sweeps(mut self, sweeps: usize) -> Self {
        self.params.num_sweeps = sweeps;
        self
    }

    /// Run the sampler on a QUBO/HOBO problem
    ///
    /// This is a generic implementation that works for both QUBO and HOBO
    /// by converting the input to a format compatible with the underlying
    /// annealing simulator.
    ///
    /// # Arguments
    ///
    /// * `matrix_or_tensor` - The problem matrix/tensor
    /// * `var_map` - The variable mapping
    /// * `shots` - The number of samples to take
    ///
    /// # Returns
    ///
    /// A vector of sample results, sorted by energy (best solutions first)
    fn run_generic<D>(
        &self,
        matrix_or_tensor: &Array<f64, D>,
        var_map: &HashMap<String, usize>,
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>>
    where
        D: scirs2_core::ndarray::Dimension + 'static,
    {
        // Make sure shots is reasonable
        let shots = std::cmp::max(shots, 1);

        // Get the problem dimension (number of variables)
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // For QUBO problems, convert to quantrs-anneal format.
        //
        // Note: SASampler cannot use the SIMD energy functions from `super::energy` because
        // it delegates all inner-loop computation to `quantrs2_anneal::ClassicalAnnealingSimulator`,
        // which manages its own internal state representation and energy bookkeeping.
        // The SIMD functions (energy_full_simd, energy_delta_simd, etc.) are designed for
        // samplers that own a flat Vec<f64> q_matrix and iterate over bool states directly
        // (see TabuSampler, PASampler, SBSampler). Integrating them here would require
        // rewriting SA from scratch and abandoning the well-tested anneal crate backend.
        if matrix_or_tensor.ndim() == 2 {
            // Convert ndarray to a QuboModel
            let mut qubo = QuboModel::new(n_vars);

            // Set linear and quadratic terms
            for i in 0..n_vars {
                let diag_val = match matrix_or_tensor.ndim() {
                    2 => {
                        // For 2D matrices (QUBO)
                        let matrix = matrix_or_tensor
                            .to_owned()
                            .into_dimensionality::<Ix2>()
                            .ok();
                        matrix.map_or(0.0, |m| m[[i, i]])
                    }
                    _ => 0.0, // For higher dimensions, assume 0 for diagonal elements
                };

                if diag_val != 0.0 {
                    qubo.set_linear(i, diag_val)?;
                }

                for j in (i + 1)..n_vars {
                    let quad_val = match matrix_or_tensor.ndim() {
                        2 => {
                            // For 2D matrices (QUBO)
                            let matrix = matrix_or_tensor
                                .to_owned()
                                .into_dimensionality::<Ix2>()
                                .ok();
                            matrix.map_or(0.0, |m| m[[i, j]])
                        }
                        _ => 0.0, // Higher dimensions would need separate handling
                    };

                    if quad_val != 0.0 {
                        qubo.set_quadratic(i, j, quad_val)?;
                    }
                }
            }

            // Configure annealing parameters
            // Note: We respect the user's configured num_repetitions instead of
            // overriding it with shots. The shots parameter in QUBO solving
            // represents the number of independent samples desired, but for now
            // we return the best solution found in the configured repetitions.
            let params = self.params.clone();

            // Create annealing simulator
            let simulator = ClassicalAnnealingSimulator::new(params)?;

            // Convert QUBO to Ising model
            let (ising_model, _) = qubo.to_ising();

            // Solve the problem
            let annealing_result = simulator.solve(&ising_model)?;

            // Convert to our result format
            let mut results = Vec::new();

            // Convert spins to binary variables
            let binary_vars: Vec<bool> = annealing_result
                .best_spins
                .iter()
                .map(|&spin| spin > 0)
                .collect();

            // Convert binary array to HashMap
            let assignments: HashMap<String, bool> = binary_vars
                .iter()
                .enumerate()
                .filter_map(|(idx, &value)| {
                    idx_to_var
                        .get(&idx)
                        .map(|var_name| (var_name.clone(), value))
                })
                .collect();

            // Create a result
            let result = SampleResult {
                assignments,
                energy: annealing_result.best_energy,
                occurrences: 1,
            };

            results.push(result);

            return Ok(results);
        }

        // For higher-order tensors (HOBO problems)
        self.run_hobo_tensor(matrix_or_tensor, var_map, shots)
    }

    /// Run simulated annealing on a HOBO problem represented as a tensor
    fn run_hobo_tensor<D>(
        &self,
        tensor: &Array<f64, D>,
        var_map: &HashMap<String, usize>,
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>>
    where
        D: scirs2_core::ndarray::Dimension + 'static,
    {
        // Get the problem dimension (number of variables)
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Create RNG with seed if provided
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let seed: u64 = thread_rng().random();
            StdRng::seed_from_u64(seed)
        };

        // Store solutions and their frequencies
        let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

        // Maximum parallel runs
        #[cfg(feature = "parallel")]
        let num_threads = scirs2_core::parallel_ops::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let num_threads = 1;

        // Divide shots across threads
        let shots_per_thread = shots / num_threads + usize::from(shots % num_threads > 0);
        let total_runs = shots_per_thread * num_threads;

        // Set up annealing parameters
        let initial_temp = 10.0;
        let final_temp = 0.1;
        let sweeps = 1000;

        // Function to evaluate HOBO energy
        let evaluate_energy = |state: &[bool]| -> f64 {
            let mut energy = 0.0;

            // We'll match based on tensor dimension to handle differently
            // Handle the tensor processing based on its dimensions
            if tensor.ndim() == 3 {
                let tensor3d = tensor
                    .to_owned()
                    .into_dimensionality::<scirs2_core::ndarray::Ix3>()
                    .ok();
                if let Some(t) = tensor3d {
                    // Calculate energy for 3D tensor
                    for i in 0..std::cmp::min(n_vars, t.dim().0) {
                        if !state[i] {
                            continue;
                        }
                        for j in 0..std::cmp::min(n_vars, t.dim().1) {
                            if !state[j] {
                                continue;
                            }
                            for k in 0..std::cmp::min(n_vars, t.dim().2) {
                                if state[k] {
                                    energy += t[[i, j, k]];
                                }
                            }
                        }
                    }
                }
            } else {
                // For other dimensions, we'll do a brute force approach
                let shape = tensor.shape();
                if shape.len() == 2 {
                    // Handle 2D specifically
                    if let Ok(tensor2d) = tensor
                        .to_owned()
                        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    {
                        for i in 0..std::cmp::min(n_vars, tensor2d.dim().0) {
                            if !state[i] {
                                continue;
                            }
                            for j in 0..std::cmp::min(n_vars, tensor2d.dim().1) {
                                if state[j] {
                                    energy += tensor2d[[i, j]];
                                }
                            }
                        }
                    }
                } else {
                    // Fallback for other dimensions - just return the energy as is
                    // This should be specialized for other tensor dimensions if needed
                    if !tensor.is_empty() {
                        println!(
                            "Warning: Processing tensor with shape {shape:?} not specifically optimized"
                        );
                    }
                }
            }

            energy
        };

        // Vector to store thread-local solutions
        #[allow(unused_assignments)]
        let mut all_solutions = Vec::with_capacity(total_runs);

        // Run annealing process
        #[cfg(feature = "parallel")]
        {
            // Create seeds for each parallel run
            let seeds: Vec<u64> = (0..total_runs)
                .map(|i| match self.seed {
                    Some(seed) => seed.wrapping_add(i as u64),
                    None => thread_rng().random(),
                })
                .collect();

            // Run in parallel
            all_solutions = seeds
                .into_par_iter()
                .map(|seed| {
                    let mut thread_rng = StdRng::seed_from_u64(seed);

                    // Initialize random state
                    let mut state = vec![false; n_vars];
                    for bit in &mut state {
                        *bit = thread_rng.random_bool(0.5);
                    }

                    // Evaluate initial energy
                    let mut energy = evaluate_energy(&state);
                    let mut best_state = state.clone();
                    let mut best_energy = energy;

                    // Simulated annealing
                    for sweep in 0..sweeps {
                        // Calculate temperature for this step
                        let temp = initial_temp
                            * f64::powf(final_temp / initial_temp, sweep as f64 / sweeps as f64);

                        // Perform n_vars updates per sweep
                        for _ in 0..n_vars {
                            // Select random bit to flip
                            let idx = thread_rng.random_range(0..n_vars);

                            // Flip the bit
                            state[idx] = !state[idx];

                            // Calculate new energy
                            let new_energy = evaluate_energy(&state);
                            let delta_e = new_energy - energy;

                            // Metropolis acceptance criterion
                            let accept = delta_e <= 0.0
                                || thread_rng.random_range(0.0..1.0) < (-delta_e / temp).exp();

                            if accept {
                                energy = new_energy;
                                if energy < best_energy {
                                    best_energy = energy;
                                    best_state = state.clone();
                                }
                            } else {
                                // Revert flip
                                state[idx] = !state[idx];
                            }
                        }
                    }

                    (best_state, best_energy)
                })
                .collect();
        }

        #[cfg(not(feature = "parallel"))]
        {
            for _ in 0..total_runs {
                // Initialize random state
                let mut state = vec![false; n_vars];
                for bit in &mut state {
                    *bit = rng.random_bool(0.5);
                }

                // Evaluate initial energy
                let mut energy = evaluate_energy(&state);
                let mut best_state = state.clone();
                let mut best_energy = energy;

                // Simulated annealing
                for sweep in 0..sweeps {
                    // Calculate temperature for this step
                    let temp = initial_temp
                        * f64::powf(final_temp / initial_temp, sweep as f64 / sweeps as f64);

                    // Perform n_vars updates per sweep
                    for _ in 0..n_vars {
                        // Select random bit to flip
                        let mut idx = rng.random_range(0..n_vars);

                        // Flip the bit
                        state[idx] = !state[idx];

                        // Calculate new energy
                        let new_energy = evaluate_energy(&state);
                        let delta_e = new_energy - energy;

                        // Metropolis acceptance criterion
                        let accept = delta_e <= 0.0
                            || rng.random_range(0.0..1.0) < f64::exp(-delta_e / temp);

                        if accept {
                            energy = new_energy;
                            if energy < best_energy {
                                best_energy = energy;
                                best_state = state.clone();
                            }
                        } else {
                            // Revert flip
                            state[idx] = !state[idx];
                        }
                    }
                }

                all_solutions.push((best_state, best_energy));
            }
        }

        // Process results from all threads
        for (state, energy) in all_solutions {
            let entry = solution_counts.entry(state).or_insert((energy, 0));
            entry.1 += 1;
        }

        // Convert to SampleResult format
        let mut results: Vec<SampleResult> = solution_counts
            .into_iter()
            .map(|(state, (energy, count))| {
                // Convert to variable assignments
                let assignments: HashMap<String, bool> = state
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &value)| {
                        idx_to_var
                            .get(&idx)
                            .map(|var_name| (var_name.clone(), value))
                    })
                    .collect();

                SampleResult {
                    assignments,
                    energy,
                    occurrences: count,
                }
            })
            .collect();

        // Sort by energy (best solutions first)
        results.sort_by(|a, b| {
            a.energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to requested number of shots if we have more
        if results.len() > shots {
            results.truncate(shots);
        }

        Ok(results)
    }
}

impl Sampler for SASampler {
    fn run_qubo(
        &self,
        qubo: &(
            Array<f64, scirs2_core::ndarray::Ix2>,
            HashMap<String, usize>,
        ),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        self.run_generic(&qubo.0, &qubo.1, shots)
    }

    fn run_hobo(
        &self,
        hobo: &(
            Array<f64, scirs2_core::ndarray::IxDyn>,
            HashMap<String, usize>,
        ),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        self.run_generic(&hobo.0, &hobo.1, shots)
    }
}
