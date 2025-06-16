//! Simulated Annealing Sampler Implementation

use ndarray::{Array, Ix2};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashMap;

use quantrs2_anneal::{
    simulator::{AnnealingParams, ClassicalAnnealingSimulator, TemperatureSchedule},
    QuboModel,
};

use super::{SampleResult, Sampler, SamplerResult};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Simulated Annealing Sampler
///
/// This sampler uses simulated annealing to find solutions to
/// QUBO/HOBO problems. It is a local search method that uses
/// temperature to control the acceptance of worse solutions.
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
        let params = AnnealingParams::default();

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
    pub fn with_params(seed: Option<u64>, params: AnnealingParams) -> Self {
        let params = params;

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
    pub fn with_sweeps(mut self, sweeps: usize) -> Self {
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
        D: ndarray::Dimension + 'static,
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

        // For QUBO problems, convert to quantrs-anneal format
        if matrix_or_tensor.ndim() == 2 {
            // Convert ndarray to a QuboModel
            let qubo = QuboModel::new(n_vars);

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
            let params = self.params.clone();
            params.num_repetitions = shots;

            // Create annealing simulator
            let simulator = ClassicalAnnealingSimulator::new(params)?;

            // Convert QUBO to Ising model
            let (ising_model, _) = qubo.to_ising();

            // Solve the problem
            let annealing_result = simulator.solve(&ising_model)?;

            // Convert to our result format
            let results = Vec::new();

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
                .map(|(idx, &value)| {
                    let var_name = idx_to_var.get(&idx).unwrap().clone();
                    (var_name, value)
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
        D: ndarray::Dimension + 'static,
    {
        // Get the problem dimension (number of variables)
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Create RNG with seed if provided
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let seed: u64 = rand::rng().random();
                StdRng::seed_from_u64(seed)
            }
        };

        // Store solutions and their frequencies
        let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

        // Maximum parallel runs
        #[cfg(feature = "parallel")]
        let num_threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let num_threads = 1;

        // Divide shots across threads
        let shots_per_thread = shots / num_threads + if shots % num_threads > 0 { 1 } else { 0 };
        let total_runs = shots_per_thread * num_threads;

        // Set up annealing parameters
        let initial_temp = 10.0;
        let final_temp = 0.1;
        let sweeps = 1000;

        // Function to evaluate HOBO energy
        let evaluate_energy = |state: &[bool]| -> f64 {
            let energy = 0.0;

            // We'll match based on tensor dimension to handle differently
            // Handle the tensor processing based on its dimensions
            if tensor.ndim() == 3 {
                let tensor3d = tensor.to_owned().into_dimensionality::<ndarray::Ix3>().ok();
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
                    let tensor2d = tensor
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
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
                } else {
                    // Fallback for other dimensions - just return the energy as is
                    // This should be specialized for other tensor dimensions if needed
                    if !tensor.is_empty() {
                        println!(
                            "Warning: Processing tensor with shape {:?} not specifically optimized",
                            shape
                        );
                    }
                }
            }

            energy
        };

        // Vector to store thread-local solutions
        #[allow(unused_assignments)]
        let all_solutions = Vec::with_capacity(total_runs);

        // Run annealing process
        #[cfg(feature = "parallel")]
        {
            // Create seeds for each parallel run
            let seeds: Vec<u64> = (0..total_runs)
                .map(|i| match self.seed {
                    Some(seed) => seed.wrapping_add(i as u64),
                    None => rand::rng().random(),
                })
                .collect();

            // Run in parallel
            all_solutions = seeds
                .into_par_iter()
                .map(|seed| {
                    let thread_rng = StdRng::seed_from_u64(seed);

                    // Initialize random state
                    let mut state = vec![false; n_vars];
                    for bit in &mut state {
                        *bit = thread_rng.random_bool(0.5);
                    }

                    // Evaluate initial energy
                    let energy = evaluate_energy(&state);
                    let best_state = state.clone();
                    let best_energy = energy;

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
                let energy = evaluate_energy(&state);
                let best_state = state.clone();
                let best_energy = energy;

                // Simulated annealing
                for sweep in 0..sweeps {
                    // Calculate temperature for this step
                    let temp = initial_temp
                        * f64::powf(final_temp / initial_temp, sweep as f64 / sweeps as f64);

                    // Perform n_vars updates per sweep
                    for _ in 0..n_vars {
                        // Select random bit to flip
                        let idx = rng.random_range(0..n_vars);

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
                    .map(|(idx, &value)| {
                        let var_name = idx_to_var.get(&idx).unwrap().clone();
                        (var_name, value)
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
        results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

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
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        self.run_generic(&qubo.0, &qubo.1, shots)
    }

    fn run_hobo(
        &self,
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        self.run_generic(&hobo.0, &hobo.1, shots)
    }
}
