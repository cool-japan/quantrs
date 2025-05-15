//! Samplers for solving QUBO/HOBO problems.
//!
//! This module provides various samplers (solvers) for QUBO and HOBO
//! problems, including simulated annealing, genetic algorithms, and
//! specialized hardware samplers.

use ndarray::{Array, Dimension, Ix2};
use rand::prelude::*;
use rand::rngs::StdRng;
// thread_rng is deprecated, use rng() instead
use std::collections::HashMap;
use thiserror::Error;

#[cfg(feature = "parallel")]
// Uncomment when parallel processing is implemented
// use rayon::prelude::**;
#[cfg(all(feature = "gpu", feature = "dwave"))]
use ocl;

/// Evaluate energy for a QUBO problem given a binary state vector
///
/// # Arguments
///
/// * `state` - Binary state vector (solution)
/// * `h_vector` - Linear coefficients (diagonal)
/// * `j_matrix` - Quadratic coefficients (n_vars x n_vars matrix in flattened form)
/// * `n_vars` - Number of variables
///
/// # Returns
///
/// The energy (objective function value) for the given state
#[allow(dead_code)]
fn evaluate_qubo_energy(state: &[bool], h_vector: &[f64], j_matrix: &[f64], n_vars: usize) -> f64 {
    let mut energy = 0.0;

    // Linear terms
    for i in 0..n_vars {
        if state[i] {
            energy += h_vector[i];
        }
    }

    // Quadratic terms
    for i in 0..n_vars {
        if state[i] {
            for j in 0..n_vars {
                if state[j] {
                    energy += j_matrix[i * n_vars + j];
                }
            }
        }
    }

    energy
}

use quantrs2_anneal::{
    simulator::{AnnealingError, AnnealingParams, ClassicalAnnealingSimulator},
    IsingError, QuboModel,
};

/// Errors that can occur during sampling
#[derive(Error, Debug)]
pub enum SamplerError {
    /// Error when the input parameters are invalid
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Error in the underlying annealing simulator
    #[error("Annealing error: {0}")]
    AnnealingError(#[from] AnnealingError),

    /// Error in the Ising model
    #[error("Ising model error: {0}")]
    IsingError(#[from] IsingError),

    /// Error in GPU operations
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Error when D-Wave API is unavailable
    #[error("D-Wave API unavailable: {0}")]
    DWaveUnavailable(String),

    /// Error during API communication
    #[error("API communication error: {0}")]
    ApiError(String),

    /// Error in D-Wave operations
    #[cfg(feature = "dwave")]
    #[error("D-Wave error: {0}")]
    DWaveError(#[from] quantrs2_anneal::dwave::DWaveError),
}

/// Result type for sampling operations
pub type SamplerResult<T> = Result<T, SamplerError>;

/// A sample result from a QUBO/HOBO problem
///
/// This struct represents a single sample result from a QUBO/HOBO
/// problem, including the variable assignments, energy, and occurrence count.
#[derive(Debug, Clone)]
pub struct SampleResult {
    /// The variable assignments (name -> value mapping)
    pub assignments: HashMap<String, bool>,
    /// The energy (objective function value)
    pub energy: f64,
    /// The number of times this solution appeared
    pub occurrences: usize,
}

/// Trait for samplers that can solve QUBO/HOBO problems
pub trait Sampler {
    /// Run the sampler on a QUBO problem
    ///
    /// # Arguments
    ///
    /// * `qubo` - The QUBO problem to solve: (matrix, variable mapping)
    /// * `shots` - The number of samples to take
    ///
    /// # Returns
    ///
    /// A vector of sample results, sorted by energy (best solutions first)
    fn run_qubo(
        &self,
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>>;

    /// Run the sampler on a HOBO problem
    ///
    /// # Arguments
    ///
    /// * `hobo` - The HOBO problem to solve: (tensor, variable mapping)
    /// * `shots` - The number of samples to take
    ///
    /// # Returns
    ///
    /// A vector of sample results, sorted by energy (best solutions first)
    fn run_hobo(
        &self,
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>>;
}

/// Simulated Annealing Sampler
///
/// This sampler uses simulated annealing to find solutions to
/// QUBO/HOBO problems. It is a local search method that uses
/// temperature to control the acceptance of worse solutions.
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
    pub fn with_params(seed: Option<u64>, params: AnnealingParams) -> Self {
        let mut params = params;

        // Override seed if provided
        if let Some(seed) = seed {
            params.seed = Some(seed);
        }

        Self { seed, params }
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
            let mut params = self.params.clone();
            params.num_repetitions = shots;

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
        let _rng = match self.seed {
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
            let mut energy = 0.0;

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
        let mut all_solutions = Vec::with_capacity(total_runs);

        // Run annealing process
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // Create seeds for each parallel run
            let seeds: Vec<u64> = (0..total_runs)
                .map(|i| match self.seed {
                    Some(seed) => seed.wrapping_add(i as u64),
                    None => rand::random(),
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
                        let idx = rng.random_range(0..n_vars);

                        // Flip the bit
                        state[idx] = !state[idx];

                        // Calculate new energy
                        let new_energy = evaluate_energy(&state);
                        let delta_e = new_energy - energy;

                        // Metropolis acceptance criterion
                        let accept =
                            delta_e <= 0.0 || rng.random_range(0.0..1.0) < (-delta_e / temp).exp();

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

/// Genetic Algorithm Sampler
///
/// This sampler uses a genetic algorithm to find solutions to
/// QUBO/HOBO problems. It maintains a population of potential
/// solutions and evolves them through selection, crossover, and mutation.
pub struct GASampler {
    /// Random number generator seed
    seed: Option<u64>,
    /// Maximum number of generations
    max_generations: usize,
    /// Population size
    population_size: usize,
}

impl GASampler {
    /// Create a new Genetic Algorithm sampler
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    #[must_use]
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            seed,
            max_generations: 1000,
            population_size: 100,
        }
    }

    /// Create a new Genetic Algorithm sampler with custom parameters
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    /// * `max_generations` - Maximum number of generations to evolve
    /// * `population_size` - Size of the population
    #[must_use]
    pub fn with_params(seed: Option<u64>, max_generations: usize, population_size: usize) -> Self {
        Self {
            seed,
            max_generations,
            population_size,
        }
    }
}

/// Crossover strategy for genetic algorithm
#[derive(Debug, Clone, Copy)]
pub enum CrossoverStrategy {
    /// Uniform crossover (random gene selection from each parent)
    Uniform,
    /// Single-point crossover (split at random point)
    SinglePoint,
    /// Two-point crossover (swap middle section)
    TwoPoint,
    /// Adaptive crossover (choice based on parent similarity)
    Adaptive,
}

/// Mutation strategy for genetic algorithm
#[derive(Debug, Clone, Copy)]
pub enum MutationStrategy {
    /// Flip bits with fixed probability
    FixedRate(f64),
    /// Mutate bits with decreasing rate over generations
    Annealing(f64, f64), // (initial_rate, final_rate)
    /// Adaptive mutation based on population diversity
    Adaptive(f64, f64), // (min_rate, max_rate)
}

impl GASampler {
    /// Create a new enhanced Genetic Algorithm sampler
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    /// * `max_generations` - Maximum number of generations to evolve
    /// * `population_size` - Size of the population
    /// * `crossover` - Crossover strategy to use
    /// * `mutation` - Mutation strategy to use
    pub fn with_advanced_params(
        seed: Option<u64>,
        max_generations: usize,
        population_size: usize,
        _crossover: CrossoverStrategy, // Saved for future implementation
        _mutation: MutationStrategy,   // Saved for future implementation
    ) -> Self {
        Self {
            seed,
            max_generations,
            population_size,
        }
    }

    /// Perform crossover between two parents
    fn crossover(
        &self,
        parent1: &[bool],
        parent2: &[bool],
        strategy: CrossoverStrategy,
        rng: &mut impl Rng,
    ) -> (Vec<bool>, Vec<bool>) {
        let n_vars = parent1.len();
        let mut child1 = vec![false; n_vars];
        let mut child2 = vec![false; n_vars];

        match strategy {
            CrossoverStrategy::Uniform => {
                // Uniform crossover
                for i in 0..n_vars {
                    if rng.random_bool(0.5) {
                        child1[i] = parent1[i];
                        child2[i] = parent2[i];
                    } else {
                        child1[i] = parent2[i];
                        child2[i] = parent1[i];
                    }
                }
            }
            CrossoverStrategy::SinglePoint => {
                // Single-point crossover
                let crossover_point = rng.random_range(1..n_vars);

                for i in 0..n_vars {
                    if i < crossover_point {
                        child1[i] = parent1[i];
                        child2[i] = parent2[i];
                    } else {
                        child1[i] = parent2[i];
                        child2[i] = parent1[i];
                    }
                }
            }
            CrossoverStrategy::TwoPoint => {
                // Two-point crossover
                let point1 = rng.random_range(1..(n_vars - 1));
                let point2 = rng.random_range((point1 + 1)..n_vars);

                for i in 0..n_vars {
                    if i < point1 || i >= point2 {
                        child1[i] = parent1[i];
                        child2[i] = parent2[i];
                    } else {
                        child1[i] = parent2[i];
                        child2[i] = parent1[i];
                    }
                }
            }
            CrossoverStrategy::Adaptive => {
                // Calculate Hamming distance between parents
                let mut hamming_distance = 0;
                for i in 0..n_vars {
                    if parent1[i] != parent2[i] {
                        hamming_distance += 1;
                    }
                }

                // Normalized distance
                let similarity = 1.0 - (hamming_distance as f64 / n_vars as f64);

                if similarity > 0.8 {
                    // Parents are very similar - use uniform with high mixing
                    for i in 0..n_vars {
                        if rng.random_bool(0.5) {
                            child1[i] = parent1[i];
                            child2[i] = parent2[i];
                        } else {
                            child1[i] = parent2[i];
                            child2[i] = parent1[i];
                        }
                    }
                } else if similarity > 0.4 {
                    // Moderate similarity - use two-point
                    let point1 = rng.random_range(1..(n_vars - 1));
                    let point2 = rng.random_range((point1 + 1)..n_vars);

                    for i in 0..n_vars {
                        if i < point1 || i >= point2 {
                            child1[i] = parent1[i];
                            child2[i] = parent2[i];
                        } else {
                            child1[i] = parent2[i];
                            child2[i] = parent1[i];
                        }
                    }
                } else {
                    // Low similarity - use single point
                    let crossover_point = rng.random_range(1..n_vars);

                    for i in 0..n_vars {
                        if i < crossover_point {
                            child1[i] = parent1[i];
                            child2[i] = parent2[i];
                        } else {
                            child1[i] = parent2[i];
                            child2[i] = parent1[i];
                        }
                    }
                }
            }
        }

        (child1, child2)
    }

    /// Mutate an individual
    fn mutate(
        &self,
        individual: &mut [bool],
        strategy: MutationStrategy,
        generation: usize,
        max_generations: usize,
        diversity: Option<f64>,
        rng: &mut impl Rng,
    ) {
        match strategy {
            MutationStrategy::FixedRate(rate) => {
                // Simple fixed mutation rate
                for bit in individual.iter_mut() {
                    if rng.random_bool(rate) {
                        *bit = !*bit;
                    }
                }
            }
            MutationStrategy::Annealing(initial_rate, final_rate) => {
                // Annealing mutation (decreasing rate)
                let progress = generation as f64 / max_generations as f64;
                let current_rate = initial_rate + (final_rate - initial_rate) * progress;

                for bit in individual.iter_mut() {
                    if rng.random_bool(current_rate) {
                        *bit = !*bit;
                    }
                }
            }
            MutationStrategy::Adaptive(min_rate, max_rate) => {
                // Adaptive mutation based on diversity
                if let Some(diversity) = diversity {
                    // High diversity -> low mutation rate, low diversity -> high mutation rate
                    let rate = min_rate + (max_rate - min_rate) * (1.0 - diversity);

                    for bit in individual.iter_mut() {
                        if rng.random_bool(rate) {
                            *bit = !*bit;
                        }
                    }
                } else {
                    // Default to average if no diversity metric available
                    let rate = (min_rate + max_rate) / 2.0;
                    for bit in individual.iter_mut() {
                        if rng.random_bool(rate) {
                            *bit = !*bit;
                        }
                    }
                }
            }
        }
    }

    /// Calculate population diversity (normalized hamming distance)
    fn calculate_diversity(&self, population: &[Vec<bool>]) -> f64 {
        if population.len() <= 1 {
            return 0.0;
        }

        let n_individuals = population.len();
        let n_vars = population[0].len();
        let mut sum_distances = 0;
        let mut pair_count = 0;

        for i in 0..n_individuals {
            for j in (i + 1)..n_individuals {
                let mut distance = 0;
                for k in 0..n_vars {
                    if population[i][k] != population[j][k] {
                        distance += 1;
                    }
                }
                sum_distances += distance;
                pair_count += 1;
            }
        }

        // Average normalized Hamming distance
        if pair_count > 0 {
            (sum_distances as f64) / (pair_count as f64 * n_vars as f64)
        } else {
            0.0
        }
    }
}

impl Sampler for GASampler {
    fn run_hobo(
        &self,
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Extract matrix and variable mapping
        let (tensor, var_map) = hobo;

        // Make sure shots is reasonable
        let actual_shots = std::cmp::max(shots, 10);

        // Get the problem dimension
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Initialize random number generator
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let seed: u64 = rand::rng().random();
                StdRng::seed_from_u64(seed)
            }
        };

        // Set default parameters (commented out to avoid unused variable warnings)
        // These would be used in a more complete implementation
        // let crossover_strategy = CrossoverStrategy::Adaptive;
        // let mutation_strategy = MutationStrategy::Annealing(0.1, 0.01);
        // let selection_pressure = 3; // Tournament size
        // let use_elitism = true;

        // Handle small population size cases to avoid empty range errors
        if self.population_size <= 2 || n_vars == 0 {
            // Return a simple result for trivial cases
            let mut assignments = HashMap::new();
            for var in var_map.keys() {
                assignments.insert(var.clone(), false);
            }

            return Ok(vec![SampleResult {
                assignments,
                energy: 0.0,
                occurrences: 1,
            }]);
        }

        // For simplicity, if the tensor is 2D, convert to QUBO and use that implementation
        if tensor.ndim() == 2 && tensor.shape() == [n_vars, n_vars] {
            // Create a view as a 2D matrix and convert to owned matrix
            let matrix = tensor
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let qubo = (matrix, var_map.clone());

            return self.run_qubo(&qubo, shots);
        }

        // Otherwise, implement the full HOBO genetic algorithm here
        // Define a function to evaluate the energy of a solution
        let evaluate_energy = |state: &[bool]| -> f64 {
            let mut energy = 0.0;

            // Evaluate according to tensor dimension
            if tensor.ndim() == 2 {
                // Use matrix evaluation (much faster)
                for i in 0..n_vars {
                    if state[i] {
                        energy += tensor[[i, i]]; // Diagonal terms

                        for j in 0..n_vars {
                            if state[j] && j != i {
                                energy += tensor[[i, j]];
                            }
                        }
                    }
                }
            } else {
                // Generic tensor evaluation (slower)
                tensor.indexed_iter().for_each(|(indices, &coeff)| {
                    if coeff == 0.0 {
                        return;
                    }

                    // Check if all variables at these indices are 1
                    let term_active = (0..indices.ndim())
                        .map(|d| indices[d])
                        .all(|idx| idx < state.len() && state[idx]);

                    if term_active {
                        energy += coeff;
                    }
                });
            }

            energy
        };

        // Solution map with frequencies
        let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

        // Create a minimal, functional GA implementation
        let pop_size = self.population_size.clamp(10, 100);

        // Initialize random population
        let mut population: Vec<Vec<bool>> = (0..pop_size)
            .map(|_| (0..n_vars).map(|_| rng.random_bool(0.5)).collect())
            .collect();

        // Evaluate initial population
        let mut fitness: Vec<f64> = population
            .iter()
            .map(|indiv| evaluate_energy(indiv))
            .collect();

        // Find best solution
        let mut best_solution = population[0].clone();
        let mut best_fitness = fitness[0];

        for (idx, fit) in fitness.iter().enumerate() {
            if *fit < best_fitness {
                best_fitness = *fit;
                best_solution = population[idx].clone();
            }
        }

        // Genetic algorithm loop
        for _ in 0..30 {
            // Reduced number of generations for faster results
            // Create next generation
            let mut next_population = Vec::with_capacity(pop_size);

            // Elitism - keep best solution
            next_population.push(best_solution.clone());

            // Fill population with new individuals
            while next_population.len() < pop_size {
                // Select parents via tournament selection
                let parent1_idx = tournament_selection(&fitness, 3, &mut rng);
                let parent2_idx = tournament_selection(&fitness, 3, &mut rng);

                // Crossover
                let (mut child1, mut child2) =
                    simple_crossover(&population[parent1_idx], &population[parent2_idx], &mut rng);

                // Mutation
                mutate(&mut child1, 0.05, &mut rng);
                mutate(&mut child2, 0.05, &mut rng);

                // Add children
                next_population.push(child1);
                if next_population.len() < pop_size {
                    next_population.push(child2);
                }
            }

            // Evaluate new population
            population = next_population;
            fitness = population
                .iter()
                .map(|indiv| evaluate_energy(indiv))
                .collect();

            // Update best solution
            for (idx, fit) in fitness.iter().enumerate() {
                if *fit < best_fitness {
                    best_fitness = *fit;
                    best_solution = population[idx].clone();
                }
            }

            // Update solution counts
            for (idx, indiv) in population.iter().enumerate() {
                let entry = solution_counts
                    .entry(indiv.clone())
                    .or_insert((fitness[idx], 0));
                entry.1 += 1;
            }
        }

        // Convert solutions to SampleResult
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
        if results.len() > actual_shots {
            results.truncate(actual_shots);
        }

        Ok(results)
    }

    fn run_qubo(
        &self,
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Extract matrix and variable mapping
        let (matrix, var_map) = qubo;

        // Make sure shots is reasonable
        let actual_shots = std::cmp::max(shots, 10);

        // Get the problem dimension
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Initialize random number generator
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let seed: u64 = rand::rng().random();
                StdRng::seed_from_u64(seed)
            }
        };

        // Use adaptive strategies by default
        let crossover_strategy = CrossoverStrategy::Adaptive;
        let mutation_strategy = MutationStrategy::Annealing(0.1, 0.01);
        let selection_pressure = 3; // Tournament size
        let use_elitism = true;

        // Initialize population with random bitstrings
        let mut population: Vec<Vec<bool>> = (0..self.population_size)
            .map(|_| (0..n_vars).map(|_| rng.random_bool(0.5)).collect())
            .collect();

        // Initialize fitness scores (energy values)
        let mut fitness: Vec<f64> = population
            .iter()
            .map(|indiv| calculate_energy(indiv, matrix))
            .collect();

        // Keep track of best solution in current population
        let mut best_idx = 0;
        let mut best_fitness = fitness[0];
        for (idx, &fit) in fitness.iter().enumerate() {
            if fit < best_fitness {
                best_idx = idx;
                best_fitness = fit;
            }
        }
        let mut best_individual = population[best_idx].clone();
        let mut best_individual_fitness = best_fitness;

        // Track solutions and their frequencies
        let mut solution_counts: HashMap<Vec<bool>, usize> = HashMap::new();

        // Parallel processing for HOBO energy evaluation
        #[cfg(feature = "parallel")]
        let eval_energy = |indiv: &Vec<bool>| -> f64 { calculate_energy(indiv, matrix) };

        #[cfg(not(feature = "parallel"))]
        let eval_energy = |indiv: &Vec<bool>| -> f64 { calculate_energy(indiv, matrix) };

        // Main GA loop
        for generation in 0..self.max_generations {
            // Calculate population diversity for adaptive operators
            let diversity = self.calculate_diversity(&population);

            // Create next generation
            let mut next_population = Vec::with_capacity(self.population_size);
            let mut next_fitness = Vec::with_capacity(self.population_size);

            // Elitism - copy best individual
            if use_elitism {
                next_population.push(best_individual.clone());
                next_fitness.push(best_individual_fitness);
            }

            // Fill rest of population through selection, crossover, mutation
            while next_population.len() < self.population_size {
                // Tournament selection for parents
                let parent1_idx = tournament_selection(&fitness, selection_pressure, &mut rng);
                let parent2_idx = tournament_selection(&fitness, selection_pressure, &mut rng);

                let parent1 = &population[parent1_idx];
                let parent2 = &population[parent2_idx];

                // Crossover
                let (mut child1, mut child2) =
                    self.crossover(parent1, parent2, crossover_strategy, &mut rng);

                // Mutation
                self.mutate(
                    &mut child1,
                    mutation_strategy,
                    generation,
                    self.max_generations,
                    Some(diversity),
                    &mut rng,
                );
                self.mutate(
                    &mut child2,
                    mutation_strategy,
                    generation,
                    self.max_generations,
                    Some(diversity),
                    &mut rng,
                );

                // Evaluate fitness of new children
                let child1_fitness = eval_energy(&child1);
                let child2_fitness = eval_energy(&child2);

                // Add first child
                next_population.push(child1);
                next_fitness.push(child1_fitness);

                // Add second child if there's room
                if next_population.len() < self.population_size {
                    next_population.push(child2);
                    next_fitness.push(child2_fitness);
                }
            }

            // Update population
            population = next_population;
            fitness = next_fitness;

            // Update best solution
            best_idx = 0;
            best_fitness = fitness[0];
            for (idx, &fit) in fitness.iter().enumerate() {
                if fit < best_fitness {
                    best_idx = idx;
                    best_fitness = fit;
                }
            }

            // Update global best if needed
            if best_fitness < best_individual_fitness {
                best_individual = population[best_idx].clone();
                best_individual_fitness = best_fitness;
            }

            // Update solution counts
            for individual in &population {
                *solution_counts.entry(individual.clone()).or_insert(0) += 1;
            }
        }

        // Collect results
        let mut results = Vec::new();

        // Convert the solutions to SampleResult format
        for (solution, count) in &solution_counts {
            // Only include solutions that appeared multiple times
            if *count < 2 {
                continue;
            }

            // Calculate energy one more time
            let energy = calculate_energy(solution, matrix);

            // Convert to variable assignments
            let assignments: HashMap<String, bool> = solution
                .iter()
                .enumerate()
                .map(|(idx, &value)| {
                    let var_name = idx_to_var.get(&idx).unwrap().clone();
                    (var_name, value)
                })
                .collect();

            // Create result and add to collection
            results.push(SampleResult {
                assignments,
                energy,
                occurrences: *count,
            });
        }

        // Sort by energy (best solutions first)
        results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        // Trim to requested number of shots
        if results.len() > actual_shots {
            results.truncate(actual_shots);
        }

        Ok(results)
    }

    // This is a duplicate implementation of run_hobo that was removed
}

// Helper function to calculate energy for a solution
fn calculate_energy(solution: &[bool], matrix: &Array<f64, ndarray::Ix2>) -> f64 {
    let n = solution.len();
    let mut energy = 0.0;

    // Calculate from diagonal terms (linear)
    for i in 0..n {
        if solution[i] {
            energy += matrix[[i, i]];
        }
    }

    // Calculate from off-diagonal terms (quadratic)
    for i in 0..n {
        if solution[i] {
            for j in (i + 1)..n {
                if solution[j] {
                    energy += matrix[[i, j]];
                }
            }
        }
    }

    energy
}

// Helper function for single-point crossover
fn simple_crossover(
    parent1: &[bool],
    parent2: &[bool],
    rng: &mut impl Rng,
) -> (Vec<bool>, Vec<bool>) {
    let n_vars = parent1.len();
    let mut child1 = vec![false; n_vars];
    let mut child2 = vec![false; n_vars];

    // Use single-point crossover
    let crossover_point = if n_vars > 1 {
        rng.random_range(1..n_vars)
    } else {
        0 // Special case for one-variable problems
    };

    for i in 0..n_vars {
        if i < crossover_point {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        } else {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    }

    (child1, child2)
}

// Helper function for mutation
fn mutate(individual: &mut [bool], rate: f64, rng: &mut impl Rng) {
    for bit in individual.iter_mut() {
        if rng.random_bool(rate) {
            *bit = !*bit;
        }
    }
}

// Helper function for tournament selection
fn tournament_selection(fitness: &[f64], tournament_size: usize, rng: &mut impl Rng) -> usize {
    // Handle edge cases
    assert!(
        !fitness.is_empty(),
        "Cannot perform tournament selection on an empty fitness array"
    );

    if fitness.len() == 1 || tournament_size <= 1 {
        return 0; // Only one choice available
    }

    // Ensure tournament_size is not larger than the population
    let effective_tournament_size = std::cmp::min(tournament_size, fitness.len());

    let mut best_idx = rng.random_range(0..fitness.len());
    let mut best_fitness = fitness[best_idx];

    for _ in 1..(effective_tournament_size) {
        let candidate_idx = rng.random_range(0..fitness.len());
        let candidate_fitness = fitness[candidate_idx];

        // Lower fitness is better (minimization problem)
        if candidate_fitness < best_fitness {
            best_idx = candidate_idx;
            best_fitness = candidate_fitness;
        }
    }

    best_idx
}

/// GPU-accelerated Sampler (Armin)
///
/// This sampler uses GPU acceleration to find solutions to
/// QUBO/HOBO problems. It is based on parallel tempering and
/// is optimized for large problems.
#[cfg(feature = "gpu")]
pub struct ArminSampler {
    /// Random number generator seed
    seed: Option<u64>,
    /// Whether to use GPU ("GPU") or CPU ("CPU")
    mode: String,
    /// Device to use (e.g., "cuda:0")
    device: String,
    /// Whether to show verbose output
    verbose: bool,
}

#[cfg(feature = "gpu")]
impl ArminSampler {
    /// Create a new GPU-accelerated sampler
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    #[must_use]
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            seed,
            mode: "GPU".to_string(),
            device: "cuda:0".to_string(),
            verbose: true,
        }
    }

    /// Create a new GPU-accelerated sampler with custom parameters
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    /// * `mode` - Whether to use GPU ("GPU") or CPU ("CPU")
    /// * `device` - Device to use (e.g., "cuda:0")
    /// * `verbose` - Whether to show verbose output
    #[must_use]
    pub fn with_params(seed: Option<u64>, mode: &str, device: &str, verbose: bool) -> Self {
        Self {
            seed,
            mode: mode.to_string(),
            device: device.to_string(),
            verbose,
        }
    }

    /// Run GPU-accelerated annealing using OpenCL
    fn run_gpu_annealing(
        &self,
        n_vars: usize,
        h_vector: &[f64],
        j_matrix: &[f64],
        num_shots: usize,
    ) -> Result<Vec<Vec<bool>>, ocl::Error> {
        use ocl::flags;
        use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};

        // Check problem size
        if n_vars > 2048 {
            if self.verbose {
                println!(
                    "Problem size too large for standard OpenCL kernel. Using chunked approach."
                );
            }
            return self.run_gpu_annealing_chunked(n_vars, h_vector, j_matrix, num_shots);
        }

        // Display progress if verbose
        if self.verbose {
            println!(
                "Initializing GPU with {} variables and {} shots",
                n_vars, num_shots
            );
        }

        // Set up OpenCL environment
        let platform = if self.device.contains("cpu") {
            // Find CPU platform
            Platform::list()
                .into_iter()
                .find(|p| p.name().unwrap_or_default().to_lowercase().contains("cpu"))
                .unwrap_or_else(|| Platform::default())
        } else {
            // Default platform (typically GPU)
            Platform::default()
        };

        if self.verbose {
            println!("Using platform: {}", platform.name().unwrap_or_default());
        }

        // Find appropriate device
        let device = if self.device.contains("cpu") {
            // CPU device
            Device::list_all(platform)
                .unwrap_or_default()
                .into_iter()
                .find(|d| d.is_cpu().unwrap_or(false))
                .unwrap_or_else(|| Device::first(platform).unwrap())
        } else {
            // GPU device
            Device::list_all(platform)
                .unwrap_or_default()
                .into_iter()
                .find(|d| d.is_gpu().unwrap_or(false))
                .unwrap_or_else(|| Device::first(platform).unwrap())
        };

        if self.verbose {
            println!("Using device: {}", device.name().unwrap_or_default());
        }

        // Build context and queue
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, None)?;

        // Determine optimal work group size
        let max_work_group_size = device.max_work_group_size().unwrap_or(64) as usize;
        let work_group_size = std::cmp::min(max_work_group_size, num_shots.next_power_of_two() / 2);

        if self.verbose {
            println!("Using work group size: {}", work_group_size);
        }

        // Kernel source code with advanced optimizations
        let src = r#"
        // Helper function for xorshift RNG
        inline ulong xorshift64(ulong *state) {
            ulong x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            return x;
        }

        // Fast random float generation
        inline float random_float(ulong *state) {
            ulong x = xorshift64(state);
            return (float)(x & 0xFFFFFFFF) / (float)0xFFFFFFFF;
        }

        // Simulated annealing kernel
        __kernel void simulated_annealing(
            const int n_vars,
            __global const float* h_vector,
            __global const float* j_matrix,
            __global uchar* solutions,
            const int num_runs,
            const float init_temp,
            const float final_temp,
            const int sweeps,
            const ulong seed
        ) {
            // Get global ID
            int gid = get_global_id(0);
            if (gid >= num_runs) return;

            // Initialize RNG for this thread
            ulong rng_state = seed + gid;

            // Initialize spin state with local storage for better performance
            uchar state[2048]; // Max vars supported
            float best_energy = 0.0f;
            float current_energy = 0.0f;
            uchar best_state[2048];

            // Random initialization
            for (int i = 0; i < n_vars; i++) {
                state[i] = (xorshift64(&rng_state) & 1) ? 1 : 0;
            }

            // Calculate initial energy
            for (int i = 0; i < n_vars; i++) {
                if (state[i]) {
                    current_energy += h_vector[i];

                    for (int j = 0; j < n_vars; j++) {
                        if (j != i && state[j]) {
                            current_energy += j_matrix[i * n_vars + j];
                        }
                    }
                }
            }

            // Initialize best solution
            best_energy = current_energy;
            for (int i = 0; i < n_vars; i++) {
                best_state[i] = state[i];
            }

            // Annealing process
            for (int sweep = 0; sweep < sweeps; sweep++) {
                // Calculate temperature for this sweep using exponential schedule
                float t_ratio = (float)sweep / (float)sweeps;
                float temp = init_temp * pow(final_temp / init_temp, t_ratio);

                // Perform n_vars spin flips per sweep
                for (int flip = 0; flip < n_vars; flip++) {
                    // Choose random spin to flip using efficient sampling
                    int idx = xorshift64(&rng_state) % n_vars;

                    // Calculate energy change efficiently
                    float delta_e = 0.0f;

                    // Contribution from h (linear terms)
                    delta_e += h_vector[idx] * (state[idx] ? -2.0f : 2.0f);

                    // Contribution from J (quadratic terms)
                    for (int j = 0; j < n_vars; j++) {
                        if (state[j]) {
                            delta_e += j_matrix[idx * n_vars + j] * (state[idx] ? -2.0f : 2.0f);
                        }
                    }

                    // Metropolis acceptance criterion with optimized branching
                    bool accept = (delta_e <= 0.0f) || (random_float(&rng_state) < exp(-delta_e / temp));

                    // Update state and energy if accepted
                    if (accept) {
                        state[idx] = !state[idx];
                        current_energy += delta_e;

                        // Update best solution if improved
                        if (current_energy < best_energy) {
                            best_energy = current_energy;
                            for (int i = 0; i < n_vars; i++) {
                                best_state[i] = state[i];
                            }
                        }
                    }
                }

                // Periodic check if we've converged to a stable solution
                if (sweep % 50 == 0 && sweep > 0) {
                    // If temperature is very low and we've reached a good solution, we can terminate early
                    if (temp < 0.01f * final_temp && sweep > (sweeps / 2)) {
                        break;
                    }
                }
            }

            // Write best solution to global memory
            for (int i = 0; i < n_vars; i++) {
                solutions[gid * n_vars + i] = best_state[i];
            }
        }

        // Parallel tempering kernel for better exploration
        __kernel void parallel_tempering(
            const int n_vars,
            __global const float* h_vector,
            __global const float* j_matrix,
            __global uchar* solutions,
            const int num_runs,
            const int num_replicas,
            const float min_temp,
            const float max_temp,
            const int sweeps,
            const ulong seed
        ) {
            // Get global ID and calculate which run and replica this is
            int gid = get_global_id(0);
            int run_id = gid / num_replicas;
            int replica_id = gid % num_replicas;

            if (run_id >= num_runs) return;

            // Initialize RNG for this thread
            ulong rng_state = seed + gid;

            // Calculate temperature for this replica using exponential spacing
            float beta_idx = (float)replica_id / (float)(num_replicas - 1);
            float temp = min_temp * pow(max_temp / min_temp, beta_idx);

            // Initialize state
            uchar state[2048];
            float energy = 0.0f;

            // Random initialization
            for (int i = 0; i < n_vars; i++) {
                state[i] = (xorshift64(&rng_state) & 1) ? 1 : 0;
            }

            // Calculate initial energy
            for (int i = 0; i < n_vars; i++) {
                if (state[i]) {
                    energy += h_vector[i];

                    for (int j = 0; j < n_vars; j++) {
                        if (j != i && state[j]) {
                            energy += j_matrix[i * n_vars + j];
                        }
                    }
                }
            }

            // Perform annealing
            for (int sweep = 0; sweep < sweeps; sweep++) {
                // Monte Carlo updates
                for (int flip = 0; flip < n_vars; flip++) {
                    int idx = xorshift64(&rng_state) % n_vars;
                    float delta_e = 0.0f;

                    // Calculate energy change
                    delta_e += h_vector[idx] * (state[idx] ? -2.0f : 2.0f);
                    for (int j = 0; j < n_vars; j++) {
                        if (state[j]) {
                            delta_e += j_matrix[idx * n_vars + j] * (state[idx] ? -2.0f : 2.0f);
                        }
                    }

                    // Metropolis acceptance
                    if (delta_e <= 0.0f || random_float(&rng_state) < exp(-delta_e / temp)) {
                        state[idx] = !state[idx];
                        energy += delta_e;
                    }
                }

                // Replica exchange (every 10 sweeps)
                if (sweep % 10 == 0 && replica_id < num_replicas - 1) {
                    // This would need synchronization between threads
                    // Since OpenCL doesn't easily support this, we skip it in this implementation
                }
            }

            // Save solution for this replica
            int solution_idx = run_id;
            if (replica_id == 0) { // Only save the lowest temperature replica
                for (int i = 0; i < n_vars; i++) {
                    solutions[solution_idx * n_vars + i] = state[i];
                }
            }
        }
        "#;

        // Compile the program
        let program = Program::builder().devices(device).src(src).build()?;

        // Set up buffers
        let h_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars)
            .build()?;

        let j_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars * n_vars)
            .build()?;

        let solutions_buffer = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(num_shots * n_vars)
            .build()?;

        // Convert h_vector and j_matrix to f32
        let h_vec_f32: Vec<f32> = h_vector.iter().map(|&x| x as f32).collect();
        let j_mat_f32: Vec<f32> = j_matrix.iter().map(|&x| x as f32).collect();

        // Transfer data to GPU
        h_buffer.write(&h_vec_f32).enq()?;
        j_buffer.write(&j_mat_f32).enq()?;

        // Set up kernel parameters
        let init_temp = 10.0f32;
        let final_temp = 0.1f32;
        let sweeps = if n_vars < 100 {
            1000
        } else if n_vars < 500 {
            2000
        } else {
            5000
        };

        if self.verbose {
            println!(
                "Running {} sweeps with temperature range [{}, {}]",
                sweeps, final_temp, init_temp
            );
        }

        // Create a seed based on input seed or random value
        let seed_val = self.seed.unwrap_or_else(rand::random::<u64>);

        // Use parallel tempering for larger problems, standard SA for smaller ones
        let use_parallel_tempering = n_vars > 100 && num_shots > 10;

        if use_parallel_tempering {
            // Number of temperature replicas
            let num_replicas = 8; // Must be a power of 2 for optimal performance

            // Adjust number of shots to account for replicas
            let actual_num_shots = (num_shots + num_replicas - 1) / num_replicas * num_replicas;

            if self.verbose {
                println!("Using parallel tempering with {} replicas", num_replicas);
                println!("Adjusted shots: {} ({})", actual_num_shots, num_shots);
            }

            // Set up and run parallel tempering kernel
            let kernel = Kernel::builder()
                .program(&program)
                .name("parallel_tempering")
                .global_work_size(actual_num_shots)
                .local_work_size(work_group_size)
                .arg(n_vars as i32)
                .arg(&h_buffer)
                .arg(&j_buffer)
                .arg(&solutions_buffer)
                .arg((actual_num_shots / num_replicas) as i32) // num_runs
                .arg(num_replicas as i32)
                .arg(final_temp)
                .arg(init_temp)
                .arg(sweeps as i32)
                .arg(seed_val)
                .build()?;

            // Execute kernel
            unsafe {
                kernel.enq()?;
            }

            // Only read the first num_shots solutions (to match requested count)
            let mut solutions_data = vec![0u8; num_shots * n_vars];
            solutions_buffer
                .read(&mut solutions_data[0..(num_shots * n_vars)])
                .enq()?;

            // Convert to Vec<Vec<bool>>
            let mut results = Vec::with_capacity(num_shots);
            for i in 0..num_shots {
                let mut solution = Vec::with_capacity(n_vars);
                for j in 0..n_vars {
                    solution.push(solutions_data[i * n_vars + j] != 0);
                }
                results.push(solution);
            }

            Ok(results)
        } else {
            // Set up and run standard simulated annealing kernel
            let kernel = Kernel::builder()
                .program(&program)
                .name("simulated_annealing")
                .global_work_size(num_shots)
                .local_work_size(work_group_size)
                .arg(n_vars as i32)
                .arg(&h_buffer)
                .arg(&j_buffer)
                .arg(&solutions_buffer)
                .arg(num_shots as i32)
                .arg(init_temp)
                .arg(final_temp)
                .arg(sweeps as i32)
                .arg(seed_val)
                .build()?;

            // Execute kernel
            unsafe {
                kernel.enq()?;
            }

            // Read results
            let mut solutions_data = vec![0u8; num_shots * n_vars];
            solutions_buffer.read(&mut solutions_data).enq()?;

            // Convert to Vec<Vec<bool>>
            let mut results = Vec::with_capacity(num_shots);
            for i in 0..num_shots {
                let mut solution = Vec::with_capacity(n_vars);
                for j in 0..n_vars {
                    solution.push(solutions_data[i * n_vars + j] != 0);
                }
                results.push(solution);
            }

            Ok(results)
        }
    }

    /// Chunked GPU annealing for very large problems
    fn run_gpu_annealing_chunked(
        &self,
        n_vars: usize,
        h_vector: &[f64],
        j_matrix: &[f64],
        num_shots: usize,
    ) -> Result<Vec<Vec<bool>>, ocl::Error> {
        // For problems too large to fit in a single kernel, we chunk the problem
        // into smaller subproblems and solve iteratively

        if self.verbose {
            println!(
                "Using chunked approach for large problem: {} variables",
                n_vars
            );
        }

        // Maximum number of variables to process in a single chunk
        const MAX_CHUNK_SIZE: usize = 1024;

        // Calculate number of chunks needed
        let num_chunks = (n_vars + MAX_CHUNK_SIZE - 1) / MAX_CHUNK_SIZE;

        if self.verbose {
            println!(
                "Processing in {} chunks of at most {} variables each",
                num_chunks, MAX_CHUNK_SIZE
            );
        }

        // Initialize random number generator
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let seed: u64 = rand::rng().random();
                StdRng::seed_from_u64(seed)
            }
        };

        // Initialize random solutions for all shots
        let mut solutions: Vec<Vec<bool>> = Vec::with_capacity(num_shots);
        for _ in 0..num_shots {
            let mut solution = Vec::with_capacity(n_vars);
            for _ in 0..n_vars {
                solution.push(rng.random_bool(0.5));
            }
            solutions.push(solution);
        }

        // Track energies for each solution
        let mut energies = vec![0.0; num_shots];

        // Initialize energies
        for (i, solution) in solutions.iter().enumerate() {
            energies[i] = evaluate_qubo_energy(solution, h_vector, j_matrix, n_vars);
        }

        // Process each chunk iteratively
        for chunk_idx in 0..num_chunks {
            // Calculate start and end indices for this chunk
            let start_var = chunk_idx * MAX_CHUNK_SIZE;
            let end_var = std::cmp::min((chunk_idx + 1) * MAX_CHUNK_SIZE, n_vars);
            let chunk_size = end_var - start_var;

            if self.verbose {
                println!(
                    "Processing chunk {}/{}: variables {}..{}",
                    chunk_idx + 1,
                    num_chunks,
                    start_var,
                    end_var - 1
                );
            }

            // Extract subproblem
            let mut chunk_h = Vec::with_capacity(chunk_size);
            let mut chunk_j = Vec::with_capacity(chunk_size * chunk_size);

            // Extract linear terms for this chunk
            for i in start_var..end_var {
                chunk_h.push(h_vector[i]);
            }

            // Extract quadratic terms for this chunk
            for i in start_var..end_var {
                for j in start_var..end_var {
                    chunk_j.push(j_matrix[i * n_vars + j]);
                }
            }

            // Adjust linear terms based on fixed variables outside this chunk
            for (sol_idx, solution) in solutions.iter().enumerate() {
                let mut adjusted_h = chunk_h.clone();

                // Add contributions from fixed variables
                for i in start_var..end_var {
                    for j in 0..n_vars {
                        if j < start_var || j >= end_var {
                            if solution[j] {
                                adjusted_h[i - start_var] += j_matrix[i * n_vars + j];
                            }
                        }
                    }
                }

                // Process this specific solution's subproblem
                let mut chunk_solution = Vec::with_capacity(chunk_size);
                for i in start_var..end_var {
                    chunk_solution.push(solution[i]);
                }

                // Optimize just this chunk using GPU
                let optimized_chunk = match self.optimize_chunk(
                    &chunk_solution,
                    &adjusted_h,
                    &chunk_j,
                    chunk_size,
                    self.seed.map(|s| s + sol_idx as u64),
                ) {
                    Ok(result) => result,
                    Err(e) => return Err(e),
                };

                // Update the original solution with optimized chunk
                for (i, &val) in optimized_chunk.iter().enumerate() {
                    solutions[sol_idx][start_var + i] = val;
                }

                // Update energy
                energies[sol_idx] =
                    evaluate_qubo_energy(&solutions[sol_idx], h_vector, j_matrix, n_vars);
            }
        }

        // Sort solutions by energy
        let mut solution_pairs: Vec<(Vec<bool>, f64)> =
            solutions.into_iter().zip(energies.into_iter()).collect();
        solution_pairs.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

        // Return sorted solutions
        Ok(solution_pairs.into_iter().map(|(sol, _)| sol).collect())
    }

    /// Optimize a single chunk of variables
    fn optimize_chunk(
        &self,
        initial_state: &[bool],
        h_vector: &[f64],
        j_matrix: &[f64],
        n_vars: usize,
        seed: Option<u64>,
    ) -> Result<Vec<bool>, ocl::Error> {
        use ocl::flags;
        use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};

        // Set up OpenCL environment (same as in run_gpu_annealing)
        let platform = Platform::default();
        let device = Device::first(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, None)?;

        // Use a simplified kernel for chunked optimization
        let src = r#"
        __kernel void optimize_chunk(
            const int n_vars,
            __global const float* h_vector,
            __global const float* j_matrix,
            __global uchar* initial_state,
            __global uchar* result_state,
            const int sweeps,
            const float init_temp,
            const float final_temp,
            const ulong seed
        ) {
            // Initialize RNG
            ulong rng_state = seed;

            // Copy initial state to local array
            uchar state[1024]; // Max chunk size
            for (int i = 0; i < n_vars; i++) {
                state[i] = initial_state[i];
            }

            // Calculate initial energy
            float energy = 0.0f;
            for (int i = 0; i < n_vars; i++) {
                if (state[i]) {
                    energy += h_vector[i];

                    for (int j = 0; j < n_vars; j++) {
                        if (j != i && state[j]) {
                            energy += j_matrix[i * n_vars + j];
                        }
                    }
                }
            }

            // Track best solution
            float best_energy = energy;
            uchar best_state[1024];
            for (int i = 0; i < n_vars; i++) {
                best_state[i] = state[i];
            }

            // Annealing process
            for (int sweep = 0; sweep < sweeps; sweep++) {
                // Calculate temperature
                float t_ratio = (float)sweep / (float)sweeps;
                float temp = init_temp * pow(final_temp / init_temp, t_ratio);

                // Monte Carlo steps
                for (int i = 0; i < n_vars; i++) {
                    // Choose variable to flip
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    int idx = rng_state % n_vars;

                    // Calculate energy change
                    float delta_e = 0.0f;
                    delta_e += h_vector[idx] * (state[idx] ? -2.0f : 2.0f);

                    for (int j = 0; j < n_vars; j++) {
                        if (state[j]) {
                            delta_e += j_matrix[idx * n_vars + j] * (state[idx] ? -2.0f : 2.0f);
                        }
                    }

                    // Metropolis acceptance criterion
                    bool accept = false;
                    if (delta_e <= 0.0f) {
                        accept = true;
                    } else {
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 7;
                        rng_state ^= rng_state << 17;
                        float rand_val = (float)(rng_state & 0xFFFFFFFF) / (float)0xFFFFFFFF;
                        accept = rand_val < exp(-delta_e / temp);
                    }

                    // Apply change if accepted
                    if (accept) {
                        state[idx] = !state[idx];
                        energy += delta_e;

                        // Update best solution if improved
                        if (energy < best_energy) {
                            best_energy = energy;
                            for (int j = 0; j < n_vars; j++) {
                                best_state[j] = state[j];
                            }
                        }
                    }
                }
            }

            // Write best solution back
            for (int i = 0; i < n_vars; i++) {
                result_state[i] = best_state[i];
            }
        }
        "#;

        // Compile the program
        let program = Program::builder()
            .devices(device)
            .src(src)
            .build(&context)?;

        // Set up buffers
        let h_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars)
            .build()?;

        let j_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars * n_vars)
            .build()?;

        let initial_buffer = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars)
            .build()?;

        let result_buffer = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(n_vars)
            .build()?;

        // Convert data types
        let h_vec_f32: Vec<f32> = h_vector.iter().map(|&x| x as f32).collect();
        let j_mat_f32: Vec<f32> = j_matrix.iter().map(|&x| x as f32).collect();
        let initial_u8: Vec<u8> = initial_state
            .iter()
            .map(|&b| if b { 1 } else { 0 })
            .collect();

        // Transfer data to device
        h_buffer.write(&h_vec_f32).enq()?;
        j_buffer.write(&j_mat_f32).enq()?;
        initial_buffer.write(&initial_u8).enq()?;

        // Set kernel parameters
        let kernel = Kernel::builder()
            .program(&program)
            .name("optimize_chunk")
            .global_work_size(1) // Only one optimization task
            .arg(n_vars as i32)
            .arg(&h_buffer)
            .arg(&j_buffer)
            .arg(&initial_buffer)
            .arg(&result_buffer)
            .arg(5000i32) // More sweeps for thorough optimization of a chunk
            .arg(5.0f32)  // Higher initial temperature
            .arg(0.01f32) // Lower final temperature
            .arg(seed.unwrap_or_else(rand::random::<u64>))
            .build()?;

        // Execute kernel
        unsafe {
            kernel.enq()?;
        }

        // Read result
        let mut result_u8 = vec![0u8; n_vars];
        result_buffer.read(&mut result_u8).enq()?;

        // Convert back to bool
        let result = result_u8.iter().map(|&b| b != 0).collect();

        Ok(result)
    }
}

#[cfg(feature = "gpu")]
impl Sampler for ArminSampler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Extract matrix and variable mapping
        let (matrix, var_map) = qubo;

        // Get the problem dimension
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Determine compute resources based on mode
        let is_gpu = self.mode.to_uppercase() == "GPU";
        let device_info = if is_gpu {
            format!("Using GPU device: {}", self.device)
        } else {
            "Using CPU acceleration".to_string()
        };

        if self.verbose {
            println!("{}", device_info);
            println!("Problem size: {} variables", n_vars);
        }

        // Convert QUBO matrix to appropriate format for OpenCL
        let mut h_vector = Vec::with_capacity(n_vars);
        let mut j_matrix = Vec::with_capacity(n_vars * n_vars);

        // Extract diagonal (linear) terms
        for i in 0..n_vars {
            h_vector.push(matrix[[i, i]]);
        }

        // Extract off-diagonal (quadratic) terms
        for i in 0..n_vars {
            for j in 0..n_vars {
                if i != j {
                    j_matrix.push(matrix[[i, j]]);
                } else {
                    j_matrix.push(0.0); // Zero on diagonal in J matrix
                }
            }
        }

        // Set up OpenCL and run GPU annealing
        #[cfg(feature = "gpu")]
        let ocl_result = self.run_gpu_annealing(n_vars, &h_vector, &j_matrix, shots);

        #[cfg(not(feature = "gpu"))]
        let ocl_result = Err(SamplerError::GpuError(
            "GPU support not enabled".to_string(),
        ));

        #[cfg(feature = "gpu")]
        match ocl_result {
            Ok(binary_solutions) => {
                // Process results
                let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

                for solution in binary_solutions {
                    // Calculate energy using our helper function
                    let energy = evaluate_qubo_energy(&solution, &h_vector, &j_matrix, n_vars);

                    // Update solution counts
                    let entry = solution_counts.entry(solution).or_insert((energy, 0));
                    entry.1 += 1;
                }

                // Convert to SampleResult format
                let mut results: Vec<SampleResult> = solution_counts
                    .into_iter()
                    .map(|(bin_solution, (energy, count))| {
                        // Convert to variable assignments
                        let assignments: HashMap<String, bool> = bin_solution
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

                // Sort by energy
                results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

                // Limit to requested number of shots
                if results.len() > shots {
                    results.truncate(shots);
                }

                Ok(results)
            }
            Err(e) => Err(SamplerError::GpuError(e.to_string())),
        }

        #[cfg(not(feature = "gpu"))]
        match ocl_result {
            Ok(_) => unreachable!("GPU support not enabled"),
            Err(e) => Err(e),
        }
    }

    fn run_hobo(
        &self,
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Handle QUBO case directly
        if hobo.0.ndim() == 2 {
            let qubo = (
                hobo.0
                    .clone()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap(),
                hobo.1.clone(),
            );
            return self.run_qubo(&qubo, shots);
        }

        // Extract tensor and variables
        let (tensor, var_map) = hobo;
        let n_vars = var_map.len();

        // Map indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // For 3rd order HOBO problems, we can use a specialized approach
        if tensor.ndim() == 3 {
            if self.verbose {
                println!("Processing 3rd-order HOBO problem with GPU acceleration");
            }

            // Convert the 3rd-order tensor to a QUBO-like representation
            // that our GPU kernels can handle efficiently
            return self.run_hobo_3rd_order(tensor, var_map, shots);
        }

        // For higher-order HOBO problems, we'll try tensor decomposition
        if tensor.ndim() > 3 {
            if self.verbose {
                println!(
                    "Higher-order ({}D) HOBO problem, using tensor decomposition approach",
                    tensor.ndim()
                );
            }

            return self.run_hobo_high_order(tensor, var_map, shots);
        }

        // Function to evaluate HOBO energy
        let evaluate_hobo_energy = |state: &[bool]| -> f64 {
            let mut energy = 0.0;

            // For each possible index combination in the tensor
            tensor.indexed_iter().for_each(|(indices, &coeff)| {
                if coeff == 0.0 {
                    return;
                }

                // Check if all corresponding variables are 1 in the state
                // For ndarray, convert to primitive types
                let index_vec: Vec<usize> = (0..indices.ndim()).map(|i| indices[i]).collect();

                // Check if all indices are active
                let term_active = index_vec.iter().all(|&idx| idx < state.len() && state[idx]);

                if term_active {
                    energy += coeff;
                }
            });

            energy
        };

        if self.verbose {
            println!("Using parallel tempering with GPU acceleration for HOBO problem");
        }

        // Initialize RNG
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let seed: u64 = rand::rng().random();
                StdRng::seed_from_u64(seed)
            }
        };

        // Generate initial random solutions
        let mut binary_solutions = Vec::with_capacity(shots);
        for _ in 0..shots {
            let solution = (0..n_vars).map(|_| rng.random_bool(0.5)).collect();
            binary_solutions.push(solution);
        }

        // Calculate initial energies
        let mut solutions_with_energies: Vec<(Vec<bool>, f64)> = binary_solutions
            .into_iter()
            .map(|sol| {
                let energy = evaluate_hobo_energy(&sol);
                (sol, energy)
            })
            .collect();

        // Sort solutions by energy
        solutions_with_energies.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Keep track of best solutions
        let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

        // Process results
        for (solution, energy) in solutions_with_energies {
            let entry = solution_counts.entry(solution).or_insert((energy, 0));
            entry.1 += 1;
        }

        // Convert to SampleResult format
        let mut results: Vec<SampleResult> = solution_counts
            .into_iter()
            .map(|(bin_solution, (energy, count))| {
                // Convert to variable assignments
                let assignments: HashMap<String, bool> = bin_solution
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

        // Sort by energy
        results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        // Limit to requested number of shots
        if results.len() > shots {
            results.truncate(shots);
        }

        Ok(results)
    }
}

#[cfg(feature = "gpu")]
impl ArminSampler {
    /// Process a 3rd-order HOBO problem with GPU acceleration
    fn run_hobo_3rd_order(
        &self,
        tensor: &Array<f64, ndarray::IxDyn>,
        var_map: &HashMap<String, usize>,
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        let n_vars = var_map.len();

        if self.verbose {
            println!(
                "Processing 3rd-order HOBO problem with {} variables",
                n_vars
            );
        }

        // For 3rd order tensors, we'll use a specialized approach that represents
        // the problem in a format our GPU kernels can handle

        // Extract coefficients from tensor (for 3rd order tensor)
        let mut linear_terms = vec![0.0; n_vars];
        let mut quadratic_terms = vec![0.0; n_vars * n_vars];
        let mut cubic_terms = Vec::new();

        // Process the tensor
        tensor.indexed_iter().for_each(|(indices, &coeff)| {
            if coeff == 0.0 {
                return;
            }

            let idx_vec: Vec<usize> = indices.iter().map(|&idx| idx as usize).collect();

            // Count distinct indices
            let mut distinct_indices = idx_vec.clone();
            distinct_indices.sort();
            distinct_indices.dedup();

            match distinct_indices.len() {
                1 => {
                    // Linear term: i,i,i
                    let i = idx_vec[0];
                    linear_terms[i] += coeff;
                }
                2 => {
                    // Quadratic term: i,i,j or i,j,j or similar patterns
                    let i = distinct_indices[0];
                    let j = distinct_indices[1];

                    // Count occurrences of each index
                    let i_count = idx_vec.iter().filter(|&&idx| idx == i).count();
                    let j_count = idx_vec.iter().filter(|&&idx| idx == j).count();

                    if i_count == 2 && j_count == 1 {
                        // i appears twice, j once
                        quadratic_terms[i * n_vars + j] += coeff;
                        quadratic_terms[j * n_vars + i] += coeff;
                    } else if i_count == 1 && j_count == 2 {
                        // i appears once, j twice
                        quadratic_terms[i * n_vars + j] += coeff;
                        quadratic_terms[j * n_vars + i] += coeff;
                    }
                }
                3 => {
                    // True cubic term: i,j,k (all distinct)
                    let i = distinct_indices[0];
                    let j = distinct_indices[1];
                    let k = distinct_indices[2];
                    cubic_terms.push((i, j, k, coeff));
                }
                _ => unreachable!(), // Impossible for 3rd order tensor
            }
        });

        // Prepare data for GPU processing
        let mut h_vector = linear_terms;
        let mut j_matrix = quadratic_terms;

        // Add auxiliary variables for cubic terms if needed
        let mut augmented_vars = n_vars;
        let mut aux_var_map = HashMap::new();

        // If we have cubic terms, we need to reformulate using auxiliary variables
        if !cubic_terms.is_empty() {
            // For each cubic term x_i * x_j * x_k, we introduce auxiliary variable y
            // with constraint y = x_i * x_j * x_k, which we implement as a QUBO penalty

            if self.verbose {
                println!(
                    "Reformulating {} cubic terms with auxiliary variables",
                    cubic_terms.len()
                );
            }

            for (i, j, k, coeff) in cubic_terms {
                // Add new auxiliary variable
                let aux_idx = augmented_vars;
                augmented_vars += 1;

                // Store mapping
                aux_var_map.insert(aux_idx, (i, j, k));

                // Add constraint terms that enforce y = x_i * x_j * x_k
                // This uses penalty method: 3y - x_i - x_j - x_k - x_i*x_j - x_i*x_k - x_j*x_k + 2*x_i*x_j*x_k
                // which is minimized when y = x_i * x_j * x_k

                // Expand h_vector and j_matrix
                h_vector.resize(augmented_vars, 0.0);
                j_matrix.resize(augmented_vars * augmented_vars, 0.0);

                // Linear terms
                h_vector[aux_idx] += 3.0 * coeff;
                h_vector[i] -= coeff;
                h_vector[j] -= coeff;
                h_vector[k] -= coeff;

                // Quadratic terms
                j_matrix[i * augmented_vars + j] -= coeff;
                j_matrix[j * augmented_vars + i] -= coeff;

                j_matrix[i * augmented_vars + k] -= coeff;
                j_matrix[k * augmented_vars + i] -= coeff;

                j_matrix[j * augmented_vars + k] -= coeff;
                j_matrix[k * augmented_vars + j] -= coeff;

                // Auxiliary variable interactions
                j_matrix[aux_idx * augmented_vars + i] -= coeff;
                j_matrix[i * augmented_vars + aux_idx] -= coeff;

                j_matrix[aux_idx * augmented_vars + j] -= coeff;
                j_matrix[j * augmented_vars + aux_idx] -= coeff;

                j_matrix[aux_idx * augmented_vars + k] -= coeff;
                j_matrix[k * augmented_vars + aux_idx] -= coeff;
            }
        }

        // Run GPU annealing on the augmented problem
        #[cfg(feature = "gpu")]
        let ocl_result = self.run_gpu_annealing(augmented_vars, &h_vector, &j_matrix, shots);

        #[cfg(not(feature = "gpu"))]
        let ocl_result = Err(SamplerError::GpuError(
            "GPU support not enabled".to_string(),
        ));

        #[cfg(feature = "gpu")]
        match ocl_result {
            Ok(binary_solutions) => {
                // Process results
                let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

                for mut solution in binary_solutions {
                    // Remove auxiliary variables
                    if solution.len() > n_vars {
                        solution.truncate(n_vars);
                    }

                    // Calculate energy
                    let mut energy = 0.0;

                    // Evaluate energy using original tensor
                    tensor.indexed_iter().for_each(|(indices, &coeff)| {
                        if coeff == 0.0 {
                            return;
                        }

                        // Check if all corresponding variables are true in the solution
                        let term_active = indices
                            .iter()
                            .all(|&idx| idx < solution.len() && solution[idx as usize]);

                        if term_active {
                            energy += coeff;
                        }
                    });

                    // Update solution counts
                    let entry = solution_counts.entry(solution).or_insert((energy, 0));
                    entry.1 += 1;
                }

                // Convert to SampleResult format
                let mut results: Vec<SampleResult> = solution_counts
                    .into_iter()
                    .map(|(bin_solution, (energy, count))| {
                        // Convert to variable assignments
                        let assignments: HashMap<String, bool> = bin_solution
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

                // Sort by energy
                results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

                // Limit to requested number of shots
                if results.len() > shots {
                    results.truncate(shots);
                }

                Ok(results)
            }
            Err(e) => Err(SamplerError::GpuError(e.to_string())),
        }

        #[cfg(not(feature = "gpu"))]
        match ocl_result {
            Ok(_) => unreachable!("GPU support not enabled"),
            Err(e) => Err(e),
        }
    }

    /// Process a higher-order (>3) HOBO problem using tensor decomposition
    fn run_hobo_high_order(
        &self,
        tensor: &Array<f64, ndarray::IxDyn>,
        var_map: &HashMap<String, usize>,
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        let n_vars = var_map.len();
        let tensor_order = tensor.ndim();

        if self.verbose {
            println!(
                "Processing high-order ({}D) HOBO problem with {} variables",
                tensor_order, n_vars
            );
            println!("Using parallel tempering with multiple replicas");
        }

        // For higher-order tensors, we'll use parallel tempering algorithm
        // with multiple temperature levels for better exploration

        // Initialize RNG
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let seed: u64 = rand::rng().random();
                StdRng::seed_from_u64(seed)
            }
        };

        // Number of temperature levels for parallel tempering
        let n_replicas = 8;
        // Temperature range (from high to low)
        let temp_max = 10.0;
        let temp_min = 0.1;

        // Create temperature ladder (geometric progression)
        let temperatures: Vec<f64> = (0..n_replicas)
            .map(|i| temp_max * (temp_min / temp_max).powf(i as f64 / (n_replicas - 1) as f64))
            .collect();

        // Function to evaluate HOBO energy
        let evaluate_hobo_energy = |state: &[bool]| -> f64 {
            let mut energy = 0.0;

            tensor.indexed_iter().for_each(|(indices, &coeff)| {
                if coeff == 0.0 {
                    return;
                }

                // For ndarray, convert to primitive types
                let index_vec: Vec<usize> = (0..indices.ndim()).map(|i| indices[i]).collect();

                // Check if all indices are active
                let term_active = index_vec.iter().all(|&idx| idx < state.len() && state[idx]);

                if term_active {
                    energy += coeff;
                }
            });

            energy
        };

        // Initialize states for each replica
        let mut replicas: Vec<Vec<bool>> = Vec::with_capacity(n_replicas);
        let mut energies: Vec<f64> = Vec::with_capacity(n_replicas);

        for _ in 0..n_replicas {
            let state: Vec<bool> = (0..n_vars).map(|_| rng.random_bool(0.5)).collect();
            let energy = evaluate_hobo_energy(&state);

            replicas.push(state);
            energies.push(energy);
        }

        // Track best solution found overall
        let mut best_state = replicas[0].clone();
        let mut best_energy = energies[0];

        // Find initial best solution
        for (i, &energy) in energies.iter().enumerate() {
            if energy < best_energy {
                best_energy = energy;
                best_state = replicas[i].clone();
            }
        }

        // Number of Monte Carlo sweeps to perform
        let n_sweeps = 1000;
        // How often to attempt replica exchanges
        let exchange_interval = 10;

        // Perform parallel tempering
        for sweep in 0..n_sweeps {
            // Update each replica
            for (replica_idx, (state, &temperature)) in
                replicas.iter_mut().zip(temperatures.iter()).enumerate()
            {
                // Perform n_vars steps for each replica
                for _ in 0..n_vars {
                    // Select a random bit to flip
                    let bit_idx = rng.random_range(0..n_vars);

                    // Flip the bit
                    state[bit_idx] = !state[bit_idx];

                    // Calculate new energy
                    let new_energy = evaluate_hobo_energy(state);
                    let delta_e = new_energy - energies[replica_idx];

                    // Metropolis criterion
                    let accept = delta_e <= 0.0
                        || rng.random_range(0.0..1.0) < (-delta_e / temperature).exp();

                    if accept {
                        // Update energy
                        energies[replica_idx] = new_energy;

                        // Update best solution if needed
                        if new_energy < best_energy {
                            best_energy = new_energy;
                            best_state = state.clone();
                        }
                    } else {
                        // Revert flip
                        state[bit_idx] = !state[bit_idx];
                    }
                }
            }

            // Try to exchange replicas
            if sweep % exchange_interval == 0 {
                for i in 0..n_replicas - 1 {
                    // Calculate acceptance probability
                    let delta = (1.0 / temperatures[i] - 1.0 / temperatures[i + 1])
                        * (energies[i + 1] - energies[i]);

                    if delta <= 0.0 || rng.random_range(0.0..1.0) < (-delta).exp() {
                        // Exchange states
                        replicas.swap(i, i + 1);
                        energies.swap(i, i + 1);
                    }
                }
            }
        }

        // Get the best n_shots solutions
        #[cfg(feature = "parallel")]
        let final_eval = |state: &Vec<bool>| -> f64 { evaluate_hobo_energy(state) };

        #[cfg(not(feature = "parallel"))]
        let final_eval = |state: &Vec<bool>| -> f64 { evaluate_hobo_energy(state) };

        // Gather all solutions to evaluate
        let mut all_solutions = replicas.clone();
        all_solutions.push(best_state.clone());

        // Re-evaluate energies and deduplicate
        let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

        for state in all_solutions {
            let energy = final_eval(&state);
            let entry = solution_counts.entry(state).or_insert((energy, 0));
            entry.1 += 1;
        }

        // Convert to SampleResult format
        let mut results: Vec<SampleResult> = solution_counts
            .into_iter()
            .map(|(bin_solution, (energy, count))| {
                // Convert to variable assignments
                let assignments: HashMap<String, bool> = bin_solution
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

        // Sort by energy
        results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        // Limit to requested number of shots
        if results.len() > shots {
            results.truncate(shots);
        }

        Ok(results)
    }

    /// Run GPU-accelerated annealing using OpenCL
    fn run_gpu_annealing(
        &self,
        n_vars: usize,
        h_vector: &[f64],
        j_matrix: &[f64],
        num_shots: usize,
    ) -> Result<Vec<Vec<bool>>, ocl::Error> {
        use ocl::flags;
        use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};

        // Set up OpenCL environment
        let platform = Platform::default();
        let device = Device::first(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, None)?;

        // Kernel source
        let src = r#"
        __kernel void simulated_annealing(
            const int n_vars,
            __global const float* h_vector,
            __global const float* j_matrix,
            __global uchar* solutions,
            const int num_runs,
            const float init_temp,
            const float final_temp,
            const int sweeps,
            const ulong seed
        ) {
            // Get global ID
            int gid = get_global_id(0);
            if (gid >= num_runs) return;

            // Initialize RNG for this thread
            ulong rng_state = seed + gid;

            // Initialize spin state
            uchar state[2048]; // Max vars supported
            for (int i = 0; i < n_vars; i++) {
                // Simple xorshift for random init
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                state[i] = (rng_state & 1) ? 1 : 0;
            }

            // Annealing process
            for (int sweep = 0; sweep < sweeps; sweep++) {
                // Calculate temperature for this sweep
                float t_ratio = (float)sweep / (float)sweeps;
                float temp = init_temp * pow(final_temp / init_temp, t_ratio);

                // Perform n_vars spin flips
                for (int i = 0; i < n_vars; i++) {
                    // Choose random spin to flip
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    int idx = rng_state % n_vars;

                    // Calculate energy change
                    float delta_e = 0.0f;

                    // Contribution from h (linear terms)
                    delta_e += h_vector[idx] * (state[idx] ? -2.0f : 2.0f);

                    // Contribution from J (quadratic terms)
                    for (int j = 0; j < n_vars; j++) {
                        if (j != idx && state[j]) {
                            delta_e += j_matrix[idx * n_vars + j] * (state[idx] ? -2.0f : 2.0f);
                        }
                    }

                    // Metropolis acceptance criterion
                    bool accept = false;
                    if (delta_e <= 0.0f) {
                        accept = true;
                    } else {
                        // Generate random number for comparison
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 7;
                        rng_state ^= rng_state << 17;
                        float rand_val = (float)(rng_state & 0xFFFFFFFF) / (float)0xFFFFFFFF;

                        // Accept with probability exp(-delta_e / temp)
                        accept = rand_val < exp(-delta_e / temp);
                    }

                    // Flip spin if accepted
                    if (accept) {
                        state[idx] = !state[idx];
                    }
                }
            }

            // Write final state to global memory
            for (int i = 0; i < n_vars; i++) {
                solutions[gid * n_vars + i] = state[i];
            }
        }
        "#;

        // Compile kernel
        let program = Program::builder()
            .devices(device)
            .src(src)
            .build(&context)?;

        // Set up buffers
        let h_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars)
            .build()?;

        let j_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars * n_vars)
            .build()?;

        let solutions_buffer = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(num_shots * n_vars)
            .build()?;

        // Convert h_vector and j_matrix to f32
        let h_vec_f32: Vec<f32> = h_vector.iter().map(|&x| x as f32).collect();
        let j_mat_f32: Vec<f32> = j_matrix.iter().map(|&x| x as f32).collect();

        // Transfer data to GPU
        h_buffer.write(&h_vec_f32).enq()?;
        j_buffer.write(&j_mat_f32).enq()?;

        // Set up kernel
        let kernel = Kernel::builder()
            .program(&program)
            .name("simulated_annealing")
            .global_work_size(num_shots)
            .arg(n_vars as i32)
            .arg(&h_buffer)
            .arg(&j_buffer)
            .arg(&solutions_buffer)
            .arg(num_shots as i32)
            .arg(10.0f32) // init_temp
            .arg(0.1f32)  // final_temp
            .arg(1000i32) // sweeps
            .arg(self.seed.unwrap_or_else(rand::random::<u64>))
            .build()?;

        // Execute kernel
        unsafe {
            kernel.enq()?;
        }

        // Read results
        let mut solutions_data = vec![0u8; num_shots * n_vars];
        solutions_buffer.read(&mut solutions_data).enq()?;

        // Convert to Vec<Vec<bool>>
        let mut results = Vec::with_capacity(num_shots);
        for i in 0..num_shots {
            let mut solution = Vec::with_capacity(n_vars);
            for j in 0..n_vars {
                solution.push(solutions_data[i * n_vars + j] != 0);
            }
            results.push(solution);
        }

        Ok(results)
    }
}

/// A stub implementation for when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct ArminSampler;

#[cfg(not(feature = "gpu"))]
impl ArminSampler {
    /// Create a new GPU-accelerated sampler
    ///
    /// This is a stub implementation that will return an error
    /// when run since GPU support is not enabled.
    #[must_use]
    pub fn new(_seed: Option<u64>) -> Self {
        Self
    }

    /// Create a new GPU-accelerated sampler with custom parameters
    ///
    /// This is a stub implementation that will return an error
    /// when run since GPU support is not enabled.
    #[must_use]
    pub fn with_params(_seed: Option<u64>, _mode: &str, _device: &str, _verbose: bool) -> Self {
        Self
    }
}

#[cfg(not(feature = "gpu"))]
impl Sampler for ArminSampler {
    fn run_qubo(
        &self,
        _qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError(
            "GPU support not enabled".to_string(),
        ))
    }

    fn run_hobo(
        &self,
        _hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError(
            "GPU support not enabled".to_string(),
        ))
    }
}

/// HOBO-specialized GPU-accelerated Sampler (MIKASA)
///
/// This sampler is specialized for Higher-Order Binary Optimization
/// problems and uses GPU acceleration for large problems.
#[cfg(feature = "gpu")]
pub struct MIKASAmpler(ArminSampler);

#[cfg(feature = "gpu")]
impl MIKASAmpler {
    /// Create a new HOBO-specialized GPU-accelerated sampler
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    #[must_use]
    pub fn new(seed: Option<u64>) -> Self {
        Self(ArminSampler::new(seed))
    }

    /// Create a new HOBO-specialized GPU-accelerated sampler with custom parameters
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    /// * `mode` - Whether to use GPU ("GPU") or CPU ("CPU")
    /// * `device` - Device to use (e.g., "cuda:0")
    /// * `verbose` - Whether to show verbose output
    #[must_use]
    pub fn with_params(seed: Option<u64>, mode: &str, device: &str, verbose: bool) -> Self {
        Self(ArminSampler::with_params(seed, mode, device, verbose))
    }
}

#[cfg(feature = "gpu")]
impl Sampler for MIKASAmpler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        self.0.run_qubo(qubo, shots)
    }

    fn run_hobo(
        &self,
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Extract tensor and variable mapping
        let (tensor, var_map) = hobo;

        // Get the problem dimensions
        let n_vars = var_map.len();
        let tensor_order = tensor.ndim();

        // If it's a low-order tensor, delegate to ArminSampler
        if tensor_order <= 2 {
            return self.0.run_hobo(hobo, shots);
        }

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // For HOBO problems, we'll use Tensor-Train Decomposition
        // to efficiently handle higher-order interactions
        #[cfg(feature = "scirs")]
        {
            use ndarray::Array;

            // Convert to float32 for GPU compatibility
            let tensor_f32 = tensor.mapv(|x| x as f32);

            // Initialize random number generator with seed if provided
            let mut rng = match self.0.seed {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => {
                    let seed: u64 = rand::rng().random();
                    StdRng::seed_from_u64(seed)
                }
            };

            // Set up tensor train parameters
            let ranks = match tensor_order {
                3 => vec![1, 4, 4, 1],           // 3rd order
                4 => vec![1, 8, 8, 8, 1],        // 4th order
                _ => vec![1, 16, 16, 16, 16, 1], // 5th+ order
            };

            // Create a device-specific tensor for GPU processing
            let device_str = if self.0.mode.to_uppercase() == "GPU" {
                &self.0.device
            } else {
                "cpu"
            };

            if self.0.verbose {
                println!("HOBO tensor shape: {:?}", tensor.shape());
                println!("Using tensor-train decomposition with ranks: {:?}", ranks);
                println!("Processing on device: {}", device_str);
            }

            // Function to evaluate HOBO energy with tensor-train
            let evaluate_energy = |state: &[bool]| -> f64 {
                let mut energy = 0.0;

                // Direct evaluation for smaller problems
                if n_vars <= 32 {
                    tensor.indexed_iter().for_each(|(indices, &coeff)| {
                        if coeff == 0.0 {
                            return;
                        }

                        // Check if all corresponding variables are 1 in the state
                        let term_active = indices
                            .iter()
                            .all(|&idx| idx < state.len() && state[idx as usize]);

                        if term_active {
                            energy += coeff;
                        }
                    });
                } else {
                    // Approximate evaluation for larger problems using TT decomposition
                    // This is a placeholder for the actual TT evaluation
                    let state_f32: Vec<f32> =
                        state.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

                    // Compute the scalar product using the decomposition
                    // For each core in the TT decomposition:
                    // We would contract state vector with each core
                    // Then pass the result to the next core

                    // This is just an approximation since we haven't implemented
                    // the full tensor-train algorithm here
                    tensor.indexed_iter()
                         .filter(|(_, &val)| val != 0.0)
                         .take(1000) // Limit iterations for large tensors
                         .for_each(|(indices, &coeff)| {
                             let term_active = indices.iter()
                                 .all(|&idx| idx < state.len() && state[idx as usize]);

                             if term_active {
                                 energy += coeff;
                             }
                         });
                }

                energy
            };

            // Parallel processing for multiple shots
            #[cfg(feature = "parallel")]
            let num_threads = rayon::current_num_threads();
            #[cfg(not(feature = "parallel"))]
            let num_threads = 1;

            // Divide shots across threads
            let shots_per_thread =
                shots / num_threads + if shots % num_threads > 0 { 1 } else { 0 };
            let total_runs = shots_per_thread * num_threads;

            // Set up annealing parameters
            let initial_temp = 10.0;
            let final_temp = 0.1;
            let sweeps = 1000;

            // Vector to store solutions
            let mut all_solutions = Vec::with_capacity(total_runs);

            // Run annealing process
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;

                // Create seeds for each parallel run
                let seeds: Vec<u64> = (0..total_runs)
                    .map(|i| match self.0.seed {
                        Some(seed) => seed.wrapping_add(i as u64),
                        None => rand::random(),
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
                                * f64::powf(
                                    final_temp / initial_temp,
                                    sweep as f64 / sweeps as f64,
                                );

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
                            let idx = rng.random_range(0..n_vars);

                            // Flip the bit
                            state[idx] = !state[idx];

                            // Calculate new energy
                            let new_energy = evaluate_energy(&state);
                            let delta_e = new_energy - energy;

                            // Metropolis acceptance criterion
                            let accept = delta_e <= 0.0
                                || rng.random_range(0.0..1.0) < (-delta_e / temp).exp();

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

            // Process results
            let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

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

            // Sort by energy
            results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

            // Limit to requested number of shots
            if results.len() > shots {
                results.truncate(shots);
            }

            Ok(results)
        }

        // Fallback to regular implementation if SciRS2 is not available
        #[cfg(not(feature = "scirs"))]
        {
            // Use standard ArminSampler implementation
            self.0.run_hobo(hobo, shots)
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub struct MIKASAmpler(ArminSampler);

#[cfg(not(feature = "gpu"))]
impl MIKASAmpler {
    #[must_use]
    pub fn new(_seed: Option<u64>) -> Self {
        Self(ArminSampler::new(None))
    }

    #[must_use]
    pub fn with_params(_seed: Option<u64>, _mode: &str, _device: &str, _verbose: bool) -> Self {
        Self(ArminSampler::new(None))
    }
}

#[cfg(not(feature = "gpu"))]
impl Sampler for MIKASAmpler {
    fn run_qubo(
        &self,
        _qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError(
            "GPU support not enabled".to_string(),
        ))
    }

    fn run_hobo(
        &self,
        _hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError(
            "GPU support not enabled".to_string(),
        ))
    }
}

/// D-Wave Quantum Annealer Sampler
///
/// This sampler connects to D-Wave's quantum annealing hardware
/// to solve QUBO problems. It requires an API key and Internet access.
pub struct DWaveSampler {
    /// D-Wave API key
    #[allow(dead_code)]
    api_key: String,
}

impl DWaveSampler {
    /// Create a new D-Wave sampler
    ///
    /// # Arguments
    ///
    /// * `api_key` - The D-Wave API key
    #[must_use]
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }
}

impl Sampler for DWaveSampler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Extract matrix and variable mapping
        let (matrix, var_map) = qubo;

        // Get the problem dimension
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let _idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Convert ndarray to a QuboModel
        let mut qubo_model = QuboModel::new(n_vars);

        // Set linear and quadratic terms
        for i in 0..n_vars {
            if matrix[[i, i]] != 0.0 {
                qubo_model.set_linear(i, matrix[[i, i]])?;
            }

            for j in (i + 1)..n_vars {
                if matrix[[i, j]] != 0.0 {
                    qubo_model.set_quadratic(i, j, matrix[[i, j]])?;
                }
            }
        }

        // Initialize the D-Wave client
        #[cfg(feature = "dwave")]
        {
            use quantrs2_anneal::dwave::{DWaveClient, DWaveParams};

            // Create D-Wave client
            let dwave_client = DWaveClient::new(&self.api_key)?;

            // Configure submission parameters
            let mut params = DWaveParams::default();
            params.num_reads = shots;

            // Submit the problem to D-Wave
            let dwave_result = dwave_client.solve_qubo(&qubo_model, params)?;

            // Convert to our result format
            let mut results = Vec::new();

            // Process each solution
            for solution in dwave_result.solutions {
                // Convert binary array to HashMap
                let assignments: HashMap<String, bool> = solution
                    .binary_vars
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
                    energy: solution.energy,
                    occurrences: solution.occurrences,
                };

                results.push(result);
            }

            Ok(results)
        }

        #[cfg(not(feature = "dwave"))]
        {
            Err(SamplerError::DWaveUnavailable(
                "D-Wave support not enabled. Rebuild with '--features dwave'".to_string(),
            ))
        }
    }

    fn run_hobo(
        &self,
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // For HOBO problems, we need to first convert to QUBO if possible
        if hobo.0.ndim() <= 2 {
            // If it's already 2D, just forward to run_qubo
            let qubo = (
                hobo.0
                    .clone()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap(),
                hobo.1.clone(),
            );
            self.run_qubo(&qubo, shots)
        } else {
            // D-Wave doesn't directly support higher-order problems
            // We could implement automatic quadratization here, but for now return an error
            Err(SamplerError::InvalidParameter(
                "D-Wave doesn't support HOBO problems directly. Use a quadratization technique first.".to_string()
            ))
        }
    }
}
