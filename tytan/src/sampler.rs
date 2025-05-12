//! Samplers for solving QUBO/HOBO problems.
//!
//! This module provides various samplers (solvers) for QUBO and HOBO
//! problems, including simulated annealing, genetic algorithms, and
//! specialized hardware samplers.

use std::collections::HashMap;
use ndarray::Array;
use thiserror::Error;
use rand::prelude::*;
use rand::rngs::StdRng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "gpu")]
use ocl;

use quantrs_anneal::{
    simulator::{AnnealingParams, AnnealingError, ClassicalAnnealingSimulator, AnnealingSolution},
    QuboModel, IsingModel, IsingError,
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
    #[cfg(feature = "gpu")]
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
    DWaveError(#[from] quantrs_anneal::dwave::DWaveError),
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
    fn run_qubo(&self, 
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>), 
        shots: usize
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
    fn run_hobo(&self, 
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>), 
        shots: usize
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
    pub fn new(seed: Option<u64>) -> Self {
        // Create default annealing parameters
        let mut params = AnnealingParams::default();
        
        // Customize based on seed
        if let Some(seed) = seed {
            params.random_seed = Some(seed);
        }
        
        Self {
            seed,
            params,
        }
    }
    
    /// Create a new Simulated Annealing sampler with custom parameters
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    /// * `params` - Custom annealing parameters
    pub fn with_params(seed: Option<u64>, params: AnnealingParams) -> Self {
        let mut params = params;
        
        // Override seed if provided
        if let Some(seed) = seed {
            params.random_seed = Some(seed);
        }
        
        Self {
            seed,
            params,
        }
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
        shots: usize
    ) -> SamplerResult<Vec<SampleResult>>
    where
        D: ndarray::Dimension,
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
                if matrix_or_tensor[[i, i]] != 0.0 {
                    qubo.set_linear(i, matrix_or_tensor[[i, i]])?;
                }

                for j in (i+1)..n_vars {
                    if matrix_or_tensor[[i, j]] != 0.0 {
                        qubo.set_quadratic(i, j, matrix_or_tensor[[i, j]])?;
                    }
                }
            }

            // Configure annealing parameters
            let mut params = self.params.clone();
            params.num_repetitions = shots;

            // Create annealing simulator
            let simulator = ClassicalAnnealingSimulator::new(params)?;

            // Solve the problem
            let annealing_result = simulator.solve_qubo(&qubo)?;

            // Convert to our result format
            let mut results = Vec::new();

            // Extract solutions and energies
            for solution in annealing_result.solutions {
                // Convert binary array to HashMap
                let assignments: HashMap<String, bool> = solution.binary_vars
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

            return Ok(results);
        } else {
            // For higher-order tensors, we need a more specialized algorithm
            // TODO: Implement tensor-based annealing for HOBO

            // For now, return a placeholder
            return Err(SamplerError::InvalidParameter(
                "Higher-order tensor sampling not yet implemented".to_string()
            ));
        }
    }
}

impl Sampler for SASampler {
    fn run_qubo(&self, 
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>), 
        shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        self.run_generic(&qubo.0, &qubo.1, shots)
    }
    
    fn run_hobo(&self, 
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>), 
        shots: usize
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
    pub fn with_params(seed: Option<u64>, max_generations: usize, population_size: usize) -> Self {
        Self {
            seed,
            max_generations,
            population_size,
        }
    }
}

impl Sampler for GASampler {
    fn run_qubo(&self,
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        shots: usize
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
            None => StdRng::from_entropy(),
        };

        // Initialize population with random bitstrings
        let mut population: Vec<Vec<bool>> = (0..self.population_size)
            .map(|_| (0..n_vars).map(|_| rng.gen_bool(0.5)).collect())
            .collect();

        // Initialize fitness scores (energy values)
        let mut fitness: Vec<f64> = population.iter()
            .map(|indiv| calculate_energy(indiv, matrix))
            .collect();

        // Track the best solution seen
        let mut best_solutions: Vec<(Vec<bool>, f64, usize)> = Vec::new();
        let mut solution_counts: HashMap<Vec<bool>, usize> = HashMap::new();

        // Main GA loop
        for _gen in 0..self.max_generations {
            // Tournament selection for parents
            let parent1_idx = tournament_selection(&fitness, 3, &mut rng);
            let parent2_idx = tournament_selection(&fitness, 3, &mut rng);

            let parent1 = &population[parent1_idx];
            let parent2 = &population[parent2_idx];

            // Crossover (uniform)
            let mut child1 = Vec::with_capacity(n_vars);
            let mut child2 = Vec::with_capacity(n_vars);

            for i in 0..n_vars {
                if rng.gen_bool(0.5) {
                    child1.push(parent1[i]);
                    child2.push(parent2[i]);
                } else {
                    child1.push(parent2[i]);
                    child2.push(parent1[i]);
                }
            }

            // Mutation (flip bits with low probability)
            for bit in &mut child1 {
                if rng.gen_bool(0.05) { // 5% mutation rate
                    *bit = !*bit;
                }
            }

            for bit in &mut child2 {
                if rng.gen_bool(0.05) { // 5% mutation rate
                    *bit = !*bit;
                }
            }

            // Calculate fitness of new children
            let child1_fitness = calculate_energy(&child1, matrix);
            let child2_fitness = calculate_energy(&child2, matrix);

            // Replace worst individuals in population (steady-state GA)
            if let Some(worst_idx) = fitness.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx) {
                population[worst_idx] = child1;
                fitness[worst_idx] = child1_fitness;
            }

            if let Some(second_worst_idx) = fitness.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx) {
                population[second_worst_idx] = child2;
                fitness[second_worst_idx] = child2_fitness;
            }

            // Update best solutions
            for (indiv, energy) in population.iter().zip(fitness.iter()) {
                *solution_counts.entry(indiv.clone()).or_insert(0) += 1;
            }
        }

        // Collect results
        let mut results = Vec::new();

        // Convert the best solutions to SampleResult format
        for (solution, count) in solution_counts.iter() {
            // Only include solutions that appeared multiple times
            if *count < 2 {
                continue;
            }

            // Calculate energy one more time
            let energy = calculate_energy(solution, matrix);

            // Convert to variable assignments
            let assignments: HashMap<String, bool> = solution.iter()
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

    fn run_hobo(&self,
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        // For HOBO, we currently only support conversion to QUBO
        // with compatibility for 2nd order terms only
        if hobo.0.ndim() <= 2 {
            // Just forward to run_qubo
            let qubo = (hobo.0.clone().into_dimensionality::<ndarray::Ix2>().unwrap(), hobo.1.clone());
            self.run_qubo(&qubo, shots)
        } else {
            Err(SamplerError::InvalidParameter(
                "Higher-order optimization not yet implemented for genetic algorithm".to_string()
            ))
        }
    }
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
            for j in (i+1)..n {
                if solution[j] {
                    energy += matrix[[i, j]];
                }
            }
        }
    }

    energy
}

// Helper function for tournament selection
fn tournament_selection(fitness: &[f64], tournament_size: usize, rng: &mut impl Rng) -> usize {
    let mut best_idx = rng.gen_range(0..fitness.len());
    let mut best_fitness = fitness[best_idx];

    for _ in 1..tournament_size {
        let candidate_idx = rng.gen_range(0..fitness.len());
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
    pub fn with_params(seed: Option<u64>, mode: &str, device: &str, verbose: bool) -> Self {
        Self {
            seed,
            mode: mode.to_string(),
            device: device.to_string(),
            verbose,
        }
    }
}

#[cfg(feature = "gpu")]
impl Sampler for ArminSampler {
    fn run_qubo(&self, 
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>), 
        shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        // TODO: Implement the GPU-accelerated sampler for QUBO
        
        // For now, return a placeholder
        let result = Vec::new();
        Ok(result)
    }
    
    fn run_hobo(&self, 
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>), 
        shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        // TODO: Implement the GPU-accelerated sampler for HOBO
        
        // For now, return a placeholder
        let result = Vec::new();
        Ok(result)
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
    pub fn new(_seed: Option<u64>) -> Self {
        Self
    }
    
    /// Create a new GPU-accelerated sampler with custom parameters
    ///
    /// This is a stub implementation that will return an error
    /// when run since GPU support is not enabled.
    pub fn with_params(_seed: Option<u64>, _mode: &str, _device: &str, _verbose: bool) -> Self {
        Self
    }
}

#[cfg(not(feature = "gpu"))]
impl Sampler for ArminSampler {
    fn run_qubo(&self, 
        _qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>), 
        _shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError("GPU support not enabled".to_string()))
    }
    
    fn run_hobo(&self, 
        _hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>), 
        _shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError("GPU support not enabled".to_string()))
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
    pub fn with_params(seed: Option<u64>, mode: &str, device: &str, verbose: bool) -> Self {
        Self(ArminSampler::with_params(seed, mode, device, verbose))
    }
}

#[cfg(feature = "gpu")]
impl Sampler for MIKASAmpler {
    fn run_qubo(&self, 
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>), 
        shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        self.0.run_qubo(qubo, shots)
    }
    
    fn run_hobo(&self, 
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>), 
        shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        // This has specialized optimizations for HOBO problems
        // For now, just delegate to ArminSampler
        self.0.run_hobo(hobo, shots)
    }
}

#[cfg(not(feature = "gpu"))]
pub struct MIKASAmpler(ArminSampler);

#[cfg(not(feature = "gpu"))]
impl MIKASAmpler {
    pub fn new(_seed: Option<u64>) -> Self {
        Self(ArminSampler::new(None))
    }
    
    pub fn with_params(_seed: Option<u64>, _mode: &str, _device: &str, _verbose: bool) -> Self {
        Self(ArminSampler::new(None))
    }
}

#[cfg(not(feature = "gpu"))]
impl Sampler for MIKASAmpler {
    fn run_qubo(&self, 
        _qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>), 
        _shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError("GPU support not enabled".to_string()))
    }
    
    fn run_hobo(&self, 
        _hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>), 
        _shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError("GPU support not enabled".to_string()))
    }
}

/// D-Wave Quantum Annealer Sampler
///
/// This sampler connects to D-Wave's quantum annealing hardware
/// to solve QUBO problems. It requires an API key and Internet access.
pub struct DWaveSampler {
    /// D-Wave API key
    api_key: String,
}

impl DWaveSampler {
    /// Create a new D-Wave sampler
    ///
    /// # Arguments
    ///
    /// * `api_key` - The D-Wave API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }
}

impl Sampler for DWaveSampler {
    fn run_qubo(&self,
        qubo: &(Array<f64, ndarray::Ix2>, HashMap<String, usize>),
        shots: usize
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

        // Convert ndarray to a QuboModel
        let mut qubo_model = QuboModel::new(n_vars);

        // Set linear and quadratic terms
        for i in 0..n_vars {
            if matrix[[i, i]] != 0.0 {
                qubo_model.set_linear(i, matrix[[i, i]])?;
            }

            for j in (i+1)..n_vars {
                if matrix[[i, j]] != 0.0 {
                    qubo_model.set_quadratic(i, j, matrix[[i, j]])?;
                }
            }
        }

        // Initialize the D-Wave client
        #[cfg(feature = "dwave")]
        {
            use quantrs_anneal::dwave::{DWaveClient, DWaveParams};

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
                let assignments: HashMap<String, bool> = solution.binary_vars
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
            Err(SamplerError::DWaveUnavailable("D-Wave support not enabled. Rebuild with '--features dwave'".to_string()))
        }
    }

    fn run_hobo(&self,
        hobo: &(Array<f64, ndarray::IxDyn>, HashMap<String, usize>),
        shots: usize
    ) -> SamplerResult<Vec<SampleResult>> {
        // For HOBO problems, we need to first convert to QUBO if possible
        if hobo.0.ndim() <= 2 {
            // If it's already 2D, just forward to run_qubo
            let qubo = (hobo.0.clone().into_dimensionality::<ndarray::Ix2>().unwrap(), hobo.1.clone());
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