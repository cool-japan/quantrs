//! Genetic Algorithm Sampler Implementation

use scirs2_core::ndarray::{Array, Dimension, Ix2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use std::collections::HashMap;

use super::{SampleResult, Sampler, SamplerResult};

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
    /// Create a new Genetic Algorithm sampler
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    #[must_use]
    pub const fn new(seed: Option<u64>) -> Self {
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
    pub const fn with_params(
        seed: Option<u64>,
        max_generations: usize,
        population_size: usize,
    ) -> Self {
        Self {
            seed,
            max_generations,
            population_size,
        }
    }

    /// Set population size
    pub const fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set elite fraction (placeholder method)
    pub const fn with_elite_fraction(self, _fraction: f64) -> Self {
        // Note: Elite fraction not currently implemented in struct
        // This is a placeholder to satisfy compilation
        self
    }

    /// Set mutation rate (placeholder method)
    pub const fn with_mutation_rate(self, _rate: f64) -> Self {
        // Note: Mutation rate not currently implemented in struct
        // This is a placeholder to satisfy compilation
        self
    }

    /// Create a new enhanced Genetic Algorithm sampler
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    /// * `max_generations` - Maximum number of generations to evolve
    /// * `population_size` - Size of the population
    /// * `crossover` - Crossover strategy to use
    /// * `mutation` - Mutation strategy to use
    pub const fn with_advanced_params(
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
                    if rng.gen_bool(0.5) {
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
                let crossover_point = rng.gen_range(1..n_vars);

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
                let point1 = rng.gen_range(1..(n_vars - 1));
                let point2 = rng.gen_range((point1 + 1)..n_vars);

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
                        if rng.gen_bool(0.5) {
                            child1[i] = parent1[i];
                            child2[i] = parent2[i];
                        } else {
                            child1[i] = parent2[i];
                            child2[i] = parent1[i];
                        }
                    }
                } else if similarity > 0.4 {
                    // Moderate similarity - use two-point
                    let point1 = rng.gen_range(1..(n_vars - 1));
                    let point2 = rng.gen_range((point1 + 1)..n_vars);

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
                    let crossover_point = rng.gen_range(1..n_vars);

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
                    if rng.gen_bool(rate) {
                        *bit = !*bit;
                    }
                }
            }
            MutationStrategy::Annealing(initial_rate, final_rate) => {
                // Annealing mutation (decreasing rate)
                let progress = generation as f64 / max_generations as f64;
                let current_rate = (final_rate - initial_rate).mul_add(progress, initial_rate);

                for bit in individual.iter_mut() {
                    if rng.gen_bool(current_rate) {
                        *bit = !*bit;
                    }
                }
            }
            MutationStrategy::Adaptive(min_rate, max_rate) => {
                // Adaptive mutation based on diversity
                if let Some(diversity) = diversity {
                    // High diversity -> low mutation rate, low diversity -> high mutation rate
                    let rate = (max_rate - min_rate).mul_add(1.0 - diversity, min_rate);

                    for bit in individual.iter_mut() {
                        if rng.gen_bool(rate) {
                            *bit = !*bit;
                        }
                    }
                } else {
                    // Default to average if no diversity metric available
                    let rate = f64::midpoint(min_rate, max_rate);
                    for bit in individual.iter_mut() {
                        if rng.gen_bool(rate) {
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
        hobo: &(
            Array<f64, scirs2_core::ndarray::IxDyn>,
            HashMap<String, usize>,
        ),
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
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let seed: u64 = thread_rng().random();
            StdRng::seed_from_u64(seed)
        };

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
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
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
            .map(|_| (0..n_vars).map(|_| rng.gen_bool(0.5)).collect())
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
        qubo: &(
            Array<f64, scirs2_core::ndarray::Ix2>,
            HashMap<String, usize>,
        ),
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
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let seed: u64 = thread_rng().random();
            StdRng::seed_from_u64(seed)
        };

        // Handle edge cases
        if self.population_size <= 2 || n_vars == 0 {
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

        // Use adaptive strategies by default
        let crossover_strategy = CrossoverStrategy::Adaptive;
        let mutation_strategy = MutationStrategy::Annealing(0.1, 0.01);
        let selection_pressure = 3; // Tournament size
        let use_elitism = true;

        // Initialize population with random bitstrings
        let mut population: Vec<Vec<bool>> = (0..self.population_size)
            .map(|_| (0..n_vars).map(|_| rng.gen_bool(0.5)).collect())
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
                let child1_fitness = calculate_energy(&child1, matrix);
                let child2_fitness = calculate_energy(&child2, matrix);

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
}

// Helper function to calculate energy for a solution
fn calculate_energy(solution: &[bool], matrix: &Array<f64, Ix2>) -> f64 {
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
        rng.gen_range(1..n_vars)
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
        if rng.gen_bool(rate) {
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

    let mut best_idx = rng.gen_range(0..fitness.len());
    let mut best_fitness = fitness[best_idx];

    for _ in 1..(effective_tournament_size) {
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
