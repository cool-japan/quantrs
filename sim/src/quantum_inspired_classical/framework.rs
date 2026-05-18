//! Quantum-inspired framework implementation

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use super::types::*;

impl QuantumInspiredFramework {
    /// Create a new quantum-inspired framework
    pub fn new(config: QuantumInspiredConfig) -> Result<Self> {
        let state = QuantumInspiredState {
            variables: Array1::zeros(config.num_variables),
            objective_value: f64::INFINITY,
            iteration: 0,
            best_solution: Array1::zeros(config.num_variables),
            best_objective: f64::INFINITY,
            convergence_history: Vec::new(),
            runtime_stats: RuntimeStats::default(),
        };
        let rng = thread_rng();
        Ok(Self {
            config,
            state,
            backend: None,
            stats: QuantumInspiredStats::default(),
            rng: Arc::new(Mutex::new(rng)),
        })
    }
    /// Set `SciRS2` backend for numerical operations
    pub fn set_backend(&mut self, backend: SciRS2Backend) {
        self.backend = Some(backend);
    }
    /// Run optimization algorithm
    pub fn optimize(&mut self) -> Result<OptimizationResult> {
        let start_time = std::time::Instant::now();
        match self.config.optimization_config.algorithm_type {
            OptimizationAlgorithm::QuantumGeneticAlgorithm => self.quantum_genetic_algorithm(),
            OptimizationAlgorithm::QuantumParticleSwarm => {
                self.quantum_particle_swarm_optimization()
            }
            OptimizationAlgorithm::QuantumSimulatedAnnealing => self.quantum_simulated_annealing(),
            OptimizationAlgorithm::QuantumDifferentialEvolution => {
                self.quantum_differential_evolution()
            }
            OptimizationAlgorithm::ClassicalQAOA => self.classical_qaoa_simulation(),
            OptimizationAlgorithm::ClassicalVQE => self.classical_vqe_simulation(),
            OptimizationAlgorithm::QuantumAntColony => self.quantum_ant_colony_optimization(),
            OptimizationAlgorithm::QuantumHarmonySearch => self.quantum_harmony_search(),
        }
    }
    /// Quantum-inspired genetic algorithm
    pub(super) fn quantum_genetic_algorithm(&mut self) -> Result<OptimizationResult> {
        let pop_size = self.config.algorithm_config.population_size;
        let num_vars = self.config.num_variables;
        let max_iterations = self.config.algorithm_config.max_iterations;
        let mut population = self.initialize_quantum_population(pop_size, num_vars)?;
        let mut fitness_values = vec![0.0; pop_size];
        for (i, individual) in population.iter().enumerate() {
            fitness_values[i] = self.evaluate_objective(individual)?;
            self.state.runtime_stats.function_evaluations += 1;
        }
        for generation in 0..max_iterations {
            self.state.iteration = generation;
            let parents = self.quantum_selection(&population, &fitness_values)?;
            let mut offspring = self.quantum_crossover(&parents)?;
            self.quantum_mutation(&mut offspring)?;
            let mut offspring_fitness = vec![0.0; offspring.len()];
            for (i, individual) in offspring.iter().enumerate() {
                offspring_fitness[i] = self.evaluate_objective(individual)?;
                self.state.runtime_stats.function_evaluations += 1;
            }
            self.quantum_replacement(
                &mut population,
                &mut fitness_values,
                offspring,
                offspring_fitness,
            )?;
            let best_idx = fitness_values
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if fitness_values[best_idx] < self.state.best_objective {
                self.state.best_objective = fitness_values[best_idx];
                self.state.best_solution = population[best_idx].clone();
            }
            self.state
                .convergence_history
                .push(self.state.best_objective);
            if self.check_convergence()? {
                break;
            }
        }
        Ok(OptimizationResult {
            solution: self.state.best_solution.clone(),
            objective_value: self.state.best_objective,
            iterations: self.state.iteration,
            converged: self.check_convergence()?,
            runtime_stats: self.state.runtime_stats.clone(),
            metadata: HashMap::new(),
        })
    }
    /// Initialize quantum-inspired population with superposition
    pub(super) fn initialize_quantum_population(
        &self,
        pop_size: usize,
        num_vars: usize,
    ) -> Result<Vec<Array1<f64>>> {
        let mut population = Vec::with_capacity(pop_size);
        let bounds = &self.config.optimization_config.bounds;
        let quantum_params = &self.config.algorithm_config.quantum_parameters;
        for _ in 0..pop_size {
            let mut individual = Array1::zeros(num_vars);
            for j in 0..num_vars {
                let (min_bound, max_bound) = if j < bounds.len() {
                    bounds[j]
                } else {
                    (-1.0, 1.0)
                };
                let mut rng = self.rng.lock().expect("RNG lock poisoned");
                let base_value = rng
                    .random::<f64>()
                    .mul_add(max_bound - min_bound, min_bound);
                let superposition_noise = (rng.random::<f64>() - 0.5)
                    * quantum_params.superposition_strength
                    * (max_bound - min_bound);
                individual[j] = (base_value + superposition_noise).clamp(min_bound, max_bound);
            }
            population.push(individual);
        }
        Ok(population)
    }
    /// Quantum-inspired selection using interference
    pub(super) fn quantum_selection(
        &self,
        population: &[Array1<f64>],
        fitness: &[f64],
    ) -> Result<Vec<Array1<f64>>> {
        let pop_size = population.len();
        let elite_size = (self.config.algorithm_config.elite_ratio * pop_size as f64) as usize;
        let quantum_params = &self.config.algorithm_config.quantum_parameters;
        let mut indexed_fitness: Vec<(usize, f64)> =
            fitness.iter().enumerate().map(|(i, &f)| (i, f)).collect();
        indexed_fitness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut parents = Vec::new();
        for i in 0..elite_size {
            parents.push(population[indexed_fitness[i].0].clone());
        }
        let mut rng = self.rng.lock().expect("RNG lock poisoned");
        while parents.len() < pop_size {
            let tournament_size = 3;
            let mut tournament_indices = Vec::new();
            for _ in 0..tournament_size {
                tournament_indices.push(rng.random_range(0..pop_size));
            }
            let mut selection_probabilities = vec![0.0; tournament_size];
            for (i, &idx) in tournament_indices.iter().enumerate() {
                let normalized_fitness = 1.0 / (1.0 + fitness[idx]);
                let interference_factor = (quantum_params.interference_strength
                    * (i as f64 * PI / tournament_size as f64))
                    .cos()
                    .abs();
                selection_probabilities[i] = normalized_fitness * (1.0 + interference_factor);
            }
            let sum: f64 = selection_probabilities.iter().sum();
            for prob in &mut selection_probabilities {
                *prob /= sum;
            }
            let mut cumulative = 0.0;
            let random_val = rng.random::<f64>();
            for (i, &prob) in selection_probabilities.iter().enumerate() {
                cumulative += prob;
                if random_val <= cumulative {
                    parents.push(population[tournament_indices[i]].clone());
                    break;
                }
            }
        }
        Ok(parents)
    }
    /// Quantum-inspired crossover with entanglement
    pub(super) fn quantum_crossover(&self, parents: &[Array1<f64>]) -> Result<Vec<Array1<f64>>> {
        let mut offspring = Vec::new();
        let crossover_rate = self.config.algorithm_config.crossover_rate;
        let quantum_params = &self.config.algorithm_config.quantum_parameters;
        let mut rng = self.rng.lock().expect("RNG lock poisoned");
        for i in (0..parents.len()).step_by(2) {
            if i + 1 < parents.len() && rng.random::<f64>() < crossover_rate {
                let parent1 = &parents[i];
                let parent2 = &parents[i + 1];
                let mut child1 = parent1.clone();
                let mut child2 = parent2.clone();
                for j in 0..parent1.len() {
                    let entanglement_strength = quantum_params.entanglement_strength;
                    let alpha = rng.random::<f64>();
                    let entangled_val1 = alpha.mul_add(parent1[j], (1.0 - alpha) * parent2[j]);
                    let entangled_val2 = (1.0 - alpha).mul_add(parent1[j], alpha * parent2[j]);
                    let correlation = entanglement_strength
                        * (parent1[j] - parent2[j]).abs()
                        * (rng.random::<f64>() - 0.5);
                    child1[j] = entangled_val1 + correlation;
                    child2[j] = entangled_val2 - correlation;
                }
                offspring.push(child1);
                offspring.push(child2);
            } else {
                offspring.push(parents[i].clone());
                if i + 1 < parents.len() {
                    offspring.push(parents[i + 1].clone());
                }
            }
        }
        Ok(offspring)
    }
    /// Quantum-inspired mutation with tunneling
    pub(super) fn quantum_mutation(&mut self, population: &mut [Array1<f64>]) -> Result<()> {
        let mutation_rate = self.config.algorithm_config.mutation_rate;
        let quantum_params = &self.config.algorithm_config.quantum_parameters;
        let bounds = &self.config.optimization_config.bounds;
        let mut rng = self.rng.lock().expect("RNG lock poisoned");
        for individual in population.iter_mut() {
            for j in 0..individual.len() {
                if rng.random::<f64>() < mutation_rate {
                    let (min_bound, max_bound) = if j < bounds.len() {
                        bounds[j]
                    } else {
                        (-1.0, 1.0)
                    };
                    let current_val = individual[j];
                    let range = max_bound - min_bound;
                    let gaussian_mutation =
                        rng.random::<f64>() * 0.1 * range * (rng.random::<f64>() - 0.5);
                    let tunneling_prob = quantum_params.tunneling_probability;
                    let tunneling_mutation = if rng.random::<f64>() < tunneling_prob {
                        (rng.random::<f64>() - 0.5) * range
                    } else {
                        0.0
                    };
                    individual[j] = (current_val + gaussian_mutation + tunneling_mutation)
                        .clamp(min_bound, max_bound);
                }
            }
        }
        self.state.runtime_stats.quantum_operations += population.len();
        Ok(())
    }
    /// Quantum-inspired replacement using quantum measurement
    pub(super) fn quantum_replacement(
        &self,
        population: &mut Vec<Array1<f64>>,
        fitness: &mut Vec<f64>,
        offspring: Vec<Array1<f64>>,
        offspring_fitness: Vec<f64>,
    ) -> Result<()> {
        let quantum_params = &self.config.algorithm_config.quantum_parameters;
        let measurement_prob = quantum_params.measurement_probability;
        let mut rng = self.rng.lock().expect("RNG lock poisoned");
        let mut combined_population = population.clone();
        combined_population.extend(offspring);
        let mut combined_fitness = fitness.clone();
        combined_fitness.extend(offspring_fitness);
        let pop_size = population.len();
        let mut new_population = Vec::with_capacity(pop_size);
        let mut new_fitness = Vec::with_capacity(pop_size);
        let mut indexed_combined: Vec<(usize, f64)> = combined_fitness
            .iter()
            .enumerate()
            .map(|(i, &f)| (i, f))
            .collect();
        indexed_combined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..pop_size {
            if i < indexed_combined.len() {
                let idx = indexed_combined[i].0;
                let acceptance_prob = if rng.random::<f64>() < measurement_prob {
                    1.0
                } else {
                    1.0 / (1.0 + (i as f64 / pop_size as f64))
                };
                if rng.random::<f64>() < acceptance_prob {
                    new_population.push(combined_population[idx].clone());
                    new_fitness.push(combined_fitness[idx]);
                }
            }
        }
        while new_population.len() < pop_size {
            for i in 0..indexed_combined.len() {
                if new_population.len() >= pop_size {
                    break;
                }
                let idx = indexed_combined[i].0;
                if !new_population.iter().any(|x| {
                    x.iter()
                        .zip(combined_population[idx].iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
                }) {
                    new_population.push(combined_population[idx].clone());
                    new_fitness.push(combined_fitness[idx]);
                }
            }
        }
        new_population.truncate(pop_size);
        new_fitness.truncate(pop_size);
        *population = new_population;
        *fitness = new_fitness;
        Ok(())
    }
    /// Quantum particle swarm optimization
    pub(super) fn quantum_particle_swarm_optimization(&mut self) -> Result<OptimizationResult> {
        let pop_size = self.config.algorithm_config.population_size;
        let num_vars = self.config.num_variables;
        let max_iterations = self.config.algorithm_config.max_iterations;
        let quantum_params = self.config.algorithm_config.quantum_parameters.clone();
        let bounds = self.config.optimization_config.bounds.clone();
        let mut particles = self.initialize_quantum_population(pop_size, num_vars)?;
        let mut velocities: Vec<Array1<f64>> = vec![Array1::zeros(num_vars); pop_size];
        let mut personal_best = particles.clone();
        let mut personal_best_fitness = vec![f64::INFINITY; pop_size];
        let mut global_best = Array1::zeros(num_vars);
        let mut global_best_fitness = f64::INFINITY;
        for (i, particle) in particles.iter().enumerate() {
            let fitness = self.evaluate_objective(particle)?;
            personal_best_fitness[i] = fitness;
            if fitness < global_best_fitness {
                global_best_fitness = fitness;
                global_best = particle.clone();
            }
            self.state.runtime_stats.function_evaluations += 1;
        }
        let w = 0.7;
        let c1 = 2.0;
        let c2 = 2.0;
        for iteration in 0..max_iterations {
            self.state.iteration = iteration;
            for i in 0..pop_size {
                let mut rng = self.rng.lock().expect("RNG lock poisoned");
                for j in 0..num_vars {
                    let r1 = rng.random::<f64>();
                    let r2 = rng.random::<f64>();
                    let cognitive_term = c1 * r1 * (personal_best[i][j] - particles[i][j]);
                    let social_term = c2 * r2 * (global_best[j] - particles[i][j]);
                    let quantum_fluctuation =
                        quantum_params.superposition_strength * (rng.random::<f64>() - 0.5);
                    let quantum_tunneling =
                        if rng.random::<f64>() < quantum_params.tunneling_probability {
                            (rng.random::<f64>() - 0.5) * 2.0
                        } else {
                            0.0
                        };
                    velocities[i][j] = w * velocities[i][j]
                        + cognitive_term
                        + social_term
                        + quantum_fluctuation
                        + quantum_tunneling;
                }
                for j in 0..num_vars {
                    particles[i][j] += velocities[i][j];
                    let (min_bound, max_bound) = if j < bounds.len() {
                        bounds[j]
                    } else {
                        (-10.0, 10.0)
                    };
                    particles[i][j] = particles[i][j].clamp(min_bound, max_bound);
                }
                drop(rng);
                let fitness = self.evaluate_objective(&particles[i])?;
                self.state.runtime_stats.function_evaluations += 1;
                if fitness < personal_best_fitness[i] {
                    personal_best_fitness[i] = fitness;
                    personal_best[i] = particles[i].clone();
                }
                if fitness < global_best_fitness {
                    global_best_fitness = fitness;
                    global_best = particles[i].clone();
                }
            }
            self.state.best_objective = global_best_fitness;
            self.state.best_solution = global_best.clone();
            self.state.convergence_history.push(global_best_fitness);
            if self.check_convergence()? {
                break;
            }
        }
        Ok(OptimizationResult {
            solution: global_best,
            objective_value: global_best_fitness,
            iterations: self.state.iteration,
            converged: self.check_convergence()?,
            runtime_stats: self.state.runtime_stats.clone(),
            metadata: HashMap::new(),
        })
    }
    /// Quantum-inspired simulated annealing
    pub(super) fn quantum_simulated_annealing(&mut self) -> Result<OptimizationResult> {
        let max_iterations = self.config.algorithm_config.max_iterations;
        let temperature_schedule = self.config.algorithm_config.temperature_schedule;
        let quantum_parameters = self.config.algorithm_config.quantum_parameters.clone();
        let bounds = self.config.optimization_config.bounds.clone();
        let num_vars = self.config.num_variables;
        let mut current_solution = Array1::zeros(num_vars);
        let mut rng = self.rng.lock().expect("RNG lock poisoned");
        for i in 0..num_vars {
            let (min_bound, max_bound) = if i < bounds.len() {
                bounds[i]
            } else {
                (-10.0, 10.0)
            };
            current_solution[i] = rng
                .random::<f64>()
                .mul_add(max_bound - min_bound, min_bound);
        }
        drop(rng);
        let mut current_energy = self.evaluate_objective(&current_solution)?;
        let mut best_solution = current_solution.clone();
        let mut best_energy = current_energy;
        self.state.runtime_stats.function_evaluations += 1;
        let initial_temp: f64 = 100.0;
        let final_temp: f64 = 0.01;
        for iteration in 0..max_iterations {
            self.state.iteration = iteration;
            let temp = match temperature_schedule {
                TemperatureSchedule::Exponential => {
                    initial_temp
                        * (final_temp / initial_temp).powf(iteration as f64 / max_iterations as f64)
                }
                TemperatureSchedule::Linear => (initial_temp - final_temp)
                    .mul_add(-(iteration as f64 / max_iterations as f64), initial_temp),
                TemperatureSchedule::Logarithmic => initial_temp / (1.0 + (iteration as f64).ln()),
                TemperatureSchedule::QuantumAdiabatic => {
                    let s = iteration as f64 / max_iterations as f64;
                    initial_temp.mul_add(1.0 - s, final_temp * s * (1.0 - (1.0 - s).powi(3)))
                }
                TemperatureSchedule::Custom => initial_temp * 0.95_f64.powi(iteration as i32),
            };
            let mut neighbor = current_solution.clone();
            let quantum_params = &quantum_parameters;
            let mut rng = self.rng.lock().expect("RNG lock poisoned");
            for i in 0..num_vars {
                if rng.random::<f64>() < 0.5 {
                    let (min_bound, max_bound) = if i < bounds.len() {
                        bounds[i]
                    } else {
                        (-10.0, 10.0)
                    };
                    let step_size = temp / initial_temp;
                    let gaussian_step =
                        rng.random::<f64>() * step_size * (max_bound - min_bound) * 0.1;
                    let tunneling_move =
                        if rng.random::<f64>() < quantum_params.tunneling_probability {
                            (rng.random::<f64>() - 0.5) * (max_bound - min_bound) * 0.5
                        } else {
                            0.0
                        };
                    neighbor[i] = (current_solution[i] + gaussian_step + tunneling_move)
                        .clamp(min_bound, max_bound);
                }
            }
            drop(rng);
            let neighbor_energy = self.evaluate_objective(&neighbor)?;
            self.state.runtime_stats.function_evaluations += 1;
            let delta_energy = neighbor_energy - current_energy;
            let acceptance_prob = if delta_energy < 0.0 {
                1.0
            } else {
                let boltzmann_factor = (-delta_energy / temp).exp();
                let quantum_correction = quantum_params.interference_strength
                    * (2.0 * PI * iteration as f64 / max_iterations as f64).cos()
                    * 0.1;
                (boltzmann_factor + quantum_correction).clamp(0.0, 1.0)
            };
            let mut rng = self.rng.lock().expect("RNG lock poisoned");
            if rng.random::<f64>() < acceptance_prob {
                current_solution = neighbor;
                current_energy = neighbor_energy;
                if current_energy < best_energy {
                    best_solution = current_solution.clone();
                    best_energy = current_energy;
                }
            }
            drop(rng);
            self.state.best_objective = best_energy;
            self.state.best_solution = best_solution.clone();
            self.state.convergence_history.push(best_energy);
            if temp < final_temp || self.check_convergence()? {
                break;
            }
        }
        Ok(OptimizationResult {
            solution: best_solution,
            objective_value: best_energy,
            iterations: self.state.iteration,
            converged: self.check_convergence()?,
            runtime_stats: self.state.runtime_stats.clone(),
            metadata: HashMap::new(),
        })
    }
    /// Quantum differential evolution
    pub(super) fn quantum_differential_evolution(&mut self) -> Result<OptimizationResult> {
        let pop_size = self.config.algorithm_config.population_size.max(4);
        let num_vars = self.config.num_variables;
        let max_iterations = self.config.algorithm_config.max_iterations;
        // F in DE literature: scale factor for the difference vector
        let mutation_factor = 0.5_f64;
        let crossover_rate = self.config.algorithm_config.crossover_rate;
        let bounds = self.config.optimization_config.bounds.clone();
        let quantum_params = self.config.algorithm_config.quantum_parameters.clone();

        // Initialize population using quantum-inspired initialization
        let mut population = self.initialize_quantum_population(pop_size, num_vars)?;
        let mut fitness: Vec<f64> = Vec::with_capacity(pop_size);
        let mut best_solution = population[0].clone();
        let mut best_fitness = f64::INFINITY;

        // Evaluate initial population fitness
        for individual in &population {
            let f = self.evaluate_objective(individual)?;
            fitness.push(f);
            self.state.runtime_stats.function_evaluations += 1;
            if f < best_fitness {
                best_fitness = f;
                best_solution = individual.clone();
            }
        }

        for iteration in 0..max_iterations {
            self.state.iteration = iteration;

            for i in 0..pop_size {
                // Pick 3 distinct random indices r1, r2, r3, all != i
                let mut rng = self.rng.lock().expect("RNG lock poisoned");
                let mut indices: Vec<usize> = Vec::with_capacity(3);
                let mut attempts = 0usize;
                while indices.len() < 3 && attempts < 1000 {
                    let idx = (rng.random::<f64>() * pop_size as f64) as usize % pop_size;
                    if idx != i && !indices.contains(&idx) {
                        indices.push(idx);
                    }
                    attempts += 1;
                }
                // Fallback if pop_size < 4: wrap with offset to ensure distinct-ish values
                while indices.len() < 3 {
                    let candidate = (i + indices.len() + 1) % pop_size;
                    if !indices.contains(&candidate) {
                        indices.push(candidate);
                    } else {
                        indices.push((candidate + 1) % pop_size);
                    }
                }
                let (r1, r2, r3) = (indices[0], indices[1], indices[2]);

                // Mutant vector: v = x_r1 + F * (x_r2 - x_r3)
                let mut mutant = population[r1].clone();
                for j in 0..num_vars {
                    mutant[j] =
                        mutation_factor.mul_add(population[r2][j] - population[r3][j], mutant[j]);
                }

                // Binomial crossover: trial inherits from mutant with rate CR,
                // at least one dimension guaranteed (jrand)
                let mut trial = population[i].clone();
                let guaranteed_j = (rng.random::<f64>() * num_vars as f64) as usize % num_vars;
                for j in 0..num_vars {
                    if j == guaranteed_j || rng.random::<f64>() < crossover_rate {
                        trial[j] = mutant[j];
                    }
                    // Quantum tunneling: probabilistic random reset to bypass local basins
                    if rng.random::<f64>() < quantum_params.tunneling_probability {
                        let (min_b, max_b) = if j < bounds.len() {
                            bounds[j]
                        } else {
                            (-10.0, 10.0)
                        };
                        trial[j] = rng.random::<f64>().mul_add(max_b - min_b, min_b);
                    }
                    // Clamp to feasible bounds
                    let (min_b, max_b) = if j < bounds.len() {
                        bounds[j]
                    } else {
                        (-10.0, 10.0)
                    };
                    trial[j] = trial[j].clamp(min_b, max_b);
                }
                drop(rng);

                // Greedy selection: replace parent only if trial is better
                let trial_fitness = self.evaluate_objective(&trial)?;
                self.state.runtime_stats.function_evaluations += 1;
                if trial_fitness < fitness[i] {
                    fitness[i] = trial_fitness;
                    population[i] = trial;
                    if trial_fitness < best_fitness {
                        best_fitness = trial_fitness;
                        best_solution = population[i].clone();
                    }
                }
            }

            self.state.best_objective = best_fitness;
            self.state.best_solution = best_solution.clone();
            self.state.convergence_history.push(best_fitness);

            if self.check_convergence()? {
                break;
            }
        }

        Ok(OptimizationResult {
            solution: best_solution,
            objective_value: best_fitness,
            iterations: self.state.iteration,
            converged: self.check_convergence()?,
            runtime_stats: self.state.runtime_stats.clone(),
            metadata: HashMap::new(),
        })
    }
    /// Classical QAOA simulation
    ///
    /// Implements a QAOA-inspired coordinate landscape search using alternating
    /// problem (cost) and mixing (driver) phases with sinusoidal annealing schedules.
    pub(super) fn classical_qaoa_simulation(&mut self) -> Result<OptimizationResult> {
        let num_vars = self.config.num_variables;
        let max_iterations = self.config.algorithm_config.max_iterations;
        let bounds = self.config.optimization_config.bounds.clone();
        let quantum_params = self.config.algorithm_config.quantum_parameters.clone();

        // Initialize solution with continuous relaxation in [lo, hi]
        let mut current = {
            let mut rng = self.rng.lock().expect("RNG lock poisoned");
            let mut v = Array1::zeros(num_vars);
            for j in 0..num_vars {
                let (lo, hi) = if j < bounds.len() { bounds[j] } else { (-10.0, 10.0) };
                v[j] = rng.random::<f64>().mul_add(hi - lo, lo);
            }
            v
        };
        let mut current_energy = self.evaluate_objective(&current)?;
        let mut best = current.clone();
        let mut best_energy = current_energy;
        self.state.runtime_stats.function_evaluations += 1;

        for iteration in 0..max_iterations {
            self.state.iteration = iteration;
            let layers = iteration as f64 / max_iterations.max(1) as f64;
            // gamma: problem phase angle increases with layer depth (QAOA schedule)
            let gamma = PI * layers;
            // beta: mixing angle decreases (adiabatic annealing schedule)
            let beta = std::f64::consts::FRAC_PI_2 * (1.0 - layers);

            let mut next = current.clone();
            let mut rng = self.rng.lock().expect("RNG lock poisoned");
            for j in 0..num_vars {
                let (lo, hi) = if j < bounds.len() { bounds[j] } else { (-10.0, 10.0) };
                let range = hi - lo;
                // Problem phase: gradient-guided shift using cost landscape curvature
                let shift = range * 0.05;
                let cost_shift = (gamma.sin() * shift).mul_add(rng.random::<f64>() - 0.5, 0.0);
                // Mixing phase: random exploration with decreasing strength
                let mix_shift = if rng.random::<f64>() < beta.sin().powi(2) {
                    (rng.random::<f64>() - 0.5) * range * quantum_params.superposition_strength
                } else {
                    0.0
                };
                // Quantum tunneling jump
                let tunnel_shift = if rng.random::<f64>() < quantum_params.tunneling_probability {
                    (rng.random::<f64>() - 0.5) * range
                } else {
                    0.0
                };
                next[j] = (current[j] + cost_shift + mix_shift + tunnel_shift).clamp(lo, hi);
            }
            drop(rng);

            let next_energy = self.evaluate_objective(&next)?;
            self.state.runtime_stats.function_evaluations += 1;

            // Accept if improved (greedy acceptance analogous to classical QAOA measurement)
            if next_energy < current_energy {
                current = next;
                current_energy = next_energy;
                if current_energy < best_energy {
                    best_energy = current_energy;
                    best = current.clone();
                }
            }

            self.state.best_objective = best_energy;
            self.state.best_solution = best.clone();
            self.state.convergence_history.push(best_energy);
            if self.check_convergence()? {
                break;
            }
        }

        Ok(OptimizationResult {
            solution: best,
            objective_value: best_energy,
            iterations: self.state.iteration,
            converged: self.check_convergence()?,
            runtime_stats: self.state.runtime_stats.clone(),
            metadata: HashMap::new(),
        })
    }

    /// Classical VQE simulation
    ///
    /// Implements coordinate descent with parameter-shift-inspired steps to mimic
    /// variational quantum eigensolver parameter optimization classically.
    pub(super) fn classical_vqe_simulation(&mut self) -> Result<OptimizationResult> {
        let num_vars = self.config.num_variables;
        let max_iterations = self.config.algorithm_config.max_iterations;
        let bounds = self.config.optimization_config.bounds.clone();
        let quantum_params = self.config.algorithm_config.quantum_parameters.clone();

        // Initialize parameter vector randomly
        let mut theta = {
            let mut rng = self.rng.lock().expect("RNG lock poisoned");
            let mut v = Array1::zeros(num_vars);
            for j in 0..num_vars {
                let (lo, hi) = if j < bounds.len() { bounds[j] } else { (-10.0, 10.0) };
                v[j] = rng.random::<f64>().mul_add(hi - lo, lo);
            }
            v
        };
        let mut current_energy = self.evaluate_objective(&theta)?;
        let mut best = theta.clone();
        let mut best_energy = current_energy;
        self.state.runtime_stats.function_evaluations += 1;

        for iteration in 0..max_iterations {
            self.state.iteration = iteration;
            // Interference phase oscillates like a quantum circuit expectation value
            let phase = 2.0 * PI * iteration as f64 / max_iterations.max(1) as f64;
            let interference = quantum_params.interference_strength * phase.cos();

            // Coordinate descent with parameter-shift rule (analogous to quantum gradient)
            for j in 0..num_vars {
                let (lo, hi) = if j < bounds.len() { bounds[j] } else { (-10.0, 10.0) };
                let delta = 0.1 * (hi - lo) + interference.abs() * 0.01 * (hi - lo);

                let mut plus = theta.clone();
                plus[j] = (plus[j] + delta).clamp(lo, hi);
                let mut minus = theta.clone();
                minus[j] = (minus[j] - delta).clamp(lo, hi);

                let e_plus = self.evaluate_objective(&plus)?;
                let e_minus = self.evaluate_objective(&minus)?;
                self.state.runtime_stats.function_evaluations += 2;

                if e_plus < current_energy && e_plus <= e_minus {
                    theta = plus;
                    current_energy = e_plus;
                } else if e_minus < current_energy {
                    theta = minus;
                    current_energy = e_minus;
                }
            }

            // Quantum tunneling escape from local minima
            {
                let mut rng = self.rng.lock().expect("RNG lock poisoned");
                if rng.random::<f64>() < quantum_params.tunneling_probability {
                    let j = (rng.random::<f64>() * num_vars as f64) as usize % num_vars;
                    let (lo, hi) = if j < bounds.len() { bounds[j] } else { (-10.0, 10.0) };
                    theta[j] = rng.random::<f64>().mul_add(hi - lo, lo);
                    current_energy = self.evaluate_objective(&theta)?;
                    self.state.runtime_stats.function_evaluations += 1;
                }
            }

            if current_energy < best_energy {
                best_energy = current_energy;
                best = theta.clone();
            }

            self.state.best_objective = best_energy;
            self.state.best_solution = best.clone();
            self.state.convergence_history.push(best_energy);
            if self.check_convergence()? {
                break;
            }
        }

        Ok(OptimizationResult {
            solution: best,
            objective_value: best_energy,
            iterations: self.state.iteration,
            converged: self.check_convergence()?,
            runtime_stats: self.state.runtime_stats.clone(),
            metadata: HashMap::new(),
        })
    }

    /// Quantum ant colony optimization
    ///
    /// ACO with quantum amplitude-modulated pheromones: pheromone attractiveness is
    /// boosted by quantum interference effects and tunneling enables non-greedy exploration.
    pub(super) fn quantum_ant_colony_optimization(&mut self) -> Result<OptimizationResult> {
        const N_LEVELS: usize = 10;
        let n_ants = self.config.algorithm_config.population_size.max(5);
        let num_vars = self.config.num_variables;
        let max_iterations = self.config.algorithm_config.max_iterations;
        let bounds = self.config.optimization_config.bounds.clone();
        let quantum_params = self.config.algorithm_config.quantum_parameters.clone();
        let evaporation_rate = 1.0 - self.config.algorithm_config.mutation_rate.clamp(0.01, 0.99);

        // Pheromone matrix: pheromones[j][k] = strength for variable j at discretization level k
        let mut pheromones = vec![vec![1.0f64; N_LEVELS]; num_vars];

        let mut best_solution = {
            let mut rng = self.rng.lock().expect("RNG lock poisoned");
            let mut v = Array1::zeros(num_vars);
            for j in 0..num_vars {
                let (lo, hi) = if j < bounds.len() { bounds[j] } else { (-10.0, 10.0) };
                v[j] = rng.random::<f64>().mul_add(hi - lo, lo);
            }
            v
        };
        let mut best_energy = self.evaluate_objective(&best_solution)?;
        self.state.runtime_stats.function_evaluations += 1;

        for iteration in 0..max_iterations {
            self.state.iteration = iteration;
            // Quantum interference modulation of pheromone attractiveness
            let q_phase = 2.0 * PI * iteration as f64 / max_iterations.max(1) as f64;
            let q_boost = 1.0 + quantum_params.interference_strength * q_phase.cos();

            let mut ant_solutions: Vec<Array1<f64>> = Vec::with_capacity(n_ants);
            let mut ant_energies: Vec<f64> = Vec::with_capacity(n_ants);

            // Each ant constructs a solution via pheromone-guided roulette selection
            for _ in 0..n_ants {
                let mut solution = Array1::zeros(num_vars);
                {
                    let mut rng = self.rng.lock().expect("RNG lock poisoned");
                    for j in 0..num_vars {
                        let (lo, hi) = if j < bounds.len() { bounds[j] } else { (-10.0, 10.0) };
                        // Quantum tunneling: occasionally pick uniformly regardless of pheromone
                        if rng.random::<f64>() < quantum_params.tunneling_probability {
                            solution[j] = rng.random::<f64>().mul_add(hi - lo, lo);
                            continue;
                        }
                        // Roulette wheel over N_LEVELS using quantum-boosted pheromones
                        let total: f64 = pheromones[j]
                            .iter()
                            .map(|&p| (p * q_boost).max(1e-10))
                            .sum();
                        let mut pick = rng.random::<f64>() * total;
                        let mut chosen_level = N_LEVELS - 1;
                        for (k, &p) in pheromones[j].iter().enumerate() {
                            pick -= (p * q_boost).max(1e-10);
                            if pick <= 0.0 {
                                chosen_level = k;
                                break;
                            }
                        }
                        solution[j] =
                            lo + (chosen_level as f64 + 0.5) / N_LEVELS as f64 * (hi - lo);
                    }
                }

                let energy = self.evaluate_objective(&solution)?;
                self.state.runtime_stats.function_evaluations += 1;
                if energy < best_energy {
                    best_energy = energy;
                    best_solution = solution.clone();
                }
                ant_solutions.push(solution);
                ant_energies.push(energy);
            }

            // Pheromone evaporation
            for j in 0..num_vars {
                for k in 0..N_LEVELS {
                    pheromones[j][k] *= evaporation_rate;
                }
            }

            // Pheromone deposit: lower-energy ants contribute more
            let min_e = ant_energies.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_e = ant_energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (max_e - min_e).max(1e-10);
            for (ant_idx, solution) in ant_solutions.iter().enumerate() {
                let deposit = 1.0 - (ant_energies[ant_idx] - min_e) / range;
                for j in 0..num_vars {
                    let (lo, hi) = if j < bounds.len() { bounds[j] } else { (-10.0, 10.0) };
                    let level = ((solution[j] - lo) / (hi - lo) * N_LEVELS as f64) as usize;
                    let level = level.min(N_LEVELS - 1);
                    pheromones[j][level] += deposit;
                }
            }

            self.state.best_objective = best_energy;
            self.state.best_solution = best_solution.clone();
            self.state.convergence_history.push(best_energy);
            if self.check_convergence()? {
                break;
            }
        }

        Ok(OptimizationResult {
            solution: best_solution,
            objective_value: best_energy,
            iterations: self.state.iteration,
            converged: self.check_convergence()?,
            runtime_stats: self.state.runtime_stats.clone(),
            metadata: HashMap::new(),
        })
    }
    /// Quantum harmony search
    pub(super) fn quantum_harmony_search(&mut self) -> Result<OptimizationResult> {
        let hm_size = self.config.algorithm_config.population_size.max(5);
        let num_vars = self.config.num_variables;
        let max_iterations = self.config.algorithm_config.max_iterations;
        // HMCR: probability of selecting from harmony memory (typical: 0.7–0.95)
        let hmcr = 0.9_f64;
        // PAR: probability of pitch adjustment when selected from memory (typical: 0.1–0.5)
        let par = 0.3_f64;
        let bounds = self.config.optimization_config.bounds.clone();
        let quantum_params = self.config.algorithm_config.quantum_parameters.clone();

        // Initialize harmony memory using quantum-inspired population initializer
        let mut harmony_memory = self.initialize_quantum_population(hm_size, num_vars)?;
        let mut hm_fitness: Vec<f64> = Vec::with_capacity(hm_size);
        let mut best_solution = harmony_memory[0].clone();
        let mut best_fitness = f64::INFINITY;

        // Evaluate initial harmonies
        for harmony in &harmony_memory {
            let f = self.evaluate_objective(harmony)?;
            hm_fitness.push(f);
            self.state.runtime_stats.function_evaluations += 1;
            if f < best_fitness {
                best_fitness = f;
                best_solution = harmony.clone();
            }
        }

        for iteration in 0..max_iterations {
            self.state.iteration = iteration;

            // Improvise a new harmony vector
            let mut new_harmony = Array1::zeros(num_vars);
            let mut rng = self.rng.lock().expect("RNG lock poisoned");

            for j in 0..num_vars {
                let (min_b, max_b) = if j < bounds.len() {
                    bounds[j]
                } else {
                    (-10.0, 10.0)
                };

                if rng.random::<f64>() < hmcr {
                    // Memory consideration: borrow a value from a randomly chosen harmony
                    let hm_idx = (rng.random::<f64>() * hm_size as f64) as usize % hm_size;
                    new_harmony[j] = harmony_memory[hm_idx][j];

                    // Pitch adjustment: perturb the recalled value within a bandwidth
                    if rng.random::<f64>() < par {
                        // Bandwidth proportional to 5% of the variable range
                        let bw = 0.05 * (max_b - min_b);
                        let perturbation = (rng.random::<f64>() - 0.5) * 2.0 * bw;
                        new_harmony[j] = (new_harmony[j] + perturbation).clamp(min_b, max_b);
                    }
                } else {
                    // Random selection: uniform draw within variable bounds
                    new_harmony[j] = rng.random::<f64>().mul_add(max_b - min_b, min_b);
                }

                // Quantum tunneling: escape local optima by random reset
                if rng.random::<f64>() < quantum_params.tunneling_probability {
                    new_harmony[j] = rng.random::<f64>().mul_add(max_b - min_b, min_b);
                }
            }
            drop(rng);

            // Evaluate the improvised harmony
            let new_fitness = self.evaluate_objective(&new_harmony)?;
            self.state.runtime_stats.function_evaluations += 1;

            // Update harmony memory: replace the worst harmony if the new one is better
            if let Some((worst_idx, &worst_fitness)) = hm_fitness
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                if new_fitness < worst_fitness {
                    hm_fitness[worst_idx] = new_fitness;
                    harmony_memory[worst_idx] = new_harmony;
                    if new_fitness < best_fitness {
                        best_fitness = new_fitness;
                        best_solution = harmony_memory[worst_idx].clone();
                    }
                }
            }

            self.state.best_objective = best_fitness;
            self.state.best_solution = best_solution.clone();
            self.state.convergence_history.push(best_fitness);

            if self.check_convergence()? {
                break;
            }
        }

        Ok(OptimizationResult {
            solution: best_solution,
            objective_value: best_fitness,
            iterations: self.state.iteration,
            converged: self.check_convergence()?,
            runtime_stats: self.state.runtime_stats.clone(),
            metadata: HashMap::new(),
        })
    }
    /// Evaluate objective function
    pub(super) fn evaluate_objective(&self, solution: &Array1<f64>) -> Result<f64> {
        let result = match self.config.optimization_config.objective_function {
            ObjectiveFunction::Quadratic => solution.iter().map(|&x| x * x).sum(),
            ObjectiveFunction::Rastrigin => {
                let n = solution.len() as f64;
                let a = 10.0;
                a * n
                    + solution
                        .iter()
                        .map(|&x| x.mul_add(x, -(a * (2.0 * PI * x).cos())))
                        .sum::<f64>()
            }
            ObjectiveFunction::Rosenbrock => {
                if solution.len() < 2 {
                    return Ok(0.0);
                }
                let mut result = 0.0;
                for i in 0..solution.len() - 1 {
                    let x = solution[i];
                    let y = solution[i + 1];
                    result += (1.0 - x).mul_add(1.0 - x, 100.0 * x.mul_add(-x, y).powi(2));
                }
                result
            }
            ObjectiveFunction::Ackley => {
                let n = solution.len() as f64;
                let a: f64 = 20.0;
                let b: f64 = 0.2;
                let c: f64 = 2.0 * PI;
                let sum1 = solution.iter().map(|&x| x * x).sum::<f64>() / n;
                let sum2 = solution.iter().map(|&x| (c * x).cos()).sum::<f64>() / n;
                (-a).mul_add((-b * sum1.sqrt()).exp(), -sum2.exp()) + a + std::f64::consts::E
            }
            ObjectiveFunction::Sphere => solution.iter().map(|&x| x * x).sum(),
            ObjectiveFunction::Griewank => {
                let sum_sq = solution.iter().map(|&x| x * x).sum::<f64>() / 4000.0;
                let prod_cos = solution
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| (x / ((i + 1) as f64).sqrt()).cos())
                    .product::<f64>();
                1.0 + sum_sq - prod_cos
            }
            ObjectiveFunction::Custom => solution.iter().map(|&x| x * x).sum(),
        };
        Ok(result)
    }
    /// Check convergence
    pub(super) fn check_convergence(&self) -> Result<bool> {
        if self.state.convergence_history.len() < 2 {
            return Ok(false);
        }
        let tolerance = self.config.algorithm_config.tolerance;
        let recent_improvements = &self.state.convergence_history
            [self.state.convergence_history.len().saturating_sub(10)..];
        if recent_improvements.len() < 2 {
            return Ok(false);
        }
        let last_value = recent_improvements
            .last()
            .expect("recent_improvements has at least 2 elements");
        let second_last_value = recent_improvements[recent_improvements.len() - 2];
        let change = (last_value - second_last_value).abs();
        Ok(change < tolerance)
    }
    /// Train machine learning model using quantum-inspired stochastic gradient descent.
    ///
    /// Trains a linear model `y = W * x + b` on the provided `(input, target)` pairs.
    /// Quantum tunneling perturbations are injected with probability
    /// `quantum_parameters.tunneling_probability` to help escape local minima.
    pub fn train_ml_model(
        &mut self,
        training_data: &[(Array1<f64>, Array1<f64>)],
    ) -> Result<MLTrainingResult> {
        if training_data.is_empty() {
            return Err(SimulatorError::InvalidInput(
                "Training data is empty".to_string(),
            ));
        }

        let input_dim = training_data[0].0.len();
        let output_dim = training_data[0].1.len();
        // Weight layout: for output neuron o, weights at indices
        // [o*(input_dim+1) .. o*(input_dim+1)+input_dim] and bias at
        // o*(input_dim+1)+input_dim.
        let n_params = (input_dim + 1) * output_dim;

        let lr = self
            .config
            .algorithm_config
            .quantum_parameters
            .superposition_strength
            .clamp(0.001, 0.1);
        let epochs = self.config.algorithm_config.max_iterations;
        let tunneling_prob = self
            .config
            .algorithm_config
            .quantum_parameters
            .tunneling_probability;

        // Initialise parameters with small random values.
        let mut params: Array1<f64> = {
            let mut rng = self.rng.lock().expect("RNG lock poisoned");
            let mut p = Array1::zeros(n_params);
            for v in p.iter_mut() {
                *v = (rng.random::<f64>() - 0.5) * 0.1;
            }
            p
        };

        let mut loss_history: Vec<f64> = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut total_loss = 0.0_f64;
            let mut grad: Array1<f64> = Array1::zeros(n_params);

            for (x, y_true) in training_data {
                // Forward pass: y_pred[o] = sum_i(w[o*(d+1)+i] * x[i]) + bias[o].
                let mut y_pred: Array1<f64> = Array1::zeros(output_dim);
                for o in 0..output_dim {
                    let w_start = o * (input_dim + 1);
                    for i in 0..input_dim {
                        y_pred[o] += params[w_start + i] * x[i];
                    }
                    y_pred[o] += params[o * (input_dim + 1) + input_dim];
                }

                // MSE loss per sample.
                let residual: Array1<f64> = Array1::from_iter(
                    y_pred
                        .iter()
                        .zip(y_true.iter())
                        .map(|(p, t)| *p - *t),
                );
                total_loss +=
                    residual.iter().map(|r| r * r).sum::<f64>() / output_dim as f64;

                // Accumulate gradients.
                for o in 0..output_dim {
                    let w_start = o * (input_dim + 1);
                    let err: f64 = residual[o] * 2.0 / output_dim as f64;
                    for i in 0..input_dim {
                        grad[w_start + i] += err * x[i];
                    }
                    grad[o * (input_dim + 1) + input_dim] += err;
                }
            }

            let n: f64 = training_data.len() as f64;
            total_loss /= n;

            // SGD parameter update.
            for i in 0..n_params {
                params[i] -= lr * grad[i] / n;
            }

            // Quantum tunneling: stochastic perturbation to escape local minima.
            {
                let mut rng = self.rng.lock().expect("RNG lock poisoned");
                if rng.random::<f64>() < tunneling_prob {
                    let j =
                        (rng.random::<f64>() * n_params as f64) as usize % n_params;
                    params[j] += (rng.random::<f64>() - 0.5) * 0.01;
                }
            }

            loss_history.push(total_loss);

            // Early-exit if loss is not improving.
            if epoch >= 2 && (loss_history[epoch - 1] - total_loss).abs() < 1e-8 {
                break;
            }
        }

        // Validation accuracy: fraction of samples whose mean absolute error is
        // within 10 % of the target magnitude (floor 0.1 near zero).
        let validation_accuracy = {
            let mut correct = 0_usize;
            for (x, y_true) in training_data {
                let mut y_pred: Array1<f64> = Array1::zeros(output_dim);
                for o in 0..output_dim {
                    let w_start = o * (input_dim + 1);
                    for i in 0..input_dim {
                        y_pred[o] += params[w_start + i] * x[i];
                    }
                    y_pred[o] += params[o * (input_dim + 1) + input_dim];
                }
                let err: f64 = y_pred
                    .iter()
                    .zip(y_true.iter())
                    .map(|(p, t)| (p - t).abs())
                    .sum::<f64>()
                    / output_dim as f64;
                let mag: f64 =
                    y_true.iter().map(|t| t.abs()).sum::<f64>() / output_dim as f64;
                if err < 0.1 * mag.max(1.0) {
                    correct += 1;
                }
            }
            correct as f64 / training_data.len() as f64
        };

        Ok(MLTrainingResult {
            parameters: params,
            loss_history,
            validation_accuracy,
            training_time: 0.0,
            complexity_metrics: HashMap::new(),
        })
    }

    /// Perform quantum-inspired Metropolis-Hastings MCMC sampling.
    ///
    /// Uses the framework's configured objective function as the (negative log) energy.
    /// Quantum tunneling jumps are mixed in with probability
    /// `quantum_parameters.tunneling_probability` to improve chain mixing.
    pub fn sample(&mut self) -> Result<SamplingResult> {
        let n_samples = self.config.algorithm_config.max_iterations.max(100);
        let num_vars = self.config.num_variables;
        let bounds = self.config.optimization_config.bounds.clone();
        let tunneling_prob = self
            .config
            .algorithm_config
            .quantum_parameters
            .tunneling_probability;

        let mut samples: Vec<Array1<f64>> = Vec::with_capacity(n_samples);
        let mut accepted = 0_usize;

        // Initialise current state uniformly within bounds.
        let mut current: Array1<f64> = {
            let mut rng = self.rng.lock().expect("RNG lock poisoned");
            let mut v = Array1::zeros(num_vars);
            for j in 0..num_vars {
                let (lo, hi) = if j < bounds.len() {
                    bounds[j]
                } else {
                    (-10.0, 10.0)
                };
                v[j] = rng.random::<f64>().mul_add(hi - lo, lo);
            }
            v
        };
        let mut current_energy = self.evaluate_objective(&current)?;

        // Fixed temperature — callers may extend with annealing schedules later.
        let temperature = 1.0_f64;

        for _ in 0..n_samples {
            // Propose new state: local Gaussian step or full quantum tunnel jump.
            let proposed: Array1<f64> = {
                let mut rng = self.rng.lock().expect("RNG lock poisoned");
                let mut prop = current.clone();
                for j in 0..num_vars {
                    let (lo, hi) = if j < bounds.len() {
                        bounds[j]
                    } else {
                        (-10.0, 10.0)
                    };
                    let step = (hi - lo) * 0.1;
                    if rng.random::<f64>() < tunneling_prob {
                        // Quantum tunneling: resample coordinate uniformly.
                        prop[j] = rng.random::<f64>().mul_add(hi - lo, lo);
                    } else {
                        prop[j] =
                            (current[j] + (rng.random::<f64>() - 0.5) * step)
                                .clamp(lo, hi);
                    }
                }
                prop
            };

            let proposed_energy = self.evaluate_objective(&proposed)?;
            let delta = proposed_energy - current_energy;

            let accept = {
                let mut rng = self.rng.lock().expect("RNG lock poisoned");
                delta < 0.0 || rng.random::<f64>() < (-delta / temperature).exp()
            };

            if accept {
                current = proposed;
                current_energy = proposed_energy;
                accepted += 1;
            }
            samples.push(current.clone());
        }

        let acceptance_rate = accepted as f64 / n_samples as f64;

        // Pack samples into a 2-D array (n_samples × num_vars).
        let mut samples_array = Array2::zeros((n_samples, num_vars));
        for (i, s) in samples.iter().enumerate() {
            samples_array.row_mut(i).assign(s);
        }

        // Per-variable statistics.
        let mean: Array1<f64> = samples_array
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .unwrap_or_else(|| Array1::zeros(num_vars));
        let variance: Array1<f64> =
            samples_array.var_axis(scirs2_core::ndarray::Axis(0), 0.0);

        // Skewness: E[(X-mu)^3] / sigma^3.
        let skewness: Array1<f64> = Array1::from_iter((0..num_vars).map(|j| {
            let mu = mean[j];
            let sigma = variance[j].sqrt();
            if sigma < 1e-12 {
                return 0.0;
            }
            let m3: f64 = samples_array
                .column(j)
                .iter()
                .map(|&v| (v - mu).powi(3))
                .sum::<f64>()
                / n_samples as f64;
            m3 / sigma.powi(3)
        }));

        // Excess kurtosis: E[(X-mu)^4] / sigma^4 - 3.
        let kurtosis: Array1<f64> = Array1::from_iter((0..num_vars).map(|j| {
            let mu = mean[j];
            let sigma = variance[j].sqrt();
            if sigma < 1e-12 {
                return 0.0;
            }
            let m4: f64 = samples_array
                .column(j)
                .iter()
                .map(|&v| (v - mu).powi(4))
                .sum::<f64>()
                / n_samples as f64;
            m4 / sigma.powi(4) - 3.0
        }));

        // Pearson correlation matrix (num_vars × num_vars).
        let mut correlation_matrix = Array2::zeros((num_vars, num_vars));
        for i in 0..num_vars {
            for j in 0..num_vars {
                if i == j {
                    correlation_matrix[[i, j]] = 1.0;
                    continue;
                }
                let mu_i = mean[i];
                let mu_j = mean[j];
                let si = variance[i].sqrt();
                let sj = variance[j].sqrt();
                if si < 1e-12 || sj < 1e-12 {
                    continue;
                }
                let cov: f64 = samples_array
                    .column(i)
                    .iter()
                    .zip(samples_array.column(j).iter())
                    .map(|(&a, &b)| (a - mu_i) * (b - mu_j))
                    .sum::<f64>()
                    / n_samples as f64;
                correlation_matrix[[i, j]] = cov / (si * sj);
            }
        }

        let effective_sample_size =
            (n_samples as f64 * acceptance_rate).max(1.0) as usize;
        // Autocorrelation time lower-bounded at 1.
        let autocorr_times: Array1<f64> =
            Array1::from_elem(num_vars, (1.0_f64 / acceptance_rate.max(1e-6)).max(1.0));

        Ok(SamplingResult {
            samples: samples_array,
            statistics: SampleStatistics {
                mean,
                variance,
                skewness,
                kurtosis,
                correlation_matrix,
            },
            acceptance_rate,
            effective_sample_size,
            autocorr_times,
        })
    }

    /// Solve a complex linear system `A · x = b` using a quantum-inspired iterative
    /// Gauss-Seidel method.
    ///
    /// The solver terminates early when the successive-iterate L2 difference drops
    /// below `algorithm_config.tolerance`.  Callers should check `residual_norm` to
    /// verify convergence for ill-conditioned or non-diagonally-dominant matrices.
    pub fn solve_linear_algebra(
        &mut self,
        matrix: &Array2<Complex64>,
        rhs: &Array1<Complex64>,
    ) -> Result<LinalgResult> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SimulatorError::InvalidInput(format!(
                "Matrix must be square: got {}×{}",
                n,
                matrix.ncols()
            )));
        }
        if n != rhs.len() {
            return Err(SimulatorError::InvalidInput(format!(
                "Matrix dimension {} incompatible with rhs length {}",
                n,
                rhs.len()
            )));
        }
        if n == 0 {
            return Ok(LinalgResult {
                solution: Array1::zeros(0),
                eigenvalues: None,
                eigenvectors: None,
                singular_values: None,
                residual_norm: 0.0,
                iterations: 0,
            });
        }

        let max_iter = self.config.algorithm_config.max_iterations.max(1000);
        let tol = self.config.algorithm_config.tolerance.max(1e-10);

        let mut x: Array1<Complex64> =
            Array1::from_elem(n, Complex64::new(0.0, 0.0));

        let mut converged_iter = max_iter;

        for iter in 0..max_iter {
            let x_old = x.clone();

            // Gauss-Seidel sweep.
            for i in 0..n {
                let diag = matrix[[i, i]];
                if diag.norm() < 1e-14 {
                    // Skip singular diagonal entries.
                    continue;
                }
                let mut sigma = Complex64::new(0.0, 0.0);
                for j in 0..n {
                    if j != i {
                        sigma += matrix[[i, j]] * x[j];
                    }
                }
                x[i] = (rhs[i] - sigma) / diag;
            }

            // Convergence: L2 norm of iterate change.
            let delta: f64 = x
                .iter()
                .zip(x_old.iter())
                .map(|(a, b)| (a - b).norm())
                .sum::<f64>();

            if delta < tol {
                converged_iter = iter + 1;
                break;
            }
        }

        // Final residual ‖A·x − b‖₁.
        let ax: Array1<Complex64> = Array1::from_iter((0..n).map(|i| {
            (0..n)
                .map(|j| matrix[[i, j]] * x[j])
                .fold(Complex64::new(0.0, 0.0), |acc, v| acc + v)
        }));
        let residual_norm: f64 = ax
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a - b).norm())
            .sum::<f64>();

        Ok(LinalgResult {
            solution: x,
            eigenvalues: None,
            eigenvectors: None,
            singular_values: None,
            residual_norm,
            iterations: converged_iter,
        })
    }

    /// Solve a graph optimisation problem on a weighted undirected graph represented
    /// by `adjacency_matrix`.
    ///
    /// Applies quantum-inspired greedy graph colouring and computes the induced
    /// max-cut value (sum of edge weights crossing the even/odd colour bipartition).
    /// Graph metrics (modularity, clustering coefficient, average path length,
    /// diameter) are also computed via BFS.
    pub fn solve_graph_problem(
        &mut self,
        adjacency_matrix: &Array2<f64>,
    ) -> Result<GraphResult> {
        let n = adjacency_matrix.nrows();
        if n == 0 {
            return Ok(GraphResult {
                solution: vec![],
                objective_value: 0.0,
                graph_metrics: GraphMetrics {
                    modularity: 0.0,
                    clustering_coefficient: 0.0,
                    average_path_length: 0.0,
                    diameter: 0,
                },
                walk_stats: None,
            });
        }

        // Quantum-inspired greedy graph colouring.
        // With probability `tunneling_probability` the assigned colour is incremented
        // by 1 (a small quantum perturbation that can expose better colourings).
        let tunneling_prob = self
            .config
            .algorithm_config
            .quantum_parameters
            .tunneling_probability;

        let mut coloring = vec![0_usize; n];
        let mut max_color = 0_usize;

        for v in 0..n {
            let mut neighbor_colors: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for u in 0..n {
                if adjacency_matrix[[v, u]] > 0.0 {
                    neighbor_colors.insert(coloring[u]);
                }
            }

            let mut color = 0;
            while neighbor_colors.contains(&color) {
                color += 1;
            }

            // Quantum tunneling perturbation.
            {
                let mut rng = self.rng.lock().expect("RNG lock poisoned");
                if rng.random::<f64>() < tunneling_prob && color <= max_color {
                    color = (color + 1).min(max_color + 1);
                }
            }

            coloring[v] = color;
            if color > max_color {
                max_color = color;
            }
        }

        // Max-cut: sum of weights crossing the even/odd colour bipartition.
        let mut cut_value = 0.0_f64;
        for u in 0..n {
            for v in (u + 1)..n {
                if coloring[u] % 2 != coloring[v] % 2 {
                    cut_value += adjacency_matrix[[u, v]];
                }
            }
        }

        // Degree sequence.
        let degrees: Vec<f64> = (0..n)
            .map(|v| (0..n).map(|u| adjacency_matrix[[v, u]].abs()).sum::<f64>())
            .collect();

        // Local clustering coefficient averaged over all vertices.
        let clustering_coefficient = {
            let mut total_cc = 0.0_f64;
            for v in 0..n {
                let neighbors: Vec<usize> = (0..n)
                    .filter(|&u| u != v && adjacency_matrix[[v, u]] > 0.0)
                    .collect();
                let k = neighbors.len();
                if k < 2 {
                    continue;
                }
                let mut triangles = 0_usize;
                for i in 0..k {
                    for j in (i + 1)..k {
                        if adjacency_matrix[[neighbors[i], neighbors[j]]] > 0.0 {
                            triangles += 1;
                        }
                    }
                }
                total_cc += triangles as f64 / (k * (k - 1) / 2) as f64;
            }
            total_cc / n as f64
        };

        // Average path length and diameter via BFS on the unweighted adjacency.
        let (average_path_length, diameter) = {
            let mut total_dist = 0_usize;
            let mut max_dist = 0_usize;
            let mut reachable_pairs = 0_usize;

            for src in 0..n {
                let mut dist = vec![usize::MAX; n];
                dist[src] = 0;
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(src);
                while let Some(u) = queue.pop_front() {
                    for v in 0..n {
                        if adjacency_matrix[[u, v]] > 0.0 && dist[v] == usize::MAX {
                            dist[v] = dist[u] + 1;
                            queue.push_back(v);
                        }
                    }
                }
                for dst in 0..n {
                    if dst != src && dist[dst] != usize::MAX {
                        total_dist += dist[dst];
                        reachable_pairs += 1;
                        if dist[dst] > max_dist {
                            max_dist = dist[dst];
                        }
                    }
                }
            }

            let avg = if reachable_pairs > 0 {
                total_dist as f64 / reachable_pairs as f64
            } else {
                0.0
            };
            (avg, max_dist)
        };

        // Newman-Girvan modularity for the colour partition.
        let total_weight: f64 = degrees.iter().sum::<f64>() / 2.0;
        let modularity = if total_weight > 1e-12 {
            let two_m = 2.0 * total_weight;
            let mut q = 0.0_f64;
            for u in 0..n {
                for v in 0..n {
                    if coloring[u] == coloring[v] {
                        let a_uv = adjacency_matrix[[u, v]];
                        let expected = degrees[u] * degrees[v] / two_m;
                        q += a_uv - expected;
                    }
                }
            }
            q / two_m
        } else {
            0.0
        };

        Ok(GraphResult {
            solution: coloring,
            objective_value: cut_value,
            graph_metrics: GraphMetrics {
                modularity,
                clustering_coefficient,
                average_path_length,
                diameter,
            },
            walk_stats: None,
        })
    }
    /// Get current statistics
    #[must_use]
    pub const fn get_stats(&self) -> &QuantumInspiredStats {
        &self.stats
    }
    /// Get current state
    #[must_use]
    pub const fn get_state(&self) -> &QuantumInspiredState {
        &self.state
    }
    /// Get mutable state access
    pub const fn get_state_mut(&mut self) -> &mut QuantumInspiredState {
        &mut self.state
    }
    /// Evaluate objective function (public version)
    pub fn evaluate_objective_public(&mut self, solution: &Array1<f64>) -> Result<f64> {
        self.evaluate_objective(solution)
    }
    /// Check convergence (public version)
    pub fn check_convergence_public(&self) -> Result<bool> {
        self.check_convergence()
    }
    /// Reset framework state
    pub fn reset(&mut self) {
        self.state = QuantumInspiredState {
            variables: Array1::zeros(self.config.num_variables),
            objective_value: f64::INFINITY,
            iteration: 0,
            best_solution: Array1::zeros(self.config.num_variables),
            best_objective: f64::INFINITY,
            convergence_history: Vec::new(),
            runtime_stats: RuntimeStats::default(),
        };
        self.stats = QuantumInspiredStats::default();
    }
}
