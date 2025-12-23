//! Grover-Inspired Amplitude Amplification for QUBO Optimization
//!
//! This module implements a quantum-inspired classical algorithm that uses
//! Grover's amplitude amplification concepts to boost the probability of
//! finding optimal solutions to QUBO problems.
//!
//! # Theory
//!
//! Grover's algorithm amplifies the amplitude of target states through:
//! 1. Oracle marking (inverting phase of target states)
//! 2. Diffusion operator (inversion about average)
//!
//! For classical QUBO optimization, we adapt this by:
//! - Using energy-based marking (lower energy = higher weight)
//! - Employing iterative refinement with population diversity
//! - Applying amplitude-inspired probability distributions

use crate::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayD};
use scirs2_core::random::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Grover-inspired amplitude amplification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroverAmplificationConfig {
    /// Number of Grover-like iterations
    pub grover_iterations: usize,
    /// Population size for each iteration
    pub population_size: usize,
    /// Number of elite solutions to preserve
    pub elite_count: usize,
    /// Temperature for probabilistic selection
    pub temperature: f64,
    /// Amplification factor (analogous to quantum amplitude boost)
    pub amplification_factor: f64,
    /// Enable adaptive amplification
    pub adaptive_amplification: bool,
    /// Diversity preservation weight
    pub diversity_weight: f64,
}

impl Default for GroverAmplificationConfig {
    fn default() -> Self {
        Self {
            grover_iterations: 20,
            population_size: 100,
            elite_count: 10,
            temperature: 1.0,
            amplification_factor: 2.0,
            adaptive_amplification: true,
            diversity_weight: 0.1,
        }
    }
}

/// State representation in the amplitude amplification process
#[derive(Debug, Clone)]
struct AmplitudeState {
    /// Binary state vector
    state: Vec<bool>,
    /// Energy of the state
    energy: f64,
    /// Amplitude (probability weight)
    amplitude: f64,
    /// Hamming distance to best known solution
    distance_to_best: usize,
}

/// Grover-inspired QUBO solver
#[derive(Debug, Clone)]
pub struct GroverAmplifiedSolver {
    config: GroverAmplificationConfig,
    /// Best solution found so far
    best_solution: Option<Vec<bool>>,
    /// Best energy found
    best_energy: f64,
    /// History of energy improvements
    energy_history: Vec<f64>,
}

impl GroverAmplifiedSolver {
    /// Create a new Grover-amplified solver
    pub const fn new(config: GroverAmplificationConfig) -> Self {
        Self {
            config,
            best_solution: None,
            best_energy: f64::INFINITY,
            energy_history: Vec::new(),
        }
    }

    /// Initialize population with random states
    fn initialize_population(&self, n_vars: usize, qubo: &Array2<f64>) -> Vec<AmplitudeState> {
        let mut rng = thread_rng();
        let mut population = Vec::new();

        for _ in 0..self.config.population_size {
            let state: Vec<bool> = (0..n_vars).map(|_| rng.gen_bool(0.5)).collect();
            let energy = self.compute_energy(&state, qubo);

            population.push(AmplitudeState {
                state,
                energy,
                amplitude: 1.0 / self.config.population_size as f64,
                distance_to_best: 0,
            });
        }

        population
    }

    /// Compute QUBO energy for a state
    fn compute_energy(&self, state: &[bool], qubo: &Array2<f64>) -> f64 {
        let n = state.len();
        let mut energy = 0.0;

        for i in 0..n {
            if state[i] {
                energy += qubo[[i, i]];
            }
            for j in (i + 1)..n {
                if state[i] && state[j] {
                    energy += qubo[[i, j]];
                }
            }
        }

        energy
    }

    /// Oracle operation: mark good solutions (low energy)
    fn oracle_marking(&self, population: &mut [AmplitudeState]) {
        // Find energy range
        let min_energy = population
            .iter()
            .map(|s| s.energy)
            .fold(f64::INFINITY, f64::min);
        let max_energy = population
            .iter()
            .map(|s| s.energy)
            .fold(f64::NEG_INFINITY, f64::max);

        let energy_range = max_energy - min_energy;

        // Mark states based on energy (lower energy = higher marking)
        for state in population.iter_mut() {
            let normalized_energy = if energy_range > 1e-10 {
                (state.energy - min_energy) / energy_range
            } else {
                0.5
            };

            // Oracle marks good states (inverts phase in quantum case,
            // here we boost amplitude)
            let marking_factor = (-normalized_energy * self.config.amplification_factor).exp();
            state.amplitude *= marking_factor;
        }
    }

    /// Diffusion operation: inversion about average
    fn diffusion_operator(&self, population: &mut [AmplitudeState]) {
        // Compute average amplitude
        let avg_amplitude: f64 =
            population.iter().map(|s| s.amplitude).sum::<f64>() / population.len() as f64;

        // Invert about average (2*avg - amplitude)
        for state in population.iter_mut() {
            state.amplitude = 2.0f64.mul_add(avg_amplitude, -state.amplitude);
            state.amplitude = state.amplitude.max(0.0); // Keep non-negative
        }

        // Renormalize amplitudes
        let total: f64 = population.iter().map(|s| s.amplitude).sum();
        if total > 1e-10 {
            for state in population.iter_mut() {
                state.amplitude /= total;
            }
        }
    }

    /// Sample new states based on amplitudes (quantum measurement analogue)
    fn amplitude_sampling(
        &mut self,
        population: &[AmplitudeState],
        n_vars: usize,
        qubo: &Array2<f64>,
    ) -> Vec<AmplitudeState> {
        let mut rng = thread_rng();
        let mut new_population = Vec::new();

        // Keep elite solutions
        let mut sorted_pop = population.to_vec();
        sorted_pop.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        for state in sorted_pop.iter().take(self.config.elite_count) {
            new_population.push(state.clone());
        }

        // Sample remaining based on amplitudes
        let cumulative: Vec<f64> = population
            .iter()
            .scan(0.0, |acc, s| {
                *acc += s.amplitude;
                Some(*acc)
            })
            .collect();

        while new_population.len() < self.config.population_size {
            // Sample parent based on amplitude
            let r: f64 = rng.gen();
            let parent_idx = cumulative
                .iter()
                .position(|&c| r <= c)
                .unwrap_or(population.len() - 1);

            // Create offspring with mutations
            let mut offspring = population[parent_idx].state.clone();

            // Mutate with probability based on amplitude
            let mutation_prob = 0.1 * (1.0 - population[parent_idx].amplitude);
            for bit in &mut offspring {
                if rng.gen_bool(mutation_prob) {
                    *bit = !*bit;
                }
            }

            let energy = self.compute_energy(&offspring, qubo);
            let distance = if let Some(ref best) = self.best_solution {
                offspring
                    .iter()
                    .zip(best.iter())
                    .filter(|(a, b)| a != b)
                    .count()
            } else {
                0
            };

            new_population.push(AmplitudeState {
                state: offspring,
                energy,
                amplitude: 1.0 / self.config.population_size as f64,
                distance_to_best: distance,
            });
        }

        new_population
    }

    /// Adaptive amplification: adjust amplification factor based on progress
    fn adaptive_adjustment(&mut self, iteration: usize) {
        if !self.config.adaptive_amplification {
            return;
        }

        // Check progress
        if self.energy_history.len() > 5 {
            let recent_improvement = self.energy_history[self.energy_history.len() - 5]
                - *self.energy_history.last().unwrap();

            // If stuck, increase amplification
            if recent_improvement.abs() < 0.01 {
                self.config.amplification_factor *= 1.1;
            } else {
                // If improving, moderate amplification
                self.config.amplification_factor *= 0.95;
            }

            // Keep in reasonable range
            self.config.amplification_factor = self.config.amplification_factor.clamp(1.5, 5.0);
        }
    }

    /// Run Grover-inspired optimization
    pub fn optimize(&mut self, qubo: &Array2<f64>) -> Result<Vec<HashMap<String, bool>>, String> {
        let n_vars = qubo.nrows();

        // Initialize population
        let mut population = self.initialize_population(n_vars, qubo);

        println!("Starting Grover-inspired amplitude amplification...");

        for iteration in 0..self.config.grover_iterations {
            // Oracle marking
            self.oracle_marking(&mut population);

            // Diffusion operator
            self.diffusion_operator(&mut population);

            // Sample new population
            population = self.amplitude_sampling(&population, n_vars, qubo);

            // Update best solution
            for state in &population {
                if state.energy < self.best_energy {
                    self.best_energy = state.energy;
                    self.best_solution = Some(state.state.clone());
                }
            }

            self.energy_history.push(self.best_energy);

            // Adaptive adjustment
            self.adaptive_adjustment(iteration);

            if (iteration + 1) % 5 == 0 {
                println!(
                    "Iteration {}/{}: Best energy = {:.4}, Amplification factor = {:.2}",
                    iteration + 1,
                    self.config.grover_iterations,
                    self.best_energy,
                    self.config.amplification_factor
                );
            }
        }

        // Convert final population to result format
        let mut results = Vec::new();
        let mut seen_energies = std::collections::HashSet::new();

        for state in &population {
            // Only return unique energy levels
            let energy_key = (state.energy * 1000.0) as i64;
            if seen_energies.insert(energy_key) {
                let mut solution = HashMap::new();
                for (i, &bit) in state.state.iter().enumerate() {
                    solution.insert(format!("x{i}"), bit);
                }
                results.push(solution);
            }
        }

        // Sort by energy (best first)
        results.sort_by_key(|sol| {
            let state: Vec<bool> = (0..n_vars)
                .map(|i| *sol.get(&format!("x{i}")).unwrap_or(&false))
                .collect();
            (self.compute_energy(&state, qubo) * 1000.0) as i64
        });

        Ok(results)
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> OptimizationStatistics {
        OptimizationStatistics {
            best_energy: self.best_energy,
            iterations: self.energy_history.len(),
            final_amplification_factor: self.config.amplification_factor,
            convergence_rate: if self.energy_history.len() > 1 {
                (self.energy_history[0] - self.best_energy) / self.energy_history.len() as f64
            } else {
                0.0
            },
        }
    }
}

/// Optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStatistics {
    pub best_energy: f64,
    pub iterations: usize,
    pub final_amplification_factor: f64,
    pub convergence_rate: f64,
}

/// Grover-amplified sampler
#[derive(Debug, Clone)]
pub struct GroverAmplifiedSampler {
    config: GroverAmplificationConfig,
}

impl GroverAmplifiedSampler {
    /// Create a new Grover-amplified sampler
    pub const fn new(config: GroverAmplificationConfig) -> Self {
        Self { config }
    }
}

impl Sampler for GroverAmplifiedSampler {
    fn run_qubo(
        &self,
        problem: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        let (matrix, _var_map) = problem;

        // Create solver
        let mut solver = GroverAmplifiedSolver::new(self.config.clone());

        // Run optimization
        let solutions = solver
            .optimize(matrix)
            .map_err(SamplerError::InvalidParameter)?;

        // Convert to SampleResult format
        let results: Vec<SampleResult> = solutions
            .into_iter()
            .take(shots)
            .map(|solution| {
                let state: Vec<bool> = (0..matrix.nrows())
                    .map(|i| *solution.get(&format!("x{i}")).unwrap_or(&false))
                    .collect();
                let energy = solver.compute_energy(&state, matrix);

                SampleResult {
                    assignments: solution,
                    energy,
                    occurrences: 1,
                }
            })
            .collect();

        Ok(results)
    }

    fn run_hobo(
        &self,
        _problem: &(ArrayD<f64>, HashMap<String, usize>),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::UnsupportedOperation(
            "HOBO sampling not yet implemented for Grover-amplified sampler".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grover_solver_creation() {
        let config = GroverAmplificationConfig::default();
        let solver = GroverAmplifiedSolver::new(config);
        assert_eq!(solver.best_energy, f64::INFINITY);
    }

    #[test]
    fn test_energy_computation() {
        let config = GroverAmplificationConfig::default();
        let solver = GroverAmplifiedSolver::new(config);

        let mut qubo = Array2::zeros((3, 3));
        qubo[[0, 0]] = -1.0;
        qubo[[1, 1]] = -1.0;
        qubo[[2, 2]] = -1.0;

        let state = vec![true, true, false];
        let energy = solver.compute_energy(&state, &qubo);
        assert_eq!(energy, -2.0);
    }

    #[test]
    fn test_small_problem_optimization() {
        let config = GroverAmplificationConfig {
            grover_iterations: 10,
            population_size: 20,
            ..Default::default()
        };
        let mut solver = GroverAmplifiedSolver::new(config);

        // Simple QUBO: minimize x0 + x1
        let mut qubo = Array2::zeros((2, 2));
        qubo[[0, 0]] = 1.0;
        qubo[[1, 1]] = 1.0;

        let _results = solver.optimize(&qubo).unwrap();

        // Best solution should be [false, false] with energy 0
        assert!(solver.best_energy <= 0.1);
    }

    #[test]
    fn test_oracle_marking() {
        let config = GroverAmplificationConfig::default();
        let solver = GroverAmplifiedSolver::new(config);

        let mut population = vec![
            AmplitudeState {
                state: vec![true, false],
                energy: -2.0,
                amplitude: 0.5,
                distance_to_best: 0,
            },
            AmplitudeState {
                state: vec![false, false],
                energy: 0.0,
                amplitude: 0.5,
                distance_to_best: 1,
            },
        ];

        solver.oracle_marking(&mut population);

        // Lower energy should have higher amplitude
        assert!(population[0].amplitude > population[1].amplitude);
    }

    #[test]
    fn test_diffusion_operator() {
        let config = GroverAmplificationConfig::default();
        let solver = GroverAmplifiedSolver::new(config);

        let mut population = vec![
            AmplitudeState {
                state: vec![true, false],
                energy: -2.0,
                amplitude: 0.8,
                distance_to_best: 0,
            },
            AmplitudeState {
                state: vec![false, false],
                energy: 0.0,
                amplitude: 0.2,
                distance_to_best: 1,
            },
        ];

        solver.diffusion_operator(&mut population);

        // Amplitudes should still sum to ~1
        let total: f64 = population.iter().map(|s| s.amplitude).sum();
        assert!((total - 1.0).abs() < 0.01);
    }
}
