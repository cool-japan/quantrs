//! Adaptive optimization strategies for quantum annealing
//!
//! This module provides adaptive algorithms that adjust parameters
//! during optimization based on performance feedback.

use crate::{
    optimization::penalty::CompiledModel,
    sampler::{SampleResult, Sampler},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "scirs")]
use crate::scirs_stub::scirs2_core::statistics::{MovingAverage, OnlineStats};

/// Adaptive strategy types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AdaptiveStrategy {
    /// Exponential decay of penalty weights
    ExponentialDecay,
    /// Adaptive penalty method (APM)
    AdaptivePenaltyMethod,
    /// Augmented Lagrangian with multiplier updates
    AugmentedLagrangian,
    /// Population-based training
    PopulationBased,
    /// Multi-armed bandit for parameter selection
    MultiArmedBandit,
}

/// Adaptive optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    pub strategy: AdaptiveStrategy,
    pub update_interval: usize,
    pub learning_rate: f64,
    pub momentum: f64,
    pub patience: usize,
    pub exploration_rate: f64,
    pub population_size: usize,
    pub history_window: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            strategy: AdaptiveStrategy::AdaptivePenaltyMethod,
            update_interval: 10,
            learning_rate: 0.1,
            momentum: 0.9,
            patience: 5,
            exploration_rate: 0.1,
            population_size: 10,
            history_window: 100,
        }
    }
}

/// Adaptive optimizer
pub struct AdaptiveOptimizer {
    config: AdaptiveConfig,
    iteration: usize,
    parameter_history: Vec<ParameterState>,
    performance_history: Vec<PerformanceMetrics>,
    lagrange_multipliers: HashMap<String, f64>,
    population: Vec<Individual>,
    #[cfg(feature = "scirs")]
    stats: OnlineStats,
}

/// Parameter state at a given iteration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParameterState {
    iteration: usize,
    parameters: HashMap<String, f64>,
    penalty_weights: HashMap<String, f64>,
    temperature: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub iteration: usize,
    pub best_energy: f64,
    pub avg_energy: f64,
    pub constraint_violations: HashMap<String, f64>,
    pub feasibility_rate: f64,
    pub diversity: f64,
}

/// Individual in population-based methods
#[derive(Debug, Clone)]
struct Individual {
    id: usize,
    parameters: HashMap<String, f64>,
    fitness: f64,
    constraint_satisfaction: f64,
}

/// Adaptive optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveResult {
    pub final_parameters: HashMap<String, f64>,
    pub final_penalty_weights: HashMap<String, f64>,
    pub convergence_history: Vec<f64>,
    pub constraint_history: Vec<HashMap<String, f64>>,
    pub total_iterations: usize,
    pub best_solution: AdaptiveSampleResult,
}

/// Sample result wrapper for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSampleResult {
    pub assignments: HashMap<String, bool>,
    pub energy: f64,
}

impl AdaptiveOptimizer {
    /// Create new adaptive optimizer
    pub fn new(config: AdaptiveConfig) -> Self {
        Self {
            config,
            iteration: 0,
            parameter_history: Vec::new(),
            performance_history: Vec::new(),
            lagrange_multipliers: HashMap::new(),
            population: Vec::new(),
            #[cfg(feature = "scirs")]
            stats: OnlineStats::new(),
        }
    }

    /// Run adaptive optimization
    pub fn optimize<S: Sampler + Clone>(
        &mut self,
        mut sampler: S,
        model: &CompiledModel,
        initial_params: HashMap<String, f64>,
        initial_penalties: HashMap<String, f64>,
        max_iterations: usize,
    ) -> Result<AdaptiveResult, Box<dyn std::error::Error>> {
        // Initialize
        let current_params = initial_params;
        let penalty_weights = initial_penalties;
        let best_solution = None;
        let best_energy = f64::INFINITY;

        // Initialize strategy-specific components
        match self.config.strategy {
            AdaptiveStrategy::PopulationBased => {
                self.initialize_population(&current_params)?;
            }
            AdaptiveStrategy::AugmentedLagrangian => {
                self.initialize_lagrange_multipliers(&penalty_weights);
            }
            _ => {}
        }

        // Main optimization loop
        let no_improvement_count = 0;

        for iter in 0..max_iterations {
            self.iteration = iter;

            // Run sampling with current parameters
            let samples =
                self.run_sampling(&mut sampler, model, &current_params, &penalty_weights)?;

            // Evaluate performance
            let metrics = self.evaluate_performance(model, &samples)?;
            self.performance_history.push(metrics.clone());

            // Update best solution
            if let Some(sample) = samples
                .iter()
                .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            {
                if sample.energy < best_energy {
                    best_energy = sample.energy;
                    best_solution = Some(AdaptiveSampleResult {
                        assignments: sample.assignments.clone(),
                        energy: sample.energy,
                    });
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                }
            }

            // Check early stopping
            if no_improvement_count > self.config.patience {
                break;
            }

            // Update parameters based on strategy
            if iter % self.config.update_interval == 0 && iter > 0 {
                self.update_parameters(&mut current_params, &mut penalty_weights, &metrics)?;
            }

            // Record state
            self.parameter_history.push(ParameterState {
                iteration: iter,
                parameters: current_params.clone(),
                penalty_weights: penalty_weights.clone(),
                temperature: self.calculate_temperature(iter, max_iterations),
            });
        }

        // Prepare result
        let convergence_history = self
            .performance_history
            .iter()
            .map(|m| m.best_energy)
            .collect();

        let constraint_history = self
            .performance_history
            .iter()
            .map(|m| m.constraint_violations.clone())
            .collect();

        Ok(AdaptiveResult {
            final_parameters: current_params,
            final_penalty_weights: penalty_weights,
            convergence_history,
            constraint_history,
            total_iterations: self.iteration,
            best_solution: best_solution.ok_or("No valid solution found")?,
        })
    }

    /// Run sampling with current parameters
    fn run_sampling<S: Sampler>(
        &self,
        sampler: &mut S,
        model: &CompiledModel,
        params: &HashMap<String, f64>,
        penalty_weights: &HashMap<String, f64>,
    ) -> Result<Vec<SampleResult>, Box<dyn std::error::Error>> {
        // Apply penalty weights to model
        let penalized_model = self.apply_penalties(model, penalty_weights)?;

        // Configure sampler with parameters
        sampler.set_parameters(params.clone());

        // Run sampling
        let num_reads = params.get("num_reads").copied().unwrap_or(100.0) as usize;

        Ok(sampler.run_qubo(&penalized_model.to_qubo(), num_reads)?)
    }

    /// Evaluate performance metrics
    fn evaluate_performance(
        &self,
        model: &CompiledModel,
        samples: &[SampleResult],
    ) -> Result<PerformanceMetrics, Box<dyn std::error::Error>> {
        let energies: Vec<f64> = samples.iter().map(|s| s.energy).collect();
        let best_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let avg_energy = energies.iter().sum::<f64>() / energies.len() as f64;

        // Evaluate constraint violations
        let constraint_violations = self.evaluate_constraint_violations(model, samples)?;

        // Calculate feasibility rate
        let feasible_count = samples
            .iter()
            .filter(|s| self.is_feasible(s, &constraint_violations).unwrap_or(false))
            .count();
        let feasibility_rate = feasible_count as f64 / samples.len() as f64;

        // Calculate diversity
        let diversity = self.calculate_diversity(samples);

        Ok(PerformanceMetrics {
            iteration: self.iteration,
            best_energy,
            avg_energy,
            constraint_violations,
            feasibility_rate,
            diversity,
        })
    }

    /// Update parameters based on adaptive strategy
    fn update_parameters(
        &mut self,
        params: &mut HashMap<String, f64>,
        penalty_weights: &mut HashMap<String, f64>,
        metrics: &PerformanceMetrics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self.config.strategy {
            AdaptiveStrategy::ExponentialDecay => {
                self.update_exponential_decay(params, penalty_weights)?;
            }
            AdaptiveStrategy::AdaptivePenaltyMethod => {
                self.update_adaptive_penalty(penalty_weights, metrics)?;
            }
            AdaptiveStrategy::AugmentedLagrangian => {
                self.update_augmented_lagrangian(penalty_weights, metrics)?;
            }
            AdaptiveStrategy::PopulationBased => {
                self.update_population_based(params, penalty_weights, metrics)?;
            }
            AdaptiveStrategy::MultiArmedBandit => {
                self.update_multi_armed_bandit(params, metrics)?;
            }
        }

        Ok(())
    }

    /// Exponential decay update
    fn update_exponential_decay(
        &self,
        params: &mut HashMap<String, f64>,
        penalty_weights: &mut HashMap<String, f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let decay_rate = 0.95;

        // Decay temperature parameter
        if let Some(temp) = params.get_mut("temperature") {
            *temp *= decay_rate;
        }

        // Optionally adjust penalty weights
        for weight in penalty_weights.values_mut() {
            *weight *= 1.0 / decay_rate.sqrt(); // Increase penalties as temperature decreases
        }

        Ok(())
    }

    /// Adaptive penalty method update
    fn update_adaptive_penalty(
        &mut self,
        penalty_weights: &mut HashMap<String, f64>,
        metrics: &PerformanceMetrics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Update penalties based on constraint violations
        for (constraint_name, &violation) in &metrics.constraint_violations {
            if let Some(weight) = penalty_weights.get_mut(constraint_name) {
                if violation > 1e-6 {
                    // Increase penalty
                    *weight *= 1.0 + self.config.learning_rate;
                } else {
                    // Decrease penalty if over-penalized
                    *weight *= 1.0 - self.config.learning_rate * 0.5;
                }

                // Apply bounds
                *weight = weight.clamp(0.001, 1000.0);
            }
        }

        #[cfg(feature = "scirs")]
        {
            // Update statistics
            self.stats.update(metrics.best_energy);
        }

        Ok(())
    }

    /// Augmented Lagrangian update
    fn update_augmented_lagrangian(
        &mut self,
        penalty_weights: &mut HashMap<String, f64>,
        metrics: &PerformanceMetrics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Update Lagrange multipliers
        for (constraint_name, &violation) in &metrics.constraint_violations {
            let multiplier = self
                .lagrange_multipliers
                .entry(constraint_name.clone())
                .or_insert(0.0);

            // Gradient ascent on multipliers
            *multiplier += self.config.learning_rate * violation;

            // Update penalty weight (augmented term)
            if let Some(weight) = penalty_weights.get_mut(constraint_name) {
                *weight = multiplier.abs() + 0.5 * weight.sqrt();
            }
        }

        Ok(())
    }

    /// Population-based update
    fn update_population_based(
        &mut self,
        params: &mut HashMap<String, f64>,
        _penalty_weights: &mut HashMap<String, f64>,
        metrics: &PerformanceMetrics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Evaluate population fitness
        let fitness_values: Vec<f64> = self
            .population
            .iter()
            .map(|individual| self.evaluate_individual_fitness(individual, metrics))
            .collect::<Result<Vec<_>, _>>()?;

        for (i, fitness) in fitness_values.into_iter().enumerate() {
            self.population[i].fitness = fitness;
        }

        // Sort by fitness
        self.population
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Exploit: copy parameters from best individuals
        if let Some(best) = self.population.first() {
            for (key, value) in &best.parameters {
                if let Some(param) = params.get_mut(key) {
                    *param = self.config.momentum * *param + (1.0 - self.config.momentum) * value;
                }
            }
        }

        // Explore: perturb bottom half of population
        let mid = self.population.len() / 2;
        let pop_len = self.population.len();
        for i in mid..pop_len {
            // Use random perturbation directly to avoid borrow issues
            use rand::Rng;
            let rng = rand::rng();

            for value in self.population[i].parameters.values_mut() {
                if rng.random::<f64>() < 0.3 {
                    let perturbation = rng.random_range(-0.3..0.3) * value.abs();
                    *value += perturbation;
                }
            }
        }

        Ok(())
    }

    /// Multi-armed bandit update
    fn update_multi_armed_bandit(
        &mut self,
        params: &mut HashMap<String, f64>,
        _metrics: &PerformanceMetrics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implement UCB or Thompson sampling for parameter selection
        // This is a simplified version

        use rand::Rng;
        let rng = rand::rng();

        for (param_name, param_value) in params.iter_mut() {
            if rng.random::<f64>() < self.config.exploration_rate {
                // Explore: random perturbation
                let perturbation = rng.random_range(-0.1..0.1) * param_value.abs();
                *param_value += perturbation;
            } else {
                // Exploit: move toward historical best
                if let Some(best_state) = self.parameter_history.iter().min_by(|a, b| {
                    let a_metrics = &self.performance_history[a.iteration];
                    let b_metrics = &self.performance_history[b.iteration];
                    a_metrics
                        .best_energy
                        .partial_cmp(&b_metrics.best_energy)
                        .unwrap()
                }) {
                    if let Some(best_value) = best_state.parameters.get(param_name) {
                        *param_value += self.config.learning_rate * (best_value - *param_value);
                    }
                }
            }
        }

        Ok(())
    }

    /// Initialize population for population-based methods
    fn initialize_population(
        &mut self,
        base_params: &HashMap<String, f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rand::Rng;
        let rng = rand::rng();

        for i in 0..self.config.population_size {
            let params = base_params.clone();

            // Add random perturbations
            for value in params.values_mut() {
                let perturbation = rng.random_range(-0.2..0.2) * value.abs();
                *value += perturbation;
            }

            self.population.push(Individual {
                id: i,
                parameters: params,
                fitness: 0.0,
                constraint_satisfaction: 0.0,
            });
        }

        Ok(())
    }

    /// Initialize Lagrange multipliers
    fn initialize_lagrange_multipliers(&mut self, penalty_weights: &HashMap<String, f64>) {
        for (constraint_name, &weight) in penalty_weights {
            self.lagrange_multipliers
                .insert(constraint_name.clone(), weight * 0.1);
        }
    }

    /// Apply penalties to model
    fn apply_penalties(
        &self,
        model: &CompiledModel,
        _penalty_weights: &HashMap<String, f64>,
    ) -> Result<CompiledModel, Box<dyn std::error::Error>> {
        // This would modify the model's QUBO matrix with penalty terms
        // For now, return the original model
        Ok(model.clone())
    }

    /// Evaluate constraint violations
    fn evaluate_constraint_violations(
        &self,
        model: &CompiledModel,
        _samples: &[SampleResult],
    ) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        // Placeholder implementation
        let violations = HashMap::new();

        for constraint_name in model.get_constraints().keys() {
            violations.insert(constraint_name.clone(), 0.0);
        }

        Ok(violations)
    }

    /// Check if solution is feasible
    fn is_feasible(
        &self,
        sample: &SampleResult,
        constraint_violations: &HashMap<String, f64>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let max_violation = constraint_violations
            .values()
            .fold(0.0f64, |a, &b| a.max(b.abs()));

        Ok(max_violation < 1e-6)
    }

    /// Calculate solution diversity
    fn calculate_diversity(&self, samples: &[SampleResult]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }

        let total_distance = 0.0;
        let count = 0;

        for i in 0..samples.len() {
            for j in i + 1..samples.len() {
                let distance = self.hamming_distance(&samples[i], &samples[j]);
                total_distance += distance as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }

    /// Calculate Hamming distance between solutions
    fn hamming_distance(&self, a: &SampleResult, b: &SampleResult) -> usize {
        a.assignments
            .iter()
            .filter(|(var, &val_a)| b.assignments.get(*var).copied().unwrap_or(false) != val_a)
            .count()
    }

    /// Calculate temperature for annealing schedule
    fn calculate_temperature(&self, iteration: usize, max_iterations: usize) -> f64 {
        let progress = iteration as f64 / max_iterations as f64;
        let initial_temp = 10.0f64;
        let final_temp = 0.01f64;

        initial_temp * (final_temp / initial_temp).powf(progress)
    }

    /// Evaluate individual fitness in population
    fn evaluate_individual_fitness(
        &self,
        _individual: &Individual,
        metrics: &PerformanceMetrics,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Combine objective value and constraint satisfaction
        let objective_score = 1.0 / (1.0 + metrics.best_energy.abs());
        let constraint_score = metrics.feasibility_rate;

        Ok(0.7 * objective_score + 0.3 * constraint_score)
    }

    /// Perturb individual in population
    fn perturb_individual(
        &self,
        individual: &mut Individual,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rand::Rng;
        let rng = rand::rng();

        for value in individual.parameters.values_mut() {
            if rng.random::<f64>() < 0.3 {
                let perturbation = rng.random_range(-0.3..0.3) * value.abs();
                *value += perturbation;
            }
        }

        Ok(())
    }

    /// Export optimization history
    pub fn export_history(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let export = AdaptiveExport {
            config: self.config.clone(),
            parameter_history: self.parameter_history.clone(),
            performance_history: self.performance_history.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        let json = serde_json::to_string_pretty(&export)?;
        std::fs::write(path, json)?;

        Ok(())
    }
}

/// Export format for adaptive optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveExport {
    pub config: AdaptiveConfig,
    pub parameter_history: Vec<ParameterState>,
    pub performance_history: Vec<PerformanceMetrics>,
    pub timestamp: std::time::SystemTime,
}

// Helper trait for sampler parameter setting
trait SamplerExt {
    fn set_parameters(&mut self, params: HashMap<String, f64>);
}

impl<S: Sampler> SamplerExt for S {
    fn set_parameters(&mut self, _params: HashMap<String, f64>) {
        // This would be implemented by specific samplers
        // For now, it's a no-op
    }
}
