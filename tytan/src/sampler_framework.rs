//! Sampler framework extensions for advanced optimization strategies.
//!
//! This module provides plugin architecture, hyperparameter optimization,
//! ensemble methods, and adaptive sampling strategies.

#[cfg(feature = "dwave")]
use crate::compile::CompiledModel;
use crate::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
use ndarray::{Array, Array2, IxDyn};
use rand::prelude::*;
use rand::thread_rng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "scirs")]
use crate::scirs_stub::{
    scirs2_ml::{CrossValidation, RandomForest},
    scirs2_optimization::bayesian::{AcquisitionFunction, BayesianOptimizer, KernelType},
};

/// Plugin trait for custom samplers
pub trait SamplerPlugin: Send + Sync {
    /// Plugin name
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// Initialize plugin
    fn initialize(&mut self, config: &HashMap<String, String>) -> Result<(), String>;

    /// Create sampler instance
    fn create_sampler(&self) -> Box<dyn Sampler>;

    /// Get default configuration
    fn default_config(&self) -> HashMap<String, String>;

    /// Validate configuration
    fn validate_config(&self, config: &HashMap<String, String>) -> Result<(), String>;
}

/// Plugin manager for dynamic sampler loading
pub struct PluginManager {
    /// Registered plugins
    plugins: HashMap<String, Box<dyn SamplerPlugin>>,
    /// Plugin configurations
    configs: HashMap<String, HashMap<String, String>>,
}

impl PluginManager {
    /// Create new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            configs: HashMap::new(),
        }
    }

    /// Register a plugin
    pub fn register_plugin(&mut self, plugin: Box<dyn SamplerPlugin>) -> Result<(), String> {
        let name = plugin.name().to_string();

        if self.plugins.contains_key(&name) {
            return Err(format!("Plugin {} already registered", name));
        }

        let default_config = plugin.default_config();
        self.configs.insert(name.clone(), default_config);
        self.plugins.insert(name, plugin);

        Ok(())
    }

    /// Configure plugin
    pub fn configure_plugin(
        &mut self,
        name: &str,
        config: HashMap<String, String>,
    ) -> Result<(), String> {
        let plugin = self
            .plugins
            .get(name)
            .ok_or_else(|| format!("Plugin {} not found", name))?;

        plugin.validate_config(&config)?;
        self.configs.insert(name.to_string(), config);

        Ok(())
    }

    /// Create sampler from plugin
    pub fn create_sampler(&mut self, name: &str) -> Result<Box<dyn Sampler>, String> {
        let plugin = self
            .plugins
            .get_mut(name)
            .ok_or_else(|| format!("Plugin {} not found", name))?;

        let config = self.configs.get(name).cloned().unwrap_or_default();
        plugin.initialize(&config)?;

        Ok(plugin.create_sampler())
    }

    /// List available plugins
    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        self.plugins
            .values()
            .map(|p| PluginInfo {
                name: p.name().to_string(),
                version: p.version().to_string(),
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
}

/// Hyperparameter optimization for samplers
pub struct HyperparameterOptimizer {
    /// Parameter search space
    search_space: HashMap<String, ParameterSpace>,
    /// Optimization method
    method: OptimizationMethod,
    /// Number of trials
    num_trials: usize,
    /// Cross-validation folds
    cv_folds: usize,
}

#[derive(Debug, Clone)]
pub enum ParameterSpace {
    /// Continuous parameter
    Continuous { min: f64, max: f64, log_scale: bool },
    /// Discrete parameter
    Discrete { values: Vec<f64> },
    /// Categorical parameter
    Categorical { options: Vec<String> },
}

#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    /// Random search
    RandomSearch,
    /// Grid search
    GridSearch { resolution: usize },
    /// Bayesian optimization
    #[cfg(feature = "scirs")]
    Bayesian {
        kernel: KernelType,
        acquisition: AcquisitionFunction,
        exploration: f64,
    },
    /// Evolutionary optimization
    Evolutionary {
        population_size: usize,
        mutation_rate: f64,
    },
}

impl HyperparameterOptimizer {
    /// Create new optimizer
    pub fn new(method: OptimizationMethod, num_trials: usize) -> Self {
        Self {
            search_space: HashMap::new(),
            method,
            num_trials,
            cv_folds: 5,
        }
    }

    /// Add parameter to search space
    pub fn add_parameter(&mut self, name: &str, space: ParameterSpace) {
        self.search_space.insert(name.to_string(), space);
    }

    /// Optimize hyperparameters
    #[cfg(feature = "dwave")]
    pub fn optimize<F>(
        &self,
        objective: F,
        validation_problems: &[CompiledModel],
    ) -> Result<OptimizationResult, String>
    where
        F: Fn(&HashMap<String, f64>) -> Box<dyn Sampler>,
    {
        match &self.method {
            OptimizationMethod::RandomSearch => self.random_search(objective, validation_problems),
            OptimizationMethod::GridSearch { resolution } => {
                self.grid_search(objective, validation_problems, *resolution)
            }
            #[cfg(feature = "scirs")]
            OptimizationMethod::Bayesian {
                kernel,
                acquisition,
                exploration,
            } => self.bayesian_optimization(
                objective,
                validation_problems,
                *kernel,
                *acquisition,
                *exploration,
            ),
            OptimizationMethod::Evolutionary {
                population_size,
                mutation_rate,
            } => self.evolutionary_optimization(
                objective,
                validation_problems,
                *population_size,
                *mutation_rate,
            ),
            _ => Err("Optimization method not available".to_string()),
        }
    }

    /// Random search implementation
    #[cfg(feature = "dwave")]
    fn random_search<F>(
        &self,
        objective: F,
        validation_problems: &[CompiledModel],
    ) -> Result<OptimizationResult, String>
    where
        F: Fn(&HashMap<String, f64>) -> Box<dyn Sampler>,
    {
        let mut rng = thread_rng();
        let mut best_params = HashMap::new();
        let mut best_score = f64::INFINITY;
        let mut history = Vec::new();

        for trial in 0..self.num_trials {
            // Sample random parameters
            let params = self.sample_parameters(&mut rng)?;

            // Evaluate
            let sampler = objective(&params);
            let score = self.evaluate_sampler(sampler, validation_problems)?;

            history.push(TrialResult {
                parameters: params.clone(),
                score,
                iteration: trial,
            });

            if score < best_score {
                best_score = score;
                best_params = params;
            }
        }

        Ok(OptimizationResult {
            best_parameters: best_params,
            best_score,
            history,
            convergence_curve: self.compute_convergence_curve(&history),
        })
    }

    /// Grid search implementation
    #[cfg(feature = "dwave")]
    fn grid_search<F>(
        &self,
        objective: F,
        validation_problems: &[CompiledModel],
        resolution: usize,
    ) -> Result<OptimizationResult, String>
    where
        F: Fn(&HashMap<String, f64>) -> Box<dyn Sampler>,
    {
        // Generate grid points
        let grid_points = self.generate_grid(resolution)?;

        let mut best_params = HashMap::new();
        let mut best_score = f64::INFINITY;
        let mut history = Vec::new();

        for (i, params) in grid_points.iter().enumerate() {
            let sampler = objective(params);
            let score = self.evaluate_sampler(sampler, validation_problems)?;

            history.push(TrialResult {
                parameters: params.clone(),
                score,
                iteration: i,
            });

            if score < best_score {
                best_score = score;
                best_params = params.clone();
            }
        }

        Ok(OptimizationResult {
            best_parameters: best_params,
            best_score,
            history,
            convergence_curve: self.compute_convergence_curve(&history),
        })
    }

    /// Bayesian optimization implementation
    #[cfg(all(feature = "scirs", feature = "dwave"))]
    fn bayesian_optimization<F>(
        &self,
        objective: F,
        validation_problems: &[CompiledModel],
        kernel: KernelType,
        acquisition: AcquisitionFunction,
        exploration: f64,
    ) -> Result<OptimizationResult, String>
    where
        F: Fn(&HashMap<String, f64>) -> Box<dyn Sampler>,
    {
        use ndarray::Array1;

        let dim = self.search_space.len();
        let mut optimizer = BayesianOptimizer::new(dim, kernel, acquisition, exploration)?;

        let mut history = Vec::new();
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        // Initial random samples
        let mut rng = thread_rng();
        for _ in 0..std::cmp::min(10, self.num_trials / 4) {
            let params = self.sample_parameters(&mut rng)?;
            let sampler = objective(&params);
            let score = self.evaluate_sampler(sampler, validation_problems)?;

            let x = self.params_to_array(&params)?;
            x_data.push(x);
            y_data.push(score);

            history.push(TrialResult {
                parameters: params,
                score,
                iteration: history.len(),
            });
        }

        // Bayesian optimization loop
        let y_array = Array1::from_vec(y_data.clone());
        optimizer.update(&x_data, &y_array)?;

        for _ in history.len()..self.num_trials {
            // Suggest next point
            let x_next = optimizer.suggest_next()?;
            let params = self.array_to_params(&x_next)?;

            // Evaluate
            let sampler = objective(&params);
            let score = self.evaluate_sampler(sampler, validation_problems)?;

            // Update model
            x_data.push(x_next);
            y_data.push(score);
            let y_array = Array1::from_vec(y_data.clone());
            optimizer.update(&x_data, &y_array)?;

            history.push(TrialResult {
                parameters: params,
                score,
                iteration: history.len(),
            });
        }

        // Find best
        let (best_idx, &best_score) = y_data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let best_params = self.array_to_params(&x_data[best_idx])?;

        Ok(OptimizationResult {
            best_parameters: best_params,
            best_score,
            history,
            convergence_curve: self.compute_convergence_curve(&history),
        })
    }

    /// Evolutionary optimization (placeholder)
    #[cfg(feature = "dwave")]
    fn evolutionary_optimization<F>(
        &self,
        _objective: F,
        _validation_problems: &[CompiledModel],
        _population_size: usize,
        _mutation_rate: f64,
    ) -> Result<OptimizationResult, String>
    where
        F: Fn(&HashMap<String, f64>) -> Box<dyn Sampler>,
    {
        Err("Evolutionary optimization not yet implemented".to_string())
    }

    /// Sample parameters from search space
    fn sample_parameters(&self, rng: &mut impl Rng) -> Result<HashMap<String, f64>, String> {
        let mut params = HashMap::new();

        for (name, space) in &self.search_space {
            let value = match space {
                ParameterSpace::Continuous {
                    min,
                    max,
                    log_scale,
                } => {
                    if *log_scale {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let log_val = rng.gen_range(log_min..log_max);
                        log_val.exp()
                    } else {
                        rng.gen_range(*min..*max)
                    }
                }
                ParameterSpace::Discrete { values } => values[rng.gen_range(0..values.len())],
                ParameterSpace::Categorical { options } => {
                    // Return index for categorical
                    rng.gen_range(0..options.len()) as f64
                }
            };

            params.insert(name.clone(), value);
        }

        Ok(params)
    }

    /// Generate grid points
    fn generate_grid(&self, resolution: usize) -> Result<Vec<HashMap<String, f64>>, String> {
        // Simplified: generate regular grid
        let mut grid_points = Vec::new();

        // This would need proper multi-dimensional grid generation
        // For now, just sample uniformly
        let total_points = resolution.pow(self.search_space.len() as u32);
        let mut rng = thread_rng();

        for _ in 0..total_points.min(self.num_trials) {
            grid_points.push(self.sample_parameters(&mut rng)?);
        }

        Ok(grid_points)
    }

    /// Convert parameters to array
    #[cfg(feature = "scirs")]
    fn params_to_array(
        &self,
        params: &HashMap<String, f64>,
    ) -> Result<ndarray::Array1<f64>, String> {
        let mut values = Vec::new();

        // Ensure consistent ordering
        let mut names: Vec<_> = self.search_space.keys().collect();
        names.sort();

        for name in names {
            values.push(params.get(name).copied().unwrap_or(0.0));
        }

        Ok(ndarray::Array1::from_vec(values))
    }

    /// Convert array to parameters
    #[cfg(feature = "scirs")]
    fn array_to_params(
        &self,
        array: &ndarray::Array1<f64>,
    ) -> Result<HashMap<String, f64>, String> {
        let mut params = HashMap::new();

        let mut names: Vec<_> = self.search_space.keys().collect();
        names.sort();

        for (i, name) in names.iter().enumerate() {
            params.insert((*name).clone(), array[i]);
        }

        Ok(params)
    }

    /// Evaluate sampler performance
    #[cfg(feature = "dwave")]
    fn evaluate_sampler(
        &self,
        mut sampler: Box<dyn Sampler>,
        problems: &[CompiledModel],
    ) -> Result<f64, String> {
        let mut scores = Vec::new();

        for problem in problems {
            let qubo = problem.to_qubo();
            let start = Instant::now();

            let results = sampler.run_qubo(&qubo, 100)?;

            let elapsed = start.elapsed();

            // Score based on solution quality and time
            let best_energy = results.first().map(|r| r.energy).unwrap_or(f64::INFINITY);

            let time_penalty = elapsed.as_secs_f64();
            let score = best_energy + 0.1 * time_penalty;

            scores.push(score);
        }

        // Return average score
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Compute convergence curve
    fn compute_convergence_curve(&self, history: &[TrialResult]) -> Vec<f64> {
        let mut curve = Vec::new();
        let mut best_so_far = f64::INFINITY;

        for trial in history {
            best_so_far = best_so_far.min(trial.score);
            curve.push(best_so_far);
        }

        curve
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_parameters: HashMap<String, f64>,
    pub best_score: f64,
    pub history: Vec<TrialResult>,
    pub convergence_curve: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrialResult {
    pub parameters: HashMap<String, f64>,
    pub score: f64,
    pub iteration: usize,
}

/// Ensemble sampler that combines multiple sampling strategies
pub struct EnsembleSampler {
    /// Base samplers
    samplers: Vec<Box<dyn Sampler>>,
    /// Combination method
    method: EnsembleMethod,
    /// Weights for weighted combination
    weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub enum EnsembleMethod {
    /// Simple voting
    Voting,
    /// Weighted voting
    WeightedVoting,
    /// Best of all
    BestOf,
    /// Sequential refinement
    Sequential,
    /// Parallel with aggregation
    Parallel,
}

impl EnsembleSampler {
    /// Create new ensemble sampler
    pub fn new(samplers: Vec<Box<dyn Sampler>>, method: EnsembleMethod) -> Self {
        Self {
            samplers,
            method,
            weights: None,
        }
    }

    /// Set weights for weighted voting
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }
}

impl Sampler for EnsembleSampler {
    fn run_qubo(
        &self,
        qubo: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        match &self.method {
            EnsembleMethod::Voting => self.voting_ensemble(qubo, shots),
            EnsembleMethod::WeightedVoting => self.weighted_voting_ensemble(qubo, shots),
            EnsembleMethod::BestOf => self.best_of_ensemble(qubo, shots),
            EnsembleMethod::Sequential => self.sequential_ensemble(qubo, shots),
            EnsembleMethod::Parallel => self.parallel_ensemble(qubo, shots),
        }
    }

    fn run_hobo(
        &self,
        hobo: &(Array<f64, IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Similar implementation for HOBO
        match &self.method {
            EnsembleMethod::Voting => self.voting_ensemble_hobo(hobo, shots),
            _ => Err(SamplerError::InvalidParameter(
                "HOBO ensemble not fully implemented".to_string(),
            )),
        }
    }
}

impl EnsembleSampler {
    /// Simple voting ensemble
    fn voting_ensemble(
        &self,
        qubo: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        let shots_per_sampler = shots / self.samplers.len();
        let mut all_results = Vec::new();

        // Run each sampler
        for sampler in &self.samplers {
            let results = sampler.run_qubo(qubo, shots_per_sampler)?;
            all_results.extend(results);
        }

        // Aggregate by voting
        let mut vote_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

        for result in all_results {
            let state: Vec<bool> = qubo
                .1
                .iter()
                .map(|(var, _)| result.assignments[var])
                .collect();

            let entry = vote_counts.entry(state).or_insert((result.energy, 0));
            entry.1 += result.occurrences;
        }

        // Convert back to results
        let mut final_results: Vec<SampleResult> = vote_counts
            .into_iter()
            .map(|(state, (energy, count))| {
                let assignments: HashMap<String, bool> = qubo
                    .1
                    .iter()
                    .zip(state.iter())
                    .map(|((var, _), &val)| (var.clone(), val))
                    .collect();

                SampleResult {
                    assignments,
                    energy,
                    occurrences: count,
                }
            })
            .collect();

        final_results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        Ok(final_results)
    }

    /// Weighted voting ensemble
    fn weighted_voting_ensemble(
        &self,
        qubo: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            SamplerError::InvalidParameter("Weights not set for weighted voting".to_string())
        })?;

        if weights.len() != self.samplers.len() {
            return Err(SamplerError::InvalidParameter(
                "Number of weights must match number of samplers".to_string(),
            ));
        }

        // Normalize weights
        let total_weight: f64 = weights.iter().sum();
        let normalized: Vec<f64> = weights.iter().map(|&w| w / total_weight).collect();

        let mut all_results = Vec::new();

        // Run each sampler with weighted shots
        for (sampler, &weight) in self.samplers.iter().zip(normalized.iter()) {
            let sampler_shots = (shots as f64 * weight).round() as usize;
            if sampler_shots > 0 {
                let results = sampler.run_qubo(qubo, sampler_shots)?;
                all_results.extend(results);
            }
        }

        // Aggregate results
        self.aggregate_results(all_results, &qubo.1)
    }

    /// Best-of ensemble
    fn best_of_ensemble(
        &self,
        qubo: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        let shots_per_sampler = shots / self.samplers.len();
        let mut best_results = Vec::new();
        let mut best_energy = f64::INFINITY;

        // Run each sampler and keep best
        for sampler in &self.samplers {
            let results = sampler.run_qubo(qubo, shots_per_sampler)?;

            if let Some(best) = results.first() {
                if best.energy < best_energy {
                    best_energy = best.energy;
                    best_results = results;
                }
            }
        }

        Ok(best_results)
    }

    /// Sequential refinement ensemble
    fn sequential_ensemble(
        &self,
        qubo: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        if self.samplers.is_empty() {
            return Ok(Vec::new());
        }

        // Start with first sampler
        let mut current_best = self.samplers[0].run_qubo(qubo, shots)?;

        // Refine with subsequent samplers
        for sampler in self.samplers.iter().skip(1) {
            // Use best solutions as warm start (if sampler supports it)
            // For now, just run independently
            let refined = sampler.run_qubo(qubo, shots / self.samplers.len())?;

            // Merge results
            current_best.extend(refined);
            current_best.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
            current_best.truncate(shots);
        }

        Ok(current_best)
    }

    /// Parallel ensemble with aggregation
    fn parallel_ensemble(
        &self,
        qubo: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        use std::thread;

        let shots_per_sampler = shots / self.samplers.len();
        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();

        // Would need to make samplers thread-safe for real parallel execution
        // For now, sequential execution
        let mut all_results = Vec::new();

        for sampler in &self.samplers {
            let results = sampler.run_qubo(qubo, shots_per_sampler)?;
            all_results.extend(results);
        }

        self.aggregate_results(all_results, &qubo.1)
    }

    /// Aggregate results from multiple samplers
    fn aggregate_results(
        &self,
        results: Vec<SampleResult>,
        var_map: &HashMap<String, usize>,
    ) -> SamplerResult<Vec<SampleResult>> {
        let mut aggregated: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

        for result in results {
            let state: Vec<bool> = var_map
                .iter()
                .map(|(var, _)| result.assignments[var])
                .collect();

            let entry = aggregated.entry(state).or_insert((result.energy, 0));

            // Keep minimum energy for duplicates
            entry.0 = entry.0.min(result.energy);
            entry.1 += result.occurrences;
        }

        let mut final_results: Vec<SampleResult> = aggregated
            .into_iter()
            .map(|(state, (energy, count))| {
                let assignments: HashMap<String, bool> = var_map
                    .iter()
                    .zip(state.iter())
                    .map(|((var, _), &val)| (var.clone(), val))
                    .collect();

                SampleResult {
                    assignments,
                    energy,
                    occurrences: count,
                }
            })
            .collect();

        final_results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        Ok(final_results)
    }

    /// Voting ensemble for HOBO
    fn voting_ensemble_hobo(
        &self,
        hobo: &(Array<f64, IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Similar to QUBO voting but for HOBO
        let shots_per_sampler = shots / self.samplers.len();
        let mut all_results = Vec::new();

        for sampler in &self.samplers {
            let results = sampler.run_hobo(hobo, shots_per_sampler)?;
            all_results.extend(results);
        }

        self.aggregate_results(all_results, &hobo.1)
    }
}

/// Adaptive sampling strategy
pub struct AdaptiveSampler<S: Sampler> {
    /// Base sampler
    base_sampler: S,
    /// Adaptation strategy
    strategy: AdaptationStrategy,
    /// Performance history
    history: Arc<Mutex<PerformanceHistory>>,
}

#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    /// Temperature adaptation
    TemperatureAdaptive {
        initial_range: (f64, f64),
        adaptation_rate: f64,
    },
    /// Population size adaptation
    PopulationAdaptive {
        min_size: usize,
        max_size: usize,
        growth_rate: f64,
    },
    /// Multi-armed bandit for strategy selection
    BanditAdaptive {
        strategies: Vec<String>,
        exploration_rate: f64,
    },
    /// Reinforcement learning based
    RLAdaptive {
        state_features: Vec<String>,
        action_space: Vec<String>,
    },
}

#[derive(Default)]
struct PerformanceHistory {
    energies: Vec<f64>,
    times: Vec<Duration>,
    improvements: Vec<f64>,
    parameters: Vec<HashMap<String, f64>>,
}

impl<S: Sampler> AdaptiveSampler<S> {
    /// Create new adaptive sampler
    pub fn new(base_sampler: S, strategy: AdaptationStrategy) -> Self {
        Self {
            base_sampler,
            strategy,
            history: Arc::new(Mutex::new(PerformanceHistory::default())),
        }
    }

    /// Adapt parameters based on performance
    fn adapt_parameters(&self) -> HashMap<String, f64> {
        let history = self.history.lock().unwrap();

        match &self.strategy {
            AdaptationStrategy::TemperatureAdaptive {
                initial_range,
                adaptation_rate,
            } => {
                let mut params = HashMap::new();

                // Adapt temperature based on acceptance rate
                let (min_temp, max_temp) = initial_range;
                let temp = if history.improvements.len() > 10 {
                    let recent_improvements: f64 =
                        history.improvements.iter().rev().take(10).sum::<f64>() / 10.0;

                    if recent_improvements < 0.1 {
                        // Low improvement: increase temperature
                        min_temp + (max_temp - min_temp) * (1.0 - adaptation_rate)
                    } else {
                        // Good improvement: decrease temperature
                        min_temp + (max_temp - min_temp) * adaptation_rate
                    }
                } else {
                    (min_temp + max_temp) / 2.0
                };

                params.insert("temperature".to_string(), temp);
                params
            }
            _ => HashMap::new(),
        }
    }
}

impl<S: Sampler> Sampler for AdaptiveSampler<S> {
    fn run_qubo(
        &self,
        qubo: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Adapt parameters
        let params = self.adapt_parameters();

        // Run base sampler (would need to apply params)
        let start = Instant::now();
        let results = self.base_sampler.run_qubo(qubo, shots)?;
        let elapsed = start.elapsed();

        // Update history
        if let Some(best) = results.first() {
            let mut history = self.history.lock().unwrap();

            let improvement = if let Some(&last) = history.energies.last() {
                (last - best.energy) / last.abs().max(1.0)
            } else {
                1.0
            };

            history.energies.push(best.energy);
            history.times.push(elapsed);
            history.improvements.push(improvement);
            history.parameters.push(params);
        }

        Ok(results)
    }

    fn run_hobo(
        &self,
        hobo: &(Array<f64, IxDyn>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Similar adaptation for HOBO
        self.base_sampler.run_hobo(hobo, shots)
    }
}

/// Cross-validation for sampler evaluation
pub struct SamplerCrossValidation {
    /// Number of folds
    n_folds: usize,
    /// Evaluation metric
    metric: EvaluationMetric,
}

#[derive(Debug, Clone)]
pub enum EvaluationMetric {
    /// Best energy found
    BestEnergy,
    /// Average of top-k energies
    TopKAverage(usize),
    /// Time to solution
    TimeToSolution(f64),
    /// Success probability
    SuccessProbability(f64),
}

impl SamplerCrossValidation {
    /// Create new cross-validation
    pub fn new(n_folds: usize, metric: EvaluationMetric) -> Self {
        Self { n_folds, metric }
    }

    /// Evaluate sampler with cross-validation
    #[cfg(feature = "dwave")]
    pub fn evaluate<S: Sampler>(
        &self,
        sampler: &S,
        problems: &[CompiledModel],
        shots_per_problem: usize,
    ) -> Result<CrossValidationResult, String> {
        let n_problems = problems.len();
        let fold_size = n_problems / self.n_folds;

        let mut fold_scores = Vec::new();

        for fold in 0..self.n_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == self.n_folds - 1 {
                n_problems
            } else {
                (fold + 1) * fold_size
            };

            let test_problems = &problems[test_start..test_end];

            // Evaluate on test fold
            let mut scores = Vec::new();
            for problem in test_problems {
                let score = self.evaluate_single(sampler, problem, shots_per_problem)?;
                scores.push(score);
            }

            let fold_score = scores.iter().sum::<f64>() / scores.len() as f64;
            fold_scores.push(fold_score);
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let variance = fold_scores
            .iter()
            .map(|&s| (s - mean_score).powi(2))
            .sum::<f64>()
            / fold_scores.len() as f64;

        Ok(CrossValidationResult {
            mean_score,
            std_error: variance.sqrt(),
            fold_scores,
        })
    }

    /// Evaluate single problem
    #[cfg(feature = "dwave")]
    fn evaluate_single<S: Sampler>(
        &self,
        sampler: &S,
        problem: &CompiledModel,
        shots: usize,
    ) -> Result<f64, String> {
        let qubo = problem.to_qubo();
        let start = Instant::now();
        let results = sampler.run_qubo(&qubo, shots)?;
        let elapsed = start.elapsed();

        match &self.metric {
            EvaluationMetric::BestEnergy => {
                Ok(results.first().map(|r| r.energy).unwrap_or(f64::INFINITY))
            }
            EvaluationMetric::TopKAverage(k) => {
                let sum: f64 = results.iter().take(*k).map(|r| r.energy).sum();
                Ok(sum / (*k).min(results.len()) as f64)
            }
            EvaluationMetric::TimeToSolution(threshold) => {
                let found = results.iter().any(|r| r.energy <= *threshold);
                Ok(if found {
                    elapsed.as_secs_f64()
                } else {
                    f64::INFINITY
                })
            }
            EvaluationMetric::SuccessProbability(threshold) => {
                let successes = results
                    .iter()
                    .filter(|r| r.energy <= *threshold)
                    .map(|r| r.occurrences)
                    .sum::<usize>();
                Ok(successes as f64 / shots as f64)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    pub mean_score: f64,
    pub std_error: f64,
    pub fold_scores: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SASampler;

    #[test]
    fn test_plugin_manager() {
        let mut manager = PluginManager::new();

        // Would need actual plugin implementation to test
        assert_eq!(manager.list_plugins().len(), 0);
    }

    #[test]
    fn test_hyperparameter_space() {
        let mut optimizer = HyperparameterOptimizer::new(OptimizationMethod::RandomSearch, 10);

        optimizer.add_parameter(
            "temperature",
            ParameterSpace::Continuous {
                min: 0.1,
                max: 10.0,
                log_scale: true,
            },
        );

        optimizer.add_parameter(
            "sweeps",
            ParameterSpace::Discrete {
                values: vec![100.0, 500.0, 1000.0],
            },
        );

        // Would need actual optimization to test further
    }

    #[test]
    fn test_ensemble_sampler() {
        let samplers: Vec<Box<dyn Sampler>> = vec![
            Box::new(SASampler::new(Some(42))),
            Box::new(SASampler::new(Some(43))),
        ];

        let ensemble = EnsembleSampler::new(samplers, EnsembleMethod::Voting);

        // Would need QUBO problem to test
    }
}
