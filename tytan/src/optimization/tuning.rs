//! Parameter tuning for quantum annealing
//!
//! This module provides automated parameter tuning using Bayesian optimization
//! and other advanced techniques with SciRS2 integration.

use crate::{
    optimization::penalty::CompiledModel,
    sampler::{SampleResult, Sampler},
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "scirs")]
use crate::scirs_stub::scirs2_optimization::bayesian::{
    AcquisitionFunction, BayesianOptimizer, GaussianProcess,
};

/// Parameter bounds definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterBounds {
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub scale: ParameterScale,
    pub integer: bool,
}

/// Parameter scaling type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ParameterScale {
    Linear,
    Logarithmic,
    Sigmoid,
}

/// Tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningConfig {
    /// Maximum number of evaluations
    pub max_evaluations: usize,
    /// Number of initial random samples
    pub initial_samples: usize,
    /// Acquisition function for Bayesian optimization
    pub acquisition: AcquisitionType,
    /// Exploration vs exploitation trade-off
    pub exploration_factor: f64,
    /// Number of parallel evaluations
    pub parallel_evaluations: usize,
    /// Early stopping tolerance
    pub early_stopping_tolerance: f64,
    /// Random seed
    pub seed: Option<u64>,
}

/// Acquisition function types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AcquisitionType {
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    ThompsonSampling,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            max_evaluations: 100,
            initial_samples: 20,
            acquisition: AcquisitionType::ExpectedImprovement,
            exploration_factor: 0.1,
            parallel_evaluations: 1,
            early_stopping_tolerance: 1e-6,
            seed: None,
        }
    }
}

/// Parameter tuner
pub struct ParameterTuner {
    config: TuningConfig,
    parameter_bounds: Vec<ParameterBounds>,
    evaluation_history: Vec<TuningEvaluation>,
    #[cfg(feature = "scirs")]
    optimizer: Option<BayesianOptimizer>,
}

/// Single tuning evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningEvaluation {
    pub parameters: HashMap<String, f64>,
    pub objective_value: f64,
    pub constraint_violations: HashMap<String, f64>,
    pub evaluation_time: std::time::Duration,
    pub iteration: usize,
}

/// Tuning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    pub best_parameters: HashMap<String, f64>,
    pub best_objective: f64,
    pub convergence_history: Vec<f64>,
    pub total_evaluations: usize,
    pub converged: bool,
    pub improvement_over_default: f64,
    pub parameter_importance: HashMap<String, f64>,
}

impl ParameterTuner {
    /// Create new parameter tuner
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            parameter_bounds: Vec::new(),
            evaluation_history: Vec::new(),
            #[cfg(feature = "scirs")]
            optimizer: None,
        }
    }

    /// Add parameter to tune
    pub fn add_parameter(&mut self, bounds: ParameterBounds) {
        self.parameter_bounds.push(bounds);
    }

    /// Add multiple parameters
    pub fn add_parameters(&mut self, bounds: Vec<ParameterBounds>) {
        self.parameter_bounds.extend(bounds);
    }

    /// Tune parameters for a sampler
    pub fn tune_sampler<S: Sampler>(
        &mut self,
        sampler_factory: impl Fn(HashMap<String, f64>) -> S,
        model: &CompiledModel,
        objective: impl Fn(&[SampleResult]) -> f64,
    ) -> Result<TuningResult, Box<dyn std::error::Error>> {
        // Initialize optimizer
        self.initialize_optimizer()?;

        // Get default parameters for baseline
        let default_params = self.get_default_parameters();
        let default_objective =
            self.evaluate_configuration(&default_params, &sampler_factory, model, &objective)?;

        // Initial random sampling
        for i in 0..self.config.initial_samples {
            let params = self.sample_random_parameters(i as u64);
            let obj_value =
                self.evaluate_configuration(&params, &sampler_factory, model, &objective)?;

            self.record_evaluation(params, obj_value, HashMap::new(), i);
        }

        // Bayesian optimization loop
        let mut best_objective = self
            .evaluation_history
            .iter()
            .map(|e| e.objective_value)
            .fold(f64::INFINITY, f64::min);

        let mut no_improvement_count = 0;

        for i in self.config.initial_samples..self.config.max_evaluations {
            // Get next parameters to evaluate
            let next_params = self.get_next_parameters()?;

            // Evaluate
            let obj_value =
                self.evaluate_configuration(&next_params, &sampler_factory, model, &objective)?;

            self.record_evaluation(next_params, obj_value, HashMap::new(), i);

            // Check for improvement
            if obj_value < best_objective - self.config.early_stopping_tolerance {
                best_objective = obj_value;
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;
            }

            // Early stopping
            if no_improvement_count > 10 {
                break;
            }
        }

        // Analyze results
        let best_eval = self
            .evaluation_history
            .iter()
            .min_by(|a, b| a.objective_value.partial_cmp(&b.objective_value).unwrap())
            .unwrap();

        let convergence_history: Vec<f64> = self
            .evaluation_history
            .iter()
            .scan(f64::INFINITY, |best, eval| {
                *best = best.min(eval.objective_value);
                Some(*best)
            })
            .collect();

        let parameter_importance = self.calculate_parameter_importance()?;

        Ok(TuningResult {
            best_parameters: best_eval.parameters.clone(),
            best_objective: best_eval.objective_value,
            convergence_history,
            total_evaluations: self.evaluation_history.len(),
            converged: no_improvement_count > 10,
            improvement_over_default: (default_objective - best_eval.objective_value)
                / default_objective.abs(),
            parameter_importance,
        })
    }

    /// Initialize optimizer
    fn initialize_optimizer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_optimization::bayesian::KernelType;

            let dim = self.parameter_bounds.len();
            let kernel = KernelType::Matern52;

            self.optimizer = Some(BayesianOptimizer::new(
                dim,
                kernel,
                self.config.acquisition.into(),
                self.config.exploration_factor,
            )?);
        }

        Ok(())
    }

    /// Get default parameters (middle of bounds)
    fn get_default_parameters(&self) -> HashMap<String, f64> {
        self.parameter_bounds
            .iter()
            .map(|b| {
                let value = match b.scale {
                    ParameterScale::Linear => (b.min + b.max) / 2.0,
                    ParameterScale::Logarithmic => { (b.min.ln() + b.max.ln()) / 2.0 }.exp(),
                    ParameterScale::Sigmoid => (b.min + b.max) / 2.0,
                };
                (
                    b.name.clone(),
                    if b.integer { value.round() } else { value },
                )
            })
            .collect()
    }

    /// Sample random parameters
    fn sample_random_parameters(&self, seed: u64) -> HashMap<String, f64> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed + self.config.seed.unwrap_or(42));

        self.parameter_bounds
            .iter()
            .map(|b| {
                let value = match b.scale {
                    ParameterScale::Linear => rng.gen_range(b.min..b.max),
                    ParameterScale::Logarithmic => {
                        let log_min = b.min.ln();
                        let log_max = b.max.ln();
                        rng.gen_range(log_min..log_max).exp()
                    }
                    ParameterScale::Sigmoid => {
                        let u: f64 = rng.gen();
                        b.min + (b.max - b.min) / (1.0 + (-4.0 * (u - 0.5)).exp())
                    }
                };
                (
                    b.name.clone(),
                    if b.integer { value.round() } else { value },
                )
            })
            .collect()
    }

    /// Get next parameters using Bayesian optimization
    fn get_next_parameters(&mut self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            // Prepare data before borrowing optimizer mutably
            let x_data: Vec<Array1<f64>> = self
                .evaluation_history
                .iter()
                .map(|e| {
                    Array1::from_vec(
                        self.parameter_bounds
                            .iter()
                            .map(|b| self.transform_to_unit(e.parameters[&b.name], b))
                            .collect(),
                    )
                })
                .collect();

            let y_data: Array1<f64> = self
                .evaluation_history
                .iter()
                .map(|e| e.objective_value)
                .collect();

            if let Some(ref mut optimizer) = self.optimizer {
                // Update GP model
                optimizer.update(&x_data, &y_data)?;

                // Get next point
                let next_point = optimizer.suggest_next()?;

                // Transform back to parameter space
                return Ok(self.transform_from_unit(&next_point));
            }
        }

        // Fallback: random sampling
        Ok(self.sample_random_parameters(self.evaluation_history.len() as u64))
    }

    /// Transform parameter to unit interval
    fn transform_to_unit(&self, value: f64, bounds: &ParameterBounds) -> f64 {
        match bounds.scale {
            ParameterScale::Linear => (value - bounds.min) / (bounds.max - bounds.min),
            ParameterScale::Logarithmic => {
                (value.ln() - bounds.min.ln()) / (bounds.max.ln() - bounds.min.ln())
            }
            ParameterScale::Sigmoid => {
                let normalized = (value - bounds.min) / (bounds.max - bounds.min);
                0.5 + 0.25 * (4.0 * (normalized - 0.5)).tanh()
            }
        }
    }

    /// Transform from unit interval to parameter space
    fn transform_from_unit(&self, unit_values: &Array1<f64>) -> HashMap<String, f64> {
        self.parameter_bounds
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let unit_val = unit_values[i].clamp(0.0, 1.0);
                let value = match b.scale {
                    ParameterScale::Linear => b.min + unit_val * (b.max - b.min),
                    ParameterScale::Logarithmic => {
                        let log_val = b.min.ln() + unit_val * (b.max.ln() - b.min.ln());
                        log_val.exp()
                    }
                    ParameterScale::Sigmoid => {
                        let t = (unit_val - 0.5) * 4.0;
                        let sigmoid = 0.5 + 0.5 * t.tanh();
                        b.min + sigmoid * (b.max - b.min)
                    }
                };
                (
                    b.name.clone(),
                    if b.integer { value.round() } else { value },
                )
            })
            .collect()
    }

    /// Evaluate a parameter configuration
    fn evaluate_configuration<S: Sampler>(
        &self,
        parameters: &HashMap<String, f64>,
        sampler_factory: &impl Fn(HashMap<String, f64>) -> S,
        model: &CompiledModel,
        objective: &impl Fn(&[SampleResult]) -> f64,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();

        // Create sampler with parameters
        let mut sampler = sampler_factory(parameters.clone());

        // Run sampling
        let num_reads = 100; // Could be a tunable parameter
        let samples = sampler.run_qubo(&model.to_qubo(), num_reads)?;

        // Evaluate objective
        let obj_value = objective(&samples);

        Ok(obj_value)
    }

    /// Record evaluation result
    fn record_evaluation(
        &mut self,
        parameters: HashMap<String, f64>,
        objective_value: f64,
        constraint_violations: HashMap<String, f64>,
        iteration: usize,
    ) {
        self.evaluation_history.push(TuningEvaluation {
            parameters,
            objective_value,
            constraint_violations,
            evaluation_time: std::time::Duration::from_secs(1), // Placeholder
            iteration,
        });
    }

    /// Calculate parameter importance using sensitivity analysis
    fn calculate_parameter_importance(
        &self,
    ) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            if let Some(ref optimizer) = self.optimizer {
                // Use GP model to calculate parameter sensitivities
                return self.calculate_importance_gp();
            }
        }

        // Fallback: correlation-based importance
        self.calculate_importance_correlation()
    }

    #[cfg(feature = "scirs")]
    /// Calculate importance using Gaussian Process
    fn calculate_importance_gp(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        // Implement Sobol indices or similar sensitivity analysis
        // For now, return uniform importance
        Ok(self
            .parameter_bounds
            .iter()
            .map(|b| (b.name.clone(), 1.0 / self.parameter_bounds.len() as f64))
            .collect())
    }

    /// Calculate importance using correlation
    fn calculate_importance_correlation(
        &self,
    ) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut importance = HashMap::new();

        for bounds in &self.parameter_bounds {
            // Calculate correlation between parameter and objective
            let param_values: Vec<f64> = self
                .evaluation_history
                .iter()
                .map(|e| e.parameters[&bounds.name])
                .collect();

            let obj_values: Vec<f64> = self
                .evaluation_history
                .iter()
                .map(|e| e.objective_value)
                .collect();

            let correlation = calculate_correlation(&param_values, &obj_values);
            importance.insert(bounds.name.clone(), correlation.abs());
        }

        // Normalize
        let total: f64 = importance.values().sum();
        if total > 0.0 {
            for value in importance.values_mut() {
                *value /= total;
            }
        }

        Ok(importance)
    }

    /// Export tuning results
    pub fn export_results(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let export = TuningExport {
            config: self.config.clone(),
            parameter_bounds: self.parameter_bounds.clone(),
            evaluation_history: self.evaluation_history.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        let json = serde_json::to_string_pretty(&export)?;
        std::fs::write(path, json)?;

        Ok(())
    }
}

/// Calculate correlation coefficient
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / n;

    let std_x = (x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / n).sqrt();
    let std_y = (y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / n).sqrt();

    if std_x > 0.0 && std_y > 0.0 {
        cov / (std_x * std_y)
    } else {
        0.0
    }
}

/// Tuning export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningExport {
    pub config: TuningConfig,
    pub parameter_bounds: Vec<ParameterBounds>,
    pub evaluation_history: Vec<TuningEvaluation>,
    pub timestamp: std::time::SystemTime,
}

#[cfg(feature = "scirs")]
impl From<AcquisitionType>
    for crate::scirs_stub::scirs2_optimization::bayesian::AcquisitionFunction
{
    fn from(acq: AcquisitionType) -> Self {
        match acq {
            AcquisitionType::ExpectedImprovement => Self::ExpectedImprovement,
            AcquisitionType::UpperConfidenceBound => Self::UCB,
            AcquisitionType::ProbabilityOfImprovement => Self::PI,
            AcquisitionType::ThompsonSampling => Self::Thompson,
        }
    }
}

/// Quick parameter tuning function
pub fn quick_tune<S: Sampler>(
    sampler_factory: impl Fn(HashMap<String, f64>) -> S,
    model: &CompiledModel,
    parameter_bounds: Vec<ParameterBounds>,
) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
    let config = TuningConfig {
        max_evaluations: 50,
        initial_samples: 10,
        ..Default::default()
    };

    let mut tuner = ParameterTuner::new(config);
    tuner.add_parameters(parameter_bounds);

    let result = tuner.tune_sampler(sampler_factory, model, |samples| {
        // Default objective: minimize average energy
        samples.iter().map(|s| s.energy).sum::<f64>() / samples.len() as f64
    })?;

    Ok(result.best_parameters)
}
