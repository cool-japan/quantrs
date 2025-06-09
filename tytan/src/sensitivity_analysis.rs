//! Sensitivity analysis for optimization parameters.
//!
//! This module provides tools for analyzing how sensitive optimization results
//! are to changes in various parameters, including penalty weights, sampler
//! parameters, and problem formulation choices.

use crate::sampler::{SampleResult, Sampler, SamplerResult};
#[cfg(feature = "dwave")]
use crate::compile::CompiledModel;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::error::Error;
use rand::thread_rng;

#[cfg(feature = "scirs")]
use crate::scirs_stub::{
    scirs2_optimization::bayesian::{BayesianOptimizer, GaussianProcess},
    scirs2_statistics::descriptive::{mean, std_dev},
    scirs2_plot::{Plot, Line, Scatter, Heatmap, Bar},
};

/// Parameter types for sensitivity analysis
#[derive(Debug, Clone)]
pub enum ParameterType {
    /// Sampler parameters (temperature, iterations, etc.)
    SamplerParameter {
        name: String,
        default_value: f64,
        min_value: f64,
        max_value: f64,
    },
    /// Penalty weights for constraints
    PenaltyWeight {
        constraint_name: String,
        default_weight: f64,
        min_weight: f64,
        max_weight: f64,
    },
    /// Problem formulation parameters
    FormulationParameter {
        name: String,
        default_value: f64,
        variation_range: (f64, f64),
    },
    /// Discrete parameter choices
    DiscreteChoice {
        name: String,
        options: Vec<String>,
        default_index: usize,
    },
}

/// Sensitivity analysis method
#[derive(Debug, Clone)]
pub enum SensitivityMethod {
    /// One-at-a-time (OAT) analysis
    OneAtATime {
        num_points: usize,
        include_interactions: bool,
    },
    /// Morris method for screening
    Morris {
        num_trajectories: usize,
        num_levels: usize,
    },
    /// Sobol indices for variance decomposition
    Sobol {
        num_samples: usize,
        compute_second_order: bool,
    },
    /// Latin hypercube sampling
    LatinHypercube {
        num_samples: usize,
    },
    /// Factorial design
    Factorial {
        levels_per_factor: usize,
    },
}

/// Sensitivity analysis results
#[derive(Debug, Clone)]
pub struct SensitivityResults {
    /// Parameter sensitivities
    pub sensitivities: HashMap<String, ParameterSensitivity>,
    /// Main effects
    pub main_effects: HashMap<String, f64>,
    /// Interaction effects (if computed)
    pub interaction_effects: Option<HashMap<(String, String), f64>>,
    /// Sobol indices (if computed)
    pub sobol_indices: Option<SobolIndices>,
    /// Optimal parameter values found
    pub optimal_parameters: HashMap<String, f64>,
    /// Robustness metrics
    pub robustness_metrics: RobustnessMetrics,
}

/// Sensitivity information for a single parameter
#[derive(Debug, Clone)]
pub struct ParameterSensitivity {
    /// Parameter name
    pub name: String,
    /// Sensitivity index (normalized)
    pub sensitivity_index: f64,
    /// Effect on objective
    pub objective_gradient: f64,
    /// Effect on constraint satisfaction
    pub constraint_impact: f64,
    /// Optimal value found
    pub optimal_value: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Response curve
    pub response_curve: Vec<(f64, f64)>,
}

/// Sobol sensitivity indices
#[derive(Debug, Clone)]
pub struct SobolIndices {
    /// First-order indices
    pub first_order: HashMap<String, f64>,
    /// Total indices
    pub total_indices: HashMap<String, f64>,
    /// Second-order indices (if computed)
    pub second_order: Option<HashMap<(String, String), f64>>,
}

/// Robustness metrics
#[derive(Debug, Clone)]
pub struct RobustnessMetrics {
    /// Coefficient of variation for objective
    pub objective_cv: f64,
    /// Probability of constraint satisfaction
    pub constraint_satisfaction_prob: f64,
    /// Worst-case performance
    pub worst_case_objective: f64,
    /// Best-case performance
    pub best_case_objective: f64,
    /// Parameter stability regions
    pub stability_regions: HashMap<String, (f64, f64)>,
}

/// Sensitivity analyzer
pub struct SensitivityAnalyzer<S: Sampler> {
    /// Base sampler
    sampler: S,
    /// Parameters to analyze
    parameters: Vec<ParameterType>,
    /// Analysis method
    method: SensitivityMethod,
    /// Number of evaluations per configuration
    num_reads_per_eval: usize,
    /// Whether to use parallel evaluation
    use_parallel: bool,
}

impl<S: Sampler> SensitivityAnalyzer<S> {
    /// Create new sensitivity analyzer
    pub fn new(
        sampler: S,
        parameters: Vec<ParameterType>,
        method: SensitivityMethod,
    ) -> Self {
        Self {
            sampler,
            parameters,
            method,
            num_reads_per_eval: 100,
            use_parallel: true,
        }
    }
    
    /// Set number of reads per evaluation
    pub fn with_reads_per_eval(mut self, num_reads: usize) -> Self {
        self.num_reads_per_eval = num_reads;
        self
    }
    
    /// Perform sensitivity analysis
    #[cfg(feature = "dwave")]
    pub fn analyze(
        &mut self,
        problem: &CompiledModel,
    ) -> Result<SensitivityResults, Box<dyn Error>> {
        match &self.method {
            SensitivityMethod::OneAtATime { num_points, include_interactions } => {
                self.one_at_a_time_analysis(problem, *num_points, *include_interactions)
            }
            SensitivityMethod::Morris { num_trajectories, num_levels } => {
                self.morris_analysis(problem, *num_trajectories, *num_levels)
            }
            SensitivityMethod::Sobol { num_samples, compute_second_order } => {
                self.sobol_analysis(problem, *num_samples, *compute_second_order)
            }
            SensitivityMethod::LatinHypercube { num_samples } => {
                self.latin_hypercube_analysis(problem, *num_samples)
            }
            SensitivityMethod::Factorial { levels_per_factor } => {
                self.factorial_analysis(problem, *levels_per_factor)
            }
        }
    }
    
    /// One-at-a-time sensitivity analysis
    #[cfg(feature = "dwave")]
    fn one_at_a_time_analysis(
        &mut self,
        problem: &CompiledModel,
        num_points: usize,
        include_interactions: bool,
    ) -> Result<SensitivityResults, Box<dyn Error>> {
        let mut sensitivities = HashMap::new();
        let mut main_effects = HashMap::new();
        let mut response_data = HashMap::new();
        
        // Baseline evaluation
        let baseline_params = self.get_default_parameters();
        let baseline_result = self.evaluate_configuration(problem, &baseline_params)?;
        let baseline_objective = baseline_result.best_objective;
        
        // Vary each parameter
        for param in &self.parameters {
            let param_name = self.get_parameter_name(param);
            let (min_val, max_val) = self.get_parameter_range(param);
            
            let mut response_curve = Vec::new();
            let mut objectives = Vec::new();
            
            // Sample parameter values
            for i in 0..num_points {
                let t = i as f64 / (num_points - 1) as f64;
                let value = min_val + t * (max_val - min_val);
                
                // Create parameter configuration
                let mut params = baseline_params.clone();
                params.insert(param_name.clone(), value);
                
                // Evaluate
                let result = self.evaluate_configuration(problem, &params)?;
                response_curve.push((value, result.best_objective));
                objectives.push(result.best_objective);
            }
            
            // Compute sensitivity metrics
            let gradient = self.compute_gradient(&response_curve);
            let sensitivity_index = self.compute_sensitivity_index(&objectives, baseline_objective);
            let (optimal_value, _) = response_curve.iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .cloned()
                .unwrap_or((baseline_params[&param_name], baseline_objective));
            
            sensitivities.insert(param_name.clone(), ParameterSensitivity {
                name: param_name.clone(),
                sensitivity_index,
                objective_gradient: gradient,
                constraint_impact: 0.0, // TODO: Compute from constraint violations
                optimal_value,
                confidence_interval: self.compute_confidence_interval(&objectives),
                response_curve: response_curve.clone(),
            });
            
            main_effects.insert(param_name.clone(), sensitivity_index);
            response_data.insert(param_name, response_curve);
        }
        
        // Compute interactions if requested
        let interaction_effects = if include_interactions {
            Some(self.compute_interaction_effects(problem, &baseline_params)?)
        } else {
            None
        };
        
        // Compute robustness metrics
        let robustness_metrics = self.compute_robustness_metrics(&response_data);
        
        Ok(SensitivityResults {
            sensitivities,
            main_effects,
            interaction_effects,
            sobol_indices: None,
            optimal_parameters: self.find_optimal_parameters(&response_data),
            robustness_metrics,
        })
    }
    
    /// Morris screening analysis
    #[cfg(feature = "dwave")]
    fn morris_analysis(
        &mut self,
        problem: &CompiledModel,
        num_trajectories: usize,
        num_levels: usize,
    ) -> Result<SensitivityResults, Box<dyn Error>> {
        let mut elementary_effects = HashMap::new();
        
        for _ in 0..num_trajectories {
            // Generate Morris trajectory
            let trajectory = self.generate_morris_trajectory(num_levels)?;
            
            // Evaluate along trajectory
            for i in 0..trajectory.len() - 1 {
                let params1 = &trajectory[i];
                let params2 = &trajectory[i + 1];
                
                let result1 = self.evaluate_configuration(problem, params1)?;
                let result2 = self.evaluate_configuration(problem, params2)?;
                
                // Find which parameter changed
                for (key, &val1) in params1 {
                    if let Some(&val2) = params2.get(key) {
                        if (val1 - val2).abs() > 1e-10 {
                            let delta = val2 - val1;
                            let effect = (result2.best_objective - result1.best_objective) / delta;
                            
                            elementary_effects.entry(key.clone())
                                .or_insert_with(Vec::new)
                                .push(effect);
                        }
                    }
                }
            }
        }
        
        // Compute Morris statistics
        let mut sensitivities = HashMap::new();
        let mut main_effects = HashMap::new();
        
        for (param_name, effects) in elementary_effects {
            let mean_effect = effects.iter().sum::<f64>() / effects.len() as f64;
            let mean_abs_effect = effects.iter().map(|e| e.abs()).sum::<f64>() / effects.len() as f64;
            let std_effect = {
                let variance = effects.iter()
                    .map(|e| (e - mean_effect).powi(2))
                    .sum::<f64>() / effects.len() as f64;
                variance.sqrt()
            };
            
            sensitivities.insert(param_name.clone(), ParameterSensitivity {
                name: param_name.clone(),
                sensitivity_index: mean_abs_effect,
                objective_gradient: mean_effect,
                constraint_impact: 0.0,
                optimal_value: 0.0, // Not applicable for Morris
                confidence_interval: (mean_effect - 2.0 * std_effect, mean_effect + 2.0 * std_effect),
                response_curve: Vec::new(),
            });
            
            main_effects.insert(param_name, mean_abs_effect);
        }
        
        Ok(SensitivityResults {
            sensitivities,
            main_effects,
            interaction_effects: None,
            sobol_indices: None,
            optimal_parameters: HashMap::new(),
            robustness_metrics: RobustnessMetrics {
                objective_cv: 0.0,
                constraint_satisfaction_prob: 1.0,
                worst_case_objective: 0.0,
                best_case_objective: 0.0,
                stability_regions: HashMap::new(),
            },
        })
    }
    
    /// Sobol sensitivity analysis
    #[cfg(feature = "dwave")]
    fn sobol_analysis(
        &mut self,
        problem: &CompiledModel,
        num_samples: usize,
        compute_second_order: bool,
    ) -> Result<SensitivityResults, Box<dyn Error>> {
        // Generate Sobol samples
        let (sample_a, sample_b) = self.generate_sobol_samples(num_samples)?;
        
        // Evaluate base samples
        let mut y_a = Vec::new();
        let mut y_b = Vec::new();
        
        for i in 0..num_samples {
            let result_a = self.evaluate_configuration(problem, &sample_a[i])?;
            let result_b = self.evaluate_configuration(problem, &sample_b[i])?;
            y_a.push(result_a.best_objective);
            y_b.push(result_b.best_objective);
        }
        
        // Compute first-order indices
        let mut first_order = HashMap::new();
        let mut total_indices = HashMap::new();
        
        let var_total = variance(&y_a);
        
        for param in &self.parameters {
            let param_name = self.get_parameter_name(param);
            
            // Create sample where only this parameter varies
            let mut y_ab_i = Vec::new();
            for i in 0..num_samples {
                let mut params_ab_i = sample_a[i].clone();
                params_ab_i.insert(param_name.clone(), sample_b[i][&param_name]);
                
                let result = self.evaluate_configuration(problem, &params_ab_i)?;
                y_ab_i.push(result.best_objective);
            }
            
            // Compute indices
            let s_i = self.compute_first_order_index(&y_a, &y_b, &y_ab_i, var_total);
            let st_i = self.compute_total_index(&y_a, &y_ab_i, var_total);
            
            first_order.insert(param_name.clone(), s_i);
            total_indices.insert(param_name, st_i);
        }
        
        // Compute second-order indices if requested
        let second_order = if compute_second_order {
            Some(self.compute_second_order_indices(
                problem,
                &sample_a,
                &sample_b,
                &y_a,
                &y_b,
                var_total,
            )?)
        } else {
            None
        };
        
        // Convert to sensitivity results
        let mut sensitivities = HashMap::new();
        for param in &self.parameters {
            let param_name = self.get_parameter_name(param);
            sensitivities.insert(param_name.clone(), ParameterSensitivity {
                name: param_name.clone(),
                sensitivity_index: first_order[&param_name],
                objective_gradient: 0.0,
                constraint_impact: 0.0,
                optimal_value: 0.0,
                confidence_interval: (0.0, 0.0),
                response_curve: Vec::new(),
            });
        }
        
        Ok(SensitivityResults {
            sensitivities,
            main_effects: first_order.clone(),
            interaction_effects: None,
            sobol_indices: Some(SobolIndices {
                first_order,
                total_indices,
                second_order,
            }),
            optimal_parameters: HashMap::new(),
            robustness_metrics: self.compute_robustness_from_samples(&y_a, &y_b),
        })
    }
    
    /// Latin hypercube sampling analysis
    #[cfg(feature = "dwave")]
    fn latin_hypercube_analysis(
        &mut self,
        problem: &CompiledModel,
        num_samples: usize,
    ) -> Result<SensitivityResults, Box<dyn Error>> {
        // Generate LHS samples
        let samples = self.generate_latin_hypercube_samples(num_samples)?;
        
        // Evaluate samples
        let mut results = Vec::new();
        for sample in &samples {
            let result = self.evaluate_configuration(problem, sample)?;
            results.push((sample.clone(), result.best_objective));
        }
        
        // Compute sensitivities using regression
        let sensitivities = self.compute_regression_sensitivities(&results)?;
        
        // Find optimal configuration
        let (optimal_params, _) = results.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .cloned()
            .unwrap();
        
        Ok(SensitivityResults {
            sensitivities: sensitivities.clone(),
            main_effects: sensitivities.iter()
                .map(|(k, v)| (k.clone(), v.sensitivity_index))
                .collect(),
            interaction_effects: None,
            sobol_indices: None,
            optimal_parameters: optimal_params,
            robustness_metrics: self.compute_robustness_from_results(&results),
        })
    }
    
    /// Factorial design analysis
    #[cfg(feature = "dwave")]
    fn factorial_analysis(
        &mut self,
        problem: &CompiledModel,
        levels_per_factor: usize,
    ) -> Result<SensitivityResults, Box<dyn Error>> {
        // Generate factorial design
        let design = self.generate_factorial_design(levels_per_factor)?;
        
        // Evaluate all combinations
        let mut results = Vec::new();
        for config in &design {
            let result = self.evaluate_configuration(problem, config)?;
            results.push((config.clone(), result.best_objective));
        }
        
        // Compute main effects and interactions
        let (main_effects, interaction_effects) = self.analyze_factorial_results(&results, levels_per_factor)?;
        
        // Convert to sensitivity results
        let mut sensitivities = HashMap::new();
        for (param_name, effect) in &main_effects {
            sensitivities.insert(param_name.clone(), ParameterSensitivity {
                name: param_name.clone(),
                sensitivity_index: effect.abs(),
                objective_gradient: *effect,
                constraint_impact: 0.0,
                optimal_value: 0.0,
                confidence_interval: (0.0, 0.0),
                response_curve: Vec::new(),
            });
        }
        
        // Find optimal configuration
        let (optimal_params, _) = results.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .cloned()
            .unwrap();
        
        Ok(SensitivityResults {
            sensitivities,
            main_effects,
            interaction_effects: Some(interaction_effects),
            sobol_indices: None,
            optimal_parameters: optimal_params,
            robustness_metrics: self.compute_robustness_from_results(&results),
        })
    }
    
    /// Evaluate a parameter configuration
    #[cfg(feature = "dwave")]
    fn evaluate_configuration(
        &mut self,
        problem: &CompiledModel,
        params: &HashMap<String, f64>,
    ) -> Result<EvaluationResult, Box<dyn Error>> {
        // Apply parameters to sampler
        self.apply_parameters(params)?;
        
        // Run sampler
        let qubo = problem.to_qubo();
        let solutions = self.sampler.run_qubo(&qubo, self.num_reads_per_eval)?;
        
        // Extract best objective
        let best_objective = solutions.iter()
            .map(|s| s.energy)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f64::INFINITY);
        
        // Compute constraint violations
        let constraint_violations = 0.0; // TODO: Implement constraint checking
        
        Ok(EvaluationResult {
            best_objective,
            avg_objective: solutions.iter().map(|s| s.energy).sum::<f64>() / solutions.len() as f64,
            constraint_violations,
            num_feasible: solutions.len(),
        })
    }
    
    /// Get parameter name
    fn get_parameter_name(&self, param: &ParameterType) -> String {
        match param {
            ParameterType::SamplerParameter { name, .. } => name.clone(),
            ParameterType::PenaltyWeight { constraint_name, .. } => format!("penalty_{}", constraint_name),
            ParameterType::FormulationParameter { name, .. } => name.clone(),
            ParameterType::DiscreteChoice { name, .. } => name.clone(),
        }
    }
    
    /// Get parameter range
    fn get_parameter_range(&self, param: &ParameterType) -> (f64, f64) {
        match param {
            ParameterType::SamplerParameter { min_value, max_value, .. } => (*min_value, *max_value),
            ParameterType::PenaltyWeight { min_weight, max_weight, .. } => (*min_weight, *max_weight),
            ParameterType::FormulationParameter { variation_range, .. } => *variation_range,
            ParameterType::DiscreteChoice { options, .. } => (0.0, options.len() as f64 - 1.0),
        }
    }
    
    /// Get default parameters
    fn get_default_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        
        for param in &self.parameters {
            let name = self.get_parameter_name(param);
            let value = match param {
                ParameterType::SamplerParameter { default_value, .. } => *default_value,
                ParameterType::PenaltyWeight { default_weight, .. } => *default_weight,
                ParameterType::FormulationParameter { default_value, .. } => *default_value,
                ParameterType::DiscreteChoice { default_index, .. } => *default_index as f64,
            };
            params.insert(name, value);
        }
        
        params
    }
    
    /// Apply parameters to sampler
    fn apply_parameters(&mut self, params: &HashMap<String, f64>) -> Result<(), Box<dyn Error>> {
        // This would need to be implemented based on the specific sampler
        // For now, we assume parameters are applied externally
        Ok(())
    }
    
    /// Compute gradient from response curve
    fn compute_gradient(&self, response_curve: &[(f64, f64)]) -> f64 {
        if response_curve.len() < 2 {
            return 0.0;
        }
        
        // Simple finite difference
        let n = response_curve.len();
        let dx = response_curve[n-1].0 - response_curve[0].0;
        let dy = response_curve[n-1].1 - response_curve[0].1;
        
        dy / dx
    }
    
    /// Compute sensitivity index
    fn compute_sensitivity_index(&self, objectives: &[f64], baseline: f64) -> f64 {
        let max_diff = objectives.iter()
            .map(|&obj| (obj - baseline).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        max_diff / baseline.abs().max(1.0)
    }
    
    /// Compute confidence interval
    fn compute_confidence_interval(&self, values: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std = {
            let variance = values.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            variance.sqrt()
        };
        
        (mean - 2.0 * std, mean + 2.0 * std)
    }
    
    // Additional helper methods would be implemented here...
    
    #[cfg(feature = "dwave")]
    fn compute_interaction_effects(
        &mut self,
        _problem: &CompiledModel,
        _baseline_params: &HashMap<String, f64>,
    ) -> Result<HashMap<(String, String), f64>, Box<dyn Error>> {
        // Simplified implementation
        Ok(HashMap::new())
    }
    
    fn compute_robustness_metrics(
        &self,
        response_data: &HashMap<String, Vec<(f64, f64)>>,
    ) -> RobustnessMetrics {
        let all_objectives: Vec<f64> = response_data.values()
            .flat_map(|curve| curve.iter().map(|(_, obj)| *obj))
            .collect();
        
        let mean_obj = all_objectives.iter().sum::<f64>() / all_objectives.len() as f64;
        let std_obj = {
            let variance = all_objectives.iter()
                .map(|obj| (obj - mean_obj).powi(2))
                .sum::<f64>() / all_objectives.len() as f64;
            variance.sqrt()
        };
        
        RobustnessMetrics {
            objective_cv: std_obj / mean_obj.abs().max(1.0),
            constraint_satisfaction_prob: 1.0, // Placeholder
            worst_case_objective: all_objectives.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            best_case_objective: all_objectives.iter().cloned().fold(f64::INFINITY, f64::min),
            stability_regions: HashMap::new(), // Placeholder
        }
    }
    
    fn find_optimal_parameters(
        &self,
        response_data: &HashMap<String, Vec<(f64, f64)>>,
    ) -> HashMap<String, f64> {
        let mut optimal = HashMap::new();
        
        for (param_name, curve) in response_data {
            if let Some((opt_val, _)) = curve.iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                optimal.insert(param_name.clone(), *opt_val);
            }
        }
        
        optimal
    }
    
    fn generate_morris_trajectory(
        &self,
        _num_levels: usize,
    ) -> Result<Vec<HashMap<String, f64>>, Box<dyn Error>> {
        // Simplified implementation
        Ok(vec![self.get_default_parameters()])
    }
    
    fn generate_sobol_samples(
        &self,
        num_samples: usize,
    ) -> Result<(Vec<HashMap<String, f64>>, Vec<HashMap<String, f64>>), Box<dyn Error>> {
        // Simplified implementation using random sampling
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let mut sample_a = Vec::new();
        let mut sample_b = Vec::new();
        
        for _ in 0..num_samples {
            let mut params_a = HashMap::new();
            let mut params_b = HashMap::new();
            
            for param in &self.parameters {
                let name = self.get_parameter_name(param);
                let (min_val, max_val) = self.get_parameter_range(param);
                
                params_a.insert(name.clone(), rng.gen_range(min_val..max_val));
                params_b.insert(name, rng.gen_range(min_val..max_val));
            }
            
            sample_a.push(params_a);
            sample_b.push(params_b);
        }
        
        Ok((sample_a, sample_b))
    }
    
    fn compute_first_order_index(
        &self,
        y_a: &[f64],
        y_b: &[f64],
        y_ab_i: &[f64],
        var_total: f64,
    ) -> f64 {
        let n = y_a.len() as f64;
        let mean_product: f64 = y_a.iter().zip(y_ab_i.iter())
            .map(|(a, ab)| a * ab)
            .sum::<f64>() / n;
        let mean_a = y_a.iter().sum::<f64>() / n;
        let mean_b = y_b.iter().sum::<f64>() / n;
        
        (mean_product - mean_a * mean_b) / var_total
    }
    
    fn compute_total_index(
        &self,
        y_a: &[f64],
        y_ab_i: &[f64],
        var_total: f64,
    ) -> f64 {
        let n = y_a.len() as f64;
        let mean_sq_diff: f64 = y_a.iter().zip(y_ab_i.iter())
            .map(|(a, ab)| (a - ab).powi(2))
            .sum::<f64>() / n;
        
        0.5 * mean_sq_diff / var_total
    }
    
    #[cfg(feature = "dwave")]
    fn compute_second_order_indices(
        &mut self,
        _problem: &CompiledModel,
        _sample_a: &[HashMap<String, f64>],
        _sample_b: &[HashMap<String, f64>],
        _y_a: &[f64],
        _y_b: &[f64],
        _var_total: f64,
    ) -> Result<HashMap<(String, String), f64>, Box<dyn Error>> {
        // Simplified implementation
        Ok(HashMap::new())
    }
    
    fn compute_robustness_from_samples(
        &self,
        y_a: &[f64],
        y_b: &[f64],
    ) -> RobustnessMetrics {
        let all_y: Vec<f64> = y_a.iter().chain(y_b.iter()).cloned().collect();
        
        let mean_y = all_y.iter().sum::<f64>() / all_y.len() as f64;
        let std_y = {
            let variance = all_y.iter()
                .map(|y| (y - mean_y).powi(2))
                .sum::<f64>() / all_y.len() as f64;
            variance.sqrt()
        };
        
        RobustnessMetrics {
            objective_cv: std_y / mean_y.abs().max(1.0),
            constraint_satisfaction_prob: 1.0,
            worst_case_objective: all_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            best_case_objective: all_y.iter().cloned().fold(f64::INFINITY, f64::min),
            stability_regions: HashMap::new(),
        }
    }
    
    fn generate_latin_hypercube_samples(
        &self,
        num_samples: usize,
    ) -> Result<Vec<HashMap<String, f64>>, Box<dyn Error>> {
        // Simplified LHS implementation
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let mut samples = Vec::new();
        let n_params = self.parameters.len();
        
        // Create permutations for each parameter
        let mut permutations: Vec<Vec<usize>> = Vec::new();
        for _ in 0..n_params {
            let mut perm: Vec<usize> = (0..num_samples).collect();
            perm.shuffle(&mut rng);
            permutations.push(perm);
        }
        
        // Generate samples
        for i in 0..num_samples {
            let mut sample = HashMap::new();
            
            for (j, param) in self.parameters.iter().enumerate() {
                let name = self.get_parameter_name(param);
                let (min_val, max_val) = self.get_parameter_range(param);
                
                let level = permutations[j][i];
                let value = min_val + (level as f64 + rng.gen::<f64>()) / num_samples as f64 * (max_val - min_val);
                
                sample.insert(name, value);
            }
            
            samples.push(sample);
        }
        
        Ok(samples)
    }
    
    fn compute_regression_sensitivities(
        &self,
        results: &[(HashMap<String, f64>, f64)],
    ) -> Result<HashMap<String, ParameterSensitivity>, Box<dyn Error>> {
        // Simplified linear regression
        let mut sensitivities = HashMap::new();
        
        for param in &self.parameters {
            let param_name = self.get_parameter_name(param);
            
            // Extract x and y values
            let x: Vec<f64> = results.iter()
                .map(|(params, _)| params[&param_name])
                .collect();
            let y: Vec<f64> = results.iter()
                .map(|(_, obj)| *obj)
                .collect();
            
            // Compute regression coefficient
            let coeff = simple_linear_regression(&x, &y);
            
            sensitivities.insert(param_name.clone(), ParameterSensitivity {
                name: param_name.clone(),
                sensitivity_index: coeff.abs(),
                objective_gradient: coeff,
                constraint_impact: 0.0,
                optimal_value: 0.0,
                confidence_interval: (0.0, 0.0),
                response_curve: Vec::new(),
            });
        }
        
        Ok(sensitivities)
    }
    
    fn compute_robustness_from_results(
        &self,
        results: &[(HashMap<String, f64>, f64)],
    ) -> RobustnessMetrics {
        let objectives: Vec<f64> = results.iter().map(|(_, obj)| *obj).collect();
        
        let mean_obj = objectives.iter().sum::<f64>() / objectives.len() as f64;
        let std_obj = {
            let variance = objectives.iter()
                .map(|obj| (obj - mean_obj).powi(2))
                .sum::<f64>() / objectives.len() as f64;
            variance.sqrt()
        };
        
        RobustnessMetrics {
            objective_cv: std_obj / mean_obj.abs().max(1.0),
            constraint_satisfaction_prob: 1.0,
            worst_case_objective: objectives.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            best_case_objective: objectives.iter().cloned().fold(f64::INFINITY, f64::min),
            stability_regions: HashMap::new(),
        }
    }
    
    fn generate_factorial_design(
        &self,
        levels_per_factor: usize,
    ) -> Result<Vec<HashMap<String, f64>>, Box<dyn Error>> {
        // Generate full factorial design
        let n_params = self.parameters.len();
        let total_combinations = levels_per_factor.pow(n_params as u32);
        
        let mut design = Vec::new();
        
        for i in 0..total_combinations {
            let mut config = HashMap::new();
            let mut idx = i;
            
            for param in &self.parameters {
                let name = self.get_parameter_name(param);
                let (min_val, max_val) = self.get_parameter_range(param);
                
                let level = idx % levels_per_factor;
                idx /= levels_per_factor;
                
                let value = if levels_per_factor == 1 {
                    (min_val + max_val) / 2.0
                } else {
                    min_val + level as f64 / (levels_per_factor - 1) as f64 * (max_val - min_val)
                };
                
                config.insert(name, value);
            }
            
            design.push(config);
        }
        
        Ok(design)
    }
    
    fn analyze_factorial_results(
        &self,
        results: &[(HashMap<String, f64>, f64)],
        levels_per_factor: usize,
    ) -> Result<(HashMap<String, f64>, HashMap<(String, String), f64>), Box<dyn Error>> {
        // Simplified factorial analysis
        let mut main_effects = HashMap::new();
        let mut interaction_effects = HashMap::new();
        
        // Compute main effects
        for param in &self.parameters {
            let param_name = self.get_parameter_name(param);
            
            let mut level_sums = vec![0.0; levels_per_factor];
            let mut level_counts = vec![0; levels_per_factor];
            
            for (config, obj) in results {
                let value = config[&param_name];
                let (min_val, max_val) = self.get_parameter_range(param);
                
                let level = if levels_per_factor == 1 {
                    0
                } else {
                    ((value - min_val) / (max_val - min_val) * (levels_per_factor - 1) as f64).round() as usize
                };
                
                if level < levels_per_factor {
                    level_sums[level] += obj;
                    level_counts[level] += 1;
                }
            }
            
            // Compute effect as difference between high and low levels
            if levels_per_factor >= 2 && level_counts[0] > 0 && level_counts[levels_per_factor - 1] > 0 {
                let low_mean = level_sums[0] / level_counts[0] as f64;
                let high_mean = level_sums[levels_per_factor - 1] / level_counts[levels_per_factor - 1] as f64;
                main_effects.insert(param_name, high_mean - low_mean);
            } else {
                main_effects.insert(param_name, 0.0);
            }
        }
        
        // Compute two-way interactions (simplified)
        for i in 0..self.parameters.len() {
            for j in i+1..self.parameters.len() {
                let param1 = self.get_parameter_name(&self.parameters[i]);
                let param2 = self.get_parameter_name(&self.parameters[j]);
                
                // Placeholder interaction effect
                interaction_effects.insert((param1, param2), 0.0);
            }
        }
        
        Ok((main_effects, interaction_effects))
    }
}

/// Evaluation result
struct EvaluationResult {
    best_objective: f64,
    avg_objective: f64,
    constraint_violations: f64,
    num_feasible: usize,
}

/// Visualization utilities
pub mod visualization {
    use super::*;
    
    /// Plot sensitivity tornado chart
    pub fn plot_tornado_chart(
        results: &SensitivityResults,
        output_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        #[cfg(feature = "scirs")]
        {
            let mut plot = Plot::new();
            
            // Sort parameters by sensitivity
            let mut params: Vec<_> = results.main_effects.iter().collect();
            params.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            
            let names: Vec<String> = params.iter().map(|(k, _)| k.to_string()).collect();
            let values: Vec<f64> = params.iter().map(|(_, v)| **v).collect();
            
            let bar = Bar::new(names, values)
                .name("Sensitivity");
            
            plot.add_trace(bar);
            plot.set_title("Parameter Sensitivity (Tornado Chart)");
            plot.set_xlabel("Main Effect");
            plot.save(output_path)?;
        }
        
        Ok(())
    }
    
    /// Plot response curves
    pub fn plot_response_curves(
        results: &SensitivityResults,
        output_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        #[cfg(feature = "scirs")]
        {
            let mut plot = Plot::new();
            
            for (param_name, sensitivity) in &results.sensitivities {
                if !sensitivity.response_curve.is_empty() {
                    let x: Vec<f64> = sensitivity.response_curve.iter().map(|(x, _)| *x).collect();
                    let y: Vec<f64> = sensitivity.response_curve.iter().map(|(_, y)| *y).collect();
                    
                    let line = Line::new(x, y)
                        .name(param_name);
                    
                    plot.add_trace(line);
                }
            }
            
            plot.set_title("Parameter Response Curves");
            plot.set_xlabel("Parameter Value");
            plot.set_ylabel("Objective");
            plot.save(output_path)?;
        }
        
        Ok(())
    }
}

/// Utility functions

fn variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

fn simple_linear_regression(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    slope
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SASampler;
    
    #[test]
    fn test_one_at_a_time_analysis() {
        let sampler = SASampler::new(None);
        let parameters = vec![
            ParameterType::SamplerParameter {
                name: "temperature".to_string(),
                default_value: 1.0,
                min_value: 0.1,
                max_value: 10.0,
            },
        ];
        
        let mut analyzer = SensitivityAnalyzer::new(
            sampler,
            parameters,
            SensitivityMethod::OneAtATime {
                num_points: 5,
                include_interactions: false,
            },
        );
        
        // Would need a real compiled model to test
        // let results = analyzer.analyze(&compiled_model).unwrap();
    }
}