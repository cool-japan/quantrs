//! Penalty function optimization for QUBO problems
//!
//! This module provides advanced penalty weight optimization using SciRS2
//! for automatic tuning and constraint satisfaction analysis.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "scirs")]
use crate::scirs_stub::{
    scirs2_linalg::norm::Norm,
    scirs2_optimization::{OptimizationProblem, Optimizer},
};

/// Penalty function configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyConfig {
    /// Initial penalty weight
    pub initial_weight: f64,
    /// Minimum penalty weight
    pub min_weight: f64,
    /// Maximum penalty weight
    pub max_weight: f64,
    /// Weight adjustment factor
    pub adjustment_factor: f64,
    /// Target constraint violation tolerance
    pub violation_tolerance: f64,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Use adaptive penalty scaling
    pub adaptive_scaling: bool,
    /// Penalty function type
    pub penalty_type: PenaltyType,
}

/// Types of penalty functions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PenaltyType {
    /// Quadratic penalty: weight * violation^2
    Quadratic,
    /// Linear penalty: weight * |violation|
    Linear,
    /// Logarithmic barrier: -weight * log(slack)
    LogBarrier,
    /// Exponential penalty: weight * exp(violation) - 1
    Exponential,
    /// Augmented Lagrangian method
    AugmentedLagrangian,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            initial_weight: 1.0,
            min_weight: 0.001,
            max_weight: 1000.0,
            adjustment_factor: 2.0,
            violation_tolerance: 1e-6,
            max_iterations: 100,
            adaptive_scaling: true,
            penalty_type: PenaltyType::Quadratic,
        }
    }
}

/// Penalty function optimizer
pub struct PenaltyOptimizer {
    config: PenaltyConfig,
    constraint_weights: HashMap<String, f64>,
    violation_history: Vec<ConstraintViolation>,
    #[cfg(feature = "scirs")]
    optimizer: Option<Box<dyn Optimizer>>,
}

/// Constraint violation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_name: String,
    pub violation_amount: f64,
    pub penalty_weight: f64,
    pub iteration: usize,
}

/// Penalty optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyOptimizationResult {
    pub optimal_weights: HashMap<String, f64>,
    pub final_violations: HashMap<String, f64>,
    pub converged: bool,
    pub iterations: usize,
    pub objective_value: f64,
    pub constraint_satisfaction: f64,
}

impl PenaltyOptimizer {
    /// Create new penalty optimizer
    pub fn new(config: PenaltyConfig) -> Self {
        Self {
            config,
            constraint_weights: HashMap::new(),
            violation_history: Vec::new(),
            #[cfg(feature = "scirs")]
            optimizer: None,
        }
    }

    /// Initialize constraint weights
    pub fn initialize_weights(&mut self, constraints: &[String]) {
        for constraint in constraints {
            self.constraint_weights
                .insert(constraint.clone(), self.config.initial_weight);
        }

        #[cfg(feature = "scirs")]
        {
            // Initialize SciRS2 optimizer
            use crate::scirs_stub::scirs2_optimization::gradient::LBFGS;
            self.optimizer = Some(Box::new(LBFGS::new(constraints.len())));
        }
    }

    /// Optimize penalty weights for a compiled model
    pub fn optimize_penalties(
        &mut self,
        model: &CompiledModel,
        sample_results: &[(Vec<bool>, f64)],
    ) -> Result<PenaltyOptimizationResult, Box<dyn std::error::Error>> {
        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.config.max_iterations && !converged {
            // Evaluate constraint violations
            let violations = self.evaluate_violations(model, sample_results)?;

            // Check convergence
            let max_violation = violations.values().map(|v| v.abs()).fold(0.0, f64::max);

            if max_violation < self.config.violation_tolerance {
                converged = true;
                break;
            }

            // Update penalty weights
            self.update_weights(&violations, iteration)?;

            // Record history
            for (name, &violation) in &violations {
                self.violation_history.push(ConstraintViolation {
                    constraint_name: name.clone(),
                    violation_amount: violation,
                    penalty_weight: self.constraint_weights[name],
                    iteration,
                });
            }

            iteration += 1;
        }

        // Calculate final metrics
        let final_violations = self.evaluate_violations(model, sample_results)?;
        let objective_value = self.calculate_objective(model, sample_results)?;
        let constraint_satisfaction = self.calculate_satisfaction_rate(&final_violations);

        Ok(PenaltyOptimizationResult {
            optimal_weights: self.constraint_weights.clone(),
            final_violations,
            converged,
            iterations: iteration,
            objective_value,
            constraint_satisfaction,
        })
    }

    /// Evaluate constraint violations
    fn evaluate_violations(
        &self,
        model: &CompiledModel,
        sample_results: &[(Vec<bool>, f64)],
    ) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut violations = HashMap::new();

        // For each constraint in the model
        for (constraint_name, constraint_expr) in model.get_constraints() {
            let mut total_violation = 0.0;
            let mut count = 0;

            // Evaluate constraint for each sample
            for (assignment, _energy) in sample_results {
                let violation = self.evaluate_constraint_violation(
                    constraint_expr,
                    assignment,
                    &model.get_variable_map(),
                )?;

                total_violation += violation;
                count += 1;
            }

            // Average violation
            violations.insert(
                constraint_name.clone(),
                if count > 0 {
                    total_violation / count as f64
                } else {
                    0.0
                },
            );
        }

        Ok(violations)
    }

    /// Evaluate single constraint violation
    fn evaluate_constraint_violation(
        &self,
        _constraint: &ConstraintExpr,
        _assignment: &[bool],
        _var_map: &HashMap<String, usize>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Placeholder evaluation - in real implementation would parse and evaluate expression
        let value: f64 = 0.0; // Placeholder

        // Calculate violation based on constraint type
        Ok(match self.config.penalty_type {
            PenaltyType::Quadratic => value.powi(2),
            PenaltyType::Linear => value.abs(),
            PenaltyType::LogBarrier => {
                if value > 0.0 {
                    -value.ln()
                } else {
                    f64::INFINITY
                }
            }
            PenaltyType::Exponential => value.exp() - 1.0,
            PenaltyType::AugmentedLagrangian => {
                // Simplified augmented Lagrangian
                value.powi(2) + value.abs()
            }
        })
    }

    /// Update penalty weights based on violations
    fn update_weights(
        &mut self,
        violations: &HashMap<String, f64>,
        _iteration: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            if self.config.adaptive_scaling && self.optimizer.is_some() {
                // Use SciRS2 optimizer for weight updates
                self.update_weights_optimized(violations, iteration)?;
                return Ok(());
            }
        }

        // Standard weight update
        for (constraint_name, &violation) in violations {
            if let Some(weight) = self.constraint_weights.get_mut(constraint_name) {
                if violation.abs() > self.config.violation_tolerance {
                    // Increase penalty weight
                    *weight = (*weight * self.config.adjustment_factor).min(self.config.max_weight);
                } else if violation.abs() < self.config.violation_tolerance * 0.1 {
                    // Decrease penalty weight if over-penalized
                    *weight = (*weight / self.config.adjustment_factor.sqrt())
                        .max(self.config.min_weight);
                }
            }
        }

        Ok(())
    }

    #[cfg(feature = "scirs")]
    /// Update weights using SciRS2 optimization
    fn update_weights_optimized(
        &mut self,
        violations: &HashMap<String, f64>,
        iteration: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::scirs_stub::scirs2_optimization::{Bounds, ObjectiveFunction};

        // Define optimization problem
        let constraint_names: Vec<_> = violations.keys().cloned().collect();
        let current_weights: Array1<f64> = constraint_names
            .iter()
            .map(|name| self.constraint_weights[name])
            .collect();

        // Objective: minimize total weighted violations
        let violations_vec: Array1<f64> = constraint_names
            .iter()
            .map(|name| violations[name].abs())
            .collect();

        let mut objective = WeightOptimizationObjective {
            violations: violations_vec,
            penalty_type: self.config.penalty_type,
            regularization: 0.01, // L2 regularization on weights
        };

        // Set bounds
        let lower_bounds = Array1::from_elem(constraint_names.len(), self.config.min_weight);
        let upper_bounds = Array1::from_elem(constraint_names.len(), self.config.max_weight);
        let bounds = Bounds::new(lower_bounds, upper_bounds);

        // Optimize
        if let Some(ref mut optimizer) = self.optimizer {
            let mut result = optimizer.minimize(&objective, &current_weights, &bounds, iteration)?;

            // Update weights
            for (i, name) in constraint_names.iter().enumerate() {
                self.constraint_weights.insert(name.clone(), result.x[i]);
            }
        }

        Ok(())
    }

    /// Calculate objective value
    fn calculate_objective(
        &self,
        model: &CompiledModel,
        sample_results: &[(Vec<bool>, f64)],
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let mut total_objective = 0.0;

        for (assignment, energy) in sample_results {
            // Original objective
            let mut penalized_objective = *energy;

            // Add penalty terms
            for (constraint_name, constraint_expr) in model.get_constraints() {
                let violation = self.evaluate_constraint_violation(
                    constraint_expr,
                    assignment,
                    &model.get_variable_map(),
                )?;

                let weight = self
                    .constraint_weights
                    .get(constraint_name)
                    .copied()
                    .unwrap_or(1.0);

                penalized_objective += weight * violation;
            }

            total_objective += penalized_objective;
        }

        Ok(total_objective / sample_results.len() as f64)
    }

    /// Calculate constraint satisfaction rate
    fn calculate_satisfaction_rate(&self, violations: &HashMap<String, f64>) -> f64 {
        let satisfied = violations
            .values()
            .filter(|&&v| v.abs() < self.config.violation_tolerance)
            .count();

        if violations.is_empty() {
            1.0
        } else {
            satisfied as f64 / violations.len() as f64
        }
    }

    /// Get penalty weight for a constraint
    pub fn get_weight(&self, constraint_name: &str) -> Option<f64> {
        self.constraint_weights.get(constraint_name).copied()
    }

    /// Get violation history
    pub fn get_violation_history(&self) -> &[ConstraintViolation] {
        &self.violation_history
    }

    /// Export penalty configuration
    pub fn export_config(&self) -> PenaltyExport {
        PenaltyExport {
            config: self.config.clone(),
            weights: self.constraint_weights.clone(),
            final_violations: self
                .violation_history
                .iter()
                .filter(|v| {
                    v.iteration
                        == self
                            .violation_history
                            .iter()
                            .map(|h| h.iteration)
                            .max()
                            .unwrap_or(0)
                })
                .map(|v| (v.constraint_name.clone(), v.violation_amount))
                .collect(),
        }
    }
}

/// Exported penalty configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyExport {
    pub config: PenaltyConfig,
    pub weights: HashMap<String, f64>,
    pub final_violations: HashMap<String, f64>,
}

#[cfg(feature = "scirs")]
/// Objective function for weight optimization
struct WeightOptimizationObjective {
    violations: Array1<f64>,
    penalty_type: PenaltyType,
    regularization: f64,
}

#[cfg(feature = "scirs")]
impl crate::scirs_stub::scirs2_optimization::ObjectiveFunction for WeightOptimizationObjective {
    fn evaluate(&self, weights: &Array1<f64>) -> f64 {
        // Weighted sum of violations
        let weighted_violations = weights * &self.violations;
        let total_violation = weighted_violations.sum();

        // Add regularization term
        let regularization = self.regularization * weights.dot(weights);

        total_violation + regularization
    }

    fn gradient(&self, weights: &Array1<f64>) -> Array1<f64> {
        // Gradient of weighted violations plus regularization
        &self.violations + 2.0 * self.regularization * weights
    }
}

/// Compiled model placeholder
#[derive(Debug, Clone)]
pub struct CompiledModel {
    constraints: HashMap<String, ConstraintExpr>,
    variable_map: HashMap<String, usize>,
}

impl CompiledModel {
    pub fn new() -> Self {
        Self {
            constraints: HashMap::new(),
            variable_map: HashMap::new(),
        }
    }

    pub fn get_constraints(&self) -> &HashMap<String, ConstraintExpr> {
        &self.constraints
    }

    pub fn get_variable_map(&self) -> &HashMap<String, usize> {
        &self.variable_map
    }

    pub fn to_qubo(&self) -> (Array2<f64>, HashMap<String, usize>) {
        let size = self.variable_map.len();
        (Array2::zeros((size, size)), self.variable_map.clone())
    }
}

/// Constraint expression placeholder
#[derive(Debug, Clone)]
pub struct ConstraintExpr {
    pub expression: String,
}

// Helper trait extension for Term evaluation
trait TermEvaluator {
    fn evaluate_with_assignment(
        &self,
        assignment: &[bool],
        var_map: &HashMap<String, usize>,
    ) -> Result<f64, Box<dyn std::error::Error>>;
}

/// Analyze penalty function behavior
pub fn analyze_penalty_landscape(config: &PenaltyConfig, violations: &[f64]) -> PenaltyAnalysis {
    let mut weights = Array1::linspace(config.min_weight, config.max_weight, 100);
    let mut penalties = Vec::new();

    for &weight in weights.iter() {
        let penalty_values: Vec<f64> = violations
            .iter()
            .map(|&v| calculate_penalty(v, weight, config.penalty_type))
            .collect();

        penalties.push(PenaltyPoint {
            weight,
            avg_penalty: penalty_values.iter().sum::<f64>() / penalty_values.len() as f64,
            max_penalty: penalty_values.iter().fold(0.0, |a, &b| a.max(b)),
            min_penalty: penalty_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        });
    }

    PenaltyAnalysis {
        penalty_points: penalties,
        optimal_weight: find_optimal_weight(&weights, violations, config),
        sensitivity: calculate_sensitivity(violations, config),
    }
}

/// Calculate penalty value
fn calculate_penalty(violation: f64, weight: f64, penalty_type: PenaltyType) -> f64 {
    weight
        * match penalty_type {
            PenaltyType::Quadratic => violation.powi(2),
            PenaltyType::Linear => violation.abs(),
            PenaltyType::LogBarrier => {
                if violation > 0.0 {
                    -violation.ln()
                } else {
                    1e10 // Large penalty for infeasible region
                }
            }
            PenaltyType::Exponential => violation.exp() - 1.0,
            PenaltyType::AugmentedLagrangian => violation.powi(2) + violation.abs(),
        }
}

/// Find optimal penalty weight
fn find_optimal_weight(weights: &Array1<f64>, violations: &[f64], config: &PenaltyConfig) -> f64 {
    // Simple heuristic: find weight that balances constraint satisfaction
    // with objective minimization
    let target_penalty = violations.len() as f64 * config.violation_tolerance;

    let mut best_weight = config.initial_weight;
    let mut best_diff = f64::INFINITY;

    for &weight in weights.iter() {
        let total_penalty: f64 = violations
            .iter()
            .map(|&v| calculate_penalty(v, weight, config.penalty_type))
            .sum();

        let diff = (total_penalty - target_penalty).abs();
        if diff < best_diff {
            best_diff = diff;
            best_weight = weight;
        }
    }

    best_weight
}

/// Calculate penalty sensitivity
fn calculate_sensitivity(violations: &[f64], config: &PenaltyConfig) -> f64 {
    if violations.is_empty() {
        return 0.0;
    }

    // Calculate derivative of penalty w.r.t. weight at current weight
    let weight = config.initial_weight;
    let penalties: Vec<f64> = violations
        .iter()
        .map(|&v| calculate_penalty(v, weight, config.penalty_type))
        .collect();

    let mut delta = 0.01 * weight;
    let penalties_delta: Vec<f64> = violations
        .iter()
        .map(|&v| calculate_penalty(v, weight + delta, config.penalty_type))
        .collect();

    let derivatives: Vec<f64> = penalties
        .iter()
        .zip(penalties_delta.iter())
        .map(|(&p1, &p2)| (p2 - p1) / delta)
        .collect();

    // Return average sensitivity
    derivatives.iter().sum::<f64>() / derivatives.len() as f64
}

/// Penalty analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyAnalysis {
    pub penalty_points: Vec<PenaltyPoint>,
    pub optimal_weight: f64,
    pub sensitivity: f64,
}

/// Penalty evaluation point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyPoint {
    pub weight: f64,
    pub avg_penalty: f64,
    pub max_penalty: f64,
    pub min_penalty: f64,
}
