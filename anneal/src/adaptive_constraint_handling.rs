//! Adaptive Constraint Handling for Quantum Annealing
//!
//! This module implements sophisticated adaptive constraint handling techniques
//! that dynamically adjust penalty coefficients, constraint relaxation, and
//! violation handling during the optimization process.
//!
//! # Key Features
//!
//! - **Dynamic Penalty Adjustment**: Automatically adjusts penalty coefficients based on constraint violations
//! - **Soft Constraints**: Support for both hard and soft constraints with priority levels
//! - **Constraint Relaxation**: Adaptive relaxation of constraints when necessary
//! - **Violation Tracking**: Comprehensive tracking of constraint violations over time
//! - **Multi-Objective Balance**: Balances objective optimization with constraint satisfaction
//!
//! # Example
//!
//! ```rust,no_run
//! use quantrs2_anneal::adaptive_constraint_handling::*;
//! use quantrs2_anneal::ising::IsingModel;
//!
//! // Create constraint handler
//! let config = AdaptiveConstraintConfig::default();
//! let mut handler = AdaptiveConstraintHandler::new(config);
//!
//! // Define constraints
//! let constraint = Constraint::new(
//!     "sum_constraint",
//!     ConstraintType::Equality,
//!     vec![0, 1, 2],
//!     2.0,
//!     ConstraintPriority::High
//! );
//!
//! handler.add_constraint(constraint);
//!
//! // During optimization, adapt penalties based on current solution
//! let solution = vec![1, 0, 1, 0]; // Example solution
//! handler.adapt_penalties(&solution);
//! ```

use scirs2_core::random::{ChaCha8Rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::applications::ApplicationResult;
use crate::ising::IsingModel;

/// Constraint type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Equality constraint (must equal target value)
    Equality,
    /// Less than or equal constraint
    LessThanOrEqual,
    /// Greater than or equal constraint
    GreaterThanOrEqual,
    /// Custom constraint with user-defined evaluation
    Custom,
}

/// Constraint priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConstraintPriority {
    /// Low priority (soft constraint)
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority (hard constraint)
    Critical,
}

/// Penalty adaptation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PenaltyStrategy {
    /// Static penalty (no adaptation)
    Static,
    /// Multiplicative increase/decrease
    Multiplicative,
    /// Additive increase/decrease
    Additive,
    /// Adaptive based on violation history
    Adaptive,
    /// Exponential increase for repeated violations
    Exponential,
}

/// Constraint relaxation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelaxationStrategy {
    /// No relaxation (hard constraints only)
    None,
    /// Linear relaxation over time
    Linear,
    /// Exponential relaxation
    Exponential,
    /// Adaptive based on convergence
    Adaptive,
}

/// A single constraint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Unique constraint identifier
    pub id: String,

    /// Type of constraint
    pub constraint_type: ConstraintType,

    /// Variables involved in constraint
    pub variables: Vec<usize>,

    /// Target value for the constraint
    pub target_value: f64,

    /// Current penalty coefficient
    pub penalty_coefficient: f64,

    /// Initial penalty coefficient
    pub initial_penalty: f64,

    /// Priority level
    pub priority: ConstraintPriority,

    /// Whether this is a soft constraint
    pub is_soft: bool,

    /// Tolerance for soft constraints
    pub tolerance: f64,

    /// Number of times violated
    pub violation_count: usize,

    /// Total violation magnitude
    pub cumulative_violation: f64,
}

/// Configuration for adaptive constraint handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConstraintConfig {
    /// Initial penalty coefficient
    pub initial_penalty: f64,

    /// Minimum penalty coefficient
    pub min_penalty: f64,

    /// Maximum penalty coefficient
    pub max_penalty: f64,

    /// Penalty adaptation strategy
    pub penalty_strategy: PenaltyStrategy,

    /// Penalty increase factor
    pub penalty_increase_factor: f64,

    /// Penalty decrease factor
    pub penalty_decrease_factor: f64,

    /// Constraint relaxation strategy
    pub relaxation_strategy: RelaxationStrategy,

    /// Relaxation rate
    pub relaxation_rate: f64,

    /// Maximum iterations for adaptation
    pub max_adaptation_iterations: usize,

    /// Violation tolerance threshold
    pub violation_tolerance: f64,

    /// Enable automatic constraint tightening
    pub enable_tightening: bool,

    /// Enable constraint prioritization
    pub enable_prioritization: bool,

    /// History window size for adaptive strategies
    pub history_window: usize,
}

/// Statistics about constraint handling
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintStatistics {
    /// Total constraints
    pub total_constraints: usize,

    /// Hard constraints
    pub hard_constraints: usize,

    /// Soft constraints
    pub soft_constraints: usize,

    /// Total violations
    pub total_violations: usize,

    /// Current violations
    pub current_violations: usize,

    /// Average penalty coefficient
    pub avg_penalty_coefficient: f64,

    /// Maximum violation magnitude
    pub max_violation: f64,

    /// Average violation magnitude
    pub avg_violation: f64,

    /// Satisfaction rate (0.0 to 1.0)
    pub satisfaction_rate: f64,

    /// Number of penalty adaptations
    pub num_adaptations: usize,

    /// Number of constraint relaxations
    pub num_relaxations: usize,
}

/// Violation record for a constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationRecord {
    /// Constraint ID
    pub constraint_id: String,

    /// Iteration when violation occurred
    pub iteration: usize,

    /// Violation magnitude
    pub magnitude: f64,

    /// Current penalty at time of violation
    pub penalty: f64,

    /// Whether violation was resolved
    pub resolved: bool,
}

/// Adaptive Constraint Handler
///
/// Main structure for managing adaptive constraint handling during optimization
pub struct AdaptiveConstraintHandler {
    /// Configuration
    pub config: AdaptiveConstraintConfig,

    /// Registered constraints
    pub constraints: HashMap<String, Constraint>,

    /// Violation history
    pub violation_history: Vec<ViolationRecord>,

    /// Statistics
    pub statistics: ConstraintStatistics,

    /// Current iteration
    pub current_iteration: usize,

    /// Random number generator
    rng: ChaCha8Rng,
}

impl Constraint {
    /// Create a new constraint
    pub fn new(
        id: impl Into<String>,
        constraint_type: ConstraintType,
        variables: Vec<usize>,
        target_value: f64,
        priority: ConstraintPriority,
    ) -> Self {
        let initial_penalty = match priority {
            ConstraintPriority::Low => 1.0,
            ConstraintPriority::Medium => 10.0,
            ConstraintPriority::High => 100.0,
            ConstraintPriority::Critical => 1000.0,
        };

        Self {
            id: id.into(),
            constraint_type,
            variables,
            target_value,
            penalty_coefficient: initial_penalty,
            initial_penalty,
            priority,
            is_soft: priority != ConstraintPriority::Critical,
            tolerance: 0.01,
            violation_count: 0,
            cumulative_violation: 0.0,
        }
    }

    /// Evaluate constraint violation for a given solution
    pub fn evaluate(&self, solution: &[i8]) -> f64 {
        // Calculate sum of variables in the constraint
        let sum: i8 = self
            .variables
            .iter()
            .filter_map(|&idx| solution.get(idx))
            .sum();

        let value = sum as f64;

        match self.constraint_type {
            ConstraintType::Equality => (value - self.target_value).abs(),
            ConstraintType::LessThanOrEqual => (value - self.target_value).max(0.0),
            ConstraintType::GreaterThanOrEqual => (self.target_value - value).max(0.0),
            ConstraintType::Custom => {
                // Custom constraints would need a callback mechanism
                // For now, treat as equality
                (value - self.target_value).abs()
            }
        }
    }

    /// Check if constraint is violated
    pub fn is_violated(&self, solution: &[i8]) -> bool {
        let violation = self.evaluate(solution);

        if self.is_soft {
            violation > self.tolerance
        } else {
            violation > 1e-10
        }
    }

    /// Get penalty term contribution
    pub fn penalty_term(&self, solution: &[i8]) -> f64 {
        let violation = self.evaluate(solution);
        self.penalty_coefficient * violation * violation
    }
}

impl Default for AdaptiveConstraintConfig {
    fn default() -> Self {
        Self {
            initial_penalty: 10.0,
            min_penalty: 0.1,
            max_penalty: 10000.0,
            penalty_strategy: PenaltyStrategy::Adaptive,
            penalty_increase_factor: 1.5,
            penalty_decrease_factor: 0.9,
            relaxation_strategy: RelaxationStrategy::Adaptive,
            relaxation_rate: 0.01,
            max_adaptation_iterations: 1000,
            violation_tolerance: 0.01,
            enable_tightening: true,
            enable_prioritization: true,
            history_window: 50,
        }
    }
}

impl AdaptiveConstraintHandler {
    /// Create a new adaptive constraint handler
    pub fn new(config: AdaptiveConstraintConfig) -> Self {
        Self {
            config,
            constraints: HashMap::new(),
            violation_history: Vec::new(),
            statistics: ConstraintStatistics::default(),
            current_iteration: 0,
            rng: ChaCha8Rng::from_seed([0u8; 32]),
        }
    }

    /// Add a constraint to the handler
    pub fn add_constraint(&mut self, constraint: Constraint) {
        let id = constraint.id.clone();
        self.constraints.insert(id, constraint);
        self.update_constraint_counts();
    }

    /// Remove a constraint
    pub fn remove_constraint(&mut self, id: &str) -> Option<Constraint> {
        let result = self.constraints.remove(id);
        self.update_constraint_counts();
        result
    }

    /// Evaluate all constraints for a solution
    pub fn evaluate_all(&self, solution: &[i8]) -> HashMap<String, f64> {
        self.constraints
            .iter()
            .map(|(id, constraint)| (id.clone(), constraint.evaluate(solution)))
            .collect()
    }

    /// Check if solution satisfies all constraints
    pub fn is_feasible(&self, solution: &[i8]) -> bool {
        self.constraints.values().all(|c| !c.is_violated(solution))
    }

    /// Get total penalty for a solution
    pub fn total_penalty(&self, solution: &[i8]) -> f64 {
        self.constraints
            .values()
            .map(|c| c.penalty_term(solution))
            .sum()
    }

    /// Adapt penalty coefficients based on violations
    pub fn adapt_penalties(&mut self, solution: &[i8]) {
        self.current_iteration += 1;

        let violations = self.evaluate_all(solution);

        // Collect adaptation data first to avoid borrow checker issues
        let adaptation_data: Vec<(String, f64, bool)> = violations
            .iter()
            .filter_map(|(id, violation)| {
                self.constraints.get(id).map(|constraint| {
                    let is_violated = *violation > constraint.tolerance;
                    (id.clone(), *violation, is_violated)
                })
            })
            .collect();

        // Copy config values to avoid borrow conflicts
        let config = self.config.clone();
        let current_iteration = self.current_iteration;

        // Now apply adaptations
        for (id, violation, is_violated) in adaptation_data {
            if let Some(constraint) = self.constraints.get_mut(&id) {
                if is_violated {
                    // Record violation
                    constraint.violation_count += 1;
                    constraint.cumulative_violation += violation;

                    self.violation_history.push(ViolationRecord {
                        constraint_id: id.clone(),
                        iteration: self.current_iteration,
                        magnitude: violation,
                        penalty: constraint.penalty_coefficient,
                        resolved: false,
                    });

                    // Adapt penalty based on strategy
                    let priority_factor = match constraint.priority {
                        ConstraintPriority::Low => 1.0,
                        ConstraintPriority::Medium => 1.5,
                        ConstraintPriority::High => 2.0,
                        ConstraintPriority::Critical => 3.0,
                    };

                    Self::apply_penalty_increase(constraint, violation, priority_factor, &config);
                } else if constraint.violation_count > 0 {
                    // Constraint is now satisfied, possibly decrease penalty
                    Self::apply_penalty_decrease(constraint, current_iteration, &config);
                }
            }
        }

        // Update statistics
        self.update_statistics(solution);
        self.statistics.num_adaptations += 1;
    }

    /// Apply penalty increase for violated constraint (static method)
    fn apply_penalty_increase(
        constraint: &mut Constraint,
        violation: f64,
        priority_factor: f64,
        config: &AdaptiveConstraintConfig,
    ) {
        match config.penalty_strategy {
            PenaltyStrategy::Static => {
                // No adaptation
            }
            PenaltyStrategy::Multiplicative => {
                constraint.penalty_coefficient *= config.penalty_increase_factor;
            }
            PenaltyStrategy::Additive => {
                let increase = config.initial_penalty * 0.1;
                constraint.penalty_coefficient += increase;
            }
            PenaltyStrategy::Adaptive => {
                // Adaptive strategy based on violation history
                let violation_factor =
                    (violation / constraint.target_value.abs().max(1.0)).min(10.0);
                let increase_mult = 1.0 + (priority_factor * violation_factor * 0.1);
                constraint.penalty_coefficient *= increase_mult;
            }
            PenaltyStrategy::Exponential => {
                // Exponential increase based on repeated violations
                let exp_factor = (constraint.violation_count as f64 * 0.1).exp();
                constraint.penalty_coefficient *= exp_factor.min(2.0);
            }
        }

        // Clamp penalty to valid range
        constraint.penalty_coefficient = constraint
            .penalty_coefficient
            .max(config.min_penalty)
            .min(config.max_penalty);
    }

    /// Apply penalty decrease for satisfied constraint (static method)
    fn apply_penalty_decrease(
        constraint: &mut Constraint,
        current_iteration: usize,
        config: &AdaptiveConstraintConfig,
    ) {
        match config.penalty_strategy {
            PenaltyStrategy::Static => {
                // No adaptation
            }
            PenaltyStrategy::Multiplicative | PenaltyStrategy::Exponential => {
                constraint.penalty_coefficient *= config.penalty_decrease_factor;
            }
            PenaltyStrategy::Additive => {
                let decrease = config.initial_penalty * 0.05;
                constraint.penalty_coefficient =
                    (constraint.penalty_coefficient - decrease).max(0.0);
            }
            PenaltyStrategy::Adaptive => {
                // Gradually decrease if satisfied for multiple iterations
                if current_iteration % 10 == 0 {
                    constraint.penalty_coefficient *= config.penalty_decrease_factor;
                }
            }
        }

        // Clamp penalty to valid range
        constraint.penalty_coefficient = constraint
            .penalty_coefficient
            .max(config.min_penalty)
            .min(config.max_penalty);
    }

    /// Apply constraint relaxation
    pub fn apply_relaxation(&mut self) {
        if self.config.relaxation_strategy == RelaxationStrategy::None {
            return;
        }

        for constraint in self.constraints.values_mut() {
            if !constraint.is_soft {
                continue; // Don't relax hard constraints
            }

            let relaxation_amount = match self.config.relaxation_strategy {
                RelaxationStrategy::None => 0.0,
                RelaxationStrategy::Linear => self.config.relaxation_rate,
                RelaxationStrategy::Exponential => {
                    constraint.tolerance * self.config.relaxation_rate
                }
                RelaxationStrategy::Adaptive => {
                    // Relax more if frequently violated
                    let violation_rate =
                        constraint.violation_count as f64 / self.current_iteration.max(1) as f64;
                    self.config.relaxation_rate * (1.0 + violation_rate)
                }
            };

            constraint.tolerance += relaxation_amount;

            // Cap tolerance at reasonable value
            constraint.tolerance = constraint.tolerance.min(1.0);
        }

        self.statistics.num_relaxations += 1;
    }

    /// Apply to Ising model by adding penalty terms
    pub fn apply_to_model(&self, model: &mut IsingModel) -> ApplicationResult<()> {
        for constraint in self.constraints.values() {
            // Add quadratic penalty terms for constraint
            for i in 0..constraint.variables.len() {
                for j in i..constraint.variables.len() {
                    let var_i = constraint.variables[i];
                    let var_j = constraint.variables[j];

                    if i == j {
                        // Linear term (bias)
                        let current_bias = model.get_bias(var_i).unwrap_or(0.0);
                        let penalty_bias = constraint.penalty_coefficient * constraint.target_value;
                        model.set_bias(var_i, current_bias - penalty_bias)?;
                    } else {
                        // Quadratic term (coupling)
                        let current_coupling = model.get_coupling(var_i, var_j).unwrap_or(0.0);
                        let penalty_coupling = constraint.penalty_coefficient;
                        model.set_coupling(var_i, var_j, current_coupling + penalty_coupling)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get statistics
    pub fn get_statistics(&self) -> &ConstraintStatistics {
        &self.statistics
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = ConstraintStatistics::default();
        self.current_iteration = 0;
        self.violation_history.clear();

        // Reset constraint violation counts
        for constraint in self.constraints.values_mut() {
            constraint.violation_count = 0;
            constraint.cumulative_violation = 0.0;
            constraint.penalty_coefficient = constraint.initial_penalty;
        }

        self.update_constraint_counts();
    }

    /// Update constraint counts in statistics
    fn update_constraint_counts(&mut self) {
        self.statistics.total_constraints = self.constraints.len();
        self.statistics.hard_constraints = self.constraints.values().filter(|c| !c.is_soft).count();
        self.statistics.soft_constraints = self.constraints.values().filter(|c| c.is_soft).count();
    }

    /// Update statistics based on current solution
    fn update_statistics(&mut self, solution: &[i8]) {
        let violations = self.evaluate_all(solution);

        let current_violations = violations
            .values()
            .zip(self.constraints.values())
            .filter(|(violation, constraint)| **violation > constraint.tolerance)
            .count();

        self.statistics.current_violations = current_violations;

        self.statistics.total_violations =
            self.constraints.values().map(|c| c.violation_count).sum();

        self.statistics.avg_penalty_coefficient = self
            .constraints
            .values()
            .map(|c| c.penalty_coefficient)
            .sum::<f64>()
            / self.constraints.len().max(1) as f64;

        let max_violation = violations.values().copied().fold(0.0, f64::max);
        self.statistics.max_violation = max_violation;

        let avg_violation = violations.values().sum::<f64>() / violations.len().max(1) as f64;
        self.statistics.avg_violation = avg_violation;

        let satisfied_count = self.constraints.len() - current_violations;
        self.statistics.satisfaction_rate =
            satisfied_count as f64 / self.constraints.len().max(1) as f64;
    }

    /// Get violation history for a specific constraint
    pub fn get_constraint_history(&self, constraint_id: &str) -> Vec<&ViolationRecord> {
        self.violation_history
            .iter()
            .filter(|r| r.constraint_id == constraint_id)
            .collect()
    }

    /// Get most violated constraints
    pub fn get_most_violated_constraints(&self, top_k: usize) -> Vec<&Constraint> {
        let mut constraints: Vec<&Constraint> = self.constraints.values().collect();
        constraints.sort_by(|a, b| b.violation_count.cmp(&a.violation_count));
        constraints.into_iter().take(top_k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_creation() {
        let constraint = Constraint::new(
            "test_constraint",
            ConstraintType::Equality,
            vec![0, 1, 2],
            2.0,
            ConstraintPriority::High,
        );

        assert_eq!(constraint.id, "test_constraint");
        assert_eq!(constraint.constraint_type, ConstraintType::Equality);
        assert_eq!(constraint.variables.len(), 3);
        assert_eq!(constraint.target_value, 2.0);
        assert!(constraint.is_soft);
    }

    #[test]
    fn test_constraint_evaluation() {
        let constraint = Constraint::new(
            "sum_constraint",
            ConstraintType::Equality,
            vec![0, 1, 2],
            2.0,
            ConstraintPriority::Medium,
        );

        let solution = vec![1, 1, 0, 0];
        let violation = constraint.evaluate(&solution);

        // Sum is 2, target is 2, violation should be 0
        assert_eq!(violation, 0.0);
    }

    #[test]
    fn test_constraint_violation_detection() {
        let constraint = Constraint::new(
            "test",
            ConstraintType::Equality,
            vec![0, 1],
            1.0,
            ConstraintPriority::High,
        );

        let satisfied_solution = vec![1, 0, 0, 0];
        let violated_solution = vec![1, 1, 0, 0];

        assert!(!constraint.is_violated(&satisfied_solution));
        assert!(constraint.is_violated(&violated_solution));
    }

    #[test]
    fn test_handler_creation() {
        let config = AdaptiveConstraintConfig::default();
        let handler = AdaptiveConstraintHandler::new(config);

        assert_eq!(handler.constraints.len(), 0);
        assert_eq!(handler.current_iteration, 0);
    }

    #[test]
    fn test_add_remove_constraints() {
        let config = AdaptiveConstraintConfig::default();
        let mut handler = AdaptiveConstraintHandler::new(config);

        let constraint = Constraint::new(
            "c1",
            ConstraintType::Equality,
            vec![0, 1],
            1.0,
            ConstraintPriority::Medium,
        );

        handler.add_constraint(constraint);
        assert_eq!(handler.constraints.len(), 1);

        handler.remove_constraint("c1");
        assert_eq!(handler.constraints.len(), 0);
    }

    #[test]
    fn test_feasibility_check() {
        let config = AdaptiveConstraintConfig::default();
        let mut handler = AdaptiveConstraintHandler::new(config);

        let constraint = Constraint::new(
            "c1",
            ConstraintType::Equality,
            vec![0, 1],
            1.0,
            ConstraintPriority::High,
        );

        handler.add_constraint(constraint);

        let feasible_solution = vec![1, 0, 0];
        let infeasible_solution = vec![1, 1, 0];

        assert!(handler.is_feasible(&feasible_solution));
        assert!(!handler.is_feasible(&infeasible_solution));
    }

    #[test]
    fn test_penalty_adaptation() {
        let config = AdaptiveConstraintConfig {
            penalty_strategy: PenaltyStrategy::Multiplicative,
            penalty_increase_factor: 2.0,
            ..Default::default()
        };

        let mut handler = AdaptiveConstraintHandler::new(config);

        let constraint = Constraint::new(
            "c1",
            ConstraintType::Equality,
            vec![0, 1],
            1.0,
            ConstraintPriority::Medium,
        );

        let initial_penalty = constraint.penalty_coefficient;
        handler.add_constraint(constraint);

        // Violate the constraint
        let violated_solution = vec![1, 1, 0];
        handler.adapt_penalties(&violated_solution);

        let updated_constraint = handler.constraints.get("c1").unwrap();
        assert!(updated_constraint.penalty_coefficient > initial_penalty);
    }

    #[test]
    fn test_total_penalty_calculation() {
        let config = AdaptiveConstraintConfig::default();
        let mut handler = AdaptiveConstraintHandler::new(config);

        let c1 = Constraint::new(
            "c1",
            ConstraintType::Equality,
            vec![0, 1],
            1.0,
            ConstraintPriority::Medium,
        );
        let c2 = Constraint::new(
            "c2",
            ConstraintType::Equality,
            vec![2, 3],
            1.0,
            ConstraintPriority::Medium,
        );

        handler.add_constraint(c1);
        handler.add_constraint(c2);

        let solution = vec![1, 1, 1, 1];
        let total_penalty = handler.total_penalty(&solution);

        // Both constraints are violated, so total penalty should be positive
        assert!(total_penalty > 0.0);
    }

    #[test]
    fn test_statistics_tracking() {
        let config = AdaptiveConstraintConfig::default();
        let mut handler = AdaptiveConstraintHandler::new(config);

        let constraint = Constraint::new(
            "c1",
            ConstraintType::Equality,
            vec![0, 1],
            1.0,
            ConstraintPriority::High,
        );
        handler.add_constraint(constraint);

        let stats = handler.get_statistics();
        assert_eq!(stats.total_constraints, 1);
        assert_eq!(stats.soft_constraints, 1);
    }

    #[test]
    fn test_constraint_relaxation() {
        let config = AdaptiveConstraintConfig {
            relaxation_strategy: RelaxationStrategy::Linear,
            relaxation_rate: 0.1,
            ..Default::default()
        };

        let mut handler = AdaptiveConstraintHandler::new(config);

        let constraint = Constraint::new(
            "c1",
            ConstraintType::Equality,
            vec![0, 1],
            1.0,
            ConstraintPriority::Low,
        );
        let initial_tolerance = constraint.tolerance;
        handler.add_constraint(constraint);

        handler.apply_relaxation();

        let updated_constraint = handler.constraints.get("c1").unwrap();
        assert!(updated_constraint.tolerance > initial_tolerance);
    }

    #[test]
    fn test_violation_history() {
        let config = AdaptiveConstraintConfig::default();
        let mut handler = AdaptiveConstraintHandler::new(config);

        let constraint = Constraint::new(
            "c1",
            ConstraintType::Equality,
            vec![0, 1],
            1.0,
            ConstraintPriority::Medium,
        );
        handler.add_constraint(constraint);

        let violated_solution = vec![1, 1, 0];
        handler.adapt_penalties(&violated_solution);

        assert!(!handler.violation_history.is_empty());

        let history = handler.get_constraint_history("c1");
        assert!(!history.is_empty());
    }
}
