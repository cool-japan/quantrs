//! Validation implementations for test results.
//!
//! This module provides validators that check test results for correctness,
//! constraint satisfaction, and solution quality.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

use super::types::{
    Constraint, ConstraintType, TestCase, TestResult, ValidationCheck, ValidationResult, Validator,
};

/// Constraint validator - checks if solutions satisfy constraints
pub struct ConstraintValidator;

impl Validator for ConstraintValidator {
    fn validate(&self, test_case: &TestCase, result: &TestResult) -> ValidationResult {
        let mut checks = Vec::new();
        let mut is_valid = true;

        for constraint in &test_case.constraints {
            let satisfied = self.check_constraint(constraint, &result.solution);

            checks.push(ValidationCheck {
                name: format!("Constraint {:?}", constraint.constraint_type),
                passed: satisfied,
                message: if satisfied {
                    "Constraint satisfied".to_string()
                } else {
                    "Constraint violated".to_string()
                },
                details: None,
            });

            is_valid &= satisfied;
        }

        ValidationResult {
            is_valid,
            checks,
            warnings: Vec::new(),
        }
    }

    fn name(&self) -> &'static str {
        "ConstraintValidator"
    }
}

impl ConstraintValidator {
    fn check_constraint(&self, constraint: &Constraint, solution: &HashMap<String, bool>) -> bool {
        match &constraint.constraint_type {
            ConstraintType::OneHot => {
                let active = constraint
                    .variables
                    .iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active == 1
            }
            ConstraintType::AtMostK { k } => {
                let active = constraint
                    .variables
                    .iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active <= *k
            }
            ConstraintType::AtLeastK { k } => {
                let active = constraint
                    .variables
                    .iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active >= *k
            }
            ConstraintType::ExactlyK { k } => {
                let active = constraint
                    .variables
                    .iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active == *k
            }
            _ => true, // Other constraints not implemented
        }
    }
}

/// Objective validator - checks solution quality
pub struct ObjectiveValidator;

impl Validator for ObjectiveValidator {
    fn validate(&self, test_case: &TestCase, result: &TestResult) -> ValidationResult {
        let mut checks = Vec::new();

        // Check if objective is better than random
        let random_value = self.estimate_random_objective(&test_case.qubo);
        let improvement = (random_value - result.objective_value) / random_value.abs();

        checks.push(ValidationCheck {
            name: "Objective improvement".to_string(),
            passed: improvement > 0.0,
            message: format!("Improvement over random: {:.2}%", improvement * 100.0),
            details: Some(format!(
                "Random: {:.4}, Found: {:.4}",
                random_value, result.objective_value
            )),
        });

        // Check against optimal if known
        if let Some(optimal_value) = test_case.optimal_value {
            let gap = (result.objective_value - optimal_value).abs() / optimal_value.abs();
            let acceptable_gap = 0.05; // 5% gap

            checks.push(ValidationCheck {
                name: "Optimality gap".to_string(),
                passed: gap <= acceptable_gap,
                message: format!("Gap to optimal: {:.2}%", gap * 100.0),
                details: Some(format!(
                    "Optimal: {:.4}, Found: {:.4}",
                    optimal_value, result.objective_value
                )),
            });
        }

        ValidationResult {
            is_valid: checks.iter().all(|c| c.passed),
            checks,
            warnings: Vec::new(),
        }
    }

    fn name(&self) -> &'static str {
        "ObjectiveValidator"
    }
}

impl ObjectiveValidator {
    fn estimate_random_objective(&self, qubo: &Array2<f64>) -> f64 {
        let n = qubo.shape()[0];
        let mut rng = thread_rng();
        let mut total = 0.0;
        let samples = 100;

        for _ in 0..samples {
            let mut x = vec![0.0; n];
            for x_item in x.iter_mut().take(n) {
                *x_item = if rng.random::<bool>() { 1.0 } else { 0.0 };
            }

            let mut value = 0.0;
            for i in 0..n {
                for j in 0..n {
                    value += qubo[[i, j]] * x[i] * x[j];
                }
            }

            total += value;
        }

        total / samples as f64
    }
}

/// Bounds validator - checks variable bounds
pub struct BoundsValidator;

impl Validator for BoundsValidator {
    fn validate(&self, test_case: &TestCase, result: &TestResult) -> ValidationResult {
        let mut checks = Vec::new();

        // Check all variables are binary (always true for bool type)
        let all_binary = true;

        checks.push(ValidationCheck {
            name: "Binary variables".to_string(),
            passed: all_binary,
            message: if all_binary {
                "All variables are binary".to_string()
            } else {
                "Non-binary values found".to_string()
            },
            details: None,
        });

        // Check variable count
        let expected_vars = test_case.var_map.len();
        let actual_vars = result.solution.len();

        checks.push(ValidationCheck {
            name: "Variable count".to_string(),
            passed: expected_vars == actual_vars,
            message: format!("Expected {expected_vars} variables, found {actual_vars}"),
            details: None,
        });

        ValidationResult {
            is_valid: checks.iter().all(|c| c.passed),
            checks,
            warnings: Vec::new(),
        }
    }

    fn name(&self) -> &'static str {
        "BoundsValidator"
    }
}

/// Symmetry validator - detects symmetry issues
pub struct SymmetryValidator;

impl Validator for SymmetryValidator {
    fn validate(&self, test_case: &TestCase, _result: &TestResult) -> ValidationResult {
        let mut warnings = Vec::new();

        // Check for symmetries in QUBO
        if self.is_symmetric(&test_case.qubo) {
            warnings.push("QUBO matrix has symmetries that might not be broken".to_string());
        }

        ValidationResult {
            is_valid: true,
            checks: Vec::new(),
            warnings,
        }
    }

    fn name(&self) -> &'static str {
        "SymmetryValidator"
    }
}

impl SymmetryValidator {
    fn is_symmetric(&self, qubo: &Array2<f64>) -> bool {
        let n = qubo.shape()[0];

        for i in 0..n {
            for j in i + 1..n {
                if (qubo[[i, j]] - qubo[[j, i]]).abs() > 1e-10 {
                    return false;
                }
            }
        }

        true
    }
}

/// Create default validators
pub fn default_validators() -> Vec<Box<dyn Validator>> {
    vec![
        Box::new(ConstraintValidator),
        Box::new(ObjectiveValidator),
        Box::new(BoundsValidator),
        Box::new(SymmetryValidator),
    ]
}
