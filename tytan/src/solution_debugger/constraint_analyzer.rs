//! Constraint analysis functionality for the solution debugger.

use super::types::{ConstraintInfo, ConstraintType};
use serde::Serialize;
use std::collections::HashMap;

/// Constraint analyzer
pub struct ConstraintAnalyzer {
    /// Tolerance for constraint satisfaction
    tolerance: f64,
    /// Violation cache
    violation_cache: HashMap<String, Vec<ConstraintViolation>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConstraintViolation {
    /// Constraint violated
    pub constraint: ConstraintInfo,
    /// Violation amount
    pub violation_amount: f64,
    /// Variables causing violation
    pub violating_variables: Vec<String>,
    /// Suggested fixes
    pub suggested_fixes: Vec<SuggestedFix>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuggestedFix {
    /// Fix description
    pub description: String,
    /// Variables to change
    pub variable_changes: HashMap<String, bool>,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Fix complexity
    pub complexity: FixComplexity,
}

#[derive(Debug, Clone, Serialize)]
pub enum FixComplexity {
    /// Single variable flip
    Simple,
    /// Multiple variable changes
    Moderate,
    /// Complex changes required
    Complex,
}

impl ConstraintAnalyzer {
    /// Create new constraint analyzer
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            violation_cache: HashMap::new(),
        }
    }

    /// Analyze constraints for a solution
    pub fn analyze(
        &mut self,
        constraints: &[ConstraintInfo],
        assignments: &HashMap<String, bool>,
    ) -> Vec<ConstraintViolation> {
        let violations = Vec::new();

        for constraint in constraints {
            if let Some(violation) = self.check_constraint(constraint, assignments) {
                violations.push(violation);
            }
        }

        violations
    }

    /// Check individual constraint
    fn check_constraint(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
    ) -> Option<ConstraintViolation> {
        match &constraint.constraint_type {
            ConstraintType::Equality { target } => {
                self.check_equality_constraint(constraint, assignments, *target)
            }
            ConstraintType::Inequality { bound, direction } => {
                self.check_inequality_constraint(constraint, assignments, *bound, direction)
            }
            ConstraintType::AllDifferent => {
                self.check_all_different_constraint(constraint, assignments)
            }
            ConstraintType::ExactlyOne => {
                self.check_exactly_one_constraint(constraint, assignments)
            }
            ConstraintType::AtMostOne => self.check_at_most_one_constraint(constraint, assignments),
            ConstraintType::Custom { .. } => {
                // Placeholder for custom constraint evaluation
                None
            }
        }
    }

    /// Check equality constraint
    fn check_equality_constraint(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
        target: f64,
    ) -> Option<ConstraintViolation> {
        let sum: f64 = constraint
            .variables
            .iter()
            .map(|var| {
                if assignments.get(var).copied().unwrap_or(false) {
                    1.0
                } else {
                    0.0
                }
            })
            .sum();

        let violation_amount = (sum - target).abs();
        if violation_amount > self.tolerance {
            Some(ConstraintViolation {
                constraint: constraint.clone(),
                violation_amount,
                violating_variables: constraint.variables.clone(),
                suggested_fixes: self.generate_equality_fixes(constraint, assignments, target, sum),
            })
        } else {
            None
        }
    }

    /// Check inequality constraint
    fn check_inequality_constraint(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
        bound: f64,
        direction: &super::types::InequalityDirection,
    ) -> Option<ConstraintViolation> {
        let sum: f64 = constraint
            .variables
            .iter()
            .map(|var| {
                if assignments.get(var).copied().unwrap_or(false) {
                    1.0
                } else {
                    0.0
                }
            })
            .sum();

        let violation_amount = match direction {
            super::types::InequalityDirection::LessEqual => {
                if sum > bound {
                    sum - bound
                } else {
                    0.0
                }
            }
            super::types::InequalityDirection::GreaterEqual => {
                if sum < bound {
                    bound - sum
                } else {
                    0.0
                }
            }
        };

        if violation_amount > self.tolerance {
            Some(ConstraintViolation {
                constraint: constraint.clone(),
                violation_amount,
                violating_variables: constraint.variables.clone(),
                suggested_fixes: self.generate_inequality_fixes(
                    constraint,
                    assignments,
                    bound,
                    direction,
                    sum,
                ),
            })
        } else {
            None
        }
    }

    /// Check all different constraint
    fn check_all_different_constraint(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
    ) -> Option<ConstraintViolation> {
        // For binary variables, all different means at most one can be true
        let true_count = constraint
            .variables
            .iter()
            .filter(|var| assignments.get(*var).copied().unwrap_or(false))
            .count();

        if true_count > 1 {
            let violation_amount = (true_count - 1) as f64;
            Some(ConstraintViolation {
                constraint: constraint.clone(),
                violation_amount,
                violating_variables: constraint
                    .variables
                    .iter()
                    .filter(|var| assignments.get(*var).copied().unwrap_or(false))
                    .cloned()
                    .collect(),
                suggested_fixes: self.generate_all_different_fixes(constraint, assignments),
            })
        } else {
            None
        }
    }

    /// Check exactly one constraint
    fn check_exactly_one_constraint(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
    ) -> Option<ConstraintViolation> {
        let true_count = constraint
            .variables
            .iter()
            .filter(|var| assignments.get(*var).copied().unwrap_or(false))
            .count();

        if true_count != 1 {
            let violation_amount = (true_count as i32 - 1).abs() as f64;
            Some(ConstraintViolation {
                constraint: constraint.clone(),
                violation_amount,
                violating_variables: constraint.variables.clone(),
                suggested_fixes: self.generate_exactly_one_fixes(constraint, assignments),
            })
        } else {
            None
        }
    }

    /// Check at most one constraint
    fn check_at_most_one_constraint(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
    ) -> Option<ConstraintViolation> {
        let true_count = constraint
            .variables
            .iter()
            .filter(|var| assignments.get(*var).copied().unwrap_or(false))
            .count();

        if true_count > 1 {
            let violation_amount = (true_count - 1) as f64;
            Some(ConstraintViolation {
                constraint: constraint.clone(),
                violation_amount,
                violating_variables: constraint
                    .variables
                    .iter()
                    .filter(|var| assignments.get(*var).copied().unwrap_or(false))
                    .cloned()
                    .collect(),
                suggested_fixes: self.generate_at_most_one_fixes(constraint, assignments),
            })
        } else {
            None
        }
    }

    /// Generate fixes for equality constraints
    fn generate_equality_fixes(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
        target: f64,
        current_sum: f64,
    ) -> Vec<SuggestedFix> {
        let fixes = Vec::new();

        let difference = target - current_sum;
        if difference > 0.0 {
            // Need to set more variables to true
            let false_vars: Vec<_> = constraint
                .variables
                .iter()
                .filter(|var| !assignments.get(*var).copied().unwrap_or(false))
                .collect();

            if false_vars.len() >= difference as usize {
                let changes = HashMap::new();
                for var in false_vars.iter().take(difference as usize) {
                    changes.insert((*var).clone(), true);
                }

                fixes.push(SuggestedFix {
                    description: format!("Set {} variables to true", difference as usize),
                    variable_changes: changes,
                    expected_improvement: difference * constraint.penalty,
                    complexity: if difference == 1.0 {
                        FixComplexity::Simple
                    } else {
                        FixComplexity::Moderate
                    },
                });
            }
        } else if difference < 0.0 {
            // Need to set some variables to false
            let true_vars: Vec<_> = constraint
                .variables
                .iter()
                .filter(|var| assignments.get(*var).copied().unwrap_or(false))
                .collect();

            if true_vars.len() >= (-difference) as usize {
                let changes = HashMap::new();
                for var in true_vars.iter().take((-difference) as usize) {
                    changes.insert((*var).clone(), false);
                }

                fixes.push(SuggestedFix {
                    description: format!("Set {} variables to false", (-difference) as usize),
                    variable_changes: changes,
                    expected_improvement: (-difference) * constraint.penalty,
                    complexity: if difference == -1.0 {
                        FixComplexity::Simple
                    } else {
                        FixComplexity::Moderate
                    },
                });
            }
        }

        fixes
    }

    /// Generate fixes for inequality constraints
    fn generate_inequality_fixes(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
        bound: f64,
        direction: &super::types::InequalityDirection,
        current_sum: f64,
    ) -> Vec<SuggestedFix> {
        let fixes = Vec::new();

        match direction {
            super::types::InequalityDirection::LessEqual => {
                if current_sum > bound {
                    let excess = current_sum - bound;
                    let true_vars: Vec<_> = constraint
                        .variables
                        .iter()
                        .filter(|var| assignments.get(*var).copied().unwrap_or(false))
                        .collect();

                    if true_vars.len() >= excess as usize {
                        let changes = HashMap::new();
                        for var in true_vars.iter().take(excess as usize) {
                            changes.insert((*var).clone(), false);
                        }

                        fixes.push(SuggestedFix {
                            description: format!(
                                "Set {} variables to false to satisfy bound",
                                excess as usize
                            ),
                            variable_changes: changes,
                            expected_improvement: excess * constraint.penalty,
                            complexity: if excess == 1.0 {
                                FixComplexity::Simple
                            } else {
                                FixComplexity::Moderate
                            },
                        });
                    }
                }
            }
            super::types::InequalityDirection::GreaterEqual => {
                if current_sum < bound {
                    let deficit = bound - current_sum;
                    let false_vars: Vec<_> = constraint
                        .variables
                        .iter()
                        .filter(|var| !assignments.get(*var).copied().unwrap_or(false))
                        .collect();

                    if false_vars.len() >= deficit as usize {
                        let changes = HashMap::new();
                        for var in false_vars.iter().take(deficit as usize) {
                            changes.insert((*var).clone(), true);
                        }

                        fixes.push(SuggestedFix {
                            description: format!(
                                "Set {} variables to true to satisfy bound",
                                deficit as usize
                            ),
                            variable_changes: changes,
                            expected_improvement: deficit * constraint.penalty,
                            complexity: if deficit == 1.0 {
                                FixComplexity::Simple
                            } else {
                                FixComplexity::Moderate
                            },
                        });
                    }
                }
            }
        }

        fixes
    }

    /// Generate fixes for all different constraints
    fn generate_all_different_fixes(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
    ) -> Vec<SuggestedFix> {
        let fixes = Vec::new();

        let true_vars: Vec<_> = constraint
            .variables
            .iter()
            .filter(|var| assignments.get(*var).copied().unwrap_or(false))
            .collect();

        if true_vars.len() > 1 {
            // Set all but one to false
            for keep_var in &true_vars {
                let changes = HashMap::new();
                for var in &true_vars {
                    if *var != *keep_var {
                        changes.insert((*var).clone(), false);
                    }
                }

                fixes.push(SuggestedFix {
                    description: format!("Keep only {} set to true", keep_var),
                    variable_changes: changes,
                    expected_improvement: (true_vars.len() - 1) as f64 * constraint.penalty,
                    complexity: if true_vars.len() == 2 {
                        FixComplexity::Simple
                    } else {
                        FixComplexity::Moderate
                    },
                });
            }
        }

        fixes
    }

    /// Generate fixes for exactly one constraints
    fn generate_exactly_one_fixes(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
    ) -> Vec<SuggestedFix> {
        let fixes = Vec::new();

        let true_count = constraint
            .variables
            .iter()
            .filter(|var| assignments.get(*var).copied().unwrap_or(false))
            .count();

        if true_count == 0 {
            // Set one variable to true
            for var in &constraint.variables {
                let changes = HashMap::new();
                changes.insert(var.clone(), true);

                fixes.push(SuggestedFix {
                    description: format!("Set {} to true", var),
                    variable_changes: changes,
                    expected_improvement: constraint.penalty,
                    complexity: FixComplexity::Simple,
                });
            }
        } else if true_count > 1 {
            // Keep only one true, set others to false
            let true_vars: Vec<_> = constraint
                .variables
                .iter()
                .filter(|var| assignments.get(*var).copied().unwrap_or(false))
                .collect();

            for keep_var in &true_vars {
                let changes = HashMap::new();
                for var in &true_vars {
                    if *var != *keep_var {
                        changes.insert((*var).clone(), false);
                    }
                }

                fixes.push(SuggestedFix {
                    description: format!("Keep only {} set to true", keep_var),
                    variable_changes: changes,
                    expected_improvement: (true_vars.len() - 1) as f64 * constraint.penalty,
                    complexity: if true_vars.len() == 2 {
                        FixComplexity::Simple
                    } else {
                        FixComplexity::Moderate
                    },
                });
            }
        }

        fixes
    }

    /// Generate fixes for at most one constraints
    fn generate_at_most_one_fixes(
        &self,
        constraint: &ConstraintInfo,
        assignments: &HashMap<String, bool>,
    ) -> Vec<SuggestedFix> {
        // Same as all different for binary variables
        self.generate_all_different_fixes(constraint, assignments)
    }
}
