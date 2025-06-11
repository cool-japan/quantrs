//! Higher-Order Binary Optimization (HOBO) support
//!
//! This module provides support for optimization problems with interactions
//! between more than two variables. These higher-order terms must be reduced
//! to quadratic form for quantum annealing hardware.

use crate::compression::CompressedQubo;
use crate::ising::{IsingError, IsingResult};
use std::collections::{HashMap, HashSet};

/// Represents a higher-order term with its coefficient
#[derive(Debug, Clone)]
pub struct HigherOrderTerm {
    /// Variables involved in this term (sorted)
    pub variables: Vec<usize>,
    /// Coefficient of the term
    pub coefficient: f64,
}

impl HigherOrderTerm {
    /// Create a new higher-order term
    pub fn new(mut variables: Vec<usize>, coefficient: f64) -> Self {
        variables.sort();
        variables.dedup();
        Self {
            variables,
            coefficient,
        }
    }

    /// Get the order of this term (number of variables)
    pub fn order(&self) -> usize {
        self.variables.len()
    }

    /// Check if this term contains a specific variable
    pub fn contains(&self, var: usize) -> bool {
        self.variables.binary_search(&var).is_ok()
    }
}

/// Higher-Order Binary Optimization problem
#[derive(Debug, Clone)]
pub struct HoboProblem {
    /// Number of variables
    pub num_vars: usize,
    /// Linear terms (order 1)
    pub linear_terms: HashMap<usize, f64>,
    /// Quadratic terms (order 2)
    pub quadratic_terms: HashMap<(usize, usize), f64>,
    /// Higher-order terms (order >= 3)
    pub higher_order_terms: Vec<HigherOrderTerm>,
    /// Constant offset
    pub offset: f64,
}

impl HoboProblem {
    /// Create a new HOBO problem
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            linear_terms: HashMap::new(),
            quadratic_terms: HashMap::new(),
            higher_order_terms: Vec::new(),
            offset: 0.0,
        }
    }

    /// Add a linear term
    pub fn add_linear(&mut self, var: usize, coefficient: f64) {
        if var >= self.num_vars {
            self.num_vars = var + 1;
        }
        *self.linear_terms.entry(var).or_insert(0.0) += coefficient;
    }

    /// Add a quadratic term
    pub fn add_quadratic(&mut self, var1: usize, var2: usize, coefficient: f64) {
        let max_var = var1.max(var2);
        if max_var >= self.num_vars {
            self.num_vars = max_var + 1;
        }

        if var1 == var2 {
            self.add_linear(var1, coefficient);
        } else {
            let (i, j) = if var1 < var2 {
                (var1, var2)
            } else {
                (var2, var1)
            };
            *self.quadratic_terms.entry((i, j)).or_insert(0.0) += coefficient;
        }
    }

    /// Add a higher-order term
    pub fn add_higher_order(&mut self, variables: Vec<usize>, coefficient: f64) {
        if variables.len() < 3 {
            // Handle as linear or quadratic
            match variables.len() {
                0 => self.offset += coefficient,
                1 => self.add_linear(variables[0], coefficient),
                2 => self.add_quadratic(variables[0], variables[1], coefficient),
                _ => unreachable!(),
            }
        } else {
            let term = HigherOrderTerm::new(variables, coefficient);

            // Update num_vars if needed
            if let Some(&max_var) = term.variables.last() {
                if max_var >= self.num_vars {
                    self.num_vars = max_var + 1;
                }
            }

            self.higher_order_terms.push(term);
        }
    }

    /// Get the maximum order of terms in this problem
    pub fn max_order(&self) -> usize {
        self.higher_order_terms
            .iter()
            .map(|term| term.order())
            .max()
            .unwrap_or(2)
            .max(if !self.quadratic_terms.is_empty() {
                2
            } else {
                0
            })
            .max(if !self.linear_terms.is_empty() { 1 } else { 0 })
    }

    /// Check if this is already a QUBO (no higher-order terms)
    pub fn is_qubo(&self) -> bool {
        self.higher_order_terms.is_empty()
    }

    /// Convert to QUBO using auxiliary variables
    pub fn to_qubo(&self, reduction_method: ReductionMethod) -> IsingResult<QuboReduction> {
        match reduction_method {
            ReductionMethod::SubstitutionMethod => self.reduce_by_substitution(),
            ReductionMethod::MinimumVertexCover => self.reduce_by_mvc(),
            ReductionMethod::BooleanProduct => self.reduce_by_boolean_product(),
        }
    }

    /// Reduce using substitution method
    fn reduce_by_substitution(&self) -> IsingResult<QuboReduction> {
        let mut reduction = QuboReduction::new(self.num_vars);

        // Copy linear and quadratic terms
        for (&var, &coeff) in &self.linear_terms {
            reduction.qubo.add_linear(var, coeff);
        }
        for (&(i, j), &coeff) in &self.quadratic_terms {
            reduction.qubo.add_quadratic(i, j, coeff);
        }
        reduction.qubo.offset = self.offset;

        // Process each higher-order term
        for hot in &self.higher_order_terms {
            if hot.order() < 3 {
                continue;
            }

            // Recursively reduce the term
            self.reduce_term_substitution(hot, &mut reduction)?;
        }

        Ok(reduction)
    }

    /// Reduce a single term using substitution
    fn reduce_term_substitution(
        &self,
        term: &HigherOrderTerm,
        reduction: &mut QuboReduction,
    ) -> IsingResult<()> {
        let mut current_vars = term.variables.clone();
        let mut current_coeff = term.coefficient;

        // Reduce order iteratively
        while current_vars.len() > 2 {
            // Take first two variables
            let v1 = current_vars[0];
            let v2 = current_vars[1];

            // Create auxiliary variable for v1 * v2
            let aux_var = reduction.qubo.num_vars;
            reduction.qubo.num_vars += 1;

            // Add reduction info
            reduction.auxiliary_vars.push(AuxiliaryVariable {
                index: aux_var,
                reduction_type: ReductionType::Product(v1, v2),
                penalty_weight: 3.0 * current_coeff.abs(),
            });

            // Add penalty term: 3*v1*v2 - 2*v1*aux - 2*v2*aux + aux
            let penalty = 3.0 * current_coeff.abs();
            reduction.qubo.add_quadratic(v1, v2, penalty * 3.0);
            reduction.qubo.add_quadratic(v1, aux_var, -penalty * 2.0);
            reduction.qubo.add_quadratic(v2, aux_var, -penalty * 2.0);
            reduction.qubo.add_linear(aux_var, penalty);

            // Replace v1 and v2 with aux_var in the term
            current_vars = current_vars[2..].to_vec();
            current_vars.push(aux_var);
            current_vars.sort();
        }

        // Add the final quadratic term
        if current_vars.len() == 2 {
            reduction
                .qubo
                .add_quadratic(current_vars[0], current_vars[1], current_coeff);
        } else if current_vars.len() == 1 {
            reduction.qubo.add_linear(current_vars[0], current_coeff);
        }

        Ok(())
    }

    /// Reduce using minimum vertex cover method
    fn reduce_by_mvc(&self) -> IsingResult<QuboReduction> {
        // This is a more advanced reduction method that finds an optimal
        // set of auxiliary variables to cover all higher-order terms
        // For now, fall back to substitution method
        self.reduce_by_substitution()
    }

    /// Reduce using boolean product method
    fn reduce_by_boolean_product(&self) -> IsingResult<QuboReduction> {
        let mut reduction = QuboReduction::new(self.num_vars);

        // Copy linear and quadratic terms
        for (&var, &coeff) in &self.linear_terms {
            reduction.qubo.add_linear(var, coeff);
        }
        for (&(i, j), &coeff) in &self.quadratic_terms {
            reduction.qubo.add_quadratic(i, j, coeff);
        }
        reduction.qubo.offset = self.offset;

        // Process each higher-order term
        for hot in &self.higher_order_terms {
            if hot.order() < 3 {
                continue;
            }

            // Create a single auxiliary variable for the entire term
            let aux_var = reduction.qubo.num_vars;
            reduction.qubo.num_vars += 1;

            // Add reduction info
            reduction.auxiliary_vars.push(AuxiliaryVariable {
                index: aux_var,
                reduction_type: ReductionType::MultiProduct(hot.variables.clone()),
                penalty_weight: hot.coefficient.abs() * (hot.order() as f64),
            });

            // Use generalized penalty for boolean product
            // aux = prod(x_i) is enforced by:
            // (k-1)*aux + sum_i(x_i) - 2*aux*sum_i(x_i) >= 0
            // with equality when constraint is satisfied
            let k = hot.order();
            let penalty = hot.coefficient.abs() * (k as f64);

            // Add penalty terms
            reduction.qubo.add_linear(aux_var, penalty * (k - 1) as f64);

            for &var in &hot.variables {
                reduction.qubo.add_linear(var, penalty);
                reduction.qubo.add_quadratic(aux_var, var, -2.0 * penalty);
            }

            // Add the original coefficient
            reduction.qubo.add_linear(aux_var, hot.coefficient);
        }

        Ok(reduction)
    }

    /// Evaluate the energy for a given solution
    pub fn evaluate(&self, solution: &[bool]) -> f64 {
        let mut energy = self.offset;

        // Linear terms
        for (&var, &coeff) in &self.linear_terms {
            if var < solution.len() && solution[var] {
                energy += coeff;
            }
        }

        // Quadratic terms
        for (&(i, j), &coeff) in &self.quadratic_terms {
            if i < solution.len() && j < solution.len() && solution[i] && solution[j] {
                energy += coeff;
            }
        }

        // Higher-order terms
        for term in &self.higher_order_terms {
            let mut product = true;
            for &var in &term.variables {
                if var >= solution.len() || !solution[var] {
                    product = false;
                    break;
                }
            }
            if product {
                energy += term.coefficient;
            }
        }

        energy
    }
}

/// Methods for reducing HOBO to QUBO
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionMethod {
    /// Simple substitution method (easy but may use many auxiliary variables)
    SubstitutionMethod,
    /// Minimum vertex cover method (more efficient)
    MinimumVertexCover,
    /// Boolean product with generalized penalties
    BooleanProduct,
}

/// Type of reduction for an auxiliary variable
#[derive(Debug, Clone)]
pub enum ReductionType {
    /// Auxiliary variable represents product of two variables
    Product(usize, usize),
    /// Auxiliary variable represents product of multiple variables
    MultiProduct(Vec<usize>),
    /// Auxiliary variable for other reduction types
    Custom(String),
}

/// Information about an auxiliary variable
#[derive(Debug, Clone)]
pub struct AuxiliaryVariable {
    /// Index of the auxiliary variable
    pub index: usize,
    /// Type of reduction this variable represents
    pub reduction_type: ReductionType,
    /// Penalty weight used for this reduction
    pub penalty_weight: f64,
}

/// Result of HOBO to QUBO reduction
#[derive(Debug, Clone)]
pub struct QuboReduction {
    /// The resulting QUBO problem
    pub qubo: CompressedQubo,
    /// Information about auxiliary variables
    pub auxiliary_vars: Vec<AuxiliaryVariable>,
    /// Mapping from original to new variables
    pub variable_mapping: HashMap<usize, usize>,
}

impl QuboReduction {
    /// Create a new QUBO reduction
    fn new(original_vars: usize) -> Self {
        let mut variable_mapping = HashMap::new();
        for i in 0..original_vars {
            variable_mapping.insert(i, i);
        }

        Self {
            qubo: CompressedQubo::new(original_vars),
            auxiliary_vars: Vec::new(),
            variable_mapping,
        }
    }

    /// Extract solution for original variables from QUBO solution
    pub fn extract_original_solution(&self, qubo_solution: &[bool]) -> Vec<bool> {
        let mut original_solution = vec![false; self.variable_mapping.len()];

        for (&orig_var, &new_var) in &self.variable_mapping {
            if new_var < qubo_solution.len() {
                original_solution[orig_var] = qubo_solution[new_var];
            }
        }

        original_solution
    }

    /// Verify that auxiliary variable constraints are satisfied
    pub fn verify_constraints(&self, solution: &[bool]) -> ConstraintViolations {
        let mut violations = ConstraintViolations::default();

        for aux in &self.auxiliary_vars {
            if aux.index >= solution.len() {
                violations.missing_variables += 1;
                continue;
            }

            let aux_value = solution[aux.index];

            match &aux.reduction_type {
                ReductionType::Product(v1, v2) => {
                    if *v1 < solution.len() && *v2 < solution.len() {
                        let expected = solution[*v1] && solution[*v2];
                        if aux_value != expected {
                            violations.product_violations += 1;
                        }
                    }
                }
                ReductionType::MultiProduct(vars) => {
                    let expected = vars.iter().all(|&v| v < solution.len() && solution[v]);
                    if aux_value != expected {
                        violations.multi_product_violations += 1;
                    }
                }
                _ => {}
            }
        }

        violations.total = violations.product_violations
            + violations.multi_product_violations
            + violations.missing_variables;

        violations
    }
}

/// Statistics about constraint violations
#[derive(Debug, Clone, Default)]
pub struct ConstraintViolations {
    /// Total number of violations
    pub total: usize,
    /// Product constraint violations
    pub product_violations: usize,
    /// Multi-product constraint violations
    pub multi_product_violations: usize,
    /// Missing variables in solution
    pub missing_variables: usize,
}

/// HOBO problem analyzer
pub struct HoboAnalyzer;

impl HoboAnalyzer {
    /// Analyze a HOBO problem and provide statistics
    pub fn analyze(problem: &HoboProblem) -> HoboStats {
        let mut stats = HoboStats::default();

        stats.num_variables = problem.num_vars;
        stats.num_linear_terms = problem.linear_terms.len();
        stats.num_quadratic_terms = problem.quadratic_terms.len();
        stats.num_higher_order_terms = problem.higher_order_terms.len();

        // Analyze orders
        let mut order_counts = HashMap::new();
        for term in &problem.higher_order_terms {
            *order_counts.entry(term.order()).or_insert(0) += 1;
        }

        if !order_counts.is_empty() {
            stats.max_order = *order_counts.keys().max().unwrap();
            stats.order_distribution = order_counts;
        } else if !problem.quadratic_terms.is_empty() {
            stats.max_order = 2;
        } else if !problem.linear_terms.is_empty() {
            stats.max_order = 1;
        }

        // Estimate reduction complexity
        stats.estimated_aux_vars_substitution = problem
            .higher_order_terms
            .iter()
            .map(|term| term.order().saturating_sub(2))
            .sum();

        stats.estimated_aux_vars_boolean = problem.higher_order_terms.len();

        stats
    }
}

/// Statistics about a HOBO problem
#[derive(Debug, Clone, Default)]
pub struct HoboStats {
    /// Number of variables
    pub num_variables: usize,
    /// Number of linear terms
    pub num_linear_terms: usize,
    /// Number of quadratic terms
    pub num_quadratic_terms: usize,
    /// Number of higher-order terms
    pub num_higher_order_terms: usize,
    /// Maximum order of any term
    pub max_order: usize,
    /// Distribution of term orders
    pub order_distribution: HashMap<usize, usize>,
    /// Estimated auxiliary variables needed (substitution method)
    pub estimated_aux_vars_substitution: usize,
    /// Estimated auxiliary variables needed (boolean product)
    pub estimated_aux_vars_boolean: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hobo_creation() {
        let mut hobo = HoboProblem::new(4);

        // Add various order terms
        hobo.add_linear(0, 1.0);
        hobo.add_quadratic(0, 1, -2.0);
        hobo.add_higher_order(vec![0, 1, 2], 3.0);
        hobo.add_higher_order(vec![1, 2, 3], -1.5);

        assert_eq!(hobo.num_vars, 4);
        assert_eq!(hobo.linear_terms.len(), 1);
        assert_eq!(hobo.quadratic_terms.len(), 1);
        assert_eq!(hobo.higher_order_terms.len(), 2);
        assert_eq!(hobo.max_order(), 3);
    }

    #[test]
    fn test_substitution_reduction() {
        let mut hobo = HoboProblem::new(3);
        hobo.add_higher_order(vec![0, 1, 2], 1.0);

        let reduction = hobo.to_qubo(ReductionMethod::SubstitutionMethod).unwrap();

        // Should have created 1 auxiliary variable
        assert_eq!(reduction.auxiliary_vars.len(), 1);
        assert_eq!(reduction.qubo.num_vars, 4); // 3 original + 1 auxiliary

        // Test that the reduction preserves the problem
        let test_solution = vec![true, true, true, true]; // All variables true
        let violations = reduction.verify_constraints(&test_solution);
        assert_eq!(violations.total, 0);
    }

    #[test]
    fn test_boolean_product_reduction() {
        let mut hobo = HoboProblem::new(4);
        hobo.add_higher_order(vec![0, 1, 2, 3], 2.0);

        let reduction = hobo.to_qubo(ReductionMethod::BooleanProduct).unwrap();

        // Should have created 1 auxiliary variable for the 4-way term
        assert_eq!(reduction.auxiliary_vars.len(), 1);
        assert_eq!(reduction.qubo.num_vars, 5); // 4 original + 1 auxiliary
    }

    #[test]
    fn test_hobo_evaluation() {
        let mut hobo = HoboProblem::new(3);
        hobo.add_linear(0, 1.0);
        hobo.add_quadratic(0, 1, -2.0);
        hobo.add_higher_order(vec![0, 1, 2], 3.0);

        // Test various solutions
        assert_eq!(hobo.evaluate(&[false, false, false]), 0.0);
        assert_eq!(hobo.evaluate(&[true, false, false]), 1.0);
        assert_eq!(hobo.evaluate(&[true, true, false]), 1.0 - 2.0);
        assert_eq!(hobo.evaluate(&[true, true, true]), 1.0 - 2.0 + 3.0);
    }

    #[test]
    fn test_hobo_analyzer() {
        let mut hobo = HoboProblem::new(5);
        hobo.add_linear(0, 1.0);
        hobo.add_quadratic(0, 1, -1.0);
        hobo.add_higher_order(vec![0, 1, 2], 1.0);
        hobo.add_higher_order(vec![1, 2, 3], 1.0);
        hobo.add_higher_order(vec![0, 1, 2, 3, 4], 1.0);

        let stats = HoboAnalyzer::analyze(&hobo);

        assert_eq!(stats.num_variables, 5);
        assert_eq!(stats.num_higher_order_terms, 3);
        assert_eq!(stats.max_order, 5);
        assert_eq!(stats.estimated_aux_vars_substitution, 5); // 1 + 1 + 3
        assert_eq!(stats.estimated_aux_vars_boolean, 3);
    }
}
