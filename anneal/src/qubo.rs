//! QUBO problem formulation for quantum annealing
//!
//! This module provides utilities for formulating optimization problems as
//! Quadratic Unconstrained Binary Optimization (QUBO) problems.

use std::collections::HashMap;
use thiserror::Error;

use crate::ising::{IsingError, QuboModel};

/// Errors that can occur when formulating QUBO problems
#[derive(Error, Debug)]
pub enum QuboError {
    /// Error in the underlying Ising model
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Error when formulating a constraint
    #[error("Constraint error: {0}")]
    ConstraintError(String),

    /// Error when a variable is already defined
    #[error("Variable {0} is already defined")]
    DuplicateVariable(String),

    /// Error when a variable is not found
    #[error("Variable {0} not found")]
    VariableNotFound(String),
}

/// Result type for QUBO problem operations
pub type QuboResult<T> = Result<T, QuboError>;

/// A variable in a QUBO problem formulation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable {
    /// Name of the variable
    pub name: String,
    /// Index of the variable in the QUBO model
    pub index: usize,
}

impl Variable {
    /// Create a new variable with the given name and index
    pub fn new(name: impl Into<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            index,
        }
    }
}

/// A builder for creating QUBO problems
///
/// This provides a more convenient interface for formulating optimization problems
/// than directly working with the QuboModel.
#[derive(Debug, Clone)]
pub struct QuboBuilder {
    /// Current number of variables
    num_vars: usize,

    /// Mapping from variable names to indices
    var_map: HashMap<String, usize>,

    /// The underlying QUBO model
    model: QuboModel,

    /// Penalty weight for constraint violations
    constraint_weight: f64,
}

impl QuboBuilder {
    /// Create a new empty QUBO builder
    pub fn new() -> Self {
        Self {
            num_vars: 0,
            var_map: HashMap::new(),
            model: QuboModel::new(0),
            constraint_weight: 10.0,
        }
    }

    /// Set the penalty weight for constraint violations
    pub fn set_constraint_weight(&mut self, weight: f64) -> QuboResult<()> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(QuboError::ConstraintError(format!(
                "Constraint weight must be positive and finite, got {}",
                weight
            )));
        }

        self.constraint_weight = weight;
        Ok(())
    }

    /// Add a new binary variable to the problem
    pub fn add_variable(&mut self, name: impl Into<String>) -> QuboResult<Variable> {
        let name = name.into();

        // Check if the variable already exists
        if self.var_map.contains_key(&name) {
            return Err(QuboError::DuplicateVariable(name));
        }

        // Add the variable
        let index = self.num_vars;
        self.var_map.insert(name.clone(), index);
        self.num_vars += 1;

        // Update the QUBO model
        self.model = QuboModel::new(self.num_vars);

        Ok(Variable::new(name, index))
    }

    /// Add multiple binary variables to the problem
    pub fn add_variables(
        &mut self,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> QuboResult<Vec<Variable>> {
        let mut variables = Vec::new();
        for name in names {
            variables.push(self.add_variable(name)?);
        }
        Ok(variables)
    }

    /// Get a variable by name
    pub fn get_variable(&self, name: &str) -> QuboResult<Variable> {
        match self.var_map.get(name) {
            Some(&index) => Ok(Variable::new(name, index)),
            None => Err(QuboError::VariableNotFound(name.to_string())),
        }
    }

    /// Set the linear coefficient for a variable
    pub fn set_linear_term(&mut self, var: &Variable, value: f64) -> QuboResult<()> {
        // Ensure the variable exists in the model
        if var.index >= self.num_vars {
            return Err(QuboError::VariableNotFound(var.name.clone()));
        }

        Ok(self.model.set_linear(var.index, value)?)
    }

    /// Set the quadratic coefficient for a pair of variables
    pub fn set_quadratic_term(
        &mut self,
        var1: &Variable,
        var2: &Variable,
        value: f64,
    ) -> QuboResult<()> {
        // Ensure the variables exist in the model
        if var1.index >= self.num_vars {
            return Err(QuboError::VariableNotFound(var1.name.clone()));
        }
        if var2.index >= self.num_vars {
            return Err(QuboError::VariableNotFound(var2.name.clone()));
        }

        // Check if the variables are the same
        if var1.index == var2.index {
            return Err(QuboError::ConstraintError(format!(
                "Cannot set quadratic term for the same variable: {}",
                var1.name
            )));
        }

        Ok(self.model.set_quadratic(var1.index, var2.index, value)?)
    }

    /// Set the offset term in the QUBO model
    pub fn set_offset(&mut self, offset: f64) -> QuboResult<()> {
        if !offset.is_finite() {
            return Err(QuboError::ConstraintError(format!(
                "Offset must be finite, got {}",
                offset
            )));
        }

        self.model.offset = offset;
        Ok(())
    }

    /// Add a linear objective term to minimize
    pub fn minimize_linear(&mut self, var: &Variable, coeff: f64) -> QuboResult<()> {
        self.set_linear_term(var, self.model.get_linear(var.index)? + coeff)
    }

    /// Add a quadratic objective term to minimize
    pub fn minimize_quadratic(
        &mut self,
        var1: &Variable,
        var2: &Variable,
        coeff: f64,
    ) -> QuboResult<()> {
        let current = self.model.get_quadratic(var1.index, var2.index)?;
        self.set_quadratic_term(var1, var2, current + coeff)
    }

    /// Add a constraint that two variables must be equal
    ///
    /// This adds a penalty term: weight * (x1 - x2)^2
    pub fn constrain_equal(&mut self, var1: &Variable, var2: &Variable) -> QuboResult<()> {
        // Penalty term: weight * (x1 - x2)^2 = weight * (x1 + x2 - 2*x1*x2)
        let weight = self.constraint_weight;

        // Add weight to var1's linear term
        self.set_linear_term(var1, self.model.get_linear(var1.index)? + weight)?;

        // Add weight to var2's linear term
        self.set_linear_term(var2, self.model.get_linear(var2.index)? + weight)?;

        // Add -2*weight to the quadratic term
        let current = self.model.get_quadratic(var1.index, var2.index)?;
        self.set_quadratic_term(var1, var2, current - 2.0 * weight)
    }

    /// Add a constraint that two variables must be different
    ///
    /// This adds a penalty term: weight * (1 - (x1 - x2)^2)
    pub fn constrain_different(&mut self, var1: &Variable, var2: &Variable) -> QuboResult<()> {
        // Penalty term: weight * (1 - (x1 - x2)^2) = weight * (1 - x1 - x2 + 2*x1*x2)
        let weight = self.constraint_weight;

        // Add -weight to var1's linear term
        self.set_linear_term(var1, self.model.get_linear(var1.index)? - weight)?;

        // Add -weight to var2's linear term
        self.set_linear_term(var2, self.model.get_linear(var2.index)? - weight)?;

        // Add 2*weight to the quadratic term
        let current = self.model.get_quadratic(var1.index, var2.index)?;
        self.set_quadratic_term(var1, var2, current + 2.0 * weight)?;

        // Add weight to the offset
        self.model.offset += weight;

        Ok(())
    }

    /// Add a constraint that exactly one of the variables must be 1
    ///
    /// This adds a penalty term: weight * (sum(x_i) - 1)^2
    pub fn constrain_one_hot(&mut self, vars: &[Variable]) -> QuboResult<()> {
        if vars.is_empty() {
            return Err(QuboError::ConstraintError(
                "Empty one-hot constraint".to_string(),
            ));
        }

        // Penalty term: weight * (sum(x_i) - 1)^2
        // = weight * (sum(x_i)^2 - 2*sum(x_i) + 1)
        // = weight * (sum(x_i) + sum(x_i*x_j for i!=j) - 2*sum(x_i) + 1)
        // = weight * (sum(x_i*x_j for i!=j) - sum(x_i) + 1)
        let weight = self.constraint_weight;

        // Add -weight to each variable's linear term
        for var in vars {
            self.set_linear_term(var, self.model.get_linear(var.index)? - weight)?;
        }

        // Add weight to each pair of variables' quadratic term
        for i in 0..vars.len() {
            for j in (i + 1)..vars.len() {
                let current = self.model.get_quadratic(vars[i].index, vars[j].index)?;
                self.set_quadratic_term(&vars[i], &vars[j], current + 2.0 * weight)?;
            }
        }

        // Add weight to the offset
        self.model.offset += weight;

        Ok(())
    }

    /// Add a constraint that at most one of the variables can be 1
    ///
    /// This adds a penalty term: weight * max(0, sum(x_i) - 1)^2
    pub fn constrain_at_most_one(&mut self, vars: &[Variable]) -> QuboResult<()> {
        if vars.is_empty() {
            return Err(QuboError::ConstraintError(
                "Empty at-most-one constraint".to_string(),
            ));
        }

        // Penalty term: weight * max(0, sum(x_i) - 1)^2
        // For binary variables, this simplifies to:
        // weight * sum(x_i*x_j for i!=j)
        let weight = self.constraint_weight;

        // Add weight to each pair of variables' quadratic term
        for i in 0..vars.len() {
            for j in (i + 1)..vars.len() {
                let current = self.model.get_quadratic(vars[i].index, vars[j].index)?;
                self.set_quadratic_term(&vars[i], &vars[j], current + 2.0 * weight)?;
            }
        }

        Ok(())
    }

    /// Add a constraint that at least one of the variables must be 1
    ///
    /// This adds a penalty term: weight * (1 - sum(x_i))^2
    pub fn constrain_at_least_one(&mut self, vars: &[Variable]) -> QuboResult<()> {
        if vars.is_empty() {
            return Err(QuboError::ConstraintError(
                "Empty at-least-one constraint".to_string(),
            ));
        }

        // Penalty term: weight * (1 - sum(x_i))^2
        // = weight * (1 - 2*sum(x_i) + sum(x_i)^2)
        // = weight * (1 - 2*sum(x_i) + sum(x_i) + sum(x_i*x_j for i!=j))
        // = weight * (1 - sum(x_i) + sum(x_i*x_j for i!=j))
        let weight = self.constraint_weight;

        // Add -weight to each variable's linear term
        for var in vars {
            self.set_linear_term(var, self.model.get_linear(var.index)? - 2.0 * weight)?;
        }

        // Add weight to each pair of variables' quadratic term
        for i in 0..vars.len() {
            for j in (i + 1)..vars.len() {
                let current = self.model.get_quadratic(vars[i].index, vars[j].index)?;
                self.set_quadratic_term(&vars[i], &vars[j], current + 2.0 * weight)?;
            }
        }

        // Add weight to the offset
        self.model.offset += weight;

        Ok(())
    }

    /// Build the final QUBO model
    pub fn build(&self) -> QuboModel {
        self.model.clone()
    }

    /// Get a map of variable names to indices
    pub fn variable_map(&self) -> HashMap<String, usize> {
        self.var_map.clone()
    }

    /// Get the total number of variables
    pub fn num_variables(&self) -> usize {
        self.num_vars
    }

    /// Get a list of all variables
    pub fn variables(&self) -> Vec<Variable> {
        self.var_map
            .iter()
            .map(|(name, &index)| Variable::new(name, index))
            .collect()
    }
}

/// Default implementation for QuboBuilder
impl Default for QuboBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for problems that can be formulated as QUBO
pub trait QuboFormulation {
    /// Formulate the problem as a QUBO
    fn to_qubo(&self) -> QuboResult<(QuboModel, HashMap<String, usize>)>;

    /// Interpret the solution to the QUBO in the context of the original problem
    fn interpret_solution(&self, binary_vars: &[bool]) -> QuboResult<Vec<(String, bool)>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubo_builder_basic() {
        let mut builder = QuboBuilder::new();

        // Add variables
        let x1 = builder.add_variable("x1").unwrap();
        let x2 = builder.add_variable("x2").unwrap();
        let x3 = builder.add_variable("x3").unwrap();

        // Set coefficients
        builder.set_linear_term(&x1, 2.0).unwrap();
        builder.set_linear_term(&x2, -1.0).unwrap();
        builder.set_quadratic_term(&x1, &x2, -4.0).unwrap();
        builder.set_quadratic_term(&x2, &x3, 2.0).unwrap();
        builder.set_offset(1.5).unwrap();

        // Build the QUBO model
        let model = builder.build();

        // Check linear terms
        assert_eq!(model.get_linear(0).unwrap(), 2.0);
        assert_eq!(model.get_linear(1).unwrap(), -1.0);
        assert_eq!(model.get_linear(2).unwrap(), 0.0);

        // Check quadratic terms
        assert_eq!(model.get_quadratic(0, 1).unwrap(), -4.0);
        assert_eq!(model.get_quadratic(1, 2).unwrap(), 2.0);

        // Check offset
        assert_eq!(model.offset, 1.5);
    }

    #[test]
    fn test_qubo_builder_objective() {
        let mut builder = QuboBuilder::new();

        // Add variables
        let x1 = builder.add_variable("x1").unwrap();
        let x2 = builder.add_variable("x2").unwrap();

        // Add objective terms
        builder.minimize_linear(&x1, 2.0).unwrap();
        builder.minimize_linear(&x2, -1.0).unwrap();
        builder.minimize_quadratic(&x1, &x2, -4.0).unwrap();

        // Build the QUBO model
        let model = builder.build();

        // Check linear terms
        assert_eq!(model.get_linear(0).unwrap(), 2.0);
        assert_eq!(model.get_linear(1).unwrap(), -1.0);

        // Check quadratic terms
        assert_eq!(model.get_quadratic(0, 1).unwrap(), -4.0);
    }

    #[test]
    fn test_qubo_builder_constraints() {
        let mut builder = QuboBuilder::new();

        // Add variables
        let x1 = builder.add_variable("x1").unwrap();
        let x2 = builder.add_variable("x2").unwrap();
        let x3 = builder.add_variable("x3").unwrap();

        // Set constraint weight
        builder.set_constraint_weight(5.0).unwrap();

        // Add equality constraint
        builder.constrain_equal(&x1, &x2).unwrap();

        // Add inequality constraint
        builder.constrain_different(&x2, &x3).unwrap();

        // Build the QUBO model
        let model = builder.build();

        // Check the model
        // x1 = x2 constraint adds: 5 * (x1 - x2)^2 = 5 * (x1 + x2 - 2*x1*x2)
        // x2 != x3 constraint adds: 5 * (1 - (x2 - x3)^2) = 5 * (1 - x2 - x3 + 2*x2*x3)

        // Check linear terms
        assert_eq!(model.get_linear(0).unwrap(), 5.0); // x1: +5 from equality
        assert_eq!(model.get_linear(1).unwrap(), 5.0 - 5.0); // x2: +5 from equality, -5 from inequality
        assert_eq!(model.get_linear(2).unwrap(), -5.0); // x3: -5 from inequality

        // Check quadratic terms
        assert_eq!(model.get_quadratic(0, 1).unwrap(), -10.0); // x1*x2: -2*5 from equality
        assert_eq!(model.get_quadratic(1, 2).unwrap(), 10.0); // x2*x3: +2*5 from inequality

        // Check offset
        assert_eq!(model.offset, 5.0); // +5 from inequality
    }

    #[test]
    fn test_qubo_builder_one_hot() {
        let mut builder = QuboBuilder::new();

        // Add variables
        let x1 = builder.add_variable("x1").unwrap();
        let x2 = builder.add_variable("x2").unwrap();
        let x3 = builder.add_variable("x3").unwrap();

        // Set constraint weight
        builder.set_constraint_weight(5.0).unwrap();

        // Add one-hot constraint
        builder
            .constrain_one_hot(&[x1.clone(), x2.clone(), x3.clone()])
            .unwrap();

        // Build the QUBO model
        let model = builder.build();

        // Check the model
        // One-hot constraint adds: 5 * (x1 + x2 + x3 - 1)^2
        // = 5 * (x1 + x2 + x3 + x1*x2 + x1*x3 + x2*x3 - 2*(x1 + x2 + x3) + 1)
        // = 5 * (x1*x2 + x1*x3 + x2*x3 - x1 - x2 - x3 + 1)

        // Check linear terms
        assert_eq!(model.get_linear(0).unwrap(), -5.0); // x1: -5 from one-hot
        assert_eq!(model.get_linear(1).unwrap(), -5.0); // x2: -5 from one-hot
        assert_eq!(model.get_linear(2).unwrap(), -5.0); // x3: -5 from one-hot

        // Check quadratic terms
        assert_eq!(model.get_quadratic(0, 1).unwrap(), 10.0); // x1*x2: +2*5 from one-hot
        assert_eq!(model.get_quadratic(0, 2).unwrap(), 10.0); // x1*x3: +2*5 from one-hot
        assert_eq!(model.get_quadratic(1, 2).unwrap(), 10.0); // x2*x3: +2*5 from one-hot

        // Check offset
        assert_eq!(model.offset, 5.0); // +5 from one-hot
    }
}
