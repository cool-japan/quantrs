//! Domain-Specific Language (DSL) for Optimization Problems
//!
//! This module provides a high-level DSL for expressing optimization problems that can be
//! automatically compiled to Ising/QUBO formulations. The DSL supports various variable types,
//! constraints, and objective functions with a natural syntax.

use std::collections::{HashMap, HashSet};
use std::fmt;
use thiserror::Error;

use crate::ising::{IsingError, IsingModel, QuboModel};

/// Errors that can occur in DSL operations
#[derive(Error, Debug)]
pub enum DslError {
    /// Variable not found
    #[error("Variable not found: {0}")]
    VariableNotFound(String),

    /// Invalid constraint
    #[error("Invalid constraint: {0}")]
    InvalidConstraint(String),

    /// Invalid objective
    #[error("Invalid objective: {0}")]
    InvalidObjective(String),

    /// Compilation error
    #[error("Compilation error: {0}")]
    CompilationError(String),

    /// Type mismatch
    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Invalid range
    #[error("Invalid range: {0}")]
    InvalidRange(String),
}

/// Result type for DSL operations
pub type DslResult<T> = Result<T, DslError>;

/// Variable types in the DSL
#[derive(Debug, Clone, PartialEq)]
pub enum VariableType {
    /// Binary variable (0 or 1)
    Binary,

    /// Integer variable with bounds
    Integer { min: i32, max: i32 },

    /// Spin variable (-1 or +1)
    Spin,

    /// Categorical variable (one-hot encoded)
    Categorical { categories: Vec<String> },

    /// Continuous variable (discretized)
    Continuous { min: f64, max: f64, steps: usize },
}

/// Variable representation in the DSL
#[derive(Debug, Clone)]
pub struct Variable {
    /// Unique identifier
    pub id: String,

    /// Variable type
    pub var_type: VariableType,

    /// Qubit indices for this variable
    pub qubit_indices: Vec<usize>,

    /// Description
    pub description: Option<String>,
}

/// Variable vector for array operations
#[derive(Debug, Clone)]
pub struct VariableVector {
    /// Variables in the vector
    pub variables: Vec<Variable>,

    /// Vector name
    pub name: String,
}

/// Expression tree for building complex expressions
#[derive(Debug, Clone)]
pub enum Expression {
    /// Constant value
    Constant(f64),

    /// Variable reference
    Variable(Variable),

    /// Sum of expressions
    Sum(Vec<Expression>),

    /// Product of expressions
    Product(Box<Expression>, Box<Expression>),

    /// Linear combination
    LinearCombination {
        weights: Vec<f64>,
        terms: Vec<Expression>,
    },

    /// Quadratic term
    Quadratic {
        var1: Variable,
        var2: Variable,
        coefficient: f64,
    },

    /// Power of expression
    Power(Box<Expression>, i32),

    /// Negation
    Negate(Box<Expression>),

    /// Absolute value
    Abs(Box<Expression>),

    /// Conditional expression
    Conditional {
        condition: Box<BooleanExpression>,
        if_true: Box<Expression>,
        if_false: Box<Expression>,
    },
}

/// Boolean expressions for constraints
#[derive(Debug, Clone)]
pub enum BooleanExpression {
    /// Always true
    True,

    /// Always false
    False,

    /// Equality comparison
    Equal(Expression, Expression),

    /// Less than comparison
    LessThan(Expression, Expression),

    /// Less than or equal comparison
    LessThanOrEqual(Expression, Expression),

    /// Greater than comparison
    GreaterThan(Expression, Expression),

    /// Greater than or equal comparison
    GreaterThanOrEqual(Expression, Expression),

    /// Logical AND
    And(Box<BooleanExpression>, Box<BooleanExpression>),

    /// Logical OR
    Or(Box<BooleanExpression>, Box<BooleanExpression>),

    /// Logical NOT
    Not(Box<BooleanExpression>),

    /// Logical XOR
    Xor(Box<BooleanExpression>, Box<BooleanExpression>),

    /// Implication (if-then)
    Implies(Box<BooleanExpression>, Box<BooleanExpression>),

    /// All different constraint
    AllDifferent(Vec<Variable>),

    /// At most one constraint
    AtMostOne(Vec<Variable>),

    /// Exactly one constraint
    ExactlyOne(Vec<Variable>),
}

/// Constraint in the optimization model
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Constraint expression
    pub expression: BooleanExpression,

    /// Constraint name
    pub name: Option<String>,

    /// Penalty weight (for soft constraints)
    pub penalty_weight: Option<f64>,

    /// Whether this is a hard constraint
    pub is_hard: bool,
}

/// Objective function direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

/// Objective function
#[derive(Debug, Clone)]
pub struct Objective {
    /// Expression to optimize
    pub expression: Expression,

    /// Direction (minimize or maximize)
    pub direction: ObjectiveDirection,

    /// Objective name
    pub name: Option<String>,
}

/// Optimization model builder
pub struct OptimizationModel {
    /// Model name
    pub name: String,

    /// Variables in the model
    variables: HashMap<String, Variable>,

    /// Variable vectors
    variable_vectors: HashMap<String, VariableVector>,

    /// Constraints
    constraints: Vec<Constraint>,

    /// Objectives (for multi-objective optimization)
    objectives: Vec<Objective>,

    /// Next available qubit index
    next_qubit: usize,

    /// Model metadata
    metadata: HashMap<String, String>,
}

impl OptimizationModel {
    /// Create a new optimization model
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variables: HashMap::new(),
            variable_vectors: HashMap::new(),
            constraints: Vec::new(),
            objectives: Vec::new(),
            next_qubit: 0,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the model
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Add a binary variable
    pub fn add_binary(&mut self, name: impl Into<String>) -> DslResult<Variable> {
        let var_name = name.into();

        if self.variables.contains_key(&var_name) {
            return Err(DslError::InvalidConstraint(format!(
                "Variable {} already exists",
                var_name
            )));
        }

        let var = Variable {
            id: var_name.clone(),
            var_type: VariableType::Binary,
            qubit_indices: vec![self.next_qubit],
            description: None,
        };

        self.next_qubit += 1;
        self.variables.insert(var_name, var.clone());

        Ok(var)
    }

    /// Add a binary variable vector
    pub fn add_binary_vector(
        &mut self,
        name: impl Into<String>,
        size: usize,
    ) -> DslResult<VariableVector> {
        let vec_name = name.into();
        let mut variables = Vec::new();

        for i in 0..size {
            let var_name = format!("{}[{}]", vec_name, i);
            let var = self.add_binary(var_name)?;
            variables.push(var);
        }

        let var_vec = VariableVector {
            variables,
            name: vec_name.clone(),
        };

        self.variable_vectors.insert(vec_name, var_vec.clone());
        Ok(var_vec)
    }

    /// Add an integer variable
    pub fn add_integer(
        &mut self,
        name: impl Into<String>,
        min: i32,
        max: i32,
    ) -> DslResult<Variable> {
        let var_name = name.into();

        if self.variables.contains_key(&var_name) {
            return Err(DslError::InvalidConstraint(format!(
                "Variable {} already exists",
                var_name
            )));
        }

        if min > max {
            return Err(DslError::InvalidRange(format!(
                "Invalid range [{}, {}]",
                min, max
            )));
        }

        // Calculate number of bits needed
        let range = (max - min) as u32;
        let num_bits = (range + 1).next_power_of_two().trailing_zeros() as usize;

        let qubit_indices: Vec<usize> = (0..num_bits)
            .map(|_| {
                let idx = self.next_qubit;
                self.next_qubit += 1;
                idx
            })
            .collect();

        let var = Variable {
            id: var_name.clone(),
            var_type: VariableType::Integer { min, max },
            qubit_indices,
            description: None,
        };

        self.variables.insert(var_name, var.clone());
        Ok(var)
    }

    /// Add a spin variable
    pub fn add_spin(&mut self, name: impl Into<String>) -> DslResult<Variable> {
        let var_name = name.into();

        if self.variables.contains_key(&var_name) {
            return Err(DslError::InvalidConstraint(format!(
                "Variable {} already exists",
                var_name
            )));
        }

        let var = Variable {
            id: var_name.clone(),
            var_type: VariableType::Spin,
            qubit_indices: vec![self.next_qubit],
            description: None,
        };

        self.next_qubit += 1;
        self.variables.insert(var_name, var.clone());

        Ok(var)
    }

    /// Add a constraint to the model
    pub fn add_constraint(&mut self, constraint: impl Into<Constraint>) -> DslResult<()> {
        let constraint = constraint.into();
        self.constraints.push(constraint);
        Ok(())
    }

    /// Add an objective function (minimize)
    pub fn minimize(&mut self, expression: impl Into<Expression>) -> DslResult<()> {
        let objective = Objective {
            expression: expression.into(),
            direction: ObjectiveDirection::Minimize,
            name: None,
        };

        self.objectives.push(objective);
        Ok(())
    }

    /// Add an objective function (maximize)
    pub fn maximize(&mut self, expression: impl Into<Expression>) -> DslResult<()> {
        let objective = Objective {
            expression: expression.into(),
            direction: ObjectiveDirection::Maximize,
            name: None,
        };

        self.objectives.push(objective);
        Ok(())
    }

    /// Compile the model to QUBO formulation
    pub fn compile_to_qubo(&self) -> DslResult<QuboModel> {
        // Create QUBO model with the right size
        let mut model = QuboModel::new(self.next_qubit);

        // Process objectives and add to QUBO
        for objective in &self.objectives {
            let sign = match objective.direction {
                ObjectiveDirection::Minimize => 1.0,
                ObjectiveDirection::Maximize => -1.0,
            };

            self.add_expression_to_qubo(&mut model, &objective.expression, sign)?;
        }

        // Process constraints and add as penalties
        for constraint in &self.constraints {
            // Hard constraints use a large penalty, soft constraints use specified weight
            let penalty_weight = if constraint.is_hard {
                1000.0 // Large penalty for hard constraints
            } else {
                constraint.penalty_weight.unwrap_or(1.0)
            };

            self.add_constraint_to_qubo(&mut model, &constraint.expression, penalty_weight)?;
        }

        Ok(model)
    }

    /// Compile the model to Ising formulation
    pub fn compile_to_ising(&self) -> DslResult<IsingModel> {
        let qubo = self.compile_to_qubo()?;

        // Convert QUBO to Ising using the standard transformation
        // QUBO: x_i ∈ {0,1}, Ising: s_i ∈ {-1,+1}
        // Transformation: x_i = (1 + s_i) / 2
        let mut ising = IsingModel::new(self.next_qubit);

        // Transform QUBO coefficients to Ising coefficients
        // For each QUBO term Q_ij x_i x_j, we get:
        // Q_ij * (1+s_i)/2 * (1+s_j)/2 = Q_ij/4 * (1 + s_i + s_j + s_i*s_j)

        let mut offset = 0.0;

        // Process linear terms (diagonal of QUBO)
        for i in 0..self.next_qubit {
            let q_val = qubo.get_linear(i)?;

            if q_val.abs() > 1e-10 {
                // Diagonal term: Q_ii x_i = Q_ii/4 * (1 + 2*s_i + s_i^2)
                // = Q_ii/4 * (2 + 2*s_i) since s_i^2 = 1
                let h_i = q_val / 2.0; // Linear term coefficient
                let current_bias = ising.get_bias(i)?;
                ising.set_bias(i, current_bias + h_i)?;
                offset += q_val / 4.0; // Constant offset
            }
        }

        // Process quadratic terms (off-diagonal of QUBO)
        for (i, j, q_val) in qubo.quadratic_terms() {
            if q_val.abs() > 1e-10 {
                // Off-diagonal term: Q_ij x_i x_j
                // = Q_ij/4 * (1 + s_i + s_j + s_i*s_j)
                let j_ij = q_val / 4.0; // Coupling coefficient
                ising.set_coupling(i, j, j_ij)?;

                // Add linear field contributions
                let bias_i = ising.get_bias(i)?;
                ising.set_bias(i, bias_i + q_val / 4.0)?;

                let bias_j = ising.get_bias(j)?;
                ising.set_bias(j, bias_j + q_val / 4.0)?;

                offset += q_val / 4.0;
            }
        }

        // Store the constant offset in metadata if needed
        // (The offset doesn't affect optimization but is needed for absolute energy)

        Ok(ising)
    }

    /// Helper: Add an expression to QUBO model
    fn add_expression_to_qubo(
        &self,
        model: &mut QuboModel,
        expr: &Expression,
        coefficient: f64,
    ) -> DslResult<()> {
        match expr {
            Expression::Constant(_c) => {
                // Constants don't affect optimization, only the absolute energy value
                Ok(())
            }
            Expression::Variable(var) => {
                // Linear term
                if let Some(&qubit_idx) = var.qubit_indices.first() {
                    model.add_linear(qubit_idx, coefficient)?;
                }
                Ok(())
            }
            Expression::Sum(terms) => {
                // Recursively add all terms
                for term in terms {
                    self.add_expression_to_qubo(model, term, coefficient)?;
                }
                Ok(())
            }
            Expression::Product(e1, e2) => {
                // Handle quadratic terms
                if let (Expression::Variable(v1), Expression::Variable(v2)) =
                    (e1.as_ref(), e2.as_ref())
                {
                    if let (Some(&q1), Some(&q2)) =
                        (v1.qubit_indices.first(), v2.qubit_indices.first())
                    {
                        if q1 == q2 {
                            // Same variable - add to linear term
                            model.add_linear(q1, coefficient)?;
                        } else {
                            // Different variables - add to quadratic term
                            let current = model.get_quadratic(q1, q2)?;
                            model.set_quadratic(q1, q2, current + coefficient)?;
                        }
                    }
                }
                Ok(())
            }
            Expression::Quadratic {
                var1,
                var2,
                coefficient: coef,
            } => {
                // Direct quadratic term
                if let (Some(&q1), Some(&q2)) =
                    (var1.qubit_indices.first(), var2.qubit_indices.first())
                {
                    if q1 == q2 {
                        // Same variable - add to linear term
                        model.add_linear(q1, coefficient * coef)?;
                    } else {
                        // Different variables - add to quadratic term
                        let current = model.get_quadratic(q1, q2)?;
                        model.set_quadratic(q1, q2, current + coefficient * coef)?;
                    }
                }
                Ok(())
            }
            Expression::LinearCombination { weights, terms } => {
                // Add weighted sum of terms
                for (weight, term) in weights.iter().zip(terms.iter()) {
                    self.add_expression_to_qubo(model, term, coefficient * weight)?;
                }
                Ok(())
            }
            Expression::Negate(e) => {
                self.add_expression_to_qubo(model, e, -coefficient)?;
                Ok(())
            }
            _ => {
                // For other expression types, return an error or handle appropriately
                Err(DslError::CompilationError(
                    "Unsupported expression type in QUBO compilation".to_string(),
                ))
            }
        }
    }

    /// Helper: Add a constraint to QUBO as a penalty term
    fn add_constraint_to_qubo(
        &self,
        model: &mut QuboModel,
        constraint: &BooleanExpression,
        penalty: f64,
    ) -> DslResult<()> {
        match constraint {
            BooleanExpression::True => Ok(()),
            BooleanExpression::False => {
                // Unsatisfiable constraint
                Err(DslError::InvalidConstraint(
                    "Unsatisfiable constraint (False)".to_string(),
                ))
            }
            BooleanExpression::Equal(e1, e2) => {
                // (e1 - e2)^2 penalty
                let diff =
                    Expression::Sum(vec![e1.clone(), Expression::Negate(Box::new(e2.clone()))]);
                let penalty_expr = Expression::Product(Box::new(diff.clone()), Box::new(diff));
                self.add_expression_to_qubo(model, &penalty_expr, penalty)
            }
            BooleanExpression::ExactlyOne(vars) => {
                // Sum of variables should equal 1: (sum - 1)^2
                let sum_expr = Expression::Sum(
                    vars.iter()
                        .map(|v| Expression::Variable(v.clone()))
                        .collect(),
                );
                let one = Expression::Constant(1.0);
                let diff = Expression::Sum(vec![sum_expr, Expression::Negate(Box::new(one))]);
                let penalty_expr = Expression::Product(Box::new(diff.clone()), Box::new(diff));
                self.add_expression_to_qubo(model, &penalty_expr, penalty)
            }
            BooleanExpression::AtMostOne(vars) => {
                // Penalize pairs: sum_{i<j} x_i * x_j
                for i in 0..vars.len() {
                    for j in (i + 1)..vars.len() {
                        if let (Some(&q1), Some(&q2)) =
                            (vars[i].qubit_indices.first(), vars[j].qubit_indices.first())
                        {
                            let current = model.get_quadratic(q1, q2)?;
                            model.set_quadratic(q1, q2, current + penalty)?;
                        }
                    }
                }
                Ok(())
            }
            BooleanExpression::And(b1, b2) => {
                // Both constraints must be satisfied
                self.add_constraint_to_qubo(model, b1, penalty)?;
                self.add_constraint_to_qubo(model, b2, penalty)?;
                Ok(())
            }
            _ => {
                // For other constraint types, return error or implement as needed
                Err(DslError::CompilationError(
                    "Unsupported constraint type in QUBO compilation".to_string(),
                ))
            }
        }
    }

    /// Get model summary
    pub fn summary(&self) -> ModelSummary {
        ModelSummary {
            name: self.name.clone(),
            num_variables: self.variables.len(),
            num_qubits: self.next_qubit,
            num_constraints: self.constraints.len(),
            num_objectives: self.objectives.len(),
            variable_types: self.count_variable_types(),
        }
    }

    /// Count variables by type
    fn count_variable_types(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for var in self.variables.values() {
            let type_name = match &var.var_type {
                VariableType::Binary => "binary",
                VariableType::Integer { .. } => "integer",
                VariableType::Spin => "spin",
                VariableType::Categorical { .. } => "categorical",
                VariableType::Continuous { .. } => "continuous",
            };

            *counts.entry(type_name.to_string()).or_insert(0) += 1;
        }

        counts
    }
}

/// Model summary information
#[derive(Debug)]
pub struct ModelSummary {
    pub name: String,
    pub num_variables: usize,
    pub num_qubits: usize,
    pub num_constraints: usize,
    pub num_objectives: usize,
    pub variable_types: HashMap<String, usize>,
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model: {}", self.name)?;
        writeln!(f, "  Variables: {}", self.num_variables)?;
        writeln!(f, "  Qubits: {}", self.num_qubits)?;
        writeln!(f, "  Constraints: {}", self.num_constraints)?;
        writeln!(f, "  Objectives: {}", self.num_objectives)?;
        writeln!(f, "  Variable types:")?;
        for (var_type, count) in &self.variable_types {
            writeln!(f, "    {}: {}", var_type, count)?;
        }
        Ok(())
    }
}

/// Expression builder helper methods
impl Expression {
    /// Create a constant expression
    pub fn constant(value: f64) -> Self {
        Expression::Constant(value)
    }

    /// Create a sum expression
    pub fn sum(terms: Vec<Expression>) -> Self {
        Expression::Sum(terms)
    }

    /// Add two expressions
    pub fn add(self, other: Expression) -> Self {
        match (self, other) {
            (Expression::Sum(mut terms), Expression::Sum(other_terms)) => {
                terms.extend(other_terms);
                Expression::Sum(terms)
            }
            (Expression::Sum(mut terms), other) => {
                terms.push(other);
                Expression::Sum(terms)
            }
            (expr, Expression::Sum(mut terms)) => {
                terms.insert(0, expr);
                Expression::Sum(terms)
            }
            (expr1, expr2) => Expression::Sum(vec![expr1, expr2]),
        }
    }

    /// Multiply expression by a constant
    pub fn scale(self, factor: f64) -> Self {
        match self {
            Expression::Constant(value) => Expression::Constant(value * factor),
            Expression::LinearCombination { weights, terms } => Expression::LinearCombination {
                weights: weights.into_iter().map(|w| w * factor).collect(),
                terms,
            },
            expr => Expression::LinearCombination {
                weights: vec![factor],
                terms: vec![expr],
            },
        }
    }

    /// Negate expression
    pub fn negate(self) -> Self {
        Expression::Negate(Box::new(self))
    }
}

/// Variable vector helper methods
impl VariableVector {
    /// Sum all variables in the vector
    pub fn sum(&self) -> Expression {
        Expression::Sum(
            self.variables
                .iter()
                .map(|v| Expression::Variable(v.clone()))
                .collect(),
        )
    }

    /// Weighted sum of variables
    pub fn weighted_sum(&self, weights: &[f64]) -> Expression {
        if weights.len() != self.variables.len() {
            // Return zero expression if dimensions don't match
            return Expression::Constant(0.0);
        }

        Expression::LinearCombination {
            weights: weights.to_vec(),
            terms: self
                .variables
                .iter()
                .map(|v| Expression::Variable(v.clone()))
                .collect(),
        }
    }

    /// Get a specific variable by index
    pub fn get(&self, index: usize) -> Option<&Variable> {
        self.variables.get(index)
    }

    /// Number of variables in the vector
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }
}

/// Constraint builder methods for expressions
impl Expression {
    /// Create equality constraint
    pub fn equals(self, other: impl Into<Expression>) -> Constraint {
        Constraint {
            expression: BooleanExpression::Equal(self, other.into()),
            name: None,
            penalty_weight: None,
            is_hard: true,
        }
    }

    /// Create less-than constraint
    pub fn less_than(self, other: impl Into<Expression>) -> Constraint {
        Constraint {
            expression: BooleanExpression::LessThan(self, other.into()),
            name: None,
            penalty_weight: None,
            is_hard: true,
        }
    }

    /// Create less-than-or-equal constraint
    pub fn less_than_or_equal(self, other: impl Into<Expression>) -> Constraint {
        Constraint {
            expression: BooleanExpression::LessThanOrEqual(self, other.into()),
            name: None,
            penalty_weight: None,
            is_hard: true,
        }
    }

    /// Create greater-than constraint
    pub fn greater_than(self, other: impl Into<Expression>) -> Constraint {
        Constraint {
            expression: BooleanExpression::GreaterThan(self, other.into()),
            name: None,
            penalty_weight: None,
            is_hard: true,
        }
    }

    /// Create greater-than-or-equal constraint
    pub fn greater_than_or_equal(self, other: impl Into<Expression>) -> Constraint {
        Constraint {
            expression: BooleanExpression::GreaterThanOrEqual(self, other.into()),
            name: None,
            penalty_weight: None,
            is_hard: true,
        }
    }
}

/// Implement `Into<Expression>` for numeric types
impl From<f64> for Expression {
    fn from(value: f64) -> Self {
        Expression::Constant(value)
    }
}

impl From<i32> for Expression {
    fn from(value: i32) -> Self {
        Expression::Constant(value as f64)
    }
}

impl From<Variable> for Expression {
    fn from(var: Variable) -> Self {
        Expression::Variable(var)
    }
}

/// Common optimization patterns
pub mod patterns {
    use super::*;

    /// Create a knapsack problem
    pub fn knapsack(
        items: &[String],
        values: &[f64],
        weights: &[f64],
        capacity: f64,
    ) -> DslResult<OptimizationModel> {
        let n = items.len();

        if values.len() != n || weights.len() != n {
            return Err(DslError::DimensionMismatch {
                expected: n,
                actual: values.len().min(weights.len()),
            });
        }

        let mut model = OptimizationModel::new("Knapsack Problem");

        // Binary variables for item selection
        let selection = model.add_binary_vector("select", n)?;

        // Constraint: total weight <= capacity
        model.add_constraint(selection.weighted_sum(weights).less_than_or_equal(capacity))?;

        // Objective: maximize total value
        model.maximize(selection.weighted_sum(values))?;

        Ok(model)
    }

    /// Create a graph coloring problem
    pub fn graph_coloring(
        vertices: &[String],
        edges: &[(usize, usize)],
        num_colors: usize,
    ) -> DslResult<OptimizationModel> {
        let n = vertices.len();

        let mut model = OptimizationModel::new("Graph Coloring");

        // Binary variables x[v][c] = 1 if vertex v has color c
        let mut x = Vec::new();
        for v in 0..n {
            let colors = model.add_binary_vector(format!("vertex_{}_color", v), num_colors)?;
            x.push(colors);
        }

        // Constraint: each vertex has exactly one color
        for v in 0..n {
            let vertex_vars: Vec<Variable> = (0..num_colors)
                .filter_map(|c| x[v].get(c).cloned())
                .collect();

            model.add_constraint(Constraint {
                expression: BooleanExpression::ExactlyOne(vertex_vars),
                name: Some(format!("vertex_{}_one_color", v)),
                penalty_weight: None,
                is_hard: true,
            })?;
        }

        // Constraint: adjacent vertices have different colors
        for &(u, v) in edges {
            for c in 0..num_colors {
                if let (Some(var_u), Some(var_v)) = (x[u].get(c), x[v].get(c)) {
                    // Both vertices cannot have the same color
                    model.add_constraint(Constraint {
                        expression: BooleanExpression::AtMostOne(vec![
                            var_u.clone(),
                            var_v.clone(),
                        ]),
                        name: Some(format!("edge_{}_{}_color_{}", u, v, c)),
                        penalty_weight: None,
                        is_hard: true,
                    })?;
                }
            }
        }

        // Objective: minimize number of colors used (optional)
        let mut color_used = Vec::new();
        for c in 0..num_colors {
            let color_var = model.add_binary(format!("color_{}_used", c))?;
            color_used.push(color_var.clone());

            // If any vertex uses color c, then color_used[c] = 1
            for v in 0..n {
                if let Some(var_vc) = x[v].get(c) {
                    // This is a simplified constraint - full implementation would be more complex
                    model.add_constraint(
                        Expression::Variable(var_vc.clone())
                            .less_than_or_equal(Expression::Variable(color_var.clone())),
                    )?;
                }
            }
        }

        model.minimize(Expression::Sum(
            color_used.into_iter().map(Expression::Variable).collect(),
        ))?;

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_variable_creation() {
        let mut model = OptimizationModel::new("Test Model");
        let var = model.add_binary("x").unwrap();

        assert_eq!(var.id, "x");
        assert_eq!(var.qubit_indices.len(), 1);
        assert!(matches!(var.var_type, VariableType::Binary));
    }

    #[test]
    fn test_binary_vector_creation() {
        let mut model = OptimizationModel::new("Test Model");
        let vec = model.add_binary_vector("x", 5).unwrap();

        assert_eq!(vec.name, "x");
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.variables[0].id, "x[0]");
        assert_eq!(vec.variables[4].id, "x[4]");
    }

    #[test]
    fn test_integer_variable_creation() {
        let mut model = OptimizationModel::new("Test Model");
        let var = model.add_integer("i", 0, 7).unwrap();

        assert_eq!(var.id, "i");
        assert_eq!(var.qubit_indices.len(), 3); // 2^3 = 8 > 7
        assert!(matches!(
            var.var_type,
            VariableType::Integer { min: 0, max: 7 }
        ));
    }

    #[test]
    fn test_expression_building() {
        let expr1 = Expression::constant(5.0);
        let expr2 = Expression::constant(3.0);

        let sum = expr1.add(expr2);
        assert!(matches!(sum, Expression::Sum(_)));

        let scaled = Expression::constant(2.0).scale(3.0);
        if let Expression::Constant(value) = scaled {
            assert_eq!(value, 6.0);
        } else {
            panic!("Expected constant expression");
        }
    }

    #[test]
    fn test_knapsack_pattern() {
        let items = vec![
            "Item1".to_string(),
            "Item2".to_string(),
            "Item3".to_string(),
        ];
        let values = vec![10.0, 20.0, 15.0];
        let weights = vec![5.0, 10.0, 7.0];
        let capacity = 15.0;

        let model = patterns::knapsack(&items, &values, &weights, capacity).unwrap();

        assert_eq!(model.name, "Knapsack Problem");
        assert_eq!(model.summary().num_variables, 3);
        assert_eq!(model.summary().num_constraints, 1);
        assert_eq!(model.summary().num_objectives, 1);
    }
}
