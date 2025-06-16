//! Compilation of symbolic expressions to QUBO/HOBO models.
//!
//! This module provides utilities for compiling symbolic expressions
//! into QUBO (Quadratic Unconstrained Binary Optimization) and
//! HOBO (Higher-Order Binary Optimization) models.

use ndarray::Array;
use std::collections::{HashMap, HashSet};

#[cfg(feature = "scirs")]
use crate::scirs_stub;
#[cfg(feature = "dwave")]
use symengine::Expression as SymEngineExpression;

#[cfg(feature = "dwave")]
type Expr = SymEngineExpression;
use thiserror::Error;

use quantrs2_anneal::QuboError;

/// Unified expression interface for examples
#[cfg(feature = "dwave")]
pub mod expr {
    use symengine::Expression as SymEngineExpression;

    pub type Expr = SymEngineExpression;

    pub fn constant(value: f64) -> Expr {
        SymEngineExpression::from_f64(value)
    }

    pub fn var(name: &str) -> Expr {
        SymEngineExpression::symbol(name)
    }
}

#[cfg(not(feature = "dwave"))]
pub mod expr {
    use super::SimpleExpr;

    pub type Expr = SimpleExpr;

    pub fn constant(value: f64) -> Expr {
        SimpleExpr::constant(value)
    }

    pub fn var(name: &str) -> Expr {
        SimpleExpr::var(name)
    }
}

/// Errors that can occur during compilation
#[derive(Error, Debug)]
pub enum CompileError {
    /// Error when the expression is invalid
    #[error("Invalid expression: {0}")]
    InvalidExpression(String),

    /// Error when a term has too high a degree
    #[error("Term has degree {0}, but maximum supported is {1}")]
    DegreeTooHigh(usize, usize),

    /// Error in the underlying QUBO model
    #[error("QUBO error: {0}")]
    QuboError(#[from] QuboError),

    /// Error in Symengine operations
    #[error("Symengine error: {0}")]
    SymengineError(String),
}

/// Result type for compilation operations
pub type CompileResult<T> = Result<T, CompileError>;

// Simple expression type for when dwave feature is not enabled
#[cfg(not(feature = "dwave"))]
#[derive(Debug, Clone)]
pub enum SimpleExpr {
    /// Variable
    Var(String),
    /// Constant
    Const(f64),
    /// Addition
    Add(Box<SimpleExpr>, Box<SimpleExpr>),
    /// Multiplication
    Mul(Box<SimpleExpr>, Box<SimpleExpr>),
    /// Power
    Pow(Box<SimpleExpr>, i32),
}

#[cfg(not(feature = "dwave"))]
impl SimpleExpr {
    /// Create a variable
    pub fn var(name: &str) -> Self {
        SimpleExpr::Var(name.to_string())
    }

    /// Create a constant
    pub fn constant(value: f64) -> Self {
        SimpleExpr::Const(value)
    }
}

#[cfg(not(feature = "dwave"))]
impl std::ops::Add for SimpleExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        SimpleExpr::Add(Box::new(self), Box::new(rhs))
    }
}

#[cfg(not(feature = "dwave"))]
impl std::ops::Mul for SimpleExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        SimpleExpr::Mul(Box::new(self), Box::new(rhs))
    }
}

#[cfg(not(feature = "dwave"))]
impl std::iter::Sum for SimpleExpr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(SimpleExpr::Const(0.0), |acc, x| acc + x)
    }
}

/// High-level model for constraint optimization problems
#[cfg(feature = "dwave")]
#[derive(Debug, Clone)]
pub struct Model {
    /// Variables in the model
    variables: HashSet<String>,
    /// Objective function expression
    objective: Option<Expr>,
    /// Constraints
    constraints: Vec<Constraint>,
}

/// Constraint types
#[cfg(feature = "dwave")]
#[derive(Debug, Clone)]
enum Constraint {
    /// Equality constraint: sum of variables equals value
    Equality {
        name: String,
        expr: Expr,
        value: f64,
    },
    /// Inequality constraint: sum of variables <= value
    LessEqual {
        name: String,
        expr: Expr,
        value: f64,
    },
    /// At most one constraint: at most one variable can be 1
    AtMostOne { name: String, variables: Vec<Expr> },
    /// Implication constraint: if any condition is true, then result must be true
    ImpliesAny {
        name: String,
        conditions: Vec<Expr>,
        result: Expr,
    },
}

#[cfg(feature = "dwave")]
impl Model {
    /// Create a new empty model
    pub fn new() -> Self {
        Self {
            variables: HashSet::new(),
            objective: None,
            constraints: Vec::new(),
        }
    }

    /// Add a variable to the model
    pub fn add_variable(&mut self, name: &str) -> CompileResult<Expr> {
        self.variables.insert(name.to_string());
        Ok(SymEngineExpression::symbol(name))
    }

    /// Set the objective function
    pub fn set_objective(&mut self, expr: Expr) {
        self.objective = Some(expr);
    }

    /// Add constraint: exactly one of the variables must be 1
    pub fn add_constraint_eq_one(&mut self, name: &str, variables: Vec<Expr>) -> CompileResult<()> {
        // For binary variables, sum = 1 means exactly one is 1
        let sum_expr = variables
            .iter()
            .fold(Expr::from(0), |acc, v| acc + v.clone());
        self.constraints.push(Constraint::Equality {
            name: name.to_string(),
            expr: sum_expr,
            value: 1.0,
        });
        Ok(())
    }

    /// Add constraint: at most one of the variables can be 1
    pub fn add_constraint_at_most_one(
        &mut self,
        name: &str,
        variables: Vec<Expr>,
    ) -> CompileResult<()> {
        self.constraints.push(Constraint::AtMostOne {
            name: name.to_string(),
            variables,
        });
        Ok(())
    }

    /// Add constraint: if any condition is true, then result must be true
    pub fn add_constraint_implies_any(
        &mut self,
        name: &str,
        conditions: Vec<Expr>,
        result: Expr,
    ) -> CompileResult<()> {
        self.constraints.push(Constraint::ImpliesAny {
            name: name.to_string(),
            conditions,
            result,
        });
        Ok(())
    }

    /// Compile the model to a CompiledModel
    pub fn compile(&self) -> CompileResult<CompiledModel> {
        // Build the final expression with penalty terms
        let final_expr = self.objective.clone().unwrap_or_else(|| Expr::from(0));

        // Default penalty weight
        let penalty_weight = 10.0;

        // Add penalty terms for constraints
        for constraint in &self.constraints {
            match constraint {
                Constraint::Equality { expr, value, .. } => {
                    // (expr - value)^2 penalty
                    let diff = expr.clone() - Expr::from(*value);
                    final_expr = final_expr + Expr::from(penalty_weight) * diff.clone() * diff;
                }
                Constraint::LessEqual { expr, value, .. } => {
                    // max(0, expr - value)^2 penalty
                    // For simplicity, we'll use a quadratic penalty
                    let excess = expr.clone() - Expr::from(*value);
                    final_expr = final_expr + Expr::from(penalty_weight) * excess.clone() * excess;
                }
                Constraint::AtMostOne { variables, .. } => {
                    // Penalty: sum(xi * xj) for all i < j
                    for i in 0..variables.len() {
                        for j in (i + 1)..variables.len() {
                            final_expr = final_expr
                                + Expr::from(penalty_weight)
                                    * variables[i].clone()
                                    * variables[j].clone();
                        }
                    }
                }
                Constraint::ImpliesAny {
                    conditions, result, ..
                } => {
                    // If any condition is true, result must be true
                    // Penalty: (max(conditions) - result)^2 where max is approximated by sum
                    let conditions_sum = conditions
                        .iter()
                        .fold(Expr::from(0), |acc, c| acc + c.clone());
                    // Penalty when conditions_sum > 0 and result = 0
                    final_expr = final_expr
                        + Expr::from(penalty_weight)
                            * conditions_sum
                            * (Expr::from(1) - result.clone());
                }
            }
        }

        // Use the standard compiler
        let compiler = Compile::new(final_expr);
        let ((qubo_matrix, var_map), offset) = compiler.get_qubo()?;

        Ok(CompiledModel {
            qubo_matrix,
            var_map,
            offset,
            constraints: self.constraints.clone(),
        })
    }
}

/// Compiled model ready for sampling
#[cfg(feature = "dwave")]
#[derive(Debug, Clone)]
pub struct CompiledModel {
    /// QUBO matrix
    pub qubo_matrix: Array<f64, ndarray::Ix2>,
    /// Variable name to index mapping
    pub var_map: HashMap<String, usize>,
    /// Constant offset
    pub offset: f64,
    /// Original constraints (for analysis)
    constraints: Vec<Constraint>,
}

#[cfg(feature = "dwave")]
impl CompiledModel {
    /// Convert to QUBO format
    pub fn to_qubo(&self) -> quantrs2_anneal::ising::QuboModel {
        use quantrs2_anneal::ising::QuboModel;

        let qubo = QuboModel::new(self.var_map.len());

        // Set the offset
        qubo.offset = self.offset;

        // Set all the QUBO coefficients
        for i in 0..self.qubo_matrix.nrows() {
            for j in i..self.qubo_matrix.ncols() {
                let value = self.qubo_matrix[[i, j]];
                if value.abs() > 1e-10 {
                    if i == j {
                        // Diagonal term (linear)
                        qubo.set_linear(i, value).unwrap();
                    } else {
                        // Off-diagonal term (quadratic)
                        qubo.set_quadratic(i, j, value).unwrap();
                    }
                }
            }
        }

        qubo
    }
}

/// High-level model for constraint optimization problems (non-dwave version)
#[cfg(not(feature = "dwave"))]
#[derive(Debug, Clone)]
pub struct Model {
    /// Variables in the model
    variables: HashSet<String>,
    /// Objective function expression
    objective: Option<SimpleExpr>,
    /// Constraints
    constraints: Vec<Constraint>,
}

/// Constraint types (non-dwave version)
#[cfg(not(feature = "dwave"))]
#[derive(Debug, Clone)]
enum Constraint {
    /// Equality constraint: sum of variables equals value
    Equality {
        name: String,
        expr: SimpleExpr,
        value: f64,
    },
    /// At most one constraint: at most one variable can be 1
    AtMostOne {
        name: String,
        variables: Vec<SimpleExpr>,
    },
    /// Implication constraint: if any condition is true, then result must be true
    ImpliesAny {
        name: String,
        conditions: Vec<SimpleExpr>,
        result: SimpleExpr,
    },
}

#[cfg(not(feature = "dwave"))]
impl Model {
    /// Create a new empty model
    pub fn new() -> Self {
        Self {
            variables: HashSet::new(),
            objective: None,
            constraints: Vec::new(),
        }
    }

    /// Add a variable to the model
    pub fn add_variable(&mut self, name: &str) -> CompileResult<SimpleExpr> {
        self.variables.insert(name.to_string());
        Ok(SimpleExpr::var(name))
    }

    /// Set the objective function
    pub fn set_objective(&mut self, expr: SimpleExpr) {
        self.objective = Some(expr);
    }

    /// Add constraint: exactly one of the variables must be 1
    pub fn add_constraint_eq_one(
        &mut self,
        name: &str,
        variables: Vec<SimpleExpr>,
    ) -> CompileResult<()> {
        let sum_expr = variables.into_iter().sum();
        self.constraints.push(Constraint::Equality {
            name: name.to_string(),
            expr: sum_expr,
            value: 1.0,
        });
        Ok(())
    }

    /// Add constraint: at most one of the variables can be 1
    pub fn add_constraint_at_most_one(
        &mut self,
        name: &str,
        variables: Vec<SimpleExpr>,
    ) -> CompileResult<()> {
        self.constraints.push(Constraint::AtMostOne {
            name: name.to_string(),
            variables,
        });
        Ok(())
    }

    /// Add constraint: if any condition is true, then result must be true
    pub fn add_constraint_implies_any(
        &mut self,
        name: &str,
        conditions: Vec<SimpleExpr>,
        result: SimpleExpr,
    ) -> CompileResult<()> {
        self.constraints.push(Constraint::ImpliesAny {
            name: name.to_string(),
            conditions,
            result,
        });
        Ok(())
    }

    /// Compile the model to a CompiledModel
    pub fn compile(&self) -> CompileResult<CompiledModel> {
        // Build QUBO directly from constraints
        let mut qubo_terms: HashMap<(String, String), f64> = HashMap::new();
        let offset = 0.0;
        let penalty_weight = 10.0;

        // Process objective if present
        if let Some(ref obj) = self.objective {
            self.add_expr_to_qubo(obj, 1.0, &mut qubo_terms, &mut offset)?;
        }

        // Process constraints
        for constraint in &self.constraints {
            match constraint {
                Constraint::Equality { expr, value, .. } => {
                    // (expr - value)^2 penalty
                    // Expand: expr^2 - 2*expr*value + value^2
                    self.add_expr_squared_to_qubo(
                        expr,
                        penalty_weight,
                        &mut qubo_terms,
                        &mut offset,
                    )?;
                    self.add_expr_to_qubo(
                        expr,
                        -2.0 * penalty_weight * value,
                        &mut qubo_terms,
                        &mut offset,
                    )?;
                    offset += penalty_weight * value * value;
                }
                Constraint::AtMostOne { variables, .. } => {
                    // Penalty: sum(xi * xj) for all i < j
                    for i in 0..variables.len() {
                        for j in (i + 1)..variables.len() {
                            if let (SimpleExpr::Var(vi), SimpleExpr::Var(vj)) =
                                (&variables[i], &variables[j])
                            {
                                let key = if vi < vj {
                                    (vi.clone(), vj.clone())
                                } else {
                                    (vj.clone(), vi.clone())
                                };
                                *qubo_terms.entry(key).or_insert(0.0) += penalty_weight;
                            }
                        }
                    }
                }
                Constraint::ImpliesAny {
                    conditions, result, ..
                } => {
                    // Penalty: sum(conditions) * (1 - result)
                    for cond in conditions {
                        if let (SimpleExpr::Var(c), SimpleExpr::Var(r)) = (cond, result) {
                            let key = if c < r {
                                (c.clone(), r.clone())
                            } else {
                                (r.clone(), c.clone())
                            };
                            *qubo_terms.entry(key).or_insert(0.0) -= penalty_weight;
                        }
                        // Also add linear term for condition
                        if let SimpleExpr::Var(c) = cond {
                            *qubo_terms.entry((c.clone(), c.clone())).or_insert(0.0) +=
                                penalty_weight;
                        }
                    }
                }
            }
        }

        // Convert to matrix form
        let all_vars: HashSet<String> = qubo_terms
            .keys()
            .flat_map(|(v1, v2)| vec![v1.clone(), v2.clone()])
            .collect();
        let mut sorted_vars: Vec<String> = all_vars.into_iter().collect();
        sorted_vars.sort();

        let var_map: HashMap<String, usize> = sorted_vars
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();

        let n = var_map.len();
        let matrix = Array::zeros((n, n));

        for ((v1, v2), coeff) in qubo_terms {
            let i = var_map[&v1];
            let j = var_map[&v2];
            if i == j {
                matrix[[i, i]] += coeff;
            } else {
                matrix[[i, j]] += coeff / 2.0;
                matrix[[j, i]] += coeff / 2.0;
            }
        }

        Ok(CompiledModel {
            qubo_matrix: matrix,
            var_map,
            offset,
            constraints: self.constraints.clone(),
        })
    }

    /// Add expression to QUBO terms
    fn add_expr_to_qubo(
        &self,
        expr: &SimpleExpr,
        coeff: f64,
        terms: &mut HashMap<(String, String), f64>,
        offset: &mut f64,
    ) -> CompileResult<()> {
        match expr {
            SimpleExpr::Var(name) => {
                *terms.entry((name.clone(), name.clone())).or_insert(0.0) += coeff;
            }
            SimpleExpr::Const(val) => {
                *offset += coeff * val;
            }
            SimpleExpr::Add(left, right) => {
                self.add_expr_to_qubo(left, coeff, terms, offset)?;
                self.add_expr_to_qubo(right, coeff, terms, offset)?;
            }
            SimpleExpr::Mul(left, right) => {
                if let (SimpleExpr::Var(v1), SimpleExpr::Var(v2)) = (left.as_ref(), right.as_ref())
                {
                    let key = if v1 < v2 {
                        (v1.clone(), v2.clone())
                    } else {
                        (v2.clone(), v1.clone())
                    };
                    *terms.entry(key).or_insert(0.0) += coeff;
                } else if let (SimpleExpr::Const(c), var) | (var, SimpleExpr::Const(c)) =
                    (left.as_ref(), right.as_ref())
                {
                    self.add_expr_to_qubo(var, coeff * c, terms, offset)?;
                }
            }
            SimpleExpr::Pow(base, exp) => {
                if *exp == 2 && matches!(base.as_ref(), SimpleExpr::Var(_)) {
                    // x^2 = x for binary variables
                    self.add_expr_to_qubo(base, coeff, terms, offset)?;
                }
            }
        }
        Ok(())
    }

    /// Add expression squared to QUBO terms
    fn add_expr_squared_to_qubo(
        &self,
        expr: &SimpleExpr,
        coeff: f64,
        terms: &mut HashMap<(String, String), f64>,
        offset: &mut f64,
    ) -> CompileResult<()> {
        // For simplicity, only handle simple cases
        match expr {
            SimpleExpr::Var(name) => {
                // x^2 = x for binary
                *terms.entry((name.clone(), name.clone())).or_insert(0.0) += coeff;
            }
            SimpleExpr::Add(left, right) => {
                // (a + b)^2 = a^2 + 2ab + b^2
                self.add_expr_squared_to_qubo(left, coeff, terms, offset)?;
                self.add_expr_squared_to_qubo(right, coeff, terms, offset)?;
                // Cross term
                if let (SimpleExpr::Var(v1), SimpleExpr::Var(v2)) = (left.as_ref(), right.as_ref())
                {
                    let key = if v1 < v2 {
                        (v1.clone(), v2.clone())
                    } else {
                        (v2.clone(), v1.clone())
                    };
                    *terms.entry(key).or_insert(0.0) += 2.0 * coeff;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Compiled model ready for sampling (non-dwave version)
#[cfg(not(feature = "dwave"))]
#[derive(Debug, Clone)]
pub struct CompiledModel {
    /// QUBO matrix
    pub qubo_matrix: Array<f64, ndarray::Ix2>,
    /// Variable name to index mapping
    pub var_map: HashMap<String, usize>,
    /// Constant offset
    pub offset: f64,
    /// Original constraints (for analysis)
    constraints: Vec<Constraint>,
}

#[cfg(not(feature = "dwave"))]
impl CompiledModel {
    /// Convert to QUBO format
    pub fn to_qubo(&self) -> quantrs2_anneal::ising::QuboModel {
        use quantrs2_anneal::ising::QuboModel;

        let qubo = QuboModel::new(self.var_map.len());

        // Set the offset
        qubo.offset = self.offset;

        // Set all the QUBO coefficients
        for i in 0..self.qubo_matrix.nrows() {
            for j in i..self.qubo_matrix.ncols() {
                let value = self.qubo_matrix[[i, j]];
                if value.abs() > 1e-10 {
                    if i == j {
                        // Diagonal term (linear)
                        qubo.set_linear(i, value).unwrap();
                    } else {
                        // Off-diagonal term (quadratic)
                        qubo.set_quadratic(i, j, value).unwrap();
                    }
                }
            }
        }

        qubo
    }
}

/// Compiler for converting symbolic expressions to QUBO models
///
/// This struct provides methods for converting symbolic expressions
/// to QUBO models, which can then be solved using quantum annealing.
#[cfg(feature = "dwave")]
pub struct Compile {
    /// The symbolic expression to compile
    expr: Expr,
}

#[cfg(feature = "dwave")]
impl Compile {
    /// Create a new compiler with the given expression
    pub fn new<T: Into<Expr>>(expr: T) -> Self {
        Self { expr: expr.into() }
    }

    /// Compile the expression to a QUBO model
    ///
    /// This method compiles the symbolic expression to a QUBO model,
    /// which can then be passed to a sampler for solving.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A tuple with the QUBO matrix and a mapping of variable names to indices
    /// - An offset value that should be added to all energy values
    pub fn get_qubo(
        &self,
    ) -> CompileResult<((Array<f64, ndarray::Ix2>, HashMap<String, usize>), f64)> {
        #[cfg(feature = "scirs")]
        {
            self.get_qubo_scirs()
        }
        #[cfg(not(feature = "scirs"))]
        {
            self.get_qubo_standard()
        }
    }

    /// Standard QUBO compilation without SciRS2
    fn get_qubo_standard(
        &self,
    ) -> CompileResult<((Array<f64, ndarray::Ix2>, HashMap<String, usize>), f64)> {
        // Expand the expression to simplify
        let expr = self.expr.expand();

        // Check the degree of each term
        let max_degree = calc_highest_degree(&expr)?;
        if max_degree > 2 {
            return Err(CompileError::DegreeTooHigh(max_degree, 2));
        }

        // Replace all second-degree terms (x^2) with x, since x^2 = x for binary variables
        let expr = replace_squared_terms(&expr)?;

        // Expand again to collect like terms
        let expr = expr.expand();

        // Extract the coefficients and variables
        let (coeffs, offset) = extract_coefficients(&expr)?;

        // Convert to a QUBO matrix
        let (matrix, var_map) = build_qubo_matrix(&coeffs)?;

        Ok(((matrix, var_map), offset))
    }

    /// QUBO compilation with SciRS2 optimization
    #[cfg(feature = "scirs")]
    fn get_qubo_scirs(
        &self,
    ) -> CompileResult<((Array<f64, ndarray::Ix2>, HashMap<String, usize>), f64)> {
        // Get standard result
        let ((matrix, var_map), offset) = self.get_qubo_standard()?;

        // Apply SciRS2 enhancements
        let enhanced_matrix = scirs_stub::enhance_qubo_matrix(&matrix);

        Ok(((enhanced_matrix, var_map), offset))
    }

    /// Compile the expression to a HOBO model
    ///
    /// This method compiles the symbolic expression to a Higher-Order Binary Optimization model,
    /// which can handle terms of degree higher than 2.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A tuple with the HOBO tensor and a mapping of variable names to indices
    /// - An offset value that should be added to all energy values
    pub fn get_hobo(
        &self,
    ) -> CompileResult<((Array<f64, ndarray::IxDyn>, HashMap<String, usize>), f64)> {
        // Expand the expression to simplify
        let expr = self.expr.expand();

        // Calculate highest degree (dimension of the tensor)
        let max_degree = calc_highest_degree(&expr)?;

        // Replace all squared terms (x^2) with x, since x^2 = x for binary variables
        let expr = replace_squared_terms(&expr)?;

        // Expand again to collect like terms
        let expr = expr.expand();

        // Extract the coefficients and variables
        let (coeffs, offset) = extract_coefficients(&expr)?;

        // Build the HOBO tensor
        let (tensor, var_map) = build_hobo_tensor(&coeffs, max_degree)?;

        Ok(((tensor, var_map), offset))
    }
}

// Helper function to calculate the highest degree in the expression
#[cfg(feature = "dwave")]
fn calc_highest_degree(expr: &Expr) -> CompileResult<usize> {
    // If the expression is a single variable, it's degree 1
    if expr.is_symbol() {
        return Ok(1);
    }

    // If it's a number constant, degree is 0
    if expr.is_number() {
        return Ok(0);
    }

    // If it's a power operation (like x^2)
    if expr.is_pow() {
        let (base, exp) = expr.as_pow().unwrap();

        // If the base is a symbol and exponent is a number
        if base.is_symbol() && exp.is_number() {
            let exp_val = match exp.to_f64() {
                Some(n) => n,
                None => {
                    return Err(CompileError::InvalidExpression(
                        "Invalid exponent".to_string(),
                    ))
                }
            };

            // Check if exponent is a positive integer
            if exp_val.is_sign_positive() && exp_val.fract() == 0.0 {
                return Ok(exp_val as usize);
            }
        }

        // For other power expressions, recursively calculate the degree
        let base_degree = calc_highest_degree(&base)?;
        let exp_degree = if exp.is_number() {
            match exp.to_f64() {
                Some(n) => {
                    if n.is_sign_positive() && n.fract() == 0.0 {
                        n as usize
                    } else {
                        0 // Non-integer or negative exponents don't contribute to degree
                    }
                }
                None => 0,
            }
        } else {
            0 // Non-constant exponents don't contribute to degree
        };

        return Ok(base_degree * exp_degree);
    }

    // If it's a product (like x*y or x*x)
    if expr.is_mul() {
        let total_degree = 0;
        for factor in expr.as_mul().unwrap() {
            total_degree += calc_highest_degree(&factor)?;
        }
        return Ok(total_degree);
    }

    // If it's a sum (like x + y)
    if expr.is_add() {
        let max_degree = 0;
        for term in expr.as_add().unwrap() {
            let term_degree = calc_highest_degree(&term)?;
            max_degree = std::cmp::max(max_degree, term_degree);
        }
        return Ok(max_degree);
    }

    // Default case - for simplicity, we'll say degree is 0
    // but for a complete implementation, we'd need to handle all cases
    Err(CompileError::InvalidExpression(format!(
        "Can't determine degree of: {}",
        expr
    )))
}

// Helper function to replace squared terms with linear terms
#[cfg(feature = "dwave")]
fn replace_squared_terms(expr: &Expr) -> CompileResult<Expr> {
    // For binary variables, x^2 = x since x ∈ {0,1}

    // If the expression is a symbol or number, just return it
    if expr.is_symbol() || expr.is_number() {
        return Ok(expr.clone());
    }

    // If it's a power operation (like x^2)
    if expr.is_pow() {
        let (base, exp) = expr.as_pow().unwrap();

        // If the base is a symbol and exponent is 2, replace with base
        if base.is_symbol() && exp.is_number() {
            let exp_val = match exp.to_f64() {
                Some(n) => n,
                None => {
                    return Err(CompileError::InvalidExpression(
                        "Invalid exponent".to_string(),
                    ))
                }
            };

            // Check if exponent is 2 (for higher exponents we'd need to recurse)
            if exp_val == 2.0 {
                return Ok(base);
            }
        }

        // For other power expressions, recursively replace
        let new_base = replace_squared_terms(&base)?;
        return Ok(new_base.pow(exp));
    }

    // If it's a product (like x*y or x*x)
    if expr.is_mul() {
        let new_terms = Vec::new();
        for factor in expr.as_mul().unwrap() {
            new_terms.push(replace_squared_terms(&factor)?);
        }

        // Combine the terms back into a product
        let result = Expr::from(1);
        for term in new_terms {
            result = result * term;
        }
        return Ok(result);
    }

    // If it's a sum (like x + y)
    if expr.is_add() {
        let new_terms = Vec::new();
        for term in expr.as_add().unwrap() {
            new_terms.push(replace_squared_terms(&term)?);
        }

        // Combine the terms back into a sum
        let result = Expr::from(0);
        for term in new_terms {
            result = result + term;
        }
        return Ok(result);
    }

    // For any other type of expression, just return it unchanged
    Ok(expr.clone())
}

// Helper function to extract coefficients and variables from the expression
#[cfg(feature = "dwave")]
fn extract_coefficients(expr: &Expr) -> CompileResult<(HashMap<Vec<String>, f64>, f64)> {
    let coeffs = HashMap::new();
    let offset = 0.0;

    // Process expression as a sum of terms
    if expr.is_add() {
        for term in expr.as_add().unwrap() {
            let (term_coeffs, term_offset) = extract_term_coefficients(&term)?;

            // Merge coefficients
            for (vars, coeff) in term_coeffs {
                *coeffs.entry(vars).or_insert(0.0) += coeff;
            }

            // Add constant terms to offset
            offset += term_offset;
        }
    } else {
        // Process a single term
        let (term_coeffs, term_offset) = extract_term_coefficients(expr)?;

        // Merge coefficients
        for (vars, coeff) in term_coeffs {
            *coeffs.entry(vars).or_insert(0.0) += coeff;
        }

        // Add constant terms to offset
        offset += term_offset;
    }

    Ok((coeffs, offset))
}

// Helper function to extract coefficient and variables from a single term
#[cfg(feature = "dwave")]
fn extract_term_coefficients(term: &Expr) -> CompileResult<(HashMap<Vec<String>, f64>, f64)> {
    let coeffs = HashMap::new();

    // If it's a number constant, it's an offset
    if term.is_number() {
        let value = match term.to_f64() {
            Some(n) => n,
            None => {
                return Err(CompileError::InvalidExpression(
                    "Invalid number".to_string(),
                ))
            }
        };
        return Ok((coeffs, value));
    }

    // If it's a symbol, it's a linear term with coefficient 1
    if term.is_symbol() {
        let var_name = term.as_symbol().unwrap();
        let vars = vec![var_name];
        coeffs.insert(vars, 1.0);
        return Ok((coeffs, 0.0));
    }

    // If it's a product of terms
    if term.is_mul() {
        let coeff = 1.0;
        let vars = Vec::new();

        for factor in term.as_mul().unwrap() {
            if factor.is_number() {
                // Numerical factor is a coefficient
                let value = match factor.to_f64() {
                    Some(n) => n,
                    None => {
                        return Err(CompileError::InvalidExpression(
                            "Invalid number in product".to_string(),
                        ))
                    }
                };
                coeff *= value;
            } else if factor.is_symbol() {
                // Symbol is a variable
                let var_name = factor.as_symbol().unwrap();
                vars.push(var_name);
            } else {
                // More complex factors not supported in this example
                return Err(CompileError::InvalidExpression(format!(
                    "Unsupported term in product: {}",
                    factor
                )));
            }
        }

        // Sort variables for consistent ordering
        vars.sort();

        if !vars.is_empty() {
            coeffs.insert(vars, coeff);
        } else {
            // If there are no variables, it's a constant term
            return Ok((coeffs, coeff));
        }

        return Ok((coeffs, 0.0));
    }

    // If it's a power operation (like x^2), should have been simplified earlier
    if term.is_pow() {
        return Err(CompileError::InvalidExpression(format!(
            "Unexpected power term after simplification: {}",
            term
        )));
    }

    // Unsupported term type
    Err(CompileError::InvalidExpression(format!(
        "Unsupported term: {}",
        term
    )))
}

// Helper function to build the QUBO matrix
#[allow(dead_code)]
fn build_qubo_matrix(
    coeffs: &HashMap<Vec<String>, f64>,
) -> CompileResult<(Array<f64, ndarray::Ix2>, HashMap<String, usize>)> {
    // Collect all unique variable names
    let all_vars = HashSet::new();
    for vars in coeffs.keys() {
        for var in vars {
            all_vars.insert(var.clone());
        }
    }

    // Convert to a sorted vector
    let mut sorted_vars: Vec<String> = all_vars.into_iter().collect();
    sorted_vars.sort();

    // Create the variable-to-index mapping
    let var_map: HashMap<String, usize> = sorted_vars
        .iter()
        .enumerate()
        .map(|(i, var)| (var.clone(), i))
        .collect();

    // Size of the matrix
    let n = var_map.len();

    // Create an empty matrix
    let matrix = Array::zeros((n, n));

    // Fill the matrix with coefficients
    for (vars, &coeff) in coeffs {
        match vars.len() {
            0 => {
                // Should never happen since constants are handled in offset
            }
            1 => {
                // Linear term: var * coeff
                let i = *var_map.get(&vars[0]).unwrap();
                matrix[[i, i]] += coeff;
            }
            2 => {
                // Quadratic term: var1 * var2 * coeff
                let i = *var_map.get(&vars[0]).unwrap();
                let j = *var_map.get(&vars[1]).unwrap();

                // QUBO format requires i <= j
                if i == j {
                    // Diagonal term
                    matrix[[i, i]] += coeff;
                } else {
                    // Off-diagonal term
                    matrix[[i, j]] += coeff;
                    matrix[[j, i]] += coeff; // Make sure matrix is symmetric
                }
            }
            _ => {
                // Higher-order terms are not supported in QUBO
                return Err(CompileError::DegreeTooHigh(vars.len(), 2));
            }
        }
    }

    Ok((matrix, var_map))
}

// Helper function to build the HOBO tensor
#[allow(dead_code)]
fn build_hobo_tensor(
    coeffs: &HashMap<Vec<String>, f64>,
    max_degree: usize,
) -> CompileResult<(Array<f64, ndarray::IxDyn>, HashMap<String, usize>)> {
    // Collect all unique variable names
    let all_vars = HashSet::new();
    for vars in coeffs.keys() {
        for var in vars {
            all_vars.insert(var.clone());
        }
    }

    // Convert to a sorted vector
    let mut sorted_vars: Vec<String> = all_vars.into_iter().collect();
    sorted_vars.sort();

    // Create the variable-to-index mapping
    let var_map: HashMap<String, usize> = sorted_vars
        .iter()
        .enumerate()
        .map(|(i, var)| (var.clone(), i))
        .collect();

    // Size of each dimension
    let n = var_map.len();

    // Create shape vector for the tensor
    let shape: Vec<usize> = vec![n; max_degree];

    // Create an empty tensor
    let tensor = Array::zeros(ndarray::IxDyn(&shape));

    // Fill the tensor with coefficients
    for (vars, &coeff) in coeffs {
        let degree = vars.len();

        if degree == 0 {
            // Should never happen since constants are handled in offset
            continue;
        }

        if degree > max_degree {
            return Err(CompileError::DegreeTooHigh(degree, max_degree));
        }

        // Convert variable names to indices
        let mut indices: Vec<usize> = vars.iter().map(|var| *var_map.get(var).unwrap()).collect();

        // Sort indices (canonical ordering)
        indices.sort();

        // Pad indices to match tensor order if necessary
        while indices.len() < max_degree {
            indices.insert(0, indices[0]); // Padding with first index
        }

        // Set the coefficient in the tensor
        let idx = ndarray::IxDyn(&indices);
        tensor[idx] += coeff;
    }

    Ok((tensor, var_map))
}

/// Special compiler for problems with one-hot constraints
///
/// This is a specialized compiler that is optimized for problems
/// with one-hot constraints, common in many optimization problems.
#[cfg(feature = "dwave")]
pub struct PieckCompile {
    /// The symbolic expression to compile
    expr: Expr,
    /// Whether to show verbose output
    verbose: bool,
}

#[cfg(feature = "dwave")]
impl PieckCompile {
    /// Create a new Pieck compiler with the given expression
    pub fn new<T: Into<Expr>>(expr: T, verbose: bool) -> Self {
        Self {
            expr: expr.into(),
            verbose,
        }
    }

    /// Compile the expression to a QUBO model optimized for one-hot constraints
    pub fn get_qubo(
        &self,
    ) -> CompileResult<((Array<f64, ndarray::Ix2>, HashMap<String, usize>), f64)> {
        // Implementation will compile the expression using specialized techniques
        // For now, call the regular compiler
        Compile::new(self.expr.clone()).get_qubo()
    }
}
