//! Constraint handling for quantum annealing
//!
//! This module provides comprehensive constraint management including
//! automatic penalty term generation and constraint analysis.

// Optimization penalty types
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Constraint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Equality constraint: expr = target
    Equality { target: f64 },
    /// Inequality constraint: expr <= bound
    LessThanOrEqual { bound: f64 },
    /// Inequality constraint: expr >= bound
    GreaterThanOrEqual { bound: f64 },
    /// Range constraint: lower <= expr <= upper
    Range { lower: f64, upper: f64 },
    /// One-hot constraint: exactly one variable true
    OneHot,
    /// Cardinality constraint: exactly k variables true
    Cardinality { k: usize },
    /// Integer encoding constraint
    IntegerEncoding { min: i32, max: i32 },
}

/// Constraint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub expression: Expression,
    pub variables: Vec<String>,
    pub penalty_weight: Option<f64>,
    pub slack_variables: Vec<String>,
}

/// Constraint handler for automatic penalty generation
pub struct ConstraintHandler {
    constraints: Vec<Constraint>,
    slack_variable_counter: usize,
    encoding_cache: HashMap<String, EncodingInfo>,
}

/// Encoding information for integer variables
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncodingInfo {
    pub variable_name: String,
    pub bit_variables: Vec<String>,
    pub min_value: i32,
    pub max_value: i32,
    pub encoding_type: EncodingType,
}

/// Integer encoding types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EncodingType {
    Binary,
    Unary,
    OneHot,
    Gray,
}

impl ConstraintHandler {
    /// Create new constraint handler
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            slack_variable_counter: 0,
            encoding_cache: HashMap::new(),
        }
    }

    /// Add constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add equality constraint
    pub fn add_equality(
        &mut self,
        name: String,
        expression: Expression,
        target: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let variables = expression.get_variables();

        self.add_constraint(Constraint {
            name,
            constraint_type: ConstraintType::Equality { target },
            expression,
            variables,
            penalty_weight: None,
            slack_variables: Vec::new(),
        });

        Ok(())
    }

    /// Add inequality constraint
    pub fn add_inequality(
        &mut self,
        name: String,
        expression: Expression,
        bound: f64,
        less_than: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let variables = expression.get_variables();
        let mut constraint = Constraint {
            name: name.clone(),
            constraint_type: if less_than {
                ConstraintType::LessThanOrEqual { bound }
            } else {
                ConstraintType::GreaterThanOrEqual { bound }
            },
            expression,
            variables,
            penalty_weight: None,
            slack_variables: Vec::new(),
        };

        // Add slack variables for inequality constraints
        if less_than {
            // expr + slack = bound, slack >= 0
            let slack_var = self.create_slack_variable(&name);
            constraint.slack_variables.push(slack_var);
        } else {
            // expr - slack = bound, slack >= 0
            let slack_var = self.create_slack_variable(&name);
            constraint.slack_variables.push(slack_var);
        }

        self.add_constraint(constraint);
        Ok(())
    }

    /// Add one-hot constraint
    pub fn add_one_hot(
        &mut self,
        name: String,
        variables: Vec<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create expression: (sum_i x_i - 1)^2
        let mut expr = Expression::zero();
        for var in &variables {
            expr = expr + Variable::new(var.clone()).into();
        }
        expr = expr - 1.0.into();

        self.add_constraint(Constraint {
            name,
            constraint_type: ConstraintType::OneHot,
            expression: expr,
            variables,
            penalty_weight: None,
            slack_variables: Vec::new(),
        });

        Ok(())
    }

    /// Add cardinality constraint
    pub fn add_cardinality(
        &mut self,
        name: String,
        variables: Vec<String>,
        k: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create expression: (sum_i x_i - k)^2
        let mut expr = Expression::zero();
        for var in &variables {
            expr = expr + Variable::new(var.clone()).into();
        }
        expr = expr - (k as f64).into();

        self.add_constraint(Constraint {
            name,
            constraint_type: ConstraintType::Cardinality { k },
            expression: expr,
            variables,
            penalty_weight: None,
            slack_variables: Vec::new(),
        });

        Ok(())
    }

    /// Add integer encoding constraint
    pub fn add_integer_encoding(
        &mut self,
        name: String,
        base_name: String,
        min: i32,
        max: i32,
        encoding_type: EncodingType,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let num_bits = ((max - min + 1) as f64).log2().ceil() as usize;
        let mut bit_variables = Vec::new();

        // Create bit variables
        for i in 0..num_bits {
            bit_variables.push(format!("{}_{}", base_name, i));
        }

        // Store encoding info
        self.encoding_cache.insert(
            base_name.clone(),
            EncodingInfo {
                variable_name: base_name,
                bit_variables: bit_variables.clone(),
                min_value: min,
                max_value: max,
                encoding_type,
            },
        );

        // Add encoding-specific constraints
        match encoding_type {
            EncodingType::Binary => {
                // No additional constraints for binary encoding
            }
            EncodingType::Unary => {
                // Unary: if x_i = 1, then x_{i-1} = 1
                for i in 1..bit_variables.len() {
                    let expr: Expression = Variable::new(bit_variables[i].clone()).into();
                    let prev_expr: Expression = Variable::new(bit_variables[i - 1].clone()).into();
                    let constraint_expr = expr - prev_expr;

                    self.add_inequality(
                        format!("{}_unary_{}", name, i),
                        constraint_expr,
                        0.0,
                        true,
                    )?;
                }
            }
            EncodingType::OneHot => {
                // Exactly one bit active
                self.add_one_hot(format!("{}_onehot", name), bit_variables.clone())?;
            }
            EncodingType::Gray => {
                // Gray code constraints are implicit in the mapping
            }
        }

        self.add_constraint(Constraint {
            name,
            constraint_type: ConstraintType::IntegerEncoding { min, max },
            expression: Expression::zero(), // Placeholder
            variables: bit_variables.clone(),
            penalty_weight: None,
            slack_variables: Vec::new(),
        });

        Ok(bit_variables)
    }

    /// Generate penalty terms for all constraints
    pub fn generate_penalty_terms(
        &self,
        penalty_weights: &HashMap<String, f64>,
    ) -> Result<Expression, Box<dyn std::error::Error>> {
        let mut total_penalty = Expression::zero();

        for constraint in &self.constraints {
            let weight = penalty_weights
                .get(&constraint.name)
                .or(constraint.penalty_weight.as_ref())
                .copied()
                .unwrap_or(1.0);

            let penalty_expr = match &constraint.constraint_type {
                ConstraintType::Equality { target } => {
                    // (expr - target)^2
                    let diff = constraint.expression.clone() - (*target).into();
                    diff.clone() * diff
                }
                ConstraintType::LessThanOrEqual { bound } => {
                    // expr + slack = bound => (expr + slack - bound)^2
                    if let Some(slack_var) = constraint.slack_variables.first() {
                        let expr_with_slack =
                            constraint.expression.clone() + Variable::new(slack_var.clone()).into();
                        let diff = expr_with_slack - (*bound).into();
                        diff.clone() * diff
                    } else {
                        // Penalty for violation: max(0, expr - bound)^2
                        self.generate_inequality_penalty(&constraint.expression, *bound, true)?
                    }
                }
                ConstraintType::GreaterThanOrEqual { bound } => {
                    // expr - slack = bound => (expr - slack - bound)^2
                    if let Some(slack_var) = constraint.slack_variables.first() {
                        let expr_with_slack =
                            constraint.expression.clone() - Variable::new(slack_var.clone()).into();
                        let diff = expr_with_slack - (*bound).into();
                        diff.clone() * diff
                    } else {
                        // Penalty for violation: max(0, bound - expr)^2
                        self.generate_inequality_penalty(&constraint.expression, *bound, false)?
                    }
                }
                ConstraintType::Range { lower, upper } => {
                    // Combine two inequality penalties
                    let lower_penalty =
                        self.generate_inequality_penalty(&constraint.expression, *lower, false)?;
                    let upper_penalty =
                        self.generate_inequality_penalty(&constraint.expression, *upper, true)?;
                    lower_penalty + upper_penalty
                }
                ConstraintType::OneHot => {
                    // (sum_i x_i - 1)^2
                    let expr = constraint.expression.clone();
                    expr.clone() * expr
                }
                ConstraintType::Cardinality { k } => {
                    // (sum_i x_i - k)^2
                    let expr = constraint.expression.clone();
                    expr.clone() * expr
                }
                ConstraintType::IntegerEncoding { .. } => {
                    // Encoding constraints are handled separately
                    Expression::zero()
                }
            };

            total_penalty = total_penalty + weight * penalty_expr;
        }

        Ok(total_penalty)
    }

    /// Generate inequality penalty using auxiliary binary expansion
    fn generate_inequality_penalty(
        &self,
        expression: &Expression,
        bound: f64,
        less_than: bool,
    ) -> Result<Expression, Box<dyn std::error::Error>> {
        // For now, return a quadratic penalty
        // In a full implementation, this would use binary expansion
        // to exactly encode the inequality

        if less_than {
            // max(0, expr - bound)^2
            Ok(Expression::zero()) // Placeholder
        } else {
            // max(0, bound - expr)^2
            Ok(Expression::zero()) // Placeholder
        }
    }

    /// Create slack variable
    fn create_slack_variable(&mut self, constraint_name: &str) -> String {
        let var_name = format!("_slack_{}_{}", constraint_name, self.slack_variable_counter);
        self.slack_variable_counter += 1;
        var_name
    }

    /// Get all variables including slack
    pub fn get_all_variables(&self) -> Vec<String> {
        let mut variables = Vec::new();

        for constraint in &self.constraints {
            variables.extend(constraint.variables.clone());
            variables.extend(constraint.slack_variables.clone());
        }

        // Include integer encoding bit variables
        for encoding in self.encoding_cache.values() {
            variables.extend(encoding.bit_variables.clone());
        }

        // Remove duplicates
        variables.sort();
        variables.dedup();

        variables
    }

    /// Decode integer value from bit assignment
    pub fn decode_integer(
        &self,
        variable_name: &str,
        assignment: &HashMap<String, bool>,
    ) -> Option<i32> {
        let encoding = self.encoding_cache.get(variable_name)?;

        match encoding.encoding_type {
            EncodingType::Binary => {
                let mut value = 0;
                for (i, bit_var) in encoding.bit_variables.iter().enumerate() {
                    if *assignment.get(bit_var).unwrap_or(&false) {
                        value += 1 << i;
                    }
                }
                Some(encoding.min_value + value)
            }
            EncodingType::Unary => {
                let mut count = 0;
                for bit_var in &encoding.bit_variables {
                    if *assignment.get(bit_var).unwrap_or(&false) {
                        count += 1;
                    } else {
                        break;
                    }
                }
                Some(encoding.min_value + count as i32)
            }
            EncodingType::OneHot => {
                for (i, bit_var) in encoding.bit_variables.iter().enumerate() {
                    if *assignment.get(bit_var).unwrap_or(&false) {
                        return Some(encoding.min_value + i as i32);
                    }
                }
                None
            }
            EncodingType::Gray => {
                // Convert Gray code to binary
                let mut gray_value = 0;
                for (i, bit_var) in encoding.bit_variables.iter().enumerate() {
                    if *assignment.get(bit_var).unwrap_or(&false) {
                        gray_value |= 1 << i;
                    }
                }

                // Gray to binary conversion
                let mut binary_value = gray_value;
                binary_value ^= binary_value >> 16;
                binary_value ^= binary_value >> 8;
                binary_value ^= binary_value >> 4;
                binary_value ^= binary_value >> 2;
                binary_value ^= binary_value >> 1;

                Some(encoding.min_value + binary_value)
            }
        }
    }

    /// Analyze constraint structure
    pub fn analyze_constraints(&self) -> ConstraintAnalysis {
        let total_constraints = self.constraints.len();
        let total_variables = self.get_all_variables().len();

        let mut type_counts = HashMap::new();
        let mut avg_variables_per_constraint = 0.0;
        let mut max_variables_in_constraint = 0;

        for constraint in &self.constraints {
            let type_name = match constraint.constraint_type {
                ConstraintType::Equality { .. } => "equality",
                ConstraintType::LessThanOrEqual { .. } => "less_than",
                ConstraintType::GreaterThanOrEqual { .. } => "greater_than",
                ConstraintType::Range { .. } => "range",
                ConstraintType::OneHot => "one_hot",
                ConstraintType::Cardinality { .. } => "cardinality",
                ConstraintType::IntegerEncoding { .. } => "integer",
            };

            *type_counts.entry(type_name.to_string()).or_insert(0) += 1;

            let var_count = constraint.variables.len();
            avg_variables_per_constraint += var_count as f64;
            max_variables_in_constraint = max_variables_in_constraint.max(var_count);
        }

        if total_constraints > 0 {
            avg_variables_per_constraint /= total_constraints as f64;
        }

        ConstraintAnalysis {
            total_constraints,
            total_variables,
            slack_variables: self.slack_variable_counter,
            constraint_types: type_counts,
            avg_variables_per_constraint,
            max_variables_in_constraint,
            encoding_info: self.encoding_cache.len(),
        }
    }
}

/// Constraint analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintAnalysis {
    pub total_constraints: usize,
    pub total_variables: usize,
    pub slack_variables: usize,
    pub constraint_types: HashMap<String, usize>,
    pub avg_variables_per_constraint: f64,
    pub max_variables_in_constraint: usize,
    pub encoding_info: usize,
}

// Helper trait implementations for Expression
trait ExpressionExt {
    fn zero() -> Self;
    fn get_variables(&self) -> Vec<String>;
}

impl ExpressionExt for Expression {
    fn zero() -> Self {
        // Placeholder implementation
        Expression::Constant(0.0)
    }

    fn get_variables(&self) -> Vec<String> {
        // Placeholder implementation
        Vec::new()
    }
}

/// Variable placeholder
#[derive(Debug, Clone)]
pub struct Variable {
    name: String,
}

impl Variable {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

/// Expression type placeholder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    Constant(f64),
    Variable(String),
    Add(Box<Expression>, Box<Expression>),
    Multiply(Box<Expression>, Box<Expression>),
}

impl From<f64> for Expression {
    fn from(value: f64) -> Self {
        Expression::Constant(value)
    }
}

impl From<Variable> for Expression {
    fn from(var: Variable) -> Self {
        Expression::Variable(var.name)
    }
}

impl std::ops::Add for Expression {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Expression::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Sub for Expression {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Expression::Add(
            Box::new(self),
            Box::new(Expression::Multiply(
                Box::new(Expression::Constant(-1.0)),
                Box::new(rhs),
            )),
        )
    }
}

impl std::ops::Mul for Expression {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Expression::Multiply(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul<Expression> for f64 {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Self::Output {
        Expression::Multiply(Box::new(Expression::Constant(self)), Box::new(rhs))
    }
}
