//! Symbolic computation module for QuantRS2
//!
//! This module provides symbolic computation capabilities using SymEngine,
//! enabling symbolic parameter manipulation, calculus operations, and
//! advanced mathematical analysis for quantum circuits and algorithms.

#[cfg(feature = "symbolic")]
pub use quantrs2_symengine::{Expression as SymEngine, SymEngineError, SymEngineResult};

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::num_traits::{One, Zero}; // SciRS2 POLICY compliant
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::fmt;

/// A symbolic expression that can represent constants, variables, or complex expressions
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpression {
    /// Constant floating-point value
    Constant(f64),

    /// Complex constant value
    ComplexConstant(Complex64),

    /// Variable with a name
    Variable(String),

    /// SymEngine expression (only available with "symbolic" feature)
    #[cfg(feature = "symbolic")]
    SymEngine(SymEngine),

    /// Simple arithmetic expression for when SymEngine is not available
    #[cfg(not(feature = "symbolic"))]
    Simple(SimpleExpression),
}

/// Simple expression representation for when SymEngine is not available
#[cfg(not(feature = "symbolic"))]
#[derive(Debug, Clone, PartialEq)]
pub enum SimpleExpression {
    Add(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Sub(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Mul(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Div(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Pow(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Sin(Box<SymbolicExpression>),
    Cos(Box<SymbolicExpression>),
    Exp(Box<SymbolicExpression>),
    Log(Box<SymbolicExpression>),
}

impl SymbolicExpression {
    /// Create a constant expression
    pub const fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    pub const fn zero() -> Self {
        Self::Constant(0.0)
    }

    /// Create a complex constant expression
    pub const fn complex_constant(value: Complex64) -> Self {
        Self::ComplexConstant(value)
    }

    /// Create a variable expression
    pub fn variable(name: &str) -> Self {
        Self::Variable(name.to_string())
    }

    /// Create a SymEngine expression (requires "symbolic" feature)
    #[cfg(feature = "symbolic")]
    pub const fn from_symengine(expr: SymEngine) -> Self {
        Self::SymEngine(expr)
    }

    /// Parse an expression from a string
    pub fn parse(expr: &str) -> QuantRS2Result<Self> {
        #[cfg(feature = "symbolic")]
        {
            match SymEngine::try_new(expr) {
                Ok(sym_expr) => Ok(Self::SymEngine(sym_expr)),
                Err(_) => {
                    // Fallback to simple parsing
                    Self::parse_simple(expr)
                }
            }
        }

        #[cfg(not(feature = "symbolic"))]
        {
            Self::parse_simple(expr)
        }
    }

    /// Simple expression parsing (fallback)
    fn parse_simple(expr: &str) -> QuantRS2Result<Self> {
        let trimmed = expr.trim();

        // Try to parse as a number
        if let Ok(value) = trimmed.parse::<f64>() {
            return Ok(Self::Constant(value));
        }

        // Otherwise treat as a variable
        Ok(Self::Variable(trimmed.to_string()))
    }

    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, variables: &HashMap<String, f64>) -> QuantRS2Result<f64> {
        match self {
            Self::Constant(value) => Ok(*value),
            Self::ComplexConstant(value) => {
                if value.im.abs() < 1e-12 {
                    Ok(value.re)
                } else {
                    Err(QuantRS2Error::InvalidInput(
                        "Cannot evaluate complex expression to real number".to_string(),
                    ))
                }
            }
            Self::Variable(name) => variables
                .get(name)
                .copied()
                .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Variable '{name}' not found"))),

            #[cfg(feature = "symbolic")]
            Self::SymEngine(expr) => {
                // For SymEngine evaluation, we would need to substitute variables
                // This is a simplified implementation
                if let Ok(value) = expr.to_string().parse::<f64>() {
                    Ok(value)
                } else {
                    Err(QuantRS2Error::UnsupportedOperation(
                        "SymEngine evaluation not yet implemented".to_string(),
                    ))
                }
            }

            #[cfg(not(feature = "symbolic"))]
            Self::Simple(simple_expr) => Self::evaluate_simple(simple_expr, variables),
        }
    }

    /// Evaluate complex expression with given variable values
    pub fn evaluate_complex(
        &self,
        variables: &HashMap<String, Complex64>,
    ) -> QuantRS2Result<Complex64> {
        match self {
            Self::Constant(value) => Ok(Complex64::new(*value, 0.0)),
            Self::ComplexConstant(value) => Ok(*value),
            Self::Variable(name) => variables
                .get(name)
                .copied()
                .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Variable '{name}' not found"))),

            #[cfg(feature = "symbolic")]
            Self::SymEngine(_) => Err(QuantRS2Error::UnsupportedOperation(
                "Complex SymEngine evaluation not yet implemented".to_string(),
            )),

            #[cfg(not(feature = "symbolic"))]
            Self::Simple(simple_expr) => Self::evaluate_simple_complex(simple_expr, variables),
        }
    }

    #[cfg(not(feature = "symbolic"))]
    fn evaluate_simple(
        expr: &SimpleExpression,
        variables: &HashMap<String, f64>,
    ) -> QuantRS2Result<f64> {
        match expr {
            SimpleExpression::Add(a, b) => Ok(a.evaluate(variables)? + b.evaluate(variables)?),
            SimpleExpression::Sub(a, b) => Ok(a.evaluate(variables)? - b.evaluate(variables)?),
            SimpleExpression::Mul(a, b) => Ok(a.evaluate(variables)? * b.evaluate(variables)?),
            SimpleExpression::Div(a, b) => {
                let b_val = b.evaluate(variables)?;
                if b_val.abs() < 1e-12 {
                    Err(QuantRS2Error::DivisionByZero)
                } else {
                    Ok(a.evaluate(variables)? / b_val)
                }
            }
            SimpleExpression::Pow(a, b) => Ok(a.evaluate(variables)?.powf(b.evaluate(variables)?)),
            SimpleExpression::Sin(a) => Ok(a.evaluate(variables)?.sin()),
            SimpleExpression::Cos(a) => Ok(a.evaluate(variables)?.cos()),
            SimpleExpression::Exp(a) => Ok(a.evaluate(variables)?.exp()),
            SimpleExpression::Log(a) => {
                let a_val = a.evaluate(variables)?;
                if a_val <= 0.0 {
                    Err(QuantRS2Error::InvalidInput(
                        "Logarithm of non-positive number".to_string(),
                    ))
                } else {
                    Ok(a_val.ln())
                }
            }
        }
    }

    #[cfg(not(feature = "symbolic"))]
    fn evaluate_simple_complex(
        expr: &SimpleExpression,
        variables: &HashMap<String, Complex64>,
    ) -> QuantRS2Result<Complex64> {
        // Convert variables to real for this simple implementation
        let real_vars: HashMap<String, f64> = variables
            .iter()
            .filter_map(|(k, v)| {
                if v.im.abs() < 1e-12 {
                    Some((k.clone(), v.re))
                } else {
                    None
                }
            })
            .collect();

        let real_result = Self::evaluate_simple(expr, &real_vars)?;
        Ok(Complex64::new(real_result, 0.0))
    }

    /// Get all variable names in the expression
    pub fn variables(&self) -> Vec<String> {
        match self {
            Self::Constant(_) | Self::ComplexConstant(_) => Vec::new(),
            Self::Variable(name) => vec![name.clone()],

            #[cfg(feature = "symbolic")]
            Self::SymEngine(_) => {
                // Would need to implement variable extraction from SymEngine
                Vec::new()
            }

            #[cfg(not(feature = "symbolic"))]
            Self::Simple(simple_expr) => Self::variables_simple(simple_expr),
        }
    }

    #[cfg(not(feature = "symbolic"))]
    fn variables_simple(expr: &SimpleExpression) -> Vec<String> {
        match expr {
            SimpleExpression::Add(a, b)
            | SimpleExpression::Sub(a, b)
            | SimpleExpression::Mul(a, b)
            | SimpleExpression::Div(a, b)
            | SimpleExpression::Pow(a, b) => {
                let mut vars = a.variables();
                vars.extend(b.variables());
                vars.sort();
                vars.dedup();
                vars
            }
            SimpleExpression::Sin(a)
            | SimpleExpression::Cos(a)
            | SimpleExpression::Exp(a)
            | SimpleExpression::Log(a) => a.variables(),
        }
    }

    /// Check if the expression is constant (has no variables)
    pub const fn is_constant(&self) -> bool {
        match self {
            Self::Constant(_) | Self::ComplexConstant(_) => true,
            Self::Variable(_) => false,

            #[cfg(feature = "symbolic")]
            Self::SymEngine(_) => {
                // Would need to check if SymEngine expression has variables
                false
            }

            #[cfg(not(feature = "symbolic"))]
            Self::Simple(_) => false,
        }
    }

    /// Substitute variables with expressions
    pub fn substitute(&self, substitutions: &HashMap<String, Self>) -> QuantRS2Result<Self> {
        match self {
            Self::Constant(_) | Self::ComplexConstant(_) => Ok(self.clone()),
            Self::Variable(name) => Ok(substitutions
                .get(name)
                .cloned()
                .unwrap_or_else(|| self.clone())),

            #[cfg(feature = "symbolic")]
            Self::SymEngine(_) => {
                // Would implement SymEngine substitution
                Err(QuantRS2Error::UnsupportedOperation(
                    "SymEngine substitution not yet implemented".to_string(),
                ))
            }

            #[cfg(not(feature = "symbolic"))]
            Self::Simple(_) => {
                // Would implement simple expression substitution
                Err(QuantRS2Error::UnsupportedOperation(
                    "Simple expression substitution not yet implemented".to_string(),
                ))
            }
        }
    }
}

// Arithmetic operations for SymbolicExpression
impl std::ops::Add for SymbolicExpression {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "symbolic")]
        {
            match (self, rhs) {
                // Optimize constant addition
                (Self::Constant(a), Self::Constant(b)) => Self::Constant(a + b),
                (Self::SymEngine(a), Self::SymEngine(b)) => Self::SymEngine(a + b),
                (a, b) => {
                    // Convert to SymEngine if possible
                    let a_sym = match a {
                        Self::Constant(val) => SymEngine::from(val),
                        Self::Variable(name) => SymEngine::symbol(name),
                        Self::SymEngine(expr) => expr,
                        _ => return Self::Constant(0.0), // Fallback
                    };
                    let b_sym = match b {
                        Self::Constant(val) => SymEngine::from(val),
                        Self::Variable(name) => SymEngine::symbol(name),
                        Self::SymEngine(expr) => expr,
                        _ => return Self::Constant(0.0), // Fallback
                    };
                    Self::SymEngine(a_sym + b_sym)
                }
            }
        }

        #[cfg(not(feature = "symbolic"))]
        {
            match (self, rhs) {
                (Self::Constant(a), Self::Constant(b)) => Self::Constant(a + b),
                (a, b) => Self::Simple(SimpleExpression::Add(Box::new(a), Box::new(b))),
            }
        }
    }
}

impl std::ops::Sub for SymbolicExpression {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "symbolic")]
        {
            match (self, rhs) {
                // Optimize constant subtraction
                (Self::Constant(a), Self::Constant(b)) => Self::Constant(a - b),
                (Self::SymEngine(a), Self::SymEngine(b)) => Self::SymEngine(a - b),
                (a, b) => {
                    let a_sym = match a {
                        Self::Constant(val) => SymEngine::from(val),
                        Self::Variable(name) => SymEngine::symbol(name),
                        Self::SymEngine(expr) => expr,
                        _ => return Self::Constant(0.0),
                    };
                    let b_sym = match b {
                        Self::Constant(val) => SymEngine::from(val),
                        Self::Variable(name) => SymEngine::symbol(name),
                        Self::SymEngine(expr) => expr,
                        _ => return Self::Constant(0.0),
                    };
                    Self::SymEngine(a_sym - b_sym)
                }
            }
        }

        #[cfg(not(feature = "symbolic"))]
        {
            match (self, rhs) {
                (Self::Constant(a), Self::Constant(b)) => Self::Constant(a - b),
                (a, b) => Self::Simple(SimpleExpression::Sub(Box::new(a), Box::new(b))),
            }
        }
    }
}

impl std::ops::Mul for SymbolicExpression {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "symbolic")]
        {
            match (self, rhs) {
                // Optimize constant multiplication
                (Self::Constant(a), Self::Constant(b)) => Self::Constant(a * b),
                (Self::SymEngine(a), Self::SymEngine(b)) => Self::SymEngine(a * b),
                (a, b) => {
                    let a_sym = match a {
                        Self::Constant(val) => SymEngine::from(val),
                        Self::Variable(name) => SymEngine::symbol(name),
                        Self::SymEngine(expr) => expr,
                        _ => return Self::Constant(0.0),
                    };
                    let b_sym = match b {
                        Self::Constant(val) => SymEngine::from(val),
                        Self::Variable(name) => SymEngine::symbol(name),
                        Self::SymEngine(expr) => expr,
                        _ => return Self::Constant(0.0),
                    };
                    Self::SymEngine(a_sym * b_sym)
                }
            }
        }

        #[cfg(not(feature = "symbolic"))]
        {
            match (self, rhs) {
                (Self::Constant(a), Self::Constant(b)) => Self::Constant(a * b),
                (a, b) => Self::Simple(SimpleExpression::Mul(Box::new(a), Box::new(b))),
            }
        }
    }
}

impl std::ops::Div for SymbolicExpression {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "symbolic")]
        {
            match (self, rhs) {
                // Optimize constant division
                (Self::Constant(a), Self::Constant(b)) => {
                    if b.abs() < 1e-12 {
                        Self::Constant(f64::INFINITY)
                    } else {
                        Self::Constant(a / b)
                    }
                }
                (Self::SymEngine(a), Self::SymEngine(b)) => Self::SymEngine(a / b),
                (a, b) => {
                    let a_sym = match a {
                        Self::Constant(val) => SymEngine::from(val),
                        Self::Variable(name) => SymEngine::symbol(name),
                        Self::SymEngine(expr) => expr,
                        _ => return Self::Constant(0.0),
                    };
                    let b_sym = match b {
                        Self::Constant(val) => SymEngine::from(val),
                        Self::Variable(name) => SymEngine::symbol(name),
                        Self::SymEngine(expr) => expr,
                        _ => return Self::Constant(1.0),
                    };
                    Self::SymEngine(a_sym / b_sym)
                }
            }
        }

        #[cfg(not(feature = "symbolic"))]
        {
            match (self, rhs) {
                (Self::Constant(a), Self::Constant(b)) => {
                    if b.abs() < 1e-12 {
                        Self::Constant(f64::INFINITY)
                    } else {
                        Self::Constant(a / b)
                    }
                }
                (a, b) => Self::Simple(SimpleExpression::Div(Box::new(a), Box::new(b))),
            }
        }
    }
}

impl fmt::Display for SymbolicExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(value) => write!(f, "{value}"),
            Self::ComplexConstant(value) => {
                if value.im == 0.0 {
                    write!(f, "{}", value.re)
                } else if value.re == 0.0 {
                    write!(f, "{}*I", value.im)
                } else {
                    write!(f, "{} + {}*I", value.re, value.im)
                }
            }
            Self::Variable(name) => write!(f, "{name}"),

            #[cfg(feature = "symbolic")]
            Self::SymEngine(expr) => write!(f, "{expr}"),

            #[cfg(not(feature = "symbolic"))]
            Self::Simple(expr) => Self::display_simple(expr, f),
        }
    }
}

#[cfg(not(feature = "symbolic"))]
impl SymbolicExpression {
    fn display_simple(expr: &SimpleExpression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match expr {
            SimpleExpression::Add(a, b) => write!(f, "({a} + {b})"),
            SimpleExpression::Sub(a, b) => write!(f, "({a} - {b})"),
            SimpleExpression::Mul(a, b) => write!(f, "({a} * {b})"),
            SimpleExpression::Div(a, b) => write!(f, "({a} / {b})"),
            SimpleExpression::Pow(a, b) => write!(f, "({a} ^ {b})"),
            SimpleExpression::Sin(a) => write!(f, "sin({a})"),
            SimpleExpression::Cos(a) => write!(f, "cos({a})"),
            SimpleExpression::Exp(a) => write!(f, "exp({a})"),
            SimpleExpression::Log(a) => write!(f, "log({a})"),
        }
    }
}

impl From<f64> for SymbolicExpression {
    fn from(value: f64) -> Self {
        Self::Constant(value)
    }
}

impl From<Complex64> for SymbolicExpression {
    fn from(value: Complex64) -> Self {
        if value.im == 0.0 {
            Self::Constant(value.re)
        } else {
            Self::ComplexConstant(value)
        }
    }
}

impl From<&str> for SymbolicExpression {
    fn from(name: &str) -> Self {
        Self::Variable(name.to_string())
    }
}

impl Zero for SymbolicExpression {
    fn zero() -> Self {
        Self::Constant(0.0)
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Constant(val) => *val == 0.0,
            Self::ComplexConstant(val) => val.is_zero(),
            _ => false,
        }
    }
}

impl One for SymbolicExpression {
    fn one() -> Self {
        Self::Constant(1.0)
    }

    fn is_one(&self) -> bool {
        match self {
            Self::Constant(val) => *val == 1.0,
            Self::ComplexConstant(val) => val.is_one(),
            _ => false,
        }
    }
}

/// Symbolic calculus operations
#[cfg(feature = "symbolic")]
pub mod calculus {
    use super::*;
    use quantrs2_symengine::ops::calculus;

    /// Differentiate an expression with respect to a variable
    pub fn diff(expr: &SymbolicExpression, var: &str) -> QuantRS2Result<SymbolicExpression> {
        match expr {
            SymbolicExpression::SymEngine(sym_expr) => {
                let var_expr = SymEngine::symbol(var);
                match calculus::diff(sym_expr, &var_expr) {
                    Ok(result) => Ok(SymbolicExpression::SymEngine(result)),
                    Err(e) => Err(QuantRS2Error::ComputationError(format!(
                        "Differentiation failed: {e}"
                    ))),
                }
            }
            _ => Err(QuantRS2Error::UnsupportedOperation(
                "Differentiation requires SymEngine expressions".to_string(),
            )),
        }
    }

    /// Integrate an expression with respect to a variable
    pub fn integrate(expr: &SymbolicExpression, var: &str) -> QuantRS2Result<SymbolicExpression> {
        match expr {
            SymbolicExpression::SymEngine(sym_expr) => {
                let var_expr = SymEngine::symbol(var);
                match calculus::integrate(sym_expr, &var_expr) {
                    Ok(result) => Ok(SymbolicExpression::SymEngine(result)),
                    Err(e) => Err(QuantRS2Error::ComputationError(format!(
                        "Integration failed: {e}"
                    ))),
                }
            }
            _ => Err(QuantRS2Error::UnsupportedOperation(
                "Integration requires SymEngine expressions".to_string(),
            )),
        }
    }

    /// Compute the limit of an expression
    pub fn limit(
        expr: &SymbolicExpression,
        var: &str,
        value: f64,
    ) -> QuantRS2Result<SymbolicExpression> {
        match expr {
            SymbolicExpression::SymEngine(sym_expr) => {
                let var_expr = SymEngine::symbol(var);
                let value_expr = SymEngine::from(value);
                match calculus::limit(sym_expr, &var_expr, &value_expr) {
                    Ok(result) => Ok(SymbolicExpression::SymEngine(result)),
                    Err(e) => Err(QuantRS2Error::ComputationError(format!(
                        "Limit computation failed: {e}"
                    ))),
                }
            }
            _ => Err(QuantRS2Error::UnsupportedOperation(
                "Limit computation requires SymEngine expressions".to_string(),
            )),
        }
    }

    /// Expand an expression
    pub fn expand(expr: &SymbolicExpression) -> QuantRS2Result<SymbolicExpression> {
        match expr {
            SymbolicExpression::SymEngine(sym_expr) => {
                Ok(SymbolicExpression::SymEngine(sym_expr.expand()))
            }
            _ => Ok(expr.clone()), // No expansion needed for simple expressions
        }
    }

    /// Simplify an expression
    pub fn simplify(expr: &SymbolicExpression) -> QuantRS2Result<SymbolicExpression> {
        match expr {
            SymbolicExpression::SymEngine(sym_expr) => {
                // SymEngine's simplify would go here
                Ok(SymbolicExpression::SymEngine(sym_expr.expand()))
            }
            _ => Ok(expr.clone()),
        }
    }
}

/// Symbolic matrix operations for quantum gates
pub mod matrix {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// A symbolic matrix for representing quantum gates
    #[derive(Debug, Clone)]
    pub struct SymbolicMatrix {
        pub rows: usize,
        pub cols: usize,
        pub elements: Vec<Vec<SymbolicExpression>>,
    }

    impl SymbolicMatrix {
        /// Create a new symbolic matrix
        pub fn new(rows: usize, cols: usize) -> Self {
            let elements = vec![vec![SymbolicExpression::zero(); cols]; rows];
            Self {
                rows,
                cols,
                elements,
            }
        }

        /// Create an identity matrix
        pub fn identity(size: usize) -> Self {
            let mut matrix = Self::new(size, size);
            for i in 0..size {
                matrix.elements[i][i] = SymbolicExpression::one();
            }
            matrix
        }

        /// Create a symbolic rotation matrix around X-axis
        #[allow(unused_variables)]
        pub fn rotation_x(theta: SymbolicExpression) -> Self {
            let mut matrix = Self::new(2, 2);

            #[cfg(feature = "symbolic")]
            {
                let half_theta = theta / SymbolicExpression::constant(2.0);
                let cos_expr = SymbolicExpression::SymEngine(
                    quantrs2_symengine::ops::trig::cos(&match &half_theta {
                        SymbolicExpression::SymEngine(expr) => expr.clone(),
                        _ => return matrix,
                    })
                    .unwrap_or_else(|_| quantrs2_symengine::Expression::from(1.0)),
                );
                let sin_expr = SymbolicExpression::SymEngine(
                    quantrs2_symengine::ops::trig::sin(&match &half_theta {
                        SymbolicExpression::SymEngine(expr) => expr.clone(),
                        _ => return matrix,
                    })
                    .unwrap_or_else(|_| quantrs2_symengine::Expression::from(0.0)),
                );

                matrix.elements[0][0] = cos_expr.clone();
                matrix.elements[0][1] =
                    SymbolicExpression::complex_constant(Complex64::new(0.0, -1.0))
                        * sin_expr.clone();
                matrix.elements[1][0] =
                    SymbolicExpression::complex_constant(Complex64::new(0.0, -1.0)) * sin_expr;
                matrix.elements[1][1] = cos_expr;
            }

            #[cfg(not(feature = "symbolic"))]
            {
                // Simplified representation
                matrix.elements[0][0] = SymbolicExpression::parse("cos(theta/2)")
                    .unwrap_or_else(|_| SymbolicExpression::one());
                matrix.elements[0][1] = SymbolicExpression::parse("-i*sin(theta/2)")
                    .unwrap_or_else(|_| SymbolicExpression::zero());
                matrix.elements[1][0] = SymbolicExpression::parse("-i*sin(theta/2)")
                    .unwrap_or_else(|_| SymbolicExpression::zero());
                matrix.elements[1][1] = SymbolicExpression::parse("cos(theta/2)")
                    .unwrap_or_else(|_| SymbolicExpression::one());
            }

            matrix
        }

        /// Evaluate the matrix with given variable values
        pub fn evaluate(
            &self,
            variables: &HashMap<String, f64>,
        ) -> QuantRS2Result<Array2<Complex64>> {
            let mut result = Array2::<Complex64>::zeros((self.rows, self.cols));

            for i in 0..self.rows {
                for j in 0..self.cols {
                    let complex_vars: HashMap<String, Complex64> = variables
                        .iter()
                        .map(|(k, v)| (k.clone(), Complex64::new(*v, 0.0)))
                        .collect();

                    let value = self.elements[i][j].evaluate_complex(&complex_vars)?;
                    result[[i, j]] = value;
                }
            }

            Ok(result)
        }

        /// Matrix multiplication
        pub fn multiply(&self, other: &Self) -> QuantRS2Result<Self> {
            if self.cols != other.rows {
                return Err(QuantRS2Error::InvalidInput(
                    "Matrix dimensions don't match for multiplication".to_string(),
                ));
            }

            let mut result = Self::new(self.rows, other.cols);

            for i in 0..self.rows {
                for j in 0..other.cols {
                    let mut sum = SymbolicExpression::zero();
                    for k in 0..self.cols {
                        let product = self.elements[i][k].clone() * other.elements[k][j].clone();
                        sum = sum + product;
                    }
                    result.elements[i][j] = sum;
                }
            }

            Ok(result)
        }
    }

    impl fmt::Display for SymbolicMatrix {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(f, "SymbolicMatrix[{}x{}]:", self.rows, self.cols)?;
            for row in &self.elements {
                write!(f, "[")?;
                for (j, elem) in row.iter().enumerate() {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{elem}")?;
                }
                writeln!(f, "]")?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_expression_creation() {
        let const_expr = SymbolicExpression::constant(std::f64::consts::PI);
        assert!(const_expr.is_constant());

        let var_expr = SymbolicExpression::variable("x");
        assert!(!var_expr.is_constant());
        assert_eq!(var_expr.variables(), vec!["x"]);
    }

    #[test]
    fn test_symbolic_arithmetic() {
        let a = SymbolicExpression::constant(2.0);
        let b = SymbolicExpression::constant(3.0);
        let sum = a + b;

        match sum {
            SymbolicExpression::Constant(value) => assert_eq!(value, 5.0),
            _ => panic!("Expected constant result"),
        }
    }

    #[test]
    fn test_symbolic_evaluation() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);

        let var_expr = SymbolicExpression::variable("x");
        let result = var_expr
            .evaluate(&vars)
            .expect("Failed to evaluate expression in test_symbolic_evaluation");
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_symbolic_matrix() {
        let matrix = matrix::SymbolicMatrix::identity(2);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert!(matrix.elements[0][0].is_one());
        assert!(matrix.elements[1][1].is_one());
        assert!(matrix.elements[0][1].is_zero());
    }

    #[cfg(feature = "symbolic")]
    #[test]
    fn test_symengine_integration() {
        let expr = SymbolicExpression::parse("x^2")
            .expect("Failed to parse expression in test_symengine_integration");
        match expr {
            SymbolicExpression::SymEngine(_) => {
                // Test SymEngine functionality
                assert!(!expr.is_constant());
            }
            _ => {
                // Fallback to simple parsing
                assert!(!expr.is_constant());
            }
        }
    }
}
