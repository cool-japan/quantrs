//! Expression AST and core types for symbolic computation.
//!
//! This module defines the symbolic expression type using egg's e-graph
//! for efficient representation and manipulation.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;
use std::sync::Arc;

use egg::{
    define_language, Analysis, CostFunction, EGraph, Id, Language, RecExpr, Rewrite, Runner, Symbol,
};
use scirs2_core::Complex64;

use crate::error::{SymEngineError, SymEngineResult};

// The symbolic expression language using egg's macro.
//
// This language supports:
// - Numeric constants (represented as strings to avoid trait issues)
// - Symbols (variables)
// - Arithmetic operations (add, mul, pow, neg, inv)
// - Transcendental functions (sin, cos, exp, log, sqrt)
// - Quantum-specific operations (commutator, anticommutator, tensor_product)
define_language! {
    /// The symbolic expression language
    pub enum ExprLang {
        // Use Symbol for both variable names and numeric literals
        // Numbers are stored as strings like "42" or "3.14"
        Num(Symbol),

        // Binary arithmetic operations
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "^" = Pow([Id; 2]),

        // Unary operations
        "neg" = Neg([Id; 1]),
        "inv" = Inv([Id; 1]),
        "abs" = Abs([Id; 1]),

        // Transcendental functions
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "tan" = Tan([Id; 1]),
        "exp" = Exp([Id; 1]),
        "log" = Log([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "asin" = Asin([Id; 1]),
        "acos" = Acos([Id; 1]),
        "atan" = Atan([Id; 1]),
        "sinh" = Sinh([Id; 1]),
        "cosh" = Cosh([Id; 1]),
        "tanh" = Tanh([Id; 1]),

        // Complex number operations
        "re" = Re([Id; 1]),
        "im" = Im([Id; 1]),
        "conj" = Conj([Id; 1]),

        // Quantum-specific operations
        "comm" = Commutator([Id; 2]),      // [A, B] = AB - BA
        "anticomm" = Anticommutator([Id; 2]), // {A, B} = AB + BA
        "tensor" = TensorProduct([Id; 2]),  // A ⊗ B
        "trace" = Trace([Id; 1]),
        "dagger" = Dagger([Id; 1]),         // Hermitian conjugate

        // Matrix operations
        "det" = Determinant([Id; 1]),
        "transpose" = Transpose([Id; 1]),
    }
}

/// A symbolic mathematical expression.
///
/// This type wraps egg's `RecExpr` and provides a user-friendly API
/// for symbolic computation.
#[derive(Clone, Debug)]
pub struct Expression {
    /// The underlying recursive expression
    expr: RecExpr<ExprLang>,
}

impl Expression {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a new symbolic variable
    ///
    /// # Example
    /// ```ignore
    /// use quantrs2_symengine_pure::Expression;
    /// let x = Expression::symbol("x");
    /// ```
    #[must_use]
    pub fn symbol(name: &str) -> Self {
        let mut expr = RecExpr::default();
        expr.add(ExprLang::Num(Symbol::from(name)));
        Self { expr }
    }

    /// Create an integer constant
    #[must_use]
    pub fn int(value: i64) -> Self {
        let mut expr = RecExpr::default();
        expr.add(ExprLang::Num(Symbol::from(value.to_string())));
        Self { expr }
    }

    /// Create a floating-point constant
    ///
    /// # Errors
    /// Returns an error if the value is NaN
    pub fn float(value: f64) -> SymEngineResult<Self> {
        if value.is_nan() {
            return Err(SymEngineError::Undefined(
                "NaN is not a valid symbolic value".into(),
            ));
        }
        let mut expr = RecExpr::default();
        expr.add(ExprLang::Num(Symbol::from(value.to_string())));
        Ok(Self { expr })
    }

    /// Create a floating-point constant, using 0 for NaN
    #[must_use]
    pub fn float_unchecked(value: f64) -> Self {
        let v = if value.is_nan() { 0.0 } else { value };
        let mut expr = RecExpr::default();
        expr.add(ExprLang::Num(Symbol::from(v.to_string())));
        Self { expr }
    }

    /// Create the constant zero
    #[must_use]
    pub fn zero() -> Self {
        Self::int(0)
    }

    /// Create the constant one
    #[must_use]
    pub fn one() -> Self {
        Self::int(1)
    }

    /// Create the imaginary unit i
    #[must_use]
    pub fn i() -> Self {
        Self::symbol("I")
    }

    /// Create the constant π
    #[must_use]
    pub fn pi() -> Self {
        Self::symbol("pi")
    }

    /// Create the constant e (Euler's number)
    #[must_use]
    pub fn e() -> Self {
        Self::symbol("e")
    }

    /// Create from a complex number
    ///
    /// If imaginary part is negligible, returns just the real part.
    #[must_use]
    pub fn from_complex64(c: Complex64) -> Self {
        const EPSILON: f64 = 1e-15;
        if c.im.abs() < EPSILON {
            Self::float_unchecked(c.re)
        } else if c.re.abs() < EPSILON {
            // Pure imaginary
            Self::float_unchecked(c.im) * Self::i()
        } else {
            // General complex
            Self::float_unchecked(c.re) + Self::float_unchecked(c.im) * Self::i()
        }
    }

    /// Parse an expression from a string
    ///
    /// # Errors
    /// Returns an error if parsing fails
    pub fn parse(input: &str) -> SymEngineResult<Self> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(SymEngineError::parse("empty expression"));
        }

        // Try to parse as number
        if let Ok(n) = trimmed.parse::<i64>() {
            return Ok(Self::int(n));
        }
        if let Ok(f) = trimmed.parse::<f64>() {
            return Self::float(f);
        }

        // Otherwise treat as symbol/expression
        Ok(Self::symbol(trimmed))
    }

    /// Create an expression from a string (alias for parse)
    #[must_use]
    pub fn new(input: impl AsRef<str>) -> Self {
        Self::parse(input.as_ref()).unwrap_or_else(|_| Self::symbol(input.as_ref()))
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the root node of the expression
    fn root(&self) -> &ExprLang {
        &self.expr[self.root_id()]
    }

    /// Get the root ID
    fn root_id(&self) -> Id {
        Id::from(self.expr.as_ref().len() - 1)
    }

    /// Check if this expression is a symbol (variable or number literal)
    #[must_use]
    pub fn is_symbol(&self) -> bool {
        matches!(self.root(), ExprLang::Num(_))
    }

    /// Check if this expression is a number
    #[must_use]
    pub fn is_number(&self) -> bool {
        if let ExprLang::Num(s) = self.root() {
            s.as_str().parse::<f64>().is_ok()
        } else {
            false
        }
    }

    /// Check if this expression is zero
    #[must_use]
    pub fn is_zero(&self) -> bool {
        if let ExprLang::Num(s) = self.root() {
            s.as_str() == "0" || s.as_str().parse::<f64>().is_ok_and(|v| v.abs() < 1e-15)
        } else {
            false
        }
    }

    /// Check if this expression is one
    #[must_use]
    pub fn is_one(&self) -> bool {
        if let ExprLang::Num(s) = self.root() {
            s.as_str() == "1"
                || s.as_str()
                    .parse::<f64>()
                    .is_ok_and(|v| (v - 1.0).abs() < 1e-15)
        } else {
            false
        }
    }

    /// Get the symbol name if this is a symbol
    #[must_use]
    pub fn as_symbol(&self) -> Option<&str> {
        if let ExprLang::Num(s) = self.root() {
            // Only return as symbol if it's not a number
            if s.as_str().parse::<f64>().is_err() {
                return Some(s.as_str());
            }
        }
        None
    }

    /// Convert to f64 if this is a numeric constant
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        if let ExprLang::Num(s) = self.root() {
            s.as_str().parse::<f64>().ok()
        } else {
            None
        }
    }

    /// Convert to i64 if this is an integer constant
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        if let ExprLang::Num(s) = self.root() {
            s.as_str().parse::<i64>().ok()
        } else {
            None
        }
    }

    /// Check if this expression is an addition operation
    #[must_use]
    pub fn is_add(&self) -> bool {
        matches!(self.root(), ExprLang::Add(_))
    }

    /// Check if this expression is a multiplication operation
    #[must_use]
    pub fn is_mul(&self) -> bool {
        matches!(self.root(), ExprLang::Mul(_))
    }

    /// Check if this expression is a power operation
    #[must_use]
    pub fn is_pow(&self) -> bool {
        matches!(self.root(), ExprLang::Pow(_))
    }

    /// Check if this expression is a negation operation
    #[must_use]
    pub fn is_neg(&self) -> bool {
        matches!(self.root(), ExprLang::Neg(_))
    }

    /// Get the inner expression if this is a negation
    #[must_use]
    pub fn as_neg(&self) -> Option<Self> {
        if let ExprLang::Neg([inner_id]) = self.root() {
            Some(self.extract_subexpr(*inner_id))
        } else {
            None
        }
    }

    /// Get the operands if this is an addition operation
    #[must_use]
    pub fn as_add(&self) -> Option<Vec<Self>> {
        if let ExprLang::Add([lhs_id, rhs_id]) = self.root() {
            Some(vec![
                self.extract_subexpr(*lhs_id),
                self.extract_subexpr(*rhs_id),
            ])
        } else {
            None
        }
    }

    /// Get the operands if this is a multiplication operation
    #[must_use]
    pub fn as_mul(&self) -> Option<Vec<Self>> {
        if let ExprLang::Mul([lhs_id, rhs_id]) = self.root() {
            Some(vec![
                self.extract_subexpr(*lhs_id),
                self.extract_subexpr(*rhs_id),
            ])
        } else {
            None
        }
    }

    /// Get the base and exponent if this is a power operation
    #[must_use]
    pub fn as_pow(&self) -> Option<(Self, Self)> {
        if let ExprLang::Pow([base_id, exp_id]) = self.root() {
            Some((
                self.extract_subexpr(*base_id),
                self.extract_subexpr(*exp_id),
            ))
        } else {
            None
        }
    }

    /// Extract a subexpression by its ID
    fn extract_subexpr(&self, id: Id) -> Self {
        let target_idx = usize::from(id);
        let mut new_expr = RecExpr::default();

        // Build a mapping from old IDs to new IDs
        let mut id_map = std::collections::HashMap::new();

        // Traverse the expression up to and including the target node
        for (idx, node) in self.expr.as_ref().iter().enumerate() {
            if idx > target_idx {
                break;
            }
            let new_node = node
                .clone()
                .map_children(|old_id| *id_map.get(&old_id).unwrap_or(&old_id));
            let new_id = new_expr.add(new_node);
            id_map.insert(Id::from(idx), new_id);
        }

        Self { expr: new_expr }
    }

    // =========================================================================
    // Basic Operations
    // =========================================================================

    /// Add two expressions
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.clone() + other.clone()
    }

    /// Subtract two expressions
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.clone() - other.clone()
    }

    /// Multiply two expressions
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        self.clone() * other.clone()
    }

    /// Divide two expressions
    #[must_use]
    pub fn div(&self, other: &Self) -> Self {
        self.clone() / other.clone()
    }

    /// Raise to a power
    #[must_use]
    pub fn pow(&self, exp: &Self) -> Self {
        let mut expr = self.expr.clone();
        let lhs_id = Id::from(expr.as_ref().len() - 1);

        // Merge the exponent expression
        let rhs_id = merge_expr(&mut expr, &exp.expr);

        expr.add(ExprLang::Pow([lhs_id, rhs_id]));
        Self { expr }
    }

    /// Negate the expression
    #[must_use]
    pub fn neg(&self) -> Self {
        let mut expr = self.expr.clone();
        let id = Id::from(expr.as_ref().len() - 1);
        expr.add(ExprLang::Neg([id]));
        Self { expr }
    }

    /// Complex conjugate
    #[must_use]
    pub fn conjugate(&self) -> Self {
        let mut expr = self.expr.clone();
        let id = Id::from(expr.as_ref().len() - 1);
        expr.add(ExprLang::Conj([id]));
        Self { expr }
    }

    // =========================================================================
    // Calculus
    // =========================================================================

    /// Compute the derivative with respect to a variable
    #[must_use]
    pub fn diff(&self, var: &Self) -> Self {
        crate::diff::differentiate(self, var)
    }

    /// Compute the gradient with respect to multiple variables
    #[must_use]
    pub fn gradient(&self, vars: &[Self]) -> Vec<Self> {
        vars.iter().map(|v| self.diff(v)).collect()
    }

    /// Compute the Hessian matrix (second derivatives)
    #[must_use]
    pub fn hessian(&self, vars: &[Self]) -> Vec<Vec<Self>> {
        let grad = self.gradient(vars);
        grad.iter().map(|g| g.gradient(vars)).collect()
    }

    // =========================================================================
    // Simplification
    // =========================================================================

    /// Expand the expression (distribute products over sums)
    #[must_use]
    pub fn expand(&self) -> Self {
        crate::simplify::expand(self)
    }

    /// Simplify the expression
    #[must_use]
    pub fn simplify(&self) -> Self {
        crate::simplify::simplify(self)
    }

    // =========================================================================
    // Evaluation
    // =========================================================================

    /// Evaluate the expression with given variable values
    ///
    /// # Errors
    /// Returns an error if a variable is not found in the values map
    pub fn eval(&self, values: &HashMap<String, f64>) -> SymEngineResult<f64> {
        crate::eval::evaluate(self, values)
    }

    /// Evaluate the expression to a complex number.
    ///
    /// This can handle expressions containing the imaginary unit `I`,
    /// which is essential for quantum computing applications.
    ///
    /// # Arguments
    /// * `values` - Map of variable names to real values
    ///
    /// # Returns
    /// The complex result of the evaluation.
    ///
    /// # Errors
    /// Returns an error if a variable is not found in the values map
    pub fn eval_complex(
        &self,
        values: &HashMap<String, f64>,
    ) -> SymEngineResult<scirs2_core::Complex64> {
        crate::eval::evaluate_complex(self, values)
    }

    /// Substitute a variable with an expression
    #[must_use]
    pub fn substitute(&self, var: &Self, value: &Self) -> Self {
        crate::simplify::substitute(self, var, value)
    }

    /// Substitute multiple variables
    #[must_use]
    pub fn substitute_many(&self, values: &HashMap<Self, Self>) -> Self {
        let mut result = self.clone();
        for (var, value) in values {
            result = result.substitute(var, value);
        }
        result
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Get the underlying RecExpr (for advanced usage)
    pub(crate) const fn as_rec_expr(&self) -> &RecExpr<ExprLang> {
        &self.expr
    }

    /// Create from RecExpr (for internal use)
    pub(crate) const fn from_rec_expr(expr: RecExpr<ExprLang>) -> Self {
        Self { expr }
    }
}

/// Merge another expression into a RecExpr and return the new root ID
fn merge_expr(target: &mut RecExpr<ExprLang>, source: &RecExpr<ExprLang>) -> Id {
    let offset = target.as_ref().len();
    for node in source.as_ref() {
        let shifted = node
            .clone()
            .map_children(|id| Id::from(usize::from(id) + offset));
        target.add(shifted);
    }
    Id::from(target.as_ref().len() - 1)
}

// =========================================================================
// Trait Implementations
// =========================================================================

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr.pretty(80))
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

impl Eq for Expression {}

impl Hash for Expression {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_string().hash(state);
    }
}

impl From<i64> for Expression {
    fn from(n: i64) -> Self {
        Self::int(n)
    }
}

impl From<i32> for Expression {
    fn from(n: i32) -> Self {
        Self::int(i64::from(n))
    }
}

impl From<f64> for Expression {
    fn from(f: f64) -> Self {
        Self::float_unchecked(f)
    }
}

impl From<Complex64> for Expression {
    fn from(c: Complex64) -> Self {
        Self::from_complex64(c)
    }
}

// Implement arithmetic operators
impl std::ops::Add for Expression {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut expr = self.expr;
        let lhs_id = Id::from(expr.as_ref().len() - 1);
        let rhs_id = merge_expr(&mut expr, &rhs.expr);
        expr.add(ExprLang::Add([lhs_id, rhs_id]));
        Self { expr }
    }
}

impl std::ops::Sub for Expression {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        self + rhs.neg()
    }
}

impl std::ops::Mul for Expression {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut expr = self.expr;
        let lhs_id = Id::from(expr.as_ref().len() - 1);
        let rhs_id = merge_expr(&mut expr, &rhs.expr);
        expr.add(ExprLang::Mul([lhs_id, rhs_id]));
        Self { expr }
    }
}

impl std::ops::Div for Expression {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        let mut expr = self.expr;
        let lhs_id = Id::from(expr.as_ref().len() - 1);
        let rhs_id = merge_expr(&mut expr, &rhs.expr);
        expr.add(ExprLang::Div([lhs_id, rhs_id]));
        Self { expr }
    }
}

impl std::ops::Neg for Expression {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_creation() {
        let x = Expression::symbol("x");
        assert!(x.is_symbol());
        assert_eq!(x.as_symbol(), Some("x"));
    }

    #[test]
    fn test_integer_creation() {
        let n = Expression::int(42);
        assert!(n.is_number());
        assert_eq!(n.to_i64(), Some(42));
    }

    #[test]
    fn test_float_creation() {
        let f = Expression::float(2.5).expect("valid float");
        assert!(f.is_number());
        let val = f.to_f64().expect("should be f64");
        assert!((val - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_zero_and_one() {
        let zero = Expression::zero();
        let one = Expression::one();

        assert!(zero.is_zero());
        assert!(!zero.is_one());
        assert!(one.is_one());
        assert!(!one.is_zero());
    }

    #[test]
    fn test_from_complex64() {
        let c = Complex64::new(3.0, 4.0);
        let expr = Expression::from_complex64(c);
        assert!(!expr.is_number());
    }

    #[test]
    fn test_arithmetic_operators() {
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");

        let sum = x.clone() + y.clone();
        let product = x.clone() * y.clone();
        let diff = x.clone() - y.clone();
        let quot = x / y;

        assert!(!sum.is_symbol());
        assert!(!product.is_symbol());
        assert!(!diff.is_symbol());
        assert!(!quot.is_symbol());
    }
}
