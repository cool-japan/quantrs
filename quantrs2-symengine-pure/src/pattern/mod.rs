//! Pattern matching for quantum expressions.
//!
//! This module provides utilities for recognizing and extracting
//! common patterns in quantum computing expressions.

use std::collections::HashMap;

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::{ExprLang, Expression};

/// A pattern that can match against expressions.
#[derive(Clone, Debug)]
pub enum Pattern {
    /// Match any expression and capture it
    Wildcard(String),
    /// Match a specific constant
    Constant(f64),
    /// Match a specific symbol
    Symbol(String),
    /// Match zero
    Zero,
    /// Match one
    One,
    /// Match an addition pattern
    Add(Box<Self>, Box<Self>),
    /// Match a multiplication pattern
    Mul(Box<Self>, Box<Self>),
    /// Match a power pattern
    Pow(Box<Self>, Box<Self>),
    /// Match a negation pattern
    Neg(Box<Self>),
    /// Match a sine pattern
    Sin(Box<Self>),
    /// Match a cosine pattern
    Cos(Box<Self>),
    /// Match an exponential pattern
    Exp(Box<Self>),
    /// Match a logarithm pattern
    Log(Box<Self>),
    /// Match a commutator pattern
    Commutator(Box<Self>, Box<Self>),
    /// Match an anticommutator pattern
    Anticommutator(Box<Self>, Box<Self>),
    /// Match a tensor product pattern
    TensorProduct(Box<Self>, Box<Self>),
    /// Match a dagger pattern
    Dagger(Box<Self>),
}

#[allow(clippy::should_implement_trait)]
impl Pattern {
    /// Create a wildcard pattern with the given name
    #[must_use]
    pub fn wildcard(name: &str) -> Self {
        Self::Wildcard(name.to_string())
    }

    /// Create a symbol pattern
    #[must_use]
    pub fn symbol(name: &str) -> Self {
        Self::Symbol(name.to_string())
    }

    /// Create a constant pattern
    #[must_use]
    pub const fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    /// Create an addition pattern
    #[must_use]
    pub fn add(left: Self, right: Self) -> Self {
        Self::Add(Box::new(left), Box::new(right))
    }

    /// Create a multiplication pattern
    #[must_use]
    pub fn mul(left: Self, right: Self) -> Self {
        Self::Mul(Box::new(left), Box::new(right))
    }

    /// Create a power pattern
    #[must_use]
    pub fn pow(base: Self, exp: Self) -> Self {
        Self::Pow(Box::new(base), Box::new(exp))
    }

    /// Create a sine pattern
    #[must_use]
    pub fn sin(arg: Self) -> Self {
        Self::Sin(Box::new(arg))
    }

    /// Create a cosine pattern
    #[must_use]
    pub fn cos(arg: Self) -> Self {
        Self::Cos(Box::new(arg))
    }

    /// Create a commutator pattern [A, B]
    #[must_use]
    pub fn commutator(a: Self, b: Self) -> Self {
        Self::Commutator(Box::new(a), Box::new(b))
    }

    /// Create an anticommutator pattern {A, B}
    #[must_use]
    pub fn anticommutator(a: Self, b: Self) -> Self {
        Self::Anticommutator(Box::new(a), Box::new(b))
    }

    /// Create a tensor product pattern A ⊗ B
    #[must_use]
    pub fn tensor(a: Self, b: Self) -> Self {
        Self::TensorProduct(Box::new(a), Box::new(b))
    }

    /// Create a dagger pattern A†
    #[must_use]
    pub fn dagger(a: Self) -> Self {
        Self::Dagger(Box::new(a))
    }
}

/// Result of pattern matching - captured expressions
pub type Captures = HashMap<String, Expression>;

/// Match a pattern against an expression
pub fn match_pattern(pattern: &Pattern, expr: &Expression) -> Option<Captures> {
    let mut captures = Captures::new();
    if match_pattern_rec(pattern, expr, &mut captures) {
        Some(captures)
    } else {
        None
    }
}

/// Recursive pattern matching helper
#[allow(clippy::option_if_let_else)]
fn match_pattern_rec(pattern: &Pattern, expr: &Expression, captures: &mut Captures) -> bool {
    match pattern {
        Pattern::Wildcard(name) => {
            // Check if already captured with different value
            if let Some(existing) = captures.get(name) {
                // Must match the same expression
                existing == expr
            } else {
                captures.insert(name.clone(), expr.clone());
                true
            }
        }

        Pattern::Constant(value) => {
            if let Some(v) = expr.to_f64() {
                (v - value).abs() < 1e-15
            } else {
                false
            }
        }

        Pattern::Symbol(name) => expr.as_symbol() == Some(name.as_str()),

        Pattern::Zero => expr.is_zero(),

        Pattern::One => expr.is_one(),

        // For compound patterns, we need to access the internal structure
        // This requires parsing the expression representation
        // For now, use string-based matching as a simple implementation
        _ => match_compound_pattern(pattern, expr, captures),
    }
}

/// Match a compound pattern against the real AST structure of an expression.
///
/// Each branch extracts the operands of the corresponding `ExprLang` node via the
/// structural accessors in [`crate::expr`] and recurses. Matching is performed on
/// the actual `RecExpr` tree (not its textual rendering), so it is exact and works
/// for arbitrarily nested expressions.
fn match_compound_pattern(pattern: &Pattern, expr: &Expression, captures: &mut Captures) -> bool {
    match pattern {
        Pattern::Neg(inner) => match_unary(inner, expr, "neg", captures),
        Pattern::Sin(inner) => match_unary(inner, expr, "sin", captures),
        Pattern::Cos(inner) => match_unary(inner, expr, "cos", captures),
        Pattern::Exp(inner) => match_unary(inner, expr, "exp", captures),
        Pattern::Log(inner) => match_unary(inner, expr, "log", captures),
        Pattern::Dagger(inner) => match_unary(inner, expr, "dagger", captures),

        Pattern::Add(left, right) => match_binary(left, right, expr, "+", captures),
        Pattern::Mul(left, right) => match_binary(left, right, expr, "*", captures),
        Pattern::Pow(base, exp) => match_binary(base, exp, expr, "^", captures),
        Pattern::Commutator(a, b) => match_binary(a, b, expr, "comm", captures),
        Pattern::Anticommutator(a, b) => match_binary(a, b, expr, "anticomm", captures),
        Pattern::TensorProduct(a, b) => match_binary(a, b, expr, "tensor", captures),

        // These are handled in the main match
        Pattern::Wildcard(_)
        | Pattern::Constant(_)
        | Pattern::Symbol(_)
        | Pattern::Zero
        | Pattern::One => unreachable!(),
    }
}

/// Match a unary pattern: extract the operand of `op` and recurse on `inner`.
fn match_unary(inner: &Pattern, expr: &Expression, op: &str, captures: &mut Captures) -> bool {
    extract_unary_arg(expr, op).is_some_and(|arg| match_pattern_rec(inner, &arg, captures))
}

/// Match a binary pattern: extract both operands of `op` and recurse on each.
fn match_binary(
    left: &Pattern,
    right: &Pattern,
    expr: &Expression,
    op: &str,
    captures: &mut Captures,
) -> bool {
    extract_binary_args(expr, op).is_some_and(|(l, r)| {
        match_pattern_rec(left, &l, captures) && match_pattern_rec(right, &r, captures)
    })
}

/// Extract the operand of a unary node whose operator is `op`.
///
/// Returns `None` when the expression's root node is not that unary operator.
fn extract_unary_arg(expr: &Expression, op: &str) -> Option<Expression> {
    expr.unary_arg(op)
}

/// Extract both operands of a binary node whose operator is `op`.
///
/// Returns `None` when the expression's root node is not that binary operator.
fn extract_binary_args(expr: &Expression, op: &str) -> Option<(Expression, Expression)> {
    expr.binary_args(op)
}

// =========================================================================
// Common Quantum Pattern Recognizers
// =========================================================================

/// Check if an expression is a rotation gate form: `exp(-i * θ * G / 2)`.
///
/// Returns `Some((angle, generator))` when the expression is structurally a
/// rotation gate, where `angle` is the rotation angle `θ` (the `1/2` factor and
/// the imaginary unit are stripped out) and `generator` is the Hermitian
/// generator `G`. Returns `None` for any expression that is not of this form.
///
/// The recognizer operates on the real expression AST: it requires an `exp`
/// node whose argument, after removing an outer negation, is a product
/// containing the imaginary unit `I`. The remaining factors are split into the
/// numeric/symbolic angle (multiplied by 2 to undo the conventional `/2`) and
/// the generator (the factor that is recognised as a Hermitian operator). A
/// genuine `None` here means "not a recognizable rotation form", which is a
/// correct negative rather than a placeholder.
#[must_use]
pub fn is_rotation_gate(expr: &Expression) -> Option<(Expression, Expression)> {
    // Must be exp(arg).
    let arg = expr.unary_arg("exp")?;

    // exp(-i θ G / 2): the conventional sign is negative, but accept either so
    // that exp(i θ G / 2) (negative rotation) is also recognised.
    let inner = arg.unary_arg("neg").unwrap_or(arg);

    // Flatten the product into its factors.
    let factors = flatten_factors(&inner);

    // A rotation generator times an imaginary unit needs at least `I` and `G`.
    let mut has_imaginary = false;
    let mut generator: Option<Expression> = None;
    let mut angle_factors: Vec<Expression> = Vec::with_capacity(factors.len());

    for factor in factors {
        if factor.as_symbol() == Some("I") {
            has_imaginary = true;
        } else if generator.is_none() && is_hermitian_form(&factor) && !factor.is_number() {
            // The first non-numeric Hermitian factor is taken as the generator.
            generator = Some(factor);
        } else {
            angle_factors.push(factor);
        }
    }

    if !has_imaginary {
        return None;
    }
    let generator = generator?;

    // Reassemble the angle from the remaining factors and undo the `/2` so the
    // returned angle is the physical rotation angle θ.
    let angle = match angle_factors.split_first() {
        Some((first, rest)) => {
            let mut acc = first.clone();
            for f in rest {
                acc = acc * f.clone();
            }
            acc * Expression::int(2)
        }
        None => Expression::int(2),
    };

    Some((angle, generator))
}

/// Flatten a (possibly nested) product into a flat list of factors.
///
/// Division `a / b` contributes `a` and `inv(b)` is not expanded here; only
/// multiplication nodes are descended into. Non-product expressions yield a
/// single-element list.
fn flatten_factors(expr: &Expression) -> Vec<Expression> {
    if let Some((left, right)) = expr.binary_args("*") {
        let mut factors = flatten_factors(&left);
        factors.extend(flatten_factors(&right));
        factors
    } else {
        vec![expr.clone()]
    }
}

/// Check if an expression represents a Hermitian operator (A = A†)
pub fn is_hermitian_form(expr: &Expression) -> bool {
    // Simple check: if it's a symbol, it could be Hermitian
    // Real numbers are Hermitian
    if expr.is_number() {
        return true;
    }
    // Pauli matrices are Hermitian
    expr.as_symbol().is_some_and(|sym| {
        matches!(
            sym,
            "sigma_x" | "sigma_y" | "sigma_z" | "X" | "Y" | "Z" | "I"
        )
    })
}

/// Check if an expression is a projector (P² = P).
///
/// The expression language has no outer-product / ket-bra (`|ψ⟩⟨ψ|`) node, so a
/// projector cannot be represented syntactically in this AST. This therefore
/// always returns `false`: it is an honest "cannot be a projector in this
/// representation" rather than a heuristic guess. Projector recognition would
/// require extending [`crate::expr::ExprLang`] with bra/ket constructs.
#[must_use]
pub const fn is_projector_form(_expr: &Expression) -> bool {
    false
}

/// Check if an expression is a pure imaginary number (i * real)
pub fn is_pure_imaginary(expr: &Expression) -> bool {
    let s = expr.to_string();
    s.contains("(* ") && s.contains(" I)") || s.contains("(* I ")
}

/// Check if an expression is a unit complex number (|z| = 1)
pub fn is_unit_complex_form(expr: &Expression) -> bool {
    let s = expr.to_string();
    // exp(i * θ) has |exp(i*θ)| = 1
    s.starts_with("(exp (* I ") || s.starts_with("(exp (* (neg I) ")
}

/// Recognize common quantum gate patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantumGatePattern {
    /// Pauli X gate
    PauliX,
    /// Pauli Y gate
    PauliY,
    /// Pauli Z gate
    PauliZ,
    /// Hadamard gate
    Hadamard,
    /// S gate (phase gate)
    SGate,
    /// T gate
    TGate,
    /// Rx rotation with angle
    Rx(Expression),
    /// Ry rotation with angle
    Ry(Expression),
    /// Rz rotation with angle
    Rz(Expression),
    /// General rotation
    Rotation(Expression, Expression, Expression), // θ, φ, λ
    /// Unknown gate
    Unknown,
}

/// Try to recognize a quantum gate from its matrix expression
pub fn recognize_gate_pattern(expr: &Expression) -> QuantumGatePattern {
    if let Some(sym) = expr.as_symbol() {
        match sym {
            "X" | "sigma_x" | "pauli_x" => return QuantumGatePattern::PauliX,
            "Y" | "sigma_y" | "pauli_y" => return QuantumGatePattern::PauliY,
            "Z" | "sigma_z" | "pauli_z" => return QuantumGatePattern::PauliZ,
            "H" | "hadamard" => return QuantumGatePattern::Hadamard,
            "S" | "s_gate" => return QuantumGatePattern::SGate,
            "T" | "t_gate" => return QuantumGatePattern::TGate,
            _ => {}
        }
    }
    QuantumGatePattern::Unknown
}

/// Recognize variational quantum circuit parameter patterns
#[derive(Debug, Clone)]
pub enum VariationalPattern {
    /// Single parameter rotation
    SingleRotation {
        axis: char, // 'x', 'y', or 'z'
        param: Expression,
    },
    /// Parametric entangling layer
    EntanglingLayer { params: Vec<Expression> },
    /// VQE ansatz pattern
    VqeAnsatz { params: Vec<Expression> },
    /// QAOA pattern
    QaoaMixer { beta: Expression },
    /// QAOA cost pattern
    QaoaCost { gamma: Expression },
}

/// Check if expression matches a VQE parameter pattern
pub fn is_vqe_parameter(expr: &Expression) -> bool {
    expr.as_symbol().is_some_and(|sym| {
        sym.starts_with("theta") || sym.starts_with("phi") || sym.starts_with("lambda")
    })
}

/// Check if expression matches a QAOA parameter
pub fn is_qaoa_parameter(expr: &Expression) -> bool {
    expr.as_symbol()
        .is_some_and(|sym| sym.starts_with("beta") || sym.starts_with("gamma"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wildcard_pattern() {
        let x = Expression::symbol("x");
        let pattern = Pattern::wildcard("a");

        let result = match_pattern(&pattern, &x);
        assert!(result.is_some());

        let captures = result.expect("should match");
        assert!(captures.contains_key("a"));
        assert_eq!(captures.get("a").expect("has a").as_symbol(), Some("x"));
    }

    #[test]
    fn test_symbol_pattern() {
        let x = Expression::symbol("x");
        let pattern = Pattern::symbol("x");

        assert!(match_pattern(&pattern, &x).is_some());

        let y = Expression::symbol("y");
        assert!(match_pattern(&pattern, &y).is_none());
    }

    #[test]
    fn test_constant_pattern() {
        let expr = Expression::float_unchecked(2.5);
        let pattern = Pattern::constant(2.5);

        assert!(match_pattern(&pattern, &expr).is_some());

        let pattern2 = Pattern::constant(3.0);
        assert!(match_pattern(&pattern2, &expr).is_none());
    }

    #[test]
    fn test_zero_one_patterns() {
        let zero = Expression::zero();
        let one = Expression::one();

        assert!(match_pattern(&Pattern::Zero, &zero).is_some());
        assert!(match_pattern(&Pattern::One, &one).is_some());
        assert!(match_pattern(&Pattern::Zero, &one).is_none());
        assert!(match_pattern(&Pattern::One, &zero).is_none());
    }

    #[test]
    fn test_gate_recognition() {
        let x = Expression::symbol("X");
        assert_eq!(recognize_gate_pattern(&x), QuantumGatePattern::PauliX);

        let y = Expression::symbol("sigma_y");
        assert_eq!(recognize_gate_pattern(&y), QuantumGatePattern::PauliY);

        let h = Expression::symbol("H");
        assert_eq!(recognize_gate_pattern(&h), QuantumGatePattern::Hadamard);
    }

    #[test]
    fn test_hermitian_recognition() {
        let x = Expression::symbol("X");
        assert!(is_hermitian_form(&x));

        let num = Expression::float_unchecked(2.5);
        assert!(is_hermitian_form(&num));
    }

    #[test]
    fn test_vqe_parameter_recognition() {
        let theta = Expression::symbol("theta_1");
        assert!(is_vqe_parameter(&theta));

        let x = Expression::symbol("x");
        assert!(!is_vqe_parameter(&x));
    }

    #[test]
    fn test_qaoa_parameter_recognition() {
        let beta = Expression::symbol("beta_0");
        assert!(is_qaoa_parameter(&beta));

        let gamma = Expression::symbol("gamma_1");
        assert!(is_qaoa_parameter(&gamma));

        let x = Expression::symbol("x");
        assert!(!is_qaoa_parameter(&x));
    }

    #[test]
    fn test_unary_compound_pattern_matches_and_captures() {
        // Regression test: compound (unary) patterns used to ALWAYS fail because
        // the operand extractor returned `None`. They must now match and capture.
        let x = Expression::symbol("x");
        let sin_x = crate::ops::trig::sin(&x);

        let pattern = Pattern::sin(Pattern::wildcard("inner"));
        let captures = match_pattern(&pattern, &sin_x).expect("sin(x) must match Sin(?inner)");
        assert_eq!(
            captures.get("inner").and_then(Expression::as_symbol),
            Some("x")
        );

        // A cos pattern must NOT match a sin expression.
        let cos_pattern = Pattern::cos(Pattern::wildcard("inner"));
        assert!(match_pattern(&cos_pattern, &sin_x).is_none());
    }

    #[test]
    fn test_binary_compound_pattern_matches_operands() {
        // Regression test: binary patterns used to always fail.
        let sum = Expression::symbol("x") + Expression::symbol("y");

        // Order-sensitive structural match: x is the left operand, y the right.
        let pattern = Pattern::add(Pattern::symbol("x"), Pattern::symbol("y"));
        assert!(match_pattern(&pattern, &sum).is_some());

        // Reversed operands must not match the concrete structure.
        let reversed = Pattern::add(Pattern::symbol("y"), Pattern::symbol("x"));
        assert!(match_pattern(&reversed, &sum).is_none());

        // A multiplication pattern must not match an addition.
        let mul_pattern = Pattern::mul(Pattern::wildcard("a"), Pattern::wildcard("b"));
        assert!(match_pattern(&mul_pattern, &sum).is_none());
    }

    #[test]
    fn test_nested_compound_pattern_with_wildcard_consistency() {
        // exp(sin(x)) must match Exp(Sin(?a)) and capture a = x.
        let x = Expression::symbol("x");
        let nested = crate::ops::trig::exp(&crate::ops::trig::sin(&x));

        let pattern = Pattern::Exp(Box::new(Pattern::sin(Pattern::wildcard("a"))));
        let captures = match_pattern(&pattern, &nested).expect("must match nested pattern");
        assert_eq!(captures.get("a").and_then(Expression::as_symbol), Some("x"));

        // Wildcard consistency: Add(?a, ?a) matches x + x but not x + y.
        let same = Pattern::add(Pattern::wildcard("a"), Pattern::wildcard("a"));
        assert!(match_pattern(&same, &(x.clone() + x.clone())).is_some());
        let y = Expression::symbol("y");
        assert!(match_pattern(&same, &(x + y)).is_none());
    }

    #[test]
    fn test_is_rotation_gate_structural() {
        // exp(-i * theta * X / 2) is a rotation gate with angle theta, generator X.
        let theta = Expression::symbol("theta");
        let generator = Expression::symbol("X");
        let half = Expression::float_unchecked(0.5);
        let arg = ((Expression::i() * theta) * generator) * half;
        let rot = crate::ops::trig::exp(&(-arg));

        let (angle, gen) =
            is_rotation_gate(&rot).expect("exp(-i theta X / 2) must be a rotation gate");
        assert_eq!(gen.as_symbol(), Some("X"));

        // The recovered angle, evaluated at theta = 1.3, must equal 1.3 (the /2 is
        // undone). This fails if the recognizer fabricates or drops the angle.
        let mut values = std::collections::HashMap::new();
        values.insert("theta".to_string(), 1.3_f64);
        let angle_val = angle.eval(&values).expect("angle must evaluate");
        assert!((angle_val - 1.3).abs() < 1e-10, "angle was {angle_val}");
    }

    #[test]
    fn test_is_rotation_gate_rejects_non_rotations() {
        let x = Expression::symbol("x");
        // exp(x): not a rotation (no imaginary unit / generator).
        assert!(is_rotation_gate(&crate::ops::trig::exp(&x)).is_none());

        // exp(-theta * X / 2): missing the imaginary unit -> not a rotation.
        let theta = Expression::symbol("theta");
        let generator = Expression::symbol("X");
        let arg = (theta * generator) * Expression::float_unchecked(0.5);
        assert!(is_rotation_gate(&crate::ops::trig::exp(&(-arg))).is_none());

        // A bare symbol is not an exp(...) at all.
        assert!(is_rotation_gate(&x).is_none());
    }
}
