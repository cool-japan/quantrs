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

/// Recognized unary AST operator names (mirrors the tokens accepted by
/// [`Expression::unary_arg`]).
const UNARY_OPS: &[&str] = &[
    "neg",
    "inv",
    "abs",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "re",
    "im",
    "conj",
    "trace",
    "dagger",
    "det",
    "transpose",
];

/// Recognized binary AST operator names (mirrors the tokens accepted by
/// [`Expression::binary_args`]).
const BINARY_OPS: &[&str] = &["+", "*", "/", "^", "comm", "anticomm", "tensor"];

/// Check whether `expr` contains the symbol `name` anywhere in its AST,
/// however deeply nested (inside sums, products, powers, trig/exp/log
/// wrappers, commutators, tensor products, ...).
///
/// This performs a full structural traversal via the `unary_arg`/`binary_args`
/// accessors rather than using [`Expression::free_symbols`], which
/// deliberately excludes the special constant `I` from its result (it treats
/// `I` like `pi`/`e`, not like a free variable).
fn contains_symbol(expr: &Expression, name: &str) -> bool {
    if expr.as_symbol() == Some(name) {
        return true;
    }
    for op in UNARY_OPS {
        if let Some(inner) = expr.unary_arg(op) {
            return contains_symbol(&inner, name);
        }
    }
    for op in BINARY_OPS {
        if let Some((left, right)) = expr.binary_args(op) {
            return contains_symbol(&left, name) || contains_symbol(&right, name);
        }
    }
    false
}

/// Check whether a flattened list of multiplication factors decomposes as
/// exactly one occurrence of the imaginary unit `I` times factors that are
/// all real, i.e. none of them contain a nested occurrence of `I` (which
/// would make the product genuinely complex rather than `i * real`).
fn is_imaginary_times_real(factors: &[Expression]) -> bool {
    let mut has_imaginary = false;
    for factor in factors {
        if factor.as_symbol() == Some("I") {
            if has_imaginary {
                // A second bare `I` factor (`I * I * rest`) collapses to a
                // real value, so this is no longer "a single imaginary unit
                // times a real factor".
                return false;
            }
            has_imaginary = true;
        } else if contains_symbol(factor, "I") {
            return false;
        }
    }
    has_imaginary
}

/// Check if an expression is a pure imaginary number, i.e. structurally
/// `I * real` (in any factor order, and optionally negated as a whole),
/// where `real` contains no nested occurrence of `I`.
///
/// This walks the real `RecExpr` structure (via [`flatten_factors`] and
/// [`contains_symbol`]) rather than matching substrings of
/// `expr.to_string()` (the previous implementation), which produced false
/// positives on compound expressions such as `x*I + y`: that expression's
/// string form `"(+ (* x I) y)"` contains the substring `"(* x I)"` even
/// though the *outer* expression is a sum, not a pure imaginary number.
#[must_use]
pub fn is_pure_imaginary(expr: &Expression) -> bool {
    let stripped = expr.unary_arg("neg").unwrap_or_else(|| expr.clone());

    // The bare imaginary unit is trivially pure imaginary (real part = 1).
    if stripped.as_symbol() == Some("I") {
        return true;
    }

    // Anything that is not (structurally) a product node after stripping an
    // outer negation cannot be `I * real` in this representation - notably
    // an `Add` node such as `x*I + y` is rejected here rather than
    // accidentally matching via substring search.
    if stripped.binary_args("*").is_none() {
        return false;
    }

    is_imaginary_times_real(&flatten_factors(&stripped))
}

/// Check if an expression is a unit-modulus complex exponential.
///
/// Structurally `exp(I * real)` (in any factor order, and with the whole
/// exponent optionally negated), where `real` contains no nested occurrence
/// of `I`. Such a "phase factor" `e^{iθ}` always has `|e^{iθ}| = 1` for real
/// `θ`.
///
/// This walks the real `RecExpr` structure rather than matching a string
/// prefix (the previous implementation), which both false-positived on
/// `exp(I * (a + I*b))` (a genuinely complex, not real, angle - modulus != 1
/// in general) and false-negatived on the commuted form `exp(θ * I)`.
#[must_use]
pub fn is_unit_complex_form(expr: &Expression) -> bool {
    let Some(arg) = expr.unary_arg("exp") else {
        return false;
    };
    let inner = arg.unary_arg("neg").unwrap_or(arg);

    // A bare `exp(I)` (angle = 1) is still a unit-modulus phase factor.
    if inner.as_symbol() == Some("I") {
        return true;
    }

    if inner.binary_args("*").is_none() {
        return false;
    }

    is_imaginary_times_real(&flatten_factors(&inner))
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

/// Try to recognize a quantum gate from its matrix/generator expression.
///
/// Bare Pauli/Clifford symbols (`X`, `H`, `S`, ...) are recognized directly.
/// Compound `exp(-i θ G / 2)` forms are recognized structurally via
/// [`is_rotation_gate`]: when the generator `G` is (up to naming) one of the
/// single-qubit Pauli operators, the angle is reported through the
/// corresponding [`QuantumGatePattern::Rx`]/[`Ry`](QuantumGatePattern::Ry)/
/// [`Rz`](QuantumGatePattern::Rz) variant; any other recognized rotation
/// generator is reported as a general [`QuantumGatePattern::Rotation`] with
/// the angle in the `θ` slot and `φ = λ = 0` (this expression language only
/// carries a single rotation angle per generator, so the Euler `φ`/`λ`
/// decomposition is not recoverable from a single `exp(...)` node).
#[must_use]
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

    if let Some((angle, generator)) = is_rotation_gate(expr) {
        return match generator.as_symbol() {
            Some("X" | "sigma_x" | "pauli_x") => QuantumGatePattern::Rx(angle),
            Some("Y" | "sigma_y" | "pauli_y") => QuantumGatePattern::Ry(angle),
            Some("Z" | "sigma_z" | "pauli_z") => QuantumGatePattern::Rz(angle),
            _ => QuantumGatePattern::Rotation(angle, Expression::zero(), Expression::zero()),
        };
    }

    QuantumGatePattern::Unknown
}

/// Recognize variational quantum circuit parameter patterns.
///
/// [`VariationalPattern::SingleRotation`], [`VariationalPattern::QaoaMixer`]
/// and [`VariationalPattern::QaoaCost`] are constructible from a single gate
/// expression and are produced by [`recognize_variational_pattern`].
///
/// [`VariationalPattern::EntanglingLayer`] and
/// [`VariationalPattern::VqeAnsatz`] are **forward-declared scaffolding for a
/// future circuit-level pattern API**: recognizing an entangling layer or a
/// full VQE ansatz genuinely requires looking at a *sequence* of gates (e.g.
/// the CNOT/CZ ladder plus the per-qubit rotations that make up one ansatz
/// layer), which cannot be read off a single [`Expression`]. No function in
/// this crate constructs these two variants today; they exist purely as
/// planned enum shape for when a multi-gate (`&[Expression]`-based)
/// recognizer is added.
#[derive(Debug, Clone)]
pub enum VariationalPattern {
    /// Single parameter rotation
    SingleRotation {
        axis: char, // 'x', 'y', or 'z'
        param: Expression,
    },
    /// Parametric entangling layer.
    ///
    /// Not yet constructible: see the enum-level doc comment.
    EntanglingLayer { params: Vec<Expression> },
    /// VQE ansatz pattern.
    ///
    /// Not yet constructible: see the enum-level doc comment.
    VqeAnsatz { params: Vec<Expression> },
    /// QAOA mixer pattern
    QaoaMixer { beta: Expression },
    /// QAOA cost pattern
    QaoaCost { gamma: Expression },
}

/// Check if expression matches a VQE parameter pattern
#[must_use]
pub fn is_vqe_parameter(expr: &Expression) -> bool {
    expr.as_symbol().is_some_and(|sym| {
        sym.starts_with("theta") || sym.starts_with("phi") || sym.starts_with("lambda")
    })
}

/// Check if expression matches a QAOA parameter
#[must_use]
pub fn is_qaoa_parameter(expr: &Expression) -> bool {
    expr.as_symbol()
        .is_some_and(|sym| sym.starts_with("beta") || sym.starts_with("gamma"))
}

/// Try to recognize a single-gate variational-circuit pattern.
///
/// Structurally recognizes `exp(-i θ G / 2)` (via [`is_rotation_gate`]) where
/// `G` is a single-qubit Pauli generator:
///
/// * if `θ` contains an [`is_qaoa_parameter`]-recognized `beta*` factor and
///   `G` is the `X` generator, this is a [`VariationalPattern::QaoaMixer`];
/// * if `θ` contains an [`is_qaoa_parameter`]-recognized `gamma*` factor and
///   `G` is the `Z` generator, this is a [`VariationalPattern::QaoaCost`];
/// * otherwise it is a generic [`VariationalPattern::SingleRotation`] around
///   the recognized axis.
///
/// Returns `None` when `expr` is not a structurally recognizable single-axis
/// rotation. This function never produces
/// [`VariationalPattern::EntanglingLayer`]/[`VariationalPattern::VqeAnsatz`];
/// see the enum-level doc comment on [`VariationalPattern`] for why those
/// require multi-gate context that this function does not have.
#[must_use]
pub fn recognize_variational_pattern(expr: &Expression) -> Option<VariationalPattern> {
    let (angle, generator) = is_rotation_gate(expr)?;
    let axis = match generator.as_symbol() {
        Some("X" | "sigma_x" | "pauli_x") => 'x',
        Some("Y" | "sigma_y" | "pauli_y") => 'y',
        Some("Z" | "sigma_z" | "pauli_z") => 'z',
        _ => return None,
    };

    let angle_factors = flatten_factors(&angle);
    let angle_has_qaoa_param = angle_factors.iter().any(is_qaoa_parameter);

    if axis == 'x' && angle_has_qaoa_param {
        return Some(VariationalPattern::QaoaMixer { beta: angle });
    }
    if axis == 'z' && angle_has_qaoa_param {
        return Some(VariationalPattern::QaoaCost { gamma: angle });
    }

    Some(VariationalPattern::SingleRotation { axis, param: angle })
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

    #[test]
    fn test_is_pure_imaginary_structural() {
        // Bare I and I * real are pure imaginary.
        assert!(is_pure_imaginary(&Expression::i()));
        let r = Expression::symbol("r");
        assert!(is_pure_imaginary(&(Expression::i() * r.clone())));
        assert!(is_pure_imaginary(&(r.clone() * Expression::i())));
        // Negated forms are still pure imaginary.
        assert!(is_pure_imaginary(&(-(Expression::i() * r.clone()))));

        // Regression: x*I + y is a SUM containing an imaginary term, not a
        // pure imaginary number. The old substring-based implementation
        // returned `true` here because "(* x I)" appears in the string form
        // "(+ (* x I) y)".
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");
        let sum = (x.clone() * Expression::i()) + y;
        assert!(
            !is_pure_imaginary(&sum),
            "x*I + y must not be recognized as pure imaginary"
        );

        // A factor that itself nests `I` (e.g. x * (I + y)) is not `I * real`.
        let nested = x * (Expression::i() + r);
        assert!(!is_pure_imaginary(&nested));

        // A plain real symbol is not pure imaginary.
        assert!(!is_pure_imaginary(&Expression::symbol("z")));
    }

    #[test]
    fn test_is_unit_complex_form_structural() {
        let theta = Expression::symbol("theta");

        // exp(I * theta) and the commuted exp(theta * I) are both unit
        // modulus phase factors; the old prefix-matching implementation
        // false-negatived on the commuted form.
        assert!(is_unit_complex_form(&crate::ops::trig::exp(
            &(Expression::i() * theta.clone())
        )));
        assert!(is_unit_complex_form(&crate::ops::trig::exp(
            &(theta.clone() * Expression::i())
        )));

        // exp(-i * theta) is also unit modulus.
        assert!(is_unit_complex_form(&crate::ops::trig::exp(
            &(-(Expression::i() * theta.clone()))
        )));

        // Regression: exp(I * (a + I*b)) has a genuinely COMPLEX angle
        // (a + i*b), so |exp(i*(a+ib))| = e^{-b} != 1 in general. The old
        // prefix-matching implementation returned `true` because the string
        // still started with "(exp (* I ".
        let a = Expression::symbol("a");
        let b = Expression::symbol("b");
        let complex_angle = a + Expression::i() * b;
        let fake_phase = crate::ops::trig::exp(&(Expression::i() * complex_angle));
        assert!(
            !is_unit_complex_form(&fake_phase),
            "exp(I * (a + I*b)) must not be recognized as unit modulus"
        );

        // exp(x) with no imaginary unit at all is not a phase factor.
        assert!(!is_unit_complex_form(&crate::ops::trig::exp(
            &Expression::symbol("x")
        )));

        // A bare (non-exp) expression is never a unit complex form.
        assert!(!is_unit_complex_form(&theta));
    }

    #[test]
    fn test_recognize_gate_pattern_rotations() {
        let theta = Expression::symbol("theta");
        let half = Expression::float_unchecked(0.5);

        let make_rotation = |generator: Expression| {
            crate::ops::trig::exp(
                &(-(((Expression::i() * theta.clone()) * generator) * half.clone())),
            )
        };

        match recognize_gate_pattern(&make_rotation(Expression::symbol("X"))) {
            QuantumGatePattern::Rx(angle) => {
                let mut values = std::collections::HashMap::new();
                values.insert("theta".to_string(), 0.7_f64);
                let v = angle.eval(&values).expect("angle must evaluate");
                assert!((v - 0.7).abs() < 1e-10, "angle was {v}");
            }
            other => panic!("expected Rx, got {other:?}"),
        }

        assert!(matches!(
            recognize_gate_pattern(&make_rotation(Expression::symbol("Y"))),
            QuantumGatePattern::Ry(_)
        ));
        assert!(matches!(
            recognize_gate_pattern(&make_rotation(Expression::symbol("Z"))),
            QuantumGatePattern::Rz(_)
        ));

        // Bare symbols still resolve to their fixed-gate variants.
        assert_eq!(
            recognize_gate_pattern(&Expression::symbol("H")),
            QuantumGatePattern::Hadamard
        );

        // A non-rotation, non-symbol expression is Unknown.
        assert_eq!(
            recognize_gate_pattern(&(Expression::symbol("x") + Expression::symbol("y"))),
            QuantumGatePattern::Unknown
        );
    }

    #[test]
    fn test_recognize_variational_pattern() {
        let half = Expression::float_unchecked(0.5);

        // exp(-i * beta_0 * X / 2) is a QAOA mixer term.
        let beta = Expression::symbol("beta_0");
        let mixer = crate::ops::trig::exp(
            &(-(((Expression::i() * beta) * Expression::symbol("X")) * half.clone())),
        );
        match recognize_variational_pattern(&mixer) {
            Some(VariationalPattern::QaoaMixer { beta }) => {
                assert!(is_qaoa_parameter(&flatten_factors(&beta)[0]));
            }
            other => panic!("expected QaoaMixer, got {other:?}"),
        }

        // exp(-i * gamma_1 * Z / 2) is a QAOA cost term.
        let gamma = Expression::symbol("gamma_1");
        let cost = crate::ops::trig::exp(
            &(-(((Expression::i() * gamma) * Expression::symbol("Z")) * half.clone())),
        );
        assert!(matches!(
            recognize_variational_pattern(&cost),
            Some(VariationalPattern::QaoaCost { .. })
        ));

        // exp(-i * theta_0 * Y / 2) with a non-QAOA-named angle is a plain
        // single rotation, not a QAOA variant.
        let theta = Expression::symbol("theta_0");
        let single = crate::ops::trig::exp(
            &(-(((Expression::i() * theta) * Expression::symbol("Y")) * half)),
        );
        match recognize_variational_pattern(&single) {
            Some(VariationalPattern::SingleRotation { axis, .. }) => assert_eq!(axis, 'y'),
            other => panic!("expected SingleRotation, got {other:?}"),
        }

        // A non-rotation expression yields no variational pattern.
        assert!(recognize_variational_pattern(&Expression::symbol("H")).is_none());
    }
}
