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

/// Match compound patterns by expression structure
fn match_compound_pattern(pattern: &Pattern, expr: &Expression, captures: &mut Captures) -> bool {
    // Get the expression string for structural analysis
    let expr_str = expr.to_string();

    match pattern {
        Pattern::Neg(inner) => {
            if expr_str.starts_with("(neg ") {
                // For negation patterns, we'd need to extract the inner expression
                // This is a simplified implementation
                let inner_expr = extract_unary_arg(expr, "neg");
                if let Some(inner_expr) = inner_expr {
                    return match_pattern_rec(inner, &inner_expr, captures);
                }
            }
            false
        }

        Pattern::Sin(inner) => {
            if expr_str.starts_with("(sin ") {
                if let Some(inner_expr) = extract_unary_arg(expr, "sin") {
                    return match_pattern_rec(inner, &inner_expr, captures);
                }
            }
            false
        }

        Pattern::Cos(inner) => {
            if expr_str.starts_with("(cos ") {
                if let Some(inner_expr) = extract_unary_arg(expr, "cos") {
                    return match_pattern_rec(inner, &inner_expr, captures);
                }
            }
            false
        }

        Pattern::Exp(inner) => {
            if expr_str.starts_with("(exp ") {
                if let Some(inner_expr) = extract_unary_arg(expr, "exp") {
                    return match_pattern_rec(inner, &inner_expr, captures);
                }
            }
            false
        }

        Pattern::Log(inner) => {
            if expr_str.starts_with("(log ") {
                if let Some(inner_expr) = extract_unary_arg(expr, "log") {
                    return match_pattern_rec(inner, &inner_expr, captures);
                }
            }
            false
        }

        Pattern::Dagger(inner) => {
            if expr_str.starts_with("(dagger ") {
                if let Some(inner_expr) = extract_unary_arg(expr, "dagger") {
                    return match_pattern_rec(inner, &inner_expr, captures);
                }
            }
            false
        }

        // Binary patterns - simplified implementation
        Pattern::Add(left, right) => {
            if expr_str.starts_with("(+ ") {
                if let Some((left_expr, right_expr)) = extract_binary_args(expr, "+") {
                    return match_pattern_rec(left, &left_expr, captures)
                        && match_pattern_rec(right, &right_expr, captures);
                }
            }
            false
        }

        Pattern::Mul(left, right) => {
            if expr_str.starts_with("(* ") {
                if let Some((left_expr, right_expr)) = extract_binary_args(expr, "*") {
                    return match_pattern_rec(left, &left_expr, captures)
                        && match_pattern_rec(right, &right_expr, captures);
                }
            }
            false
        }

        Pattern::Pow(base, exp) => {
            if expr_str.starts_with("(^ ") {
                if let Some((base_expr, exp_expr)) = extract_binary_args(expr, "^") {
                    return match_pattern_rec(base, &base_expr, captures)
                        && match_pattern_rec(exp, &exp_expr, captures);
                }
            }
            false
        }

        Pattern::Commutator(a, b) => {
            if expr_str.starts_with("(comm ") {
                if let Some((a_expr, b_expr)) = extract_binary_args(expr, "comm") {
                    return match_pattern_rec(a, &a_expr, captures)
                        && match_pattern_rec(b, &b_expr, captures);
                }
            }
            false
        }

        Pattern::Anticommutator(a, b) => {
            if expr_str.starts_with("(anticomm ") {
                if let Some((a_expr, b_expr)) = extract_binary_args(expr, "anticomm") {
                    return match_pattern_rec(a, &a_expr, captures)
                        && match_pattern_rec(b, &b_expr, captures);
                }
            }
            false
        }

        Pattern::TensorProduct(a, b) => {
            if expr_str.starts_with("(tensor ") {
                if let Some((a_expr, b_expr)) = extract_binary_args(expr, "tensor") {
                    return match_pattern_rec(a, &a_expr, captures)
                        && match_pattern_rec(b, &b_expr, captures);
                }
            }
            false
        }

        // These are handled in the main match
        Pattern::Wildcard(_)
        | Pattern::Constant(_)
        | Pattern::Symbol(_)
        | Pattern::Zero
        | Pattern::One => unreachable!(),
    }
}

/// Extract unary argument from expression (simplified)
const fn extract_unary_arg(_expr: &Expression, _op: &str) -> Option<Expression> {
    // In a full implementation, this would parse the RecExpr structure
    // For now, return None as this requires deeper integration
    None
}

/// Extract binary arguments from expression (simplified)
const fn extract_binary_args(_expr: &Expression, _op: &str) -> Option<(Expression, Expression)> {
    // In a full implementation, this would parse the RecExpr structure
    // For now, return None as this requires deeper integration
    None
}

// =========================================================================
// Common Quantum Pattern Recognizers
// =========================================================================

/// Check if an expression is a rotation gate form: exp(-i * θ * G / 2)
/// Returns the angle and generator if matched
pub fn is_rotation_gate(expr: &Expression) -> Option<(Expression, Expression)> {
    // Pattern: exp(* (neg (* ?i ?theta)) ?generator)
    // This is a simplified check
    let s = expr.to_string();
    if s.starts_with("(exp ") {
        // Could be a rotation gate form
        // For a full implementation, we'd need to check the structure
        return None;
    }
    None
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

/// Check if an expression is a projector (P² = P)
pub const fn is_projector_form(expr: &Expression) -> bool {
    // |ψ⟩⟨ψ| form is a projector
    // This would require more sophisticated pattern matching
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
}
