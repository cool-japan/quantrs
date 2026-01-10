//! Transcendental functions for symbolic expressions.

use crate::error::SymEngineResult;
use crate::expr::{ExprLang, Expression};
use egg::RecExpr;

/// Sine function
pub fn sin(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Sin)
}

/// Cosine function
pub fn cos(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Cos)
}

/// Tangent function
pub fn tan(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Tan)
}

/// Exponential function (e^x)
pub fn exp(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Exp)
}

/// Natural logarithm
pub fn log(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Log)
}

/// Logarithm with arbitrary base
pub fn log_base(x: &Expression, base: &Expression) -> Expression {
    log(x) / log(base)
}

/// Square root
pub fn sqrt(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Sqrt)
}

/// Absolute value
pub fn abs(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Abs)
}

/// Inverse sine (arcsin)
pub fn asin(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Asin)
}

/// Inverse cosine (arccos)
pub fn acos(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Acos)
}

/// Inverse tangent (arctan)
pub fn atan(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Atan)
}

/// Hyperbolic sine
pub fn sinh(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Sinh)
}

/// Hyperbolic cosine
pub fn cosh(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Cosh)
}

/// Hyperbolic tangent
pub fn tanh(x: &Expression) -> Expression {
    apply_unary(x, ExprLang::Tanh)
}

/// Helper function to apply a unary operation
fn apply_unary<F>(x: &Expression, make_node: F) -> Expression
where
    F: FnOnce([egg::Id; 1]) -> ExprLang,
{
    let mut expr = x.as_rec_expr().clone();
    let id = egg::Id::from(expr.as_ref().len() - 1);
    expr.add(make_node([id]));
    Expression::from_rec_expr(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trig_functions() {
        let x = Expression::symbol("x");

        let sin_x = sin(&x);
        let cos_x = cos(&x);
        let tan_x = tan(&x);

        assert!(!sin_x.is_symbol());
        assert!(!cos_x.is_symbol());
        assert!(!tan_x.is_symbol());
    }

    #[test]
    fn test_exp_log() {
        let x = Expression::symbol("x");

        let exp_x = exp(&x);
        let log_x = log(&x);

        assert!(!exp_x.is_symbol());
        assert!(!log_x.is_symbol());
    }
}
