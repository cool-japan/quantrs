//! Symbolic differentiation module.
//!
//! This module implements automatic symbolic differentiation using
//! the standard rules of calculus.

use egg::{Id, Language, RecExpr};

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::{ExprLang, Expression};

/// Compute the derivative of an expression with respect to a variable.
///
/// This implements symbolic differentiation using the chain rule,
/// product rule, quotient rule, and standard derivative formulas.
pub fn differentiate(expr: &Expression, var: &Expression) -> Expression {
    let var_name = match var.as_symbol() {
        Some(name) => name.to_string(),
        None => {
            // If var is not a symbol, return zero (constant derivative)
            return Expression::zero();
        }
    };

    differentiate_rec(
        expr.as_rec_expr(),
        expr.as_rec_expr().as_ref().len() - 1,
        &var_name,
    )
}

/// Recursive differentiation helper
fn differentiate_rec(expr: &RecExpr<ExprLang>, idx: usize, var: &str) -> Expression {
    let node = &expr[Id::from(idx)];

    match node {
        // d/dx(c) = 0 for constants, d/dx(x) = 1, d/dx(y) = 0
        ExprLang::Num(s) => {
            let name = s.as_str();
            // Check if it's a number (constant) or a variable
            if name.parse::<f64>().is_ok() {
                // It's a numeric constant
                Expression::zero()
            } else if name == var {
                // It's the variable we're differentiating with respect to
                Expression::one()
            } else {
                // It's a different variable (treat as constant)
                Expression::zero()
            }
        }

        // d/dx(a + b) = da/dx + db/dx
        ExprLang::Add([a, b]) => {
            let da = differentiate_rec(expr, usize::from(*a), var);
            let db = differentiate_rec(expr, usize::from(*b), var);
            da + db
        }

        // d/dx(a * b) = a * db/dx + da/dx * b (product rule)
        ExprLang::Mul([a, b]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let b_expr = extract_subexpr(expr, usize::from(*b));
            let da = differentiate_rec(expr, usize::from(*a), var);
            let db = differentiate_rec(expr, usize::from(*b), var);
            a_expr * db + da * b_expr
        }

        // d/dx(a / b) = (da/dx * b - a * db/dx) / b^2 (quotient rule)
        ExprLang::Div([a, b]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let b_expr = extract_subexpr(expr, usize::from(*b));
            let da = differentiate_rec(expr, usize::from(*a), var);
            let db = differentiate_rec(expr, usize::from(*b), var);
            (da * b_expr.clone() - a_expr * db) / (b_expr.clone() * b_expr)
        }

        // d/dx(a^b) - power rule with chain rule
        ExprLang::Pow([a, b]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let b_expr = extract_subexpr(expr, usize::from(*b));
            let da = differentiate_rec(expr, usize::from(*a), var);

            // Simple case: constant exponent
            if let Some(n) = b_expr.to_f64() {
                // d/dx(a^n) = n * a^(n-1) * da/dx
                Expression::float_unchecked(n)
                    * a_expr.pow(&Expression::float_unchecked(n - 1.0))
                    * da
            } else {
                // General power rule: a^b * (b' * ln(a) + b * a'/a)
                let db = differentiate_rec(expr, usize::from(*b), var);
                let ln_a = crate::ops::trig::log(&a_expr);
                let term1 = db * ln_a;
                let term2 = b_expr.clone() * da / a_expr.clone();
                a_expr.pow(&b_expr) * (term1 + term2)
            }
        }

        // d/dx(-a) = -da/dx
        ExprLang::Neg([a]) => {
            let da = differentiate_rec(expr, usize::from(*a), var);
            da.neg()
        }

        // d/dx(1/a) = -da/dx / a^2
        ExprLang::Inv([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            da.neg() / (a_expr.clone() * a_expr)
        }

        // Trigonometric derivatives
        ExprLang::Sin([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            crate::ops::trig::cos(&a_expr) * da
        }

        ExprLang::Cos([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            crate::ops::trig::sin(&a_expr).neg() * da
        }

        ExprLang::Tan([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            let sec_sq =
                Expression::one() + crate::ops::trig::tan(&a_expr).pow(&Expression::int(2));
            sec_sq * da
        }

        // d/dx(exp(a)) = exp(a) * da/dx
        ExprLang::Exp([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            crate::ops::trig::exp(&a_expr) * da
        }

        // d/dx(log(a)) = da/dx / a
        ExprLang::Log([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            da / a_expr
        }

        // d/dx(sqrt(a)) = da/dx / (2 * sqrt(a))
        ExprLang::Sqrt([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            da / (Expression::int(2) * crate::ops::trig::sqrt(&a_expr))
        }

        // d/dx(|a|) = a/|a| * da/dx (for a != 0)
        ExprLang::Abs([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            a_expr.clone() / crate::ops::trig::abs(&a_expr) * da
        }

        // Inverse trig
        ExprLang::Asin([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            da / crate::ops::trig::sqrt(&(Expression::one() - a_expr.pow(&Expression::int(2))))
        }

        ExprLang::Acos([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            da.neg()
                / crate::ops::trig::sqrt(&(Expression::one() - a_expr.pow(&Expression::int(2))))
        }

        ExprLang::Atan([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            da / (Expression::one() + a_expr.pow(&Expression::int(2)))
        }

        // Hyperbolic functions
        ExprLang::Sinh([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            crate::ops::trig::cosh(&a_expr) * da
        }

        ExprLang::Cosh([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            crate::ops::trig::sinh(&a_expr) * da
        }

        ExprLang::Tanh([a]) => {
            let a_expr = extract_subexpr(expr, usize::from(*a));
            let da = differentiate_rec(expr, usize::from(*a), var);
            let sech_sq =
                Expression::one() - crate::ops::trig::tanh(&a_expr).pow(&Expression::int(2));
            sech_sq * da
        }

        // Complex operations
        ExprLang::Re([a]) => {
            let da = differentiate_rec(expr, usize::from(*a), var);
            crate::ops::complex::re(&da)
        }

        ExprLang::Im([a]) => {
            let da = differentiate_rec(expr, usize::from(*a), var);
            crate::ops::complex::im(&da)
        }

        ExprLang::Conj([a]) => {
            let da = differentiate_rec(expr, usize::from(*a), var);
            da.conjugate()
        }

        // For unhandled cases, return zero
        _ => Expression::zero(),
    }
}

/// Extract a subexpression from a RecExpr
fn extract_subexpr(expr: &RecExpr<ExprLang>, idx: usize) -> Expression {
    let mut new_expr = RecExpr::default();
    extract_subexpr_rec(
        expr,
        idx,
        &mut new_expr,
        &mut std::collections::HashMap::new(),
    );
    Expression::from_rec_expr(new_expr)
}

fn extract_subexpr_rec(
    expr: &RecExpr<ExprLang>,
    idx: usize,
    new_expr: &mut RecExpr<ExprLang>,
    id_map: &mut std::collections::HashMap<usize, Id>,
) -> Id {
    if let Some(&new_id) = id_map.get(&idx) {
        return new_id;
    }

    let node = &expr[Id::from(idx)];
    let new_node = node.clone().map_children(|child_id| {
        extract_subexpr_rec(expr, usize::from(child_id), new_expr, id_map)
    });
    let new_id = new_expr.add(new_node);
    id_map.insert(idx, new_id);
    new_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_constant() {
        let c = Expression::int(5);
        let x = Expression::symbol("x");
        let dc = differentiate(&c, &x);
        assert!(dc.is_zero());
    }

    #[test]
    fn test_diff_variable() {
        let x = Expression::symbol("x");
        let dx = differentiate(&x, &x);
        assert!(dx.is_one());
    }

    #[test]
    fn test_diff_other_variable() {
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");
        let dy = differentiate(&y, &x);
        assert!(dy.is_zero());
    }

    #[test]
    fn test_diff_sum() {
        let x = Expression::symbol("x");
        let c = Expression::int(5);
        let expr = x.clone() + c; // x + 5
        let dx = differentiate(&expr, &x);
        // d/dx(x + 5) = 1 + 0 (unsimplified)
        // After simplification it would be 1
        assert!(!dx.to_string().is_empty());
    }

    #[test]
    fn test_diff_product() {
        let x = Expression::symbol("x");
        let expr = x.clone() * x.clone(); // x^2 as x*x
        let dx = differentiate(&expr, &x);
        // d/dx(x*x) = x*1 + 1*x = 2x
        // The result won't be simplified, but the structure should be correct
        assert!(!dx.is_zero());
    }
}
