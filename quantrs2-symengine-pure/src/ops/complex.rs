//! Complex number operations for symbolic expressions.

use crate::expr::{ExprLang, Expression};
use egg::RecExpr;

/// Real part of a complex expression
pub fn re(x: &Expression) -> Expression {
    let mut expr = x.as_rec_expr().clone();
    let id = egg::Id::from(expr.as_ref().len() - 1);
    expr.add(ExprLang::Re([id]));
    Expression::from_rec_expr(expr)
}

/// Imaginary part of a complex expression
pub fn im(x: &Expression) -> Expression {
    let mut expr = x.as_rec_expr().clone();
    let id = egg::Id::from(expr.as_ref().len() - 1);
    expr.add(ExprLang::Im([id]));
    Expression::from_rec_expr(expr)
}

/// Complex conjugate
pub fn conj(x: &Expression) -> Expression {
    x.conjugate()
}

/// Create a complex number from real and imaginary parts
pub fn complex(re: &Expression, im: &Expression) -> Expression {
    re.clone() + im.clone() * Expression::i()
}

/// Magnitude (absolute value) of a complex number
pub fn magnitude(x: &Expression) -> Expression {
    crate::ops::trig::sqrt(&(re(x).pow(&Expression::int(2)) + im(x).pow(&Expression::int(2))))
}

/// Phase angle (argument) of a complex number
pub fn phase(x: &Expression) -> Expression {
    crate::ops::trig::atan(&(im(x) / re(x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_parts() {
        let z = Expression::symbol("z");
        let re_z = re(&z);
        let im_z = im(&z);

        assert!(!re_z.is_symbol());
        assert!(!im_z.is_symbol());
    }
}
