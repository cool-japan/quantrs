//! Basic symbolic mathematics with quantrs2-symengine
//!
//! This example demonstrates the core features of symbolic computation:
//! - Creating symbols and expressions
//! - Algebraic operations
//! - Differentiation
//! - Substitution and evaluation

#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::suspicious_operation_groupings)]

use quantrs2_symengine::Expression;
use std::collections::HashMap;

fn main() {
    println!("=== Basic Symbolic Mathematics with QuantRS2-SymEngine ===\n");

    // Create symbolic variables
    println!("1. Creating Symbols:");
    let x = Expression::symbol("x");
    let y = Expression::symbol("y");
    println!("   x = {}", x);
    println!("   y = {}\n", y);

    // Algebraic operations
    println!("2. Algebraic Operations:");
    let expr1 =
        x.clone() * x.clone() + Expression::from(2) * x.clone() * y.clone() + y.clone() * y.clone();
    println!("   (x + y)² expanded: {}", expr1.expand());

    let expr2 = (x.clone() + y.clone()).pow(&Expression::from(3));
    println!("   (x + y)³ expanded: {}", expr2.expand());

    let expr3 = (x.clone() * x.clone() - y.clone() * y.clone()) / (x.clone() - y.clone());
    println!("   (x² - y²)/(x - y) simplified: {}\n", expr3.simplify());

    // Differentiation
    println!("3. Differentiation:");
    let f = x.pow(&Expression::from(3)) + Expression::from(3) * x.pow(&Expression::from(2))
        - Expression::from(2) * x.clone()
        + Expression::from(5);
    println!("   f(x) = {}", f);
    println!("   f'(x) = {}", f.diff(&x));
    println!("   f''(x) = {}\n", f.diff_n(&x, 2));

    // Multivariate calculus
    println!("4. Multivariate Calculus:");
    let g = x.pow(&Expression::from(2)) + y.pow(&Expression::from(2));
    println!("   g(x,y) = {}", g);

    let gradient = g.gradient(&[x.clone(), y.clone()]);
    println!("   ∇g = [{}, {}]", gradient[0], gradient[1]);

    let hessian = g.hessian(&[x.clone(), y.clone()]);
    println!("   Hessian:");
    println!("   [{}, {}]", hessian[0][0], hessian[0][1]);
    println!("   [{}, {}]\n", hessian[1][0], hessian[1][1]);

    // Substitution
    println!("5. Substitution:");
    let expr = x.pow(&Expression::from(2)) + y.pow(&Expression::from(2));
    println!("   Original: {}", expr);

    let substituted = expr.substitute(&x, &Expression::from(3));
    println!("   After x → 3: {}", substituted);

    let mut subs_map = HashMap::new();
    subs_map.insert(x.clone(), Expression::from(3));
    subs_map.insert(y.clone(), Expression::from(4));

    let fully_substituted = expr.substitute_many(&subs_map);
    println!("   After x → 3, y → 4: {}", fully_substituted.expand());

    // Numerical evaluation
    println!("\n6. Numerical Evaluation:");
    let mut values = HashMap::new();
    values.insert("x".to_string(), 3.0);
    values.insert("y".to_string(), 4.0);

    if let Some(result) = expr.eval(&values) {
        println!("   x² + y² at (3, 4) = {}", result);
    }

    // Trigonometric functions
    println!("\n7. Trigonometric Functions:");
    use quantrs2_symengine::ops::trig;

    let theta = Expression::symbol("theta");
    let sin_expr = trig::sin(&theta).expect("Failed to create sin");
    let cos_expr = trig::cos(&theta).expect("Failed to create cos");

    println!("   sin(θ) = {}", sin_expr);
    println!("   cos(θ) = {}", cos_expr);

    let identity = sin_expr.pow(&Expression::from(2)) + cos_expr.pow(&Expression::from(2));
    println!("   sin²(θ) + cos²(θ) = {}", identity.simplify());

    // Exponential and logarithm
    println!("\n8. Exponential and Logarithm:");
    use quantrs2_symengine::ops::exp_log;

    let exp_x = exp_log::exp(&x).expect("Failed to create exp");
    let ln_x = exp_log::ln(&x).expect("Failed to create ln");

    println!("   e^x = {}", exp_x);
    println!("   ln(x) = {}", ln_x);
    println!("   d/dx[e^x] = {}", exp_x.diff(&x));
    println!("   d/dx[ln(x)] = {}", ln_x.diff(&x));

    println!("\n=== Example Complete ===");
}
