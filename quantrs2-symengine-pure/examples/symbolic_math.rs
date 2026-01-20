//! Symbolic Mathematics Example.
//!
//! This example demonstrates the core symbolic mathematics capabilities:
//! - Expression parsing
//! - Symbolic differentiation
//! - Expression simplification
//! - Substitution
//! - Serialization
//!
//! Run with: cargo run --example symbolic_math

use std::collections::HashMap;

use quantrs2_symengine_pure::{
    expr::Expression, ops::trig, parser, serialize, simplify, SymEngineResult,
};

fn main() -> SymEngineResult<()> {
    println!("=== Symbolic Mathematics Demo ===\n");

    // =========================================================================
    // Expression Creation
    // =========================================================================
    println!("--- Expression Creation ---\n");

    // Using constructors
    let x = Expression::symbol("x");
    let y = Expression::symbol("y");
    let two = Expression::int(2);
    let pi = Expression::float(std::f64::consts::PI)?;

    println!("Symbols: x, y");
    println!("Integer: 2");
    println!("Float: π = {pi}\n");

    // Using operators
    let sum = x.clone() + y.clone();
    let product = x.clone() * y.clone();
    let power = x.pow(&two);

    println!("Sum: x + y = {sum}");
    println!("Product: x * y = {product}");
    println!("Power: x^2 = {power}\n");

    // =========================================================================
    // Expression Parsing
    // =========================================================================
    println!("--- Expression Parsing ---\n");

    let parsed = vec![
        "x + y",
        "x * y + z",
        "x^2 + 2*x + 1",
        "sin(x) + cos(y)",
        "exp(-x^2)",
        "sqrt(x^2 + y^2)",
    ];

    for input in &parsed {
        match parser::parse(input) {
            Ok(expr) => println!("  '{input}' → {expr}"),
            Err(e) => println!("  '{input}' → Error: {e}"),
        }
    }
    println!();

    // =========================================================================
    // Symbolic Differentiation
    // =========================================================================
    println!("--- Symbolic Differentiation ---\n");

    // d/dx(x^2 + 2x + 1)
    let poly = x.clone() * x.clone() + Expression::int(2) * x.clone() + Expression::one();
    let dpoly = poly.diff(&x);
    println!("d/dx(x² + 2x + 1) = {dpoly}");

    // Evaluate derivative at x=3
    let mut values = HashMap::new();
    values.insert("x".to_string(), 3.0);
    let dpoly_at_3 = dpoly.eval(&values)?;
    println!("  At x=3: {dpoly_at_3} (expected: 8)\n");

    // d/dx(sin(x))
    let sin_x = trig::sin(&x);
    let cos_x = sin_x.diff(&x);
    println!("d/dx(sin(x)) = cos(x)");

    values.insert("x".to_string(), 0.0);
    let cos_at_0 = cos_x.eval(&values)?;
    println!("  At x=0: {cos_at_0:.4} (expected: 1.0)\n");

    // d/dx(exp(x))
    let exp_x = trig::exp(&x);
    let dexp_x = exp_x.diff(&x);
    println!("d/dx(exp(x)) = exp(x)");

    values.insert("x".to_string(), 1.0);
    let dexp_at_1 = dexp_x.eval(&values)?;
    println!(
        "  At x=1: {dexp_at_1:.4} (expected: {:.4})\n",
        1.0_f64.exp()
    );

    // Chain rule: d/dx(sin(x^2))
    let sin_x2 = trig::sin(&(x.clone() * x.clone()));
    let dsin_x2 = sin_x2.diff(&x);
    println!("d/dx(sin(x²)) = 2x·cos(x²)");

    values.insert("x".to_string(), 0.5);
    let dsin_x2_val = dsin_x2.eval(&values)?;
    let expected = 2.0 * 0.5 * (0.25_f64).cos();
    println!("  At x=0.5: {dsin_x2_val:.6} (expected: {expected:.6})\n");

    // =========================================================================
    // Gradients and Hessians
    // =========================================================================
    println!("--- Gradients and Hessians ---\n");

    // f(x,y) = x^2 + xy + y^2
    #[allow(clippy::suspicious_operation_groupings)]
    let f = x.clone() * x.clone() + x.clone() * y.clone() + y.clone() * y.clone();
    println!("f(x,y) = x² + xy + y²");

    let vars = vec![x.clone(), y.clone()];
    let gradient = f.gradient(&vars);
    let hessian = f.hessian(&vars);

    println!("∇f = [∂f/∂x, ∂f/∂y]");
    println!("   = [2x + y, x + 2y]");

    values.insert("x".to_string(), 1.0);
    values.insert("y".to_string(), 2.0);

    let grad_x = gradient[0].eval(&values)?;
    let grad_y = gradient[1].eval(&values)?;
    println!("  At (1,2): [{grad_x}, {grad_y}] (expected: [4, 5])");

    println!("\nHessian:");
    println!("  H = [[∂²f/∂x², ∂²f/∂x∂y],");
    println!("       [∂²f/∂y∂x, ∂²f/∂y²]]");
    println!("    = [[2, 1],");
    println!("       [1, 2]]\n");

    // =========================================================================
    // Simplification
    // =========================================================================
    println!("--- Simplification ---\n");

    // x + 0 = x
    let add_zero = x.clone() + Expression::zero();
    let simplified = simplify::simplify(&add_zero);
    println!("x + 0 → {simplified}");

    // x * 1 = x
    let mul_one = x.clone() * Expression::one();
    let simplified = simplify::simplify(&mul_one);
    println!("x * 1 → {simplified}");

    // x * 0 = 0
    let mul_zero = x.clone() * Expression::zero();
    let simplified = simplify::simplify(&mul_zero);
    println!("x * 0 → {simplified}");

    // 0 * x = 0
    let zero_mul = Expression::zero() * x.clone();
    let simplified = simplify::simplify(&zero_mul);
    println!("0 * x → {simplified}\n");

    // =========================================================================
    // Substitution
    // =========================================================================
    println!("--- Substitution ---\n");

    let expr = x.clone() * x.clone() + y.clone();
    println!("Expression: x² + y");

    let substituted = expr.substitute(&x, &Expression::int(3));
    println!("Substitute x=3: {substituted}");

    values.insert("y".to_string(), 5.0);
    let result = substituted.eval(&values)?;
    println!("With y=5: {result} (expected: 14)\n");

    // Multiple substitution
    let mut subs = HashMap::new();
    subs.insert(x, Expression::int(2));
    subs.insert(y, Expression::int(3));

    let multi_sub = expr.substitute_many(&subs);
    println!("Substitute x=2, y=3: {multi_sub}");
    let result = multi_sub.eval(&HashMap::new())?;
    println!("Result: {result} (expected: 7)\n");

    // =========================================================================
    // Serialization
    // =========================================================================
    println!("--- Serialization ---\n");

    let expr = Expression::symbol("theta");

    // Binary serialization
    let bytes = serialize::to_bytes(&expr)?;
    println!("Binary size: {} bytes", bytes.len());

    let restored = serialize::from_bytes(&bytes)?;
    println!("Restored: {restored}");

    // JSON serialization
    let json = serialize::to_json(&expr);
    println!("JSON: {json}");

    let from_json = serialize::from_json(&json)?;
    println!("From JSON: {from_json}\n");

    // =========================================================================
    // Complex Numbers
    // =========================================================================
    println!("--- Complex Numbers ---\n");

    use quantrs2_symengine_pure::Complex64;

    let c = Complex64::new(3.0, 4.0);
    let expr_c = Expression::from_complex64(c);
    println!("Complex 3+4i: {expr_c}");

    let i = Expression::i();
    println!("Imaginary unit i: {i}");

    let two_i = Expression::int(2) * Expression::i();
    println!("2i: {two_i}\n");

    println!("=== Complete ===");

    Ok(())
}
