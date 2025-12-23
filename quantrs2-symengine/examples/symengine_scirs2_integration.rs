//! SciRS2 Integration Example
#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
//!
//! This example demonstrates how to integrate quantrs2-symengine with
//! SciRS2's numeric types for hybrid symbolic-numeric computation.

use quantrs2_symengine::{ops, Expression};
use scirs2_core::Complex64;

fn main() {
    println!("=== SciRS2 Integration Example ===\n");

    // Creating expressions from Complex64
    println!("1. Creating Expressions from Complex64:");
    let c1 = Complex64::new(1.0, 2.0);
    let c2 = Complex64::new(3.0, -1.0);

    let expr1 = Expression::from_complex64(c1);
    let expr2 = Expression::from_complex64(c2);

    println!("c1 = {} → expr1 = {}", c1, expr1);
    println!("c2 = {} → expr2 = {}", c2, expr2);
    println!();

    // Symbolic operations on complex numbers
    println!("2. Symbolic Operations:");
    let sum = expr1.clone() + expr2.clone();
    let product = expr1.clone() * expr2.clone();
    let quotient = expr1.clone() / expr2.clone();

    println!("expr1 + expr2 = {}", sum);
    println!("expr1 * expr2 = {}", product.expand());
    println!("expr1 / expr2 = {}", quotient);
    println!();

    // Converting back to Complex64
    println!("3. Converting Back to Complex64:");
    let result_complex = sum.to_complex64().expect("Failed to convert to Complex64");
    println!("Sum as Complex64: {}", result_complex);
    println!();

    // Symbolic computation then numeric evaluation
    println!("4. Hybrid Symbolic-Numeric Computation:");
    let x = Expression::symbol("x");
    let y = Expression::symbol("y");

    // Create symbolic expression
    let symbolic_expr = x.pow(&Expression::from(2)) + y.clone() * Expression::from(2);
    println!("f(x, y) = {}", symbolic_expr);

    // Differentiate symbolically
    let df_dx = symbolic_expr.diff(&x);
    let df_dy = symbolic_expr.diff(&y);

    println!("∂f/∂x = {}", df_dx);
    println!("∂f/∂y = {}", df_dy);
    println!();

    // Evaluate at complex values
    println!("5. Numerical Evaluation at Complex Values:");
    let x_val = Complex64::new(1.0, 0.5);
    let y_val = Complex64::new(2.0, -0.3);

    let expr_with_x = symbolic_expr.substitute(&x, &Expression::from_complex64(x_val));
    let expr_final = expr_with_x.substitute(&y, &Expression::from_complex64(y_val));

    println!("Substituting x = {}, y = {}", x_val, y_val);
    println!("f({}, {}) = {}", x_val, y_val, expr_final);
    println!();

    // Complex number operations
    println!("6. Complex Number Operations:");
    let z = Expression::symbol("z");
    let z_complex = Expression::from_complex64(Complex64::new(3.0, 4.0));

    let re_z = z_complex.re();
    let im_z = z_complex.im();
    let conj_z = z_complex.conjugate();
    let abs_z = z_complex.abs();
    let arg_z = z_complex.arg();

    println!("z = {}", z_complex);
    println!("Re(z) = {}", re_z);
    println!("Im(z) = {}", im_z);
    println!("z* = {}", conj_z);
    println!("|z| = {}", abs_z);
    println!("arg(z) = {}", arg_z);
    println!();

    // Trigonometric functions with complex arguments
    println!("7. Transcendental Functions:");
    let theta = Expression::symbol("theta");
    let complex_angle = Expression::from_complex64(Complex64::new(0.0, 1.0)); // i

    let sin_expr = ops::trig::sin(&theta).expect("Failed to create sin expression");
    let exp_expr = ops::exp_log::exp(&complex_angle).expect("Failed to create exp expression");

    println!("sin(θ) = {}", sin_expr);
    println!("e^i = {}", exp_expr);
    println!();

    // Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
    println!("8. Euler's Formula:");
    let i = Expression::new("I");
    let euler_lhs = ops::exp_log::exp(&(&i * theta.clone())).expect("Failed to create exp");
    let euler_rhs_cos = ops::trig::cos(&theta).expect("Failed to create cos");
    let euler_rhs_sin = ops::trig::sin(&theta).expect("Failed to create sin");
    let euler_rhs = euler_rhs_cos + &i * euler_rhs_sin;

    println!("e^(iθ) = {}", euler_lhs);
    println!("cos(θ) + i·sin(θ) = {}", euler_rhs);
    println!();

    // Quantum amplitude example
    println!("9. Quantum Amplitude Calculation:");
    let amplitude = Complex64::new(0.6, 0.8);
    let amp_expr = Expression::from_complex64(amplitude);
    let probability = amp_expr.abs().pow(&Expression::from(2));

    println!("Amplitude: α = {}", amplitude);
    println!("Symbolic: α = {}", amp_expr);
    println!("Probability: |α|² = {}", probability);
    println!();

    println!("=== Example Complete ===");
}
