//! VQE Gradient Computation Example
#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
//!
//! This example demonstrates symbolic gradient computation for
//! Variational Quantum Eigensolver (VQE) optimization.

use quantrs2_symengine::ndarray_integration::gradient_at;
use quantrs2_symengine::{ops, Expression};
use std::collections::HashMap;

fn main() {
    println!("=== VQE Gradient Computation Example ===\n");

    // Define variational ansatz parameters
    println!("1. Variational Ansatz Parameters:");
    let theta = Expression::symbol("theta");
    let phi = Expression::symbol("phi");
    println!("Parameters: θ, φ");
    println!();

    // Define a simple quantum energy expectation value
    // E(θ, φ) = cos(θ) + sin(φ) + θ^2 + φ^2 (toy example)
    println!("2. Energy Functional (Toy Example):");

    let cos_theta = ops::trig::cos(&theta).expect("Failed to create cos");
    let sin_phi = ops::trig::sin(&phi).expect("Failed to create sin");
    let theta_sq = theta.pow(&Expression::from(2));
    let phi_sq = phi.pow(&Expression::from(2));

    let energy = cos_theta.clone() + sin_phi.clone() + theta_sq.clone() + phi_sq.clone();
    println!("E(θ, φ) = cos(θ) + sin(φ) + θ² + φ²");
    println!("Symbolic form: {}", energy);
    println!();

    // Compute symbolic gradient
    println!("3. Symbolic Gradient Computation:");
    let params = vec![theta.clone(), phi.clone()];
    let grad = energy.gradient(&params);

    println!("∂E/∂θ = {}", grad[0]);
    println!("∂E/∂φ = {}", grad[1]);
    println!();

    // Evaluate gradient at specific parameter values
    println!("4. Numerical Gradient Evaluation:");
    let param_values = vec![
        (0.0, 0.0),
        (std::f64::consts::PI / 4.0, std::f64::consts::PI / 6.0),
        (std::f64::consts::PI / 2.0, std::f64::consts::PI / 2.0),
    ];

    for (theta_val, phi_val) in &param_values {
        let mut values = HashMap::new();
        values.insert("theta".to_string(), *theta_val);
        values.insert("phi".to_string(), *phi_val);

        let numeric_grad =
            gradient_at(&energy, &params, &values).expect("Failed to evaluate gradient");

        let energy_val = energy.eval(&values).expect("Failed to evaluate energy");

        println!("At (θ={:.4}, φ={:.4}):", theta_val, phi_val);
        println!("  E = {:.6}", energy_val);
        println!("  ∇E = [{:.6}, {:.6}]", numeric_grad[0], numeric_grad[1]);
        println!();
    }

    // Demonstrate gradient descent step
    println!("5. Gradient Descent Optimization:");
    let mut theta_curr = std::f64::consts::PI / 2.0;
    let mut phi_curr = std::f64::consts::PI / 2.0;
    let learning_rate = 0.1;
    let iterations = 5;

    println!("Initial: (θ={:.4}, φ={:.4})", theta_curr, phi_curr);

    for i in 0..iterations {
        let mut values = HashMap::new();
        values.insert("theta".to_string(), theta_curr);
        values.insert("phi".to_string(), phi_curr);

        let current_energy = energy.eval(&values).expect("Failed to evaluate energy");
        let grad_numeric =
            gradient_at(&energy, &params, &values).expect("Failed to evaluate gradient");

        println!("\nIteration {}:", i + 1);
        println!("  Current: (θ={:.4}, φ={:.4})", theta_curr, phi_curr);
        println!("  Energy: {:.6}", current_energy);
        println!(
            "  Gradient: [{:.6}, {:.6}]",
            grad_numeric[0], grad_numeric[1]
        );

        // Gradient descent update
        theta_curr -= learning_rate * grad_numeric[0];
        phi_curr -= learning_rate * grad_numeric[1];

        println!("  Updated: (θ={:.4}, φ={:.4})", theta_curr, phi_curr);
    }

    println!("\n6. Hessian Computation (Second Derivatives):");
    let hessian = energy.hessian(&params);
    println!("Hessian matrix:");
    println!("  ∂²E/∂θ² = {}", hessian[0][0]);
    println!("  ∂²E/∂θ∂φ = {}", hessian[0][1]);
    println!("  ∂²E/∂φ∂θ = {}", hessian[1][0]);
    println!("  ∂²E/∂φ² = {}", hessian[1][1]);
    println!();

    // Multi-parameter substitution
    println!("7. Multi-Parameter Substitution:");
    let mut final_params = HashMap::new();
    final_params.insert(theta.clone(), Expression::from(0.0));
    final_params.insert(phi.clone(), Expression::from(0.0));

    let optimum = energy.substitute_many(&final_params);
    println!("E(0, 0) = {}", optimum);

    if let Some(val) = optimum.to_f64() {
        println!("Numerical value: {:.6}", val);
    }
    println!();

    println!("=== Example Complete ===");
    println!("\nNote: This is a toy example. Real VQE would:");
    println!("  - Use actual quantum circuit ansatz");
    println!("  - Compute expectation values ⟨ψ(θ)|H|ψ(θ)⟩");
    println!("  - Use parameter-shift rule for gradients");
    println!("  - Interface with quantum hardware/simulators");
}
