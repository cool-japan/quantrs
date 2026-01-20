//! VQE (Variational Quantum Eigensolver) Gradient Computation Example
//!
//! This example demonstrates how to use quantrs2-symengine-pure for
//! computing gradients needed in variational quantum algorithms.
//!
//! VQE is a hybrid quantum-classical algorithm for finding the ground state
//! energy of a molecular Hamiltonian. The key requirement is computing
//! gradients of the energy expectation value with respect to variational
//! parameters.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::useless_vec)]

use std::collections::HashMap;

use quantrs2_symengine_pure::{
    optimization::{gradient_at, hessian_at, ParameterShiftRule, VqeOptimizer},
    Expression,
};

fn main() {
    println!("=== VQE Gradient Computation Example ===\n");

    // Example 1: Simple symbolic gradient computation
    example_symbolic_gradient();

    // Example 2: Parameter-shift rule for quantum gradients
    example_parameter_shift();

    // Example 3: VQE optimization step
    example_vqe_optimization();

    // Example 4: Hessian computation for second-order optimization
    example_hessian_computation();
}

/// Demonstrates symbolic gradient computation
fn example_symbolic_gradient() {
    println!("--- Example 1: Symbolic Gradient Computation ---\n");

    // Create variational parameters
    let theta = Expression::symbol("theta");
    let phi = Expression::symbol("phi");

    // Energy expectation value: E(θ, φ) = sin(θ)² + cos(φ)²
    // This is a simplified model of a VQE energy landscape
    let sin_theta = quantrs2_symengine_pure::ops::trig::sin(&theta);
    let cos_phi = quantrs2_symengine_pure::ops::trig::cos(&phi);

    let energy = sin_theta.clone() * sin_theta + cos_phi.clone() * cos_phi;

    println!("Energy expression: E(θ, φ) = sin²(θ) + cos²(φ)");

    // Compute symbolic gradients
    let grad_theta = energy.diff(&theta);
    let grad_phi = energy.diff(&phi);

    println!("∂E/∂θ (symbolic): computed");
    println!("∂E/∂φ (symbolic): computed");

    // Evaluate gradients at specific point
    let mut values = HashMap::new();
    values.insert("theta".to_string(), std::f64::consts::FRAC_PI_4); // π/4
    values.insert("phi".to_string(), std::f64::consts::FRAC_PI_3); // π/3

    let grad_theta_val = grad_theta.eval(&values);
    let grad_phi_val = grad_phi.eval(&values);

    println!("\nAt θ = π/4, φ = π/3:");
    match grad_theta_val {
        Ok(v) => println!("  ∂E/∂θ = {:.6}", v),
        Err(e) => println!("  ∂E/∂θ error: {}", e),
    }
    match grad_phi_val {
        Ok(v) => println!("  ∂E/∂φ = {:.6}", v),
        Err(e) => println!("  ∂E/∂φ error: {}", e),
    }

    println!();
}

/// Demonstrates the parameter-shift rule for quantum circuits
fn example_parameter_shift() {
    println!("--- Example 2: Parameter-Shift Rule ---\n");

    // In quantum computing, gradients of parameterized circuits
    // can be computed using the parameter-shift rule:
    //   ∂f/∂θ = [f(θ + s) - f(θ - s)] / (2 sin(s))
    //
    // where s is typically π/2

    let psr = ParameterShiftRule::new();

    // Define a simple energy function (simulating quantum circuit output)
    let energy_fn = |params: &[f64]| -> f64 {
        // E(θ₁, θ₂) = sin(θ₁) * cos(θ₂) + 0.5
        params[0].sin() * params[1].cos() + 0.5
    };

    let params = vec![0.5, 1.0]; // Initial parameters

    println!("Energy function: E(θ₁, θ₂) = sin(θ₁)·cos(θ₂) + 0.5");
    println!(
        "Initial parameters: θ₁ = {:.4}, θ₂ = {:.4}",
        params[0], params[1]
    );
    println!("Initial energy: E = {:.6}", energy_fn(&params));

    // Compute gradient using parameter-shift rule
    let gradient = psr.compute_gradient(energy_fn, &params);

    println!("\nGradient (via parameter-shift):");
    println!("  ∂E/∂θ₁ = {:.6}", gradient[0]);
    println!("  ∂E/∂θ₂ = {:.6}", gradient[1]);

    // Compare with analytical gradient
    // ∂E/∂θ₁ = cos(θ₁) * cos(θ₂)
    // ∂E/∂θ₂ = -sin(θ₁) * sin(θ₂)
    let analytical_grad = vec![
        params[0].cos() * params[1].cos(),
        -params[0].sin() * params[1].sin(),
    ];

    println!("\nAnalytical gradient:");
    println!("  ∂E/∂θ₁ = {:.6}", analytical_grad[0]);
    println!("  ∂E/∂θ₂ = {:.6}", analytical_grad[1]);

    println!("\nDifference (should be ~0):");
    println!(
        "  Δ(∂E/∂θ₁) = {:.2e}",
        (gradient[0] - analytical_grad[0]).abs()
    );
    println!(
        "  Δ(∂E/∂θ₂) = {:.2e}",
        (gradient[1] - analytical_grad[1]).abs()
    );

    println!();
}

/// Demonstrates a VQE optimization step
fn example_vqe_optimization() {
    println!("--- Example 3: VQE Optimization Step ---\n");

    // Create symbolic energy and parameters
    let theta = Expression::symbol("theta");

    // Simple 1-parameter ansatz: E(θ) = (θ - 1)²
    // Minimum at θ = 1
    let one = Expression::one();
    let diff = theta.clone() - one;
    let energy = diff.clone() * diff;

    let params = vec![theta];
    let learning_rate = 0.1;

    let optimizer = VqeOptimizer::new(energy, params, learning_rate);

    // Initial parameter values
    let mut values = HashMap::new();
    values.insert("theta".to_string(), 0.0);

    println!("Energy: E(θ) = (θ - 1)²");
    println!("Learning rate: {}", learning_rate);
    println!("Initial θ = 0.0");

    // Simulate several optimization steps
    let mut theta_val = 0.0;
    println!("\nOptimization steps:");

    for step in 0..10 {
        values.insert("theta".to_string(), theta_val);

        let grad = optimizer.compute_gradient(&values);
        match grad {
            Ok(g) => {
                let energy_val = (theta_val - 1.0).powi(2);
                println!(
                    "  Step {}: θ = {:.6}, E = {:.6}, ∂E/∂θ = {:.6}",
                    step, theta_val, energy_val, g[0]
                );
                // Gradient descent update
                theta_val -= learning_rate * g[0];
            }
            Err(e) => {
                println!("  Step {}: gradient error: {}", step, e);
                break;
            }
        }
    }

    println!("\nFinal θ = {:.6} (optimal = 1.0)", theta_val);
    println!();
}

/// Demonstrates Hessian computation for second-order optimization
fn example_hessian_computation() {
    println!("--- Example 4: Hessian Computation ---\n");

    let x = Expression::symbol("x");
    let y = Expression::symbol("y");

    // f(x, y) = x² + 2xy + 3y²
    let two = Expression::int(2);
    let three = Expression::int(3);

    let f = x.clone() * x.clone() + two * x.clone() * y.clone() + three * y.clone() * y.clone();

    println!("Function: f(x, y) = x² + 2xy + 3y²");

    let params = [x, y];
    let mut values = HashMap::new();
    values.insert("x".to_string(), 1.0);
    values.insert("y".to_string(), 2.0);

    // Compute gradient
    let grad = gradient_at(&f, &params, &values);
    match &grad {
        Ok(g) => {
            println!("\nAt (x, y) = (1, 2):");
            println!("  ∇f = [{:.4}, {:.4}]", g[0], g[1]);
            // Analytical: ∂f/∂x = 2x + 2y = 6, ∂f/∂y = 2x + 6y = 14
        }
        Err(e) => println!("Gradient error: {}", e),
    }

    // Compute Hessian
    let hess = hessian_at(&f, &params, &values);
    match hess {
        Ok(h) => {
            println!("\n  Hessian H = ");
            println!("    [{:.4}, {:.4}]", h[0][0], h[0][1]);
            println!("    [{:.4}, {:.4}]", h[1][0], h[1][1]);
            // Analytical: H = [[2, 2], [2, 6]]
        }
        Err(e) => println!("Hessian error: {}", e),
    }

    println!("\nAnalytical Hessian (for comparison):");
    println!("    [2.0000, 2.0000]");
    println!("    [2.0000, 6.0000]");

    println!();
}
