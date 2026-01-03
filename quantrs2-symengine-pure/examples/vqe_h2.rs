//! Variational Quantum Eigensolver (VQE) for H2 molecule.
//!
//! This example demonstrates using symbolic expressions to implement
//! VQE for finding the ground state energy of the hydrogen molecule.
//!
//! Run with: cargo run --example vqe_h2

use std::collections::HashMap;

use quantrs2_symengine_pure::{
    expr::Expression,
    matrix::{self, SymbolicMatrix},
    ops::trig,
    optimization::{gradient_at, ParameterShiftRule},
    SymEngineResult,
};

fn main() -> SymEngineResult<()> {
    println!("=== VQE for H2 Molecule ===\n");

    // Create symbolic parameters for the variational ansatz
    let theta = Expression::symbol("theta");

    // Create a simple one-parameter ansatz: Ry(theta) applied to |0⟩
    // This is a simplified model - real VQE uses more complex ansatze
    let ry = matrix::ry(&theta);
    println!("Ansatz (Ry gate):");
    println!("  Ry(theta) = [[cos(θ/2), -sin(θ/2)],");
    println!("               [sin(θ/2),  cos(θ/2)]]\n");

    // Define a simplified Hamiltonian for H2
    // H = g0*I + g1*Z + g2*X (coefficients from molecular integrals)
    let g0 = Expression::float(-1.0523)?;
    let g1 = Expression::float(0.3943)?;
    let g2 = Expression::float(0.1809)?;

    // Build Hamiltonian symbolically
    let identity = SymbolicMatrix::identity(2);
    let pauli_x = matrix::pauli_x();
    let pauli_z = matrix::pauli_z();

    println!("Hamiltonian:");
    println!("  H = g0*I + g1*Z + g2*X");
    println!("  g0 = -1.0523 (nuclear repulsion + core)");
    println!("  g1 =  0.3943 (Z coefficient)");
    println!("  g2 =  0.1809 (X coefficient)\n");

    // Expected energy function: ⟨ψ(θ)|H|ψ(θ)⟩
    // For Ry(θ)|0⟩ = [cos(θ/2), sin(θ/2)]^T:
    // E(θ) = g0 + g1*cos(θ) + g2*sin(θ)
    let half = Expression::float(0.5)?;
    let cos_theta = trig::cos(&theta);
    let sin_theta = trig::sin(&theta);

    let energy = g0 + g1 * cos_theta + g2 * sin_theta;

    println!("Energy expectation value:");
    println!("  E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩");
    println!("       = g0 + g1*cos(θ) + g2*sin(θ)\n");

    // Compute the symbolic gradient
    let gradient = energy.diff(&theta);
    println!("Symbolic gradient:");
    println!("  dE/dθ = -g1*sin(θ) + g2*cos(θ)\n");

    // Gradient descent optimization
    println!("=== Gradient Descent Optimization ===\n");

    let mut param_value = 0.0; // Initial parameter
    let learning_rate = 0.5;
    let n_iterations = 20;

    println!("Initial θ = {param_value:.4}");

    for iteration in 0..n_iterations {
        let mut values = HashMap::new();
        values.insert("theta".to_string(), param_value);

        let current_energy = energy.eval(&values)?;
        let grad = gradient.eval(&values)?;

        if iteration % 5 == 0 || iteration == n_iterations - 1 {
            println!(
                "Iter {iteration:2}: θ = {param_value:.4}, E(θ) = {current_energy:.6}, dE/dθ = {grad:.6}"
            );
        }

        param_value -= learning_rate * grad;
    }

    // Final evaluation
    let mut final_values = HashMap::new();
    final_values.insert("theta".to_string(), param_value);
    let final_energy = energy.eval(&final_values)?;

    println!("\n=== Results ===\n");
    println!("Optimal θ  = {param_value:.6}");
    println!("Minimum E  = {final_energy:.6}");

    // Theoretical minimum: E_min = g0 - sqrt(g1^2 + g2^2)
    let g1_val = -1.0523_f64;
    let g1_coeff = 0.3943_f64;
    let g2_coeff = 0.1809_f64;
    let theoretical_min = g1_val - g1_coeff.hypot(g2_coeff);
    println!("Theoretical minimum = {theoretical_min:.6}");
    println!("Error = {:.2e}", (final_energy - theoretical_min).abs());

    println!("\n=== Parameter-Shift Rule Verification ===\n");

    // Verify gradient using parameter-shift rule
    let psr = ParameterShiftRule::new();
    let energy_fn = |params: &[f64]| {
        0.3943_f64.mul_add(
            params[0].cos(),
            0.1809_f64.mul_add(params[0].sin(), -1.0523),
        )
    };

    let psr_grad = psr.compute_gradient(energy_fn, &[param_value]);
    let symbolic_grad = gradient.eval(&final_values)?;

    println!("Symbolic gradient:       {symbolic_grad:.6}");
    println!("Parameter-shift gradient: {:.6}", psr_grad[0]);
    println!(
        "Difference:              {:.2e}",
        (symbolic_grad - psr_grad[0]).abs()
    );

    Ok(())
}
