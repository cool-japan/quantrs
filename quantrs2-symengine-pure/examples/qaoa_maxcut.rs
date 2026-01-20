//! Quantum Approximate Optimization Algorithm (QAOA) for MaxCut.
//!
//! This example demonstrates symbolic computation for QAOA,
//! showing how to build and optimize variational quantum circuits
//! for combinatorial optimization problems.
//!
//! Run with: cargo run --example qaoa_maxcut

use std::collections::HashMap;

use quantrs2_symengine_pure::{
    expr::Expression,
    matrix::{self, SymbolicMatrix},
    ops::trig,
    simplify, SymEngineResult,
};

/// Represents an edge in the MaxCut graph.
struct Edge(usize, usize);

fn main() -> SymEngineResult<()> {
    println!("=== QAOA for MaxCut Problem ===\n");

    // Define a simple 4-node graph (square with diagonal)
    //   0 --- 1
    //   |  X  |
    //   3 --- 2
    let edges = [
        Edge(0, 1),
        Edge(1, 2),
        Edge(2, 3),
        Edge(3, 0),
        Edge(0, 2), // Diagonal
    ];

    println!("Graph: 4-node square with diagonal");
    println!("Edges: (0,1), (1,2), (2,3), (3,0), (0,2)\n");

    // QAOA parameters
    let gamma = Expression::symbol("gamma"); // Phase separation angle
    let beta = Expression::symbol("beta"); // Mixer angle

    println!("QAOA Ansatz Parameters:");
    println!("  γ (gamma) - Phase separation angle");
    println!("  β (beta)  - Mixer angle\n");

    // Build the cost Hamiltonian for MaxCut
    // H_C = Σ_{(i,j) ∈ E} (1 - Z_i Z_j) / 2
    println!("Cost Hamiltonian:");
    println!("  H_C = Σ (1 - Z_i Z_j) / 2 over all edges\n");

    // Build symbolic cost function expectation value
    // For a single edge (i,j): ⟨Z_i Z_j⟩ contributes to the cost
    let n_qubits = 4;

    // Construct the phase separator U_C(γ) = exp(-i γ H_C)
    // For each edge: exp(-i γ (1-Z_i Z_j)/2) = cos(γ/2) I - i sin(γ/2) Z_i Z_j
    println!("Phase Separator Circuit:");
    println!("  U_C(gamma) = Pi_edges exp(-i gamma Z_i Z_j / 2)");
    println!("             = Pi_edges [CNOT_ij . Rz(gamma)_j . CNOT_ij]\n");

    // Build the mixer U_B(β) = exp(-i β H_B) where H_B = Σ X_i
    // This is just individual Rx rotations: Π_i Rx(2β)
    println!("Mixer Circuit:");
    println!("  U_B(β) = Π_i Rx(2β)_i\n");

    // Symbolic representation of Rx(2*beta)
    let two_beta = Expression::int(2) * beta.clone();
    let rx_2beta = matrix::rx(&two_beta);

    println!("Mixer gate Rx(2β):");
    println!("  [[cos(β), -i·sin(β)],");
    println!("   [-i·sin(β), cos(β)]]\n");

    // Symbolic energy expectation value (simplified for 2 qubits)
    // For uniform superposition input:
    // E(γ, β) ≈ Σ_{edges} sin(4β) sin(γ) / 2
    let n_edges = edges.len() as f64;
    let half = Expression::float(0.5)?;

    let sin_4beta = trig::sin(&(Expression::int(4) * beta.clone()));
    let sin_gamma = trig::sin(&gamma);
    let cos_gamma = trig::cos(&gamma);

    // Simplified energy expression for p=1 QAOA
    let energy_contribution = half * sin_4beta.clone() * sin_gamma;

    println!("Approximate Energy (p=1 QAOA):");
    println!("  E(γ,β) ≈ (n_edges/2) · sin(4β) · sin(γ)");
    println!("  where n_edges = {n_edges}\n");

    // Compute symbolic gradients
    let grad_gamma = energy_contribution.diff(&gamma);
    let grad_beta = energy_contribution.diff(&beta);

    println!("Symbolic Gradients:");
    println!("  ∂E/∂γ = 0.5 · sin(4β) · cos(γ)");
    println!("  ∂E/∂β = 2 · cos(4β) · sin(γ)\n");

    // Numerical optimization
    println!("=== Gradient Descent Optimization ===\n");

    let mut gamma_val = 0.5;
    let mut beta_val = 0.3;
    let learning_rate = 0.1;
    let n_iterations = 30;

    println!("Initial: γ = {gamma_val:.4}, β = {beta_val:.4}");

    for iteration in 0..n_iterations {
        let mut values = HashMap::new();
        values.insert("gamma".to_string(), gamma_val);
        values.insert("beta".to_string(), beta_val);

        let current_energy = energy_contribution.eval(&values)?;
        let grad_g = grad_gamma.eval(&values)?;
        let grad_b = grad_beta.eval(&values)?;

        if iteration % 10 == 0 || iteration == n_iterations - 1 {
            println!(
                "Iter {iteration:2}: γ = {gamma_val:.4}, β = {beta_val:.4}, E = {current_energy:.6}"
            );
        }

        // Gradient ascent (maximize cut)
        gamma_val += learning_rate * grad_g;
        beta_val += learning_rate * grad_b;
    }

    println!("\n=== Results ===\n");
    println!("Optimal γ = {gamma_val:.6}");
    println!("Optimal β = {beta_val:.6}");

    let mut final_values = HashMap::new();
    final_values.insert("gamma".to_string(), gamma_val);
    final_values.insert("beta".to_string(), beta_val);
    let final_energy = energy_contribution.eval(&final_values)?;

    println!("Maximum E = {final_energy:.6}");
    println!("\nNote: Full QAOA would use this to construct a bitstring");
    println!("      that approximates the maximum cut.\n");

    // Demonstrate symbolic simplification
    println!("=== Symbolic Simplification ===\n");

    let complex_expr = sin_4beta + Expression::zero();
    let simplified = simplify::simplify(&complex_expr);
    println!("sin(4β) + 0 simplifies to: {simplified}");

    let cos_squared = trig::cos(&gamma) * trig::cos(&gamma);
    let sin_squared = trig::sin(&gamma) * trig::sin(&gamma);
    let pythagorean = cos_squared + sin_squared;
    println!("cos²(γ) + sin²(γ) = {pythagorean}");
    println!("(Note: Full trigonometric simplification requires additional rules)");

    Ok(())
}
