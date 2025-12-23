//! Hamiltonian and Time Evolution Example
#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
//!
//! This example demonstrates symbolic manipulation of quantum Hamiltonians
//! and time evolution operators.

use quantrs2_symengine::quantum::{hamiltonian, pauli};
use quantrs2_symengine::Expression;

fn main() {
    println!("=== Hamiltonian and Time Evolution ===\n");

    // Time evolution operator
    println!("1. Time Evolution Operator:");
    let h = Expression::symbol("H");
    let t = Expression::symbol("t");
    let hbar = Expression::from(1); // Set ℏ = 1 in natural units

    let u_t = hamiltonian::time_evolution(&h, &t, Some(&hbar))
        .expect("Failed to create time evolution operator");
    println!("U(t) = e^(-iHt/ℏ) = {}", u_t);
    println!();

    // Simple Hamiltonian: H = ω σz
    println!("2. Simple Spin Hamiltonian:");
    let omega = Expression::symbol("omega");
    let sigma_z = pauli::sigma_z();
    let h_spin = &omega * sigma_z;
    println!("H = ω·σz = {}", h_spin);

    let u_spin =
        hamiltonian::time_evolution(&h_spin, &t, Some(&hbar)).expect("Failed to create U(t)");
    println!("U(t) = e^(-iω·σz·t) = {}", u_spin);
    println!();

    // Ising model Hamiltonian
    println!("3. Ising Model Hamiltonian:");
    let n_sites = 4;
    let j = Expression::symbol("J");
    let h_field = Expression::symbol("h");

    let h_ising = hamiltonian::ising_model(n_sites, &j, &h_field);
    println!("H_Ising (n=4) = {}", h_ising);
    println!();

    // Heisenberg model
    println!("4. Heisenberg Model Hamiltonian:");
    let h_heisenberg = hamiltonian::heisenberg_model(n_sites, &j);
    println!("H_Heisenberg (n=4) = {}", h_heisenberg);
    println!();

    // Pauli string for two qubits
    println!("5. Pauli String (Tensor Product):");
    let sigma_x = pauli::sigma_x();
    let sigma_y = pauli::sigma_y();

    let pauli_ops = vec![sigma_x.clone(), sigma_y.clone()];
    let pauli_string =
        hamiltonian::pauli_string(&pauli_ops).expect("Failed to create Pauli string");
    println!("σx ⊗ σy = {}", pauli_string);
    println!();

    // Time-dependent expectation value
    println!("6. Time Evolution of Expectation Value:");
    let psi_0 = Expression::symbol("psi0");
    let observable = Expression::symbol("O");

    // ⟨ψ(t)|O|ψ(t)⟩ = ⟨ψ0|U†(t)·O·U(t)|ψ0⟩
    let u_dagger = u_spin.conjugate();
    let evolved_obs = u_dagger.clone() * observable.clone() * u_spin.clone();

    println!("⟨ψ(t)|O|ψ(t)⟩ = ⟨ψ0|U†(t)·O·U(t)|ψ0⟩");
    println!("U†(t)·O·U(t) = {}", evolved_obs);
    println!();

    // Substitution for specific time
    println!("7. Evaluation at Specific Time:");
    let t_val = Expression::new("pi / omega"); // t = π/ω
    let u_at_t = u_spin.substitute(&t, &t_val);
    println!("U(π/ω) = {}", u_at_t.expand());
    println!();

    // Differentiation (Heisenberg equation of motion)
    println!("8. Heisenberg Equation of Motion:");
    println!("dO/dt = (1/iℏ)[O, H]");

    // For demonstration, compute d/dt of simple observable
    let o_t = Expression::new("O(t)");
    let do_dt = o_t.diff(&t);
    println!("dO/dt = {}", do_dt);
    println!();

    println!("=== Example Complete ===");
}
