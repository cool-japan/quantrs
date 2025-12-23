//! Quantum Gates Symbolic Manipulation Example
#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
//!
//! This example demonstrates how to use quantrs2-symengine for symbolic
//! manipulation of quantum gates and operators.

use quantrs2_symengine::quantum::{gates, operators, pauli};
use quantrs2_symengine::Expression;

fn main() {
    println!("=== Quantum Gates Symbolic Manipulation ===\n");

    // Pauli matrices
    println!("1. Pauli Matrices:");
    let sigma_x = pauli::sigma_x();
    let sigma_y = pauli::sigma_y();
    let sigma_z = pauli::sigma_z();
    let identity = pauli::identity();

    println!("σx = {}", sigma_x);
    println!("σy = {}", sigma_y);
    println!("σz = {}", sigma_z);
    println!("I = {}", identity);
    println!();

    // Commutation relations
    println!("2. Pauli Commutation Relations:");
    let comm_xy = operators::commutator(&sigma_x, &sigma_y);
    let comm_yz = operators::commutator(&sigma_y, &sigma_z);
    let comm_zx = operators::commutator(&sigma_z, &sigma_x);

    println!("[σx, σy] = {}", comm_xy.expand());
    println!("[σy, σz] = {}", comm_yz.expand());
    println!("[σz, σx] = {}", comm_zx.expand());
    println!();

    // Anticommutation relations
    println!("3. Pauli Anticommutation Relations:");
    let anticomm_xx = operators::anticommutator(&sigma_x, &sigma_x);
    let anticomm_xy = operators::anticommutator(&sigma_x, &sigma_y);

    println!("{{σx, σx}} = {}", anticomm_xx.expand());
    println!("{{σx, σy}} = {}", anticomm_xy.expand());
    println!();

    // Common quantum gates
    println!("4. Common Quantum Gates:");
    let hadamard = gates::hadamard();
    let phase = gates::phase();
    let t_gate = gates::t_gate();

    println!("Hadamard (H) = {}", hadamard);
    println!("Phase (S) = {}", phase);
    println!("T gate = {}", t_gate);
    println!();

    // Parameterized rotations
    println!("5. Rotation Gates:");
    let theta = Expression::symbol("theta");
    let phi = Expression::symbol("phi");
    let lambda = Expression::symbol("lambda");

    let rx = gates::rx(&theta);
    let ry = gates::ry(&theta);
    let rz = gates::rz(&theta);

    println!("Rx(θ) = {}", rx);
    println!("Ry(θ) = {}", ry);
    println!("Rz(θ) = {}", rz);
    println!();

    // U3 gate (general single-qubit rotation)
    println!("6. General Single-Qubit Rotation (U3):");
    let u3 = gates::u3(&theta, &phi, &lambda);
    println!("U3(θ, φ, λ) = {}", u3);
    println!();

    // Substitution example
    println!("7. Numerical Evaluation:");
    let pi = Expression::new("pi");
    let theta_val = pi.clone() / Expression::from(4); // π/4

    let rx_pi4 = rx.substitute(&theta, &theta_val);
    println!("Rx(π/4) = {}", rx_pi4);
    println!();

    // Differentiation
    println!("8. Symbolic Differentiation:");
    let rx_derivative = rx.diff(&theta);
    println!("d/dθ Rx(θ) = {}", rx_derivative);
    println!();

    // CNOT gate
    println!("9. Two-Qubit Gates:");
    let cnot = gates::cnot();
    println!("CNOT = {}", cnot);
    println!();

    println!("=== Example Complete ===");
}
