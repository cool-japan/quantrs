//! Quantum States Symbolic Manipulation Example
#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
//!
//! This example demonstrates symbolic manipulation of quantum states
//! and their properties.

use quantrs2_symengine::quantum::states;
use quantrs2_symengine::Expression;

fn main() {
    println!("=== Quantum States Symbolic Manipulation ===\n");

    // Computational basis states
    println!("1. Computational Basis States:");
    let ket_0 = states::ket_0();
    let ket_1 = states::ket_1();

    println!("|0⟩ = {}", ket_0);
    println!("|1⟩ = {}", ket_1);
    println!();

    // Superposition states
    println!("2. Superposition States:");
    let ket_plus = states::ket_plus();
    let ket_minus = states::ket_minus();
    let ket_i = states::ket_i();
    let ket_minus_i = states::ket_minus_i();

    println!("|+⟩ = (|0⟩ + |1⟩)/√2 = {}", ket_plus);
    println!("|-⟩ = (|0⟩ - |1⟩)/√2 = {}", ket_minus);
    println!("|i⟩ = (|0⟩ + i|1⟩)/√2 = {}", ket_i);
    println!("|-i⟩ = (|0⟩ - i|1⟩)/√2 = {}", ket_minus_i);
    println!();

    // Bell states (maximally entangled)
    println!("3. Bell States (Maximally Entangled):");
    let bell_phi_plus = states::bell_phi_plus();
    let bell_phi_minus = states::bell_phi_minus();
    let bell_psi_plus = states::bell_psi_plus();
    let bell_psi_minus = states::bell_psi_minus();

    println!("|Φ+⟩ = (|00⟩ + |11⟩)/√2 = {}", bell_phi_plus);
    println!("|Φ-⟩ = (|00⟩ - |11⟩)/√2 = {}", bell_phi_minus);
    println!("|Ψ+⟩ = (|01⟩ + |10⟩)/√2 = {}", bell_psi_plus);
    println!("|Ψ-⟩ = (|01⟩ - |10⟩)/√2 = {}", bell_psi_minus);
    println!();

    // Bloch sphere parameterization
    println!("4. Bloch Sphere Parameterization:");
    let theta = Expression::symbol("theta");
    let phi = Expression::symbol("phi");

    let bloch_state = states::bloch_state(&theta, &phi);
    println!(
        "|ψ(θ,φ)⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩ = {}",
        bloch_state
    );
    println!();

    // Special cases on Bloch sphere
    println!("5. Special Cases on Bloch Sphere:");

    // |0⟩: theta = 0
    let theta_0 = Expression::from(0);
    let state_0 = states::bloch_state(&theta_0, &phi);
    println!("θ=0: |ψ⟩ = {}", state_0);

    // |1⟩: theta = π
    let pi = Expression::new("pi");
    let state_1 = states::bloch_state(&pi, &phi);
    println!("θ=π: |ψ⟩ = {}", state_1);

    // |+⟩: theta = π/2, phi = 0
    let theta_pi2 = pi.clone() / Expression::from(2);
    let phi_0 = Expression::from(0);
    let state_plus = states::bloch_state(&theta_pi2, &phi_0);
    println!("θ=π/2, φ=0: |ψ⟩ = {}", state_plus);
    println!();

    // State normalization
    println!("6. State Properties:");
    println!("Computing ⟨ψ|ψ⟩ (inner product - should be 1):");

    // For |+⟩ state
    let ket_plus_conj = ket_plus.conjugate();
    let inner_product = ket_plus_conj.clone() * ket_plus.clone();
    println!(
        "⟨+|+⟩ = {} * {} = {}",
        ket_plus_conj, ket_plus, inner_product
    );
    println!();

    println!("=== Example Complete ===");
}
