//! Quantum computing symbolic computation example
#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
//!
//! This example demonstrates symbolic quantum computation including:
//! - Quantum operator symbols
//! - Commutator operations
//! - Advanced quantum operators (creation, annihilation, spin)
//! - Symbolic quantum state manipulation

use quantrs2_symengine::quantum::advanced_operators;
use quantrs2_symengine::Expression;

fn main() {
    println!("=== Quantum Symbolic Computation with QuantRS2-SymEngine ===\n");

    // Creation and annihilation operators
    println!("1. Bosonic Ladder Operators:");
    let a = advanced_operators::annihilation();
    let a_dag = advanced_operators::creation();
    let n_op = advanced_operators::number_operator();

    println!("   Annihilation operator: a = {}", a);
    println!("   Creation operator: a† = {}", a_dag);
    println!("   Number operator: n = a†a = {}", n_op);
    println!(
        "   [a, a†] = {}\n",
        advanced_operators::bosonic_commutator()
    );

    // Position and momentum operators
    println!("2. Position and Momentum Operators:");
    let x_op = advanced_operators::position_operator();
    let p_op = advanced_operators::momentum_operator();

    println!("   Position: x = (a + a†)/√2 = {}", x_op);
    println!("   Momentum: p = i(a† - a)/√2 = {}\n", p_op);

    // Spin operators
    println!("3. Spin Operators:");
    let sx = advanced_operators::spin_x();
    let sy = advanced_operators::spin_y();
    let sz = advanced_operators::spin_z();
    let s_squared = advanced_operators::spin_squared();

    println!("   Sx = {}", sx);
    println!("   Sy = {}", sy);
    println!("   Sz = {}", sz);
    println!("   S² = Sx² + Sy² + Sz² = {}\n", s_squared);

    // Spin ladder operators
    println!("4. Spin Ladder Operators:");
    let s_plus = advanced_operators::spin_raising();
    let s_minus = advanced_operators::spin_lowering();

    println!("   S+ = {}", s_plus);
    println!("   S- = {}\n", s_minus);

    // Angular momentum operators
    println!("5. Angular Momentum Operators:");
    let lx = advanced_operators::angular_momentum_x();
    let ly = advanced_operators::angular_momentum_y();
    let lz = advanced_operators::angular_momentum_z();
    let l_squared = advanced_operators::angular_momentum_squared();

    println!("   Lx = {}", lx);
    println!("   Ly = {}", ly);
    println!("   Lz = {}", lz);
    println!("   L² = {}\n", l_squared);

    // Fermionic operators
    println!("6. Fermionic Operators:");
    let c = advanced_operators::fermionic_annihilation();
    let c_dag = advanced_operators::fermionic_creation();
    let n_fermi = advanced_operators::fermionic_number_operator();

    println!("   Fermionic annihilation: c = {}", c);
    println!("   Fermionic creation: c† = {}", c_dag);
    println!("   Fermionic number operator: n = c†c = {}\n", n_fermi);

    // Special operators
    println!("7. Special Quantum Operators:");
    let alpha = Expression::symbol("alpha");
    let disp_op = advanced_operators::displacement_operator(&alpha);
    println!("   Displacement operator: D(α) = {}", disp_op);

    let zeta = Expression::symbol("zeta");
    let squeeze_op = advanced_operators::squeezing_operator(&zeta);
    println!("   Squeezing operator: S(ζ) = {}", squeeze_op);

    // Symbolic quantum state manipulation
    println!("\n8. Symbolic State Evolution:");
    let psi = Expression::symbol("ψ");
    let hamiltonian = Expression::symbol("H");
    let time = Expression::symbol("t");

    // Time evolution operator: U(t) = exp(-iHt/ℏ)
    let i = Expression::new("I");
    let hbar = Expression::symbol("ℏ");
    let evolution = Expression::new(format!("exp(-{}*{}*{}/{})", i, hamiltonian, time, hbar));

    println!("   Initial state: |{}⟩", psi);
    println!("   Hamiltonian: {}", hamiltonian);
    println!("   Time evolution: U(t) = {}", evolution);
    println!("   Evolved state: |ψ(t)⟩ = U(t)|ψ(0)⟩\n");

    // Expectation values
    println!("9. Expectation Values:");
    let observable = Expression::symbol("A");
    println!("   Observable: {}", observable);
    println!("   ⟨ψ|A|ψ⟩ = ⟨{}|{}|{}⟩", psi, observable, psi);

    // Energy eigenstates
    println!("\n10. Energy Eigenstates:");
    let n = Expression::symbol("n");
    let omega = Expression::symbol("ω");

    // Harmonic oscillator energy eigenvalues
    let energy = (n.clone() + Expression::from_f64(0.5)) * hbar.clone() * omega.clone();
    println!(
        "   Harmonic oscillator energy: E_n = (n + 1/2)ℏω = {}",
        energy
    );

    // Number state using creation operator
    let vacuum = Expression::symbol("|0⟩");
    println!("   Number state: |n⟩ = (a†)^n / √(n!) |0⟩");
    println!("   Vacuum state: {}", vacuum);

    println!("\n=== Example Complete ===");
}
