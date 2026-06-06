//! Bell Pair Creation — 2-Qubit Bell State
//!
//! Demonstrates creating the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 using
//! only QuantRS2-Core primitives: gate matrices and manual state evolution.
//!
//! The circuit is:
//!   q0: ─── H ─── ●
//!                  │
//!   q1: ─────────  X
//!
//! where H is the Hadamard gate and ● is the CNOT control.
//!
//! Run with:
//!   cargo run --example bell_pair -p quantrs2-core --all-features

use quantrs2_core::{
    error::QuantRS2Result,
    gate::{multi::CNOT, single::Hadamard, GateOp},
    qubit::QubitId,
    register::Register,
};
use scirs2_core::Complex64;

fn main() -> QuantRS2Result<()> {
    println!("=== Bell Pair (|Φ+⟩) Creation ===\n");

    // ---- Step 0: initial state |00⟩ ----
    // For 2 qubits, basis states are |00⟩, |01⟩, |10⟩, |11⟩ (indices 0,1,2,3)
    let mut state = [
        Complex64::new(1.0, 0.0), // |00⟩ amplitude = 1
        Complex64::new(0.0, 0.0), // |01⟩
        Complex64::new(0.0, 0.0), // |10⟩
        Complex64::new(0.0, 0.0), // |11⟩
    ];

    println!("Initial state: |00⟩");
    print_state(&state);

    // ---- Step 1: Apply Hadamard to qubit 0 ----
    // H on qubit 0 maps:
    //   |00⟩ → (|00⟩ + |10⟩)/√2
    //
    // Hadamard matrix: 1/√2 * [[1, 1], [1, -1]]
    // Acting on qubit 0 (most significant bit in our ordering):
    //   new[0b0j] = (old[0b0j] + old[0b1j]) / √2
    //   new[0b1j] = (old[0b0j] - old[0b1j]) / √2
    let h_gate = Hadamard { target: QubitId(0) };
    let h_matrix = h_gate.matrix()?;
    apply_single_qubit_gate_q0(&mut state, &h_matrix);

    println!("\nAfter H on qubit 0:");
    print_state(&state);

    // ---- Step 2: Apply CNOT (control=q0, target=q1) ----
    // CNOT: if control=1, flip target; if control=0, do nothing
    //   |00⟩ → |00⟩
    //   |01⟩ → |01⟩
    //   |10⟩ → |11⟩   ← flip
    //   |11⟩ → |10⟩   ← flip
    let cnot_gate = CNOT {
        control: QubitId(0),
        target: QubitId(1),
    };
    // Verify gate definition is present
    let _cnot_matrix = cnot_gate.matrix()?;

    // Apply CNOT directly (swap |10⟩ and |11⟩ amplitudes)
    state.swap(2, 3);

    println!("\nAfter CNOT(control=0, target=1):");
    print_state(&state);

    // ---- Verify Bell state properties ----
    let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
    let prob_00 = state[0].norm_sqr(); // should be 0.5
    let prob_11 = state[3].norm_sqr(); // should be 0.5
    let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();

    println!("\n=== Verification ===");
    println!("P(|00⟩) = {prob_00:.6}  (expected 0.5)");
    println!("P(|11⟩) = {prob_11:.6}  (expected 0.5)");
    println!("Norm²   = {norm:.6}  (expected 1.0)");

    // ---- Build Register and use its helpers ----
    let register = Register::<2>::with_amplitudes(state.to_vec())?;
    let ez0 = register.expectation_z(0u32)?;
    let ez1 = register.expectation_z(1u32)?;
    println!("⟨Z₀⟩    = {ez0:+.6}  (expected 0.0 — maximally uncertain)");
    println!("⟨Z₁⟩    = {ez1:+.6}  (expected 0.0 — maximally uncertain)");

    // ---- Assertions ----
    assert!((prob_00 - 0.5).abs() < 1e-10, "P(|00⟩) should be 0.5");
    assert!((prob_11 - 0.5).abs() < 1e-10, "P(|11⟩) should be 0.5");
    assert!((norm - 1.0).abs() < 1e-10, "State not normalized");
    assert!(state[1].norm_sqr() < 1e-20, "P(|01⟩) should be 0");
    assert!(state[2].norm_sqr() < 1e-20, "P(|10⟩) should be 0");
    assert!(ez0.abs() < 1e-10, "⟨Z₀⟩ should be 0 for Bell state");
    assert!(ez1.abs() < 1e-10, "⟨Z₁⟩ should be 0 for Bell state");

    println!("\nAll assertions passed — Bell state |Φ+⟩ verified!");

    Ok(())
}

/// Apply a 2×2 gate matrix to qubit 0 (MSB) of a 2-qubit state
/// New[0b0j] = M[0,0]*old[0b0j] + M[0,1]*old[0b1j]
/// New[0b1j] = M[1,0]*old[0b0j] + M[1,1]*old[0b1j]
fn apply_single_qubit_gate_q0(state: &mut [Complex64; 4], matrix: &[Complex64]) {
    // Matrix is stored row-major: [M00, M01, M10, M11]
    let m00 = matrix[0];
    let m01 = matrix[1];
    let m10 = matrix[2];
    let m11 = matrix[3];

    // j = 0 (second qubit = 0)
    let (new_00, new_10) = (
        m00 * state[0] + m01 * state[2],
        m10 * state[0] + m11 * state[2],
    );
    // j = 1 (second qubit = 1)
    let (new_01, new_11) = (
        m00 * state[1] + m01 * state[3],
        m10 * state[1] + m11 * state[3],
    );
    state[0] = new_00;
    state[1] = new_01;
    state[2] = new_10;
    state[3] = new_11;
}

/// Pretty-print the 2-qubit state vector
fn print_state(state: &[Complex64; 4]) {
    for (i, &amp) in state.iter().enumerate() {
        let prob = amp.norm_sqr();
        if prob > 1e-12 {
            println!(
                "  |{:02b}⟩: {:.4}{:+.4}i   (prob = {:.6})",
                i, amp.re, amp.im, prob
            );
        }
    }
}
