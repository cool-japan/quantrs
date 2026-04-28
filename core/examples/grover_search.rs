//! Grover's Search Algorithm — 3-Qubit Example
//!
//! Grover's algorithm finds a marked item in an unsorted database of N=2^n
//! items in O(√N) steps (vs. O(N) classically).
//!
//! For n=3 qubits (N=8 states), the optimal iteration count is ⌊π/4·√N⌋ = 2.
//! We search for the state |101⟩ (decimal 5).
//!
//! The algorithm consists of:
//!   1. Hadamard transform: H⊗n creates uniform superposition
//!   2. Oracle Uω: marks |ω⟩ with a phase flip (|ω⟩ → -|ω⟩)
//!   3. Diffusion operator: 2|ψ⟩⟨ψ| - I  (inversion about average)
//!   4. Repeat steps 2-3 for k ≈ π/4·√N times
//!
//! This example uses only QuantRS2-Core primitives (gate matrices + manual
//! state-vector evolution) to remain dependency-free.
//!
//! Run with:
//!   cargo run --example grover_search -p quantrs2-core --all-features

use quantrs2_core::{
    error::QuantRS2Result,
    gate::{single::Hadamard, GateOp},
    qubit::QubitId,
    register::Register,
};
use scirs2_core::Complex64;

const N_QUBITS: usize = 3;
const DIM: usize = 1 << N_QUBITS; // 8

fn main() -> QuantRS2Result<()> {
    println!("=== Grover's Search (3 qubits, target = |101⟩) ===\n");

    let target = 0b101usize; // = 5
    println!("Target state: |{target:03b}⟩ (decimal {target})");
    println!("Search space: {DIM} states");
    println!("Optimal iterations: {}", optimal_iterations(DIM));
    println!();

    // ---- Step 1: Uniform superposition via H⊗3 ----
    let mut state = [Complex64::new(0.0, 0.0); DIM];
    state[0] = Complex64::new(1.0, 0.0); // |000⟩

    apply_hadamard_all(&mut state)?;
    println!("After H⊗3 (uniform superposition):");
    print_probabilities(&state);

    // ---- Grover iterations ----
    let k = optimal_iterations(DIM);
    for iter in 1..=k {
        // Oracle: flip phase of |target⟩
        apply_oracle(&mut state, target);

        // Diffusion operator: 2|ψ⟩⟨ψ| - I
        apply_diffusion(&mut state)?;

        println!("\nAfter iteration {iter}:");
        print_probabilities(&state);
    }

    // ---- Measure: find the highest-probability state ----
    let (max_idx, max_prob) = state
        .iter()
        .enumerate()
        .map(|(i, a)| (i, a.norm_sqr()))
        .fold((0, 0.0f64), |(bi, bp), (i, p)| {
            if p > bp { (i, p) } else { (bi, bp) }
        });

    println!("\n=== Result ===");
    println!("Most probable state: |{max_idx:03b}⟩ with P = {max_prob:.6}");

    // Build a Register to use expectation value helpers
    let register = Register::<N_QUBITS>::with_amplitudes(state.to_vec())?;
    for q in 0..N_QUBITS {
        let ez = register.expectation_z(q as u32)?;
        let bit = (target >> (N_QUBITS - 1 - q)) & 1; // expected bit value
        println!("  ⟨Z_{q}⟩ = {ez:+.4}  (target bit = {bit})");
    }

    // ---- Assertions ----
    assert_eq!(
        max_idx, target,
        "Grover should find |{target:03b}⟩ but found |{max_idx:03b}⟩"
    );
    // After 2 optimal iterations for N=8, the target amplitude is ~0.945
    assert!(
        max_prob > 0.90,
        "Target probability should be >0.90 for 3-qubit Grover, got {max_prob:.4}"
    );

    println!("\nOK — Grover found |{target:03b}⟩ with probability {max_prob:.4}");

    Ok(())
}

/// Number of optimal Grover iterations: ⌊π/4·√N⌋
fn optimal_iterations(n: usize) -> usize {
    let k = (std::f64::consts::PI / 4.0 * (n as f64).sqrt()).floor() as usize;
    k.max(1)
}

/// Apply H to all qubits in a 3-qubit state
fn apply_hadamard_all(state: &mut [Complex64; DIM]) -> QuantRS2Result<()> {
    let h = Hadamard { target: QubitId(0) };
    let mat = h.matrix()?;
    // Apply to each qubit
    for q in 0..N_QUBITS {
        apply_single_qubit(state, &mat, q);
    }
    Ok(())
}

/// Oracle Uω: multiply amplitude of |target⟩ by -1
fn apply_oracle(state: &mut [Complex64; DIM], target: usize) {
    state[target] = -state[target];
}

/// Grover diffusion operator: 2|ψ⟩⟨ψ| - I
/// = H⊗n (2|0⟩⟨0| - I) H⊗n
fn apply_diffusion(state: &mut [Complex64; DIM]) -> QuantRS2Result<()> {
    // Step 1: H⊗n
    let h = Hadamard { target: QubitId(0) };
    let mat = h.matrix()?;
    for q in 0..N_QUBITS {
        apply_single_qubit(state, &mat, q);
    }

    // Step 2: 2|0⟩⟨0| - I (phase flip every state except |000⟩)
    for item in state.iter_mut().skip(1) {
        *item = -*item;
    }

    // Step 3: H⊗n again
    for q in 0..N_QUBITS {
        apply_single_qubit(state, &mat, q);
    }

    Ok(())
}

/// Apply a 2×2 single-qubit gate to qubit `q` in a 3-qubit state
/// The qubit ordering is: qubit 0 = MSB, qubit 2 = LSB
fn apply_single_qubit(state: &mut [Complex64; DIM], matrix: &[Complex64], q: usize) {
    let m00 = matrix[0];
    let m01 = matrix[1];
    let m10 = matrix[2];
    let m11 = matrix[3];

    // Bit position from LSB perspective: qubit 0 → bit N-1-0 = 2
    let bit = N_QUBITS - 1 - q;
    let mask = 1usize << bit;

    let mut i = 0usize;
    while i < DIM {
        // Find pairs of states differing only in qubit q
        if i & mask == 0 {
            let j = i | mask; // partner state (qubit q = 1)
            let (a0, a1) = (state[i], state[j]);
            state[i] = m00 * a0 + m01 * a1;
            state[j] = m10 * a0 + m11 * a1;
        }
        i += 1;
    }
}

/// Print non-negligible probabilities
fn print_probabilities(state: &[Complex64; DIM]) {
    for (i, amp) in state.iter().enumerate() {
        let p = amp.norm_sqr();
        if p > 0.001 {
            let bar_len = (p * 20.0).round() as usize;
            let bar: String = "█".repeat(bar_len);
            println!("  |{i:03b}⟩ ({i:>2}): {p:.4}  {bar}");
        }
    }
}
