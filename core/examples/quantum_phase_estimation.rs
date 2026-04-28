//! Quantum Phase Estimation (QPE) — Estimating the phase of RZ(π/4)
//!
//! QPE estimates the eigenvalue e^{2πiφ} of a unitary U given an eigenstate |u⟩.
//!
//! For U = RZ(π/4), the eigenphases are:
//!   |0⟩ → e^{-iπ/8}  (φ = -1/16 mod 1 = 15/16)
//!   |1⟩ → e^{+iπ/8}  (φ = +1/16)
//!
//! We use t=3 precision qubits, so the readout register encodes φ ∈ {0/8, 1/8, ..., 7/8}.
//! With the eigenstate |1⟩ and φ = 1/16, the closest integer is 0 (i.e., 0/8 ≈ 0 = 0.0625).
//!
//! Circuit sketch (t=3 precision qubits + 1 eigenstate qubit = 4 qubits total):
//!
//!   q0 (prec): ──H─── C-U^1 ────────────── QFT†
//!   q1 (prec): ──H─── C-U^2 ────────────── QFT†
//!   q2 (prec): ──H─── C-U^4 ────────────── QFT†
//!   q3 (eig):  ──X─── (eigenstate |1⟩) ───────
//!
//! This example uses only QuantRS2-Core primitives for educational clarity.
//!
//! Run with:
//!   cargo run --example quantum_phase_estimation -p quantrs2-core --all-features

use quantrs2_core::{
    error::QuantRS2Result,
    gate::{single::Hadamard, GateOp},
    qubit::QubitId,
};
use scirs2_core::Complex64;

const T: usize = 3; // precision qubits
const N: usize = T + 1; // total qubits (precision + eigenstate)
const DIM: usize = 1 << N; // 16

fn main() -> QuantRS2Result<()> {
    println!("=== Quantum Phase Estimation: U = RZ(π/4), eigenstate |1⟩ ===\n");

    // The phase we want to estimate
    let theta = std::f64::consts::PI / 4.0; // RZ(π/4) acts as e^{+iπ/8} on |1⟩
    let true_phase = 1.0 / 16.0; // φ = π/8 / (2π) = 1/16
    let n_states = 1u32 << T; // 8

    println!("Unitary   : U = RZ(π/4)");
    println!("Eigenstate: |1⟩ with eigenvalue e^{{iπ/8}}");
    println!("True phase: φ = 1/16 = {true_phase:.6}");
    println!("Precision qubits: t = {T}  → resolution = 1/{n_states}\n");

    // ---- State preparation ----
    // Precision qubits (q0, q1, q2) in |0⟩; eigenstate qubit (q3) in |1⟩
    let mut state = [Complex64::new(0.0, 0.0); DIM];
    // Set the eigenstate: q3 = |1⟩, precision qubits = |000⟩
    // State index: |000|1⟩ = 0b0001 = 1
    state[1] = Complex64::new(1.0, 0.0);

    // ---- Step 1: Hadamard on all precision qubits ----
    let h = Hadamard { target: QubitId(0) };
    let h_mat = h.matrix()?;
    for q in 0..T {
        apply_single_qubit(&mut state, &h_mat, q);
    }

    // ---- Step 2: Controlled-U^{2^k} on eigenstate qubit (q3) ----
    // U = RZ(θ) acts as multiplication by e^{iθ/2} on |1⟩
    // Controlled-U^{2^k}: if precision qubit k = 1, apply phase shift to |...1⟩ components
    for k in 0..T {
        let angle = theta * (1u32 << (T - 1 - k)) as f64; // U^{2^(T-1-k)} = RZ(2^(T-1-k)*θ)
        apply_controlled_rz_on_eigenstate(&mut state, k, angle);
    }

    // ---- Step 3: Inverse QFT on precision qubits ----
    apply_iqft(&mut state, T)?;

    // ---- Measure precision register ----
    // Compute probability distribution over the 2^T = 8 precision states
    // (marginalised over the eigenstate qubit)
    let mut probs = [0.0f64; 1 << T];
    for (i, amp) in state.iter().enumerate() {
        let prec_reg = i >> 1; // top T bits (precision register; eigenstate is LSB)
        probs[prec_reg] += amp.norm_sqr();
    }

    println!("Measurement probabilities (precision register):");
    let mut max_prob = 0.0f64;
    let mut max_state = 0usize;
    for (s, &p) in probs.iter().enumerate() {
        let estimated_phase = s as f64 / n_states as f64;
        let bar_len = (p * 30.0).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  |{s:03b}⟩ → φ={estimated_phase:.4}  P={p:.4}  {bar}");
        if p > max_prob {
            max_prob = p;
            max_state = s;
        }
    }

    let estimated_phase = max_state as f64 / n_states as f64;
    println!("\nMost probable readout: |{max_state:03b}⟩ → φ ≈ {estimated_phase:.4}");
    println!("True phase           : φ = {true_phase:.4}");
    println!("Error                : {:.4}", (estimated_phase - true_phase).abs());

    // ---- Assertions ----
    // The estimated phase should be the nearest multiple of 1/8 to the true phase 1/16
    // 1/16 is equidistant between 0 and 1/8; either outcome is valid
    assert!(
        max_prob > 0.30,
        "Peak probability should be substantial, got {max_prob:.4}"
    );
    assert!(
        probs.iter().sum::<f64>() > 0.99,
        "Probabilities should sum to 1"
    );

    println!("\nOK — Phase estimation completed.");

    Ok(())
}

/// Apply RZ phase shift conditionally on precision qubit k being |1⟩
/// Only the eigenstate component (LSB = 1) is affected
fn apply_controlled_rz_on_eigenstate(state: &mut [Complex64; DIM], ctrl_qubit: usize, angle: f64) {
    // In our layout: precision qubits are q0,q1,q2 (MSB); eigenstate is q3 (LSB)
    // ctrl_qubit k corresponds to bit position T-k from LSB = 1+k  (0-indexed from LSB)
    // Wait — bit layout: index bit (N-1-q) from LSB for qubit q
    // For qubit k (precision): bit = N-1-k = 3-k
    let ctrl_bit = N - 1 - ctrl_qubit; // bit index from LSB
    let eigen_bit = 0usize; // eigenstate is qubit 3 = LSB bit 0
    let ctrl_mask = 1usize << ctrl_bit;
    let eigen_mask = 1usize << eigen_bit;

    // Phase e^{iangle/2} on |1⟩ component (RZ(angle) gives e^{-iangle/2}|0⟩, e^{iangle/2}|1⟩)
    let phase = Complex64::new(0.0, angle / 2.0).exp();

    for (i, amp) in state.iter_mut().enumerate() {
        if (i & ctrl_mask != 0) && (i & eigen_mask != 0) {
            *amp *= phase;
        }
    }
}

/// Inverse Quantum Fourier Transform on the first `t` qubits
fn apply_iqft(state: &mut [Complex64; DIM], t: usize) -> QuantRS2Result<()> {
    let h = Hadamard { target: QubitId(0) };
    let h_mat = h.matrix()?;

    // QFT inverse: from qubit t-1 down to 0
    for j in (0..t).rev() {
        // Apply inverse controlled-phase rotations
        for k in (0..j).rev() {
            let angle = -std::f64::consts::PI / (1u32 << (j - k)) as f64;
            apply_controlled_phase(state, k, j, angle);
        }
        // Apply Hadamard to qubit j
        apply_single_qubit(state, &h_mat, j);
    }

    // Bit-reverse permutation (only among precision bits)
    bit_reverse_permutation(state, t);

    Ok(())
}

/// Apply controlled phase shift: if qubit `ctrl` = |1⟩, apply phase to qubit `target` = |1⟩
fn apply_controlled_phase(state: &mut [Complex64; DIM], ctrl: usize, target: usize, angle: f64) {
    let ctrl_bit = N - 1 - ctrl;
    let tgt_bit = N - 1 - target;
    let ctrl_mask = 1usize << ctrl_bit;
    let tgt_mask = 1usize << tgt_bit;
    let phase = Complex64::new(0.0, angle).exp();

    for (i, amp) in state.iter_mut().enumerate() {
        if (i & ctrl_mask != 0) && (i & tgt_mask != 0) {
            *amp *= phase;
        }
    }
}

/// Bit-reverse permutation of the first `t` qubits (used in QFT/IQFT)
fn bit_reverse_permutation(state: &mut [Complex64; DIM], t: usize) {
    let mut visited = [false; DIM];
    for i in 0..DIM {
        if visited[i] {
            continue;
        }
        // Bit-reverse only the top `t` bits; keep the eigenstate bit intact
        let top_bits = i >> (N - t); // extract top t bits
        let bottom_bits = i & ((1 << (N - t)) - 1); // keep lower bits unchanged
        let rev_top = bit_reverse(top_bits, t);
        let j = (rev_top << (N - t)) | bottom_bits;
        if j > i {
            state.swap(i, j);
        }
        visited[i] = true;
        visited[j] = true;
    }
}

/// Reverse the lower `bits` bits of `x`
fn bit_reverse(x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    let mut val = x;
    for _ in 0..bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

/// Apply a 2×2 gate to qubit `q` (MSB = qubit 0)
fn apply_single_qubit(state: &mut [Complex64; DIM], matrix: &[Complex64], q: usize) {
    let m00 = matrix[0];
    let m01 = matrix[1];
    let m10 = matrix[2];
    let m11 = matrix[3];
    let bit = N - 1 - q;
    let mask = 1usize << bit;

    let mut i = 0usize;
    while i < DIM {
        if i & mask == 0 {
            let j = i | mask;
            let (a0, a1) = (state[i], state[j]);
            state[i] = m00 * a0 + m01 * a1;
            state[j] = m10 * a0 + m11 * a1;
        }
        i += 1;
    }
}
