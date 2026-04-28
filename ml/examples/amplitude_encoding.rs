//! Amplitude Encoding — Classical Data into Quantum Amplitudes
//!
//! Amplitude encoding maps a classical vector x ∈ ℝ^d to a quantum state
//! |ψ⟩ = Σ_i (x_i / ‖x‖) |i⟩
//!
//! This represents d-dimensional data in only ⌈log₂(d)⌉ qubits, achieving
//! an exponential reduction in storage compared to classical representations.
//!
//! This example:
//!   1. Encodes several classical vectors into quantum amplitudes
//!   2. Verifies normalisation (‖|ψ⟩‖ = 1)
//!   3. Computes quantum fidelity between similar/different vectors
//!   4. Shows inner product calculation (quantum dot product)
//!
//! Run with:
//!   cargo run --example amplitude_encoding -p quantrs2-ml --all-features

use quantrs2_ml::error::Result;
use quantrs2_ml::utils::encoding::amplitude_encode;
use scirs2_core::ndarray::{array, Array1};
use scirs2_core::Complex64;

fn main() -> Result<()> {
    println!("=== Amplitude Encoding: Classical Data → Quantum State ===\n");

    // ---- Example 1: Simple 4-dimensional vector → 2 qubits ----
    println!("--- Example 1: 4D vector (2 qubits) ---");
    let v1 = array![1.0f64, 2.0, 3.0, 4.0];
    let norm_v1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();

    let encoded1 = amplitude_encode(&v1)?;
    let norm_enc1: f64 = encoded1.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

    println!("Input vector : {:?}", v1.as_slice().expect("slice"));
    println!("Input norm   : {norm_v1:.6}");
    println!("Encoded state:");
    for (i, &amp) in encoded1.iter().enumerate() {
        println!(
            "  |{i:02b}⟩ : amplitude = {:+.4}{:+.4}i  (prob = {:.6})",
            amp.re, amp.im, amp.norm_sqr()
        );
    }
    println!("Encoded norm : {norm_enc1:.10}  (should be 1.0)");

    assert!(
        (norm_enc1 - 1.0).abs() < 1e-10,
        "Encoded state must be normalised, got norm={norm_enc1}"
    );

    // ---- Example 2: 8-dimensional image patch → 3 qubits ----
    println!("\n--- Example 2: 8D image patch (3 qubits) ---");
    // Simulating an 8-pixel grayscale image patch
    let image_patch = array![0.1f64, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4];
    let encoded2 = amplitude_encode(&image_patch)?;
    let norm_enc2: f64 = encoded2.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

    println!("Image patch (8 pixels): {:?}", image_patch.as_slice().expect("slice"));
    println!("Quantum state norms²  :");
    for (i, &amp) in encoded2.iter().enumerate() {
        let bar = "#".repeat((amp.norm_sqr() * 20.0).round() as usize);
        println!("  |{i:03b}⟩ : {:.4}  {bar}", amp.norm_sqr());
    }
    println!("Norm check           : {norm_enc2:.10}  (should be 1.0)");

    assert!((norm_enc2 - 1.0).abs() < 1e-10, "Norm must be 1.0");

    // ---- Example 3: Quantum inner product (dot product kernel) ----
    // |⟨ψ_a|ψ_b⟩|² = (a·b)² / (‖a‖² ‖b‖²) — quantum kernel
    println!("\n--- Example 3: Quantum Inner Product ---");

    let a = array![1.0f64, 0.0, 0.0, 0.0]; // basis vector e1
    let b = array![1.0f64, 0.0, 0.0, 0.0]; // identical → fidelity = 1
    let c = array![0.0f64, 1.0, 0.0, 0.0]; // orthogonal → fidelity = 0
    let d = array![1.0f64, 1.0, 0.0, 0.0]; // 45° angle → fidelity = 0.5

    let enc_a = amplitude_encode(&a)?;
    let enc_b = amplitude_encode(&b)?;
    let enc_c = amplitude_encode(&c)?;
    let enc_d = amplitude_encode(&d)?;

    let fidelity_ab = quantum_fidelity(&enc_a, &enc_b);
    let fidelity_ac = quantum_fidelity(&enc_a, &enc_c);
    let fidelity_ad = quantum_fidelity(&enc_a, &enc_d);

    println!("  ⟨e1|e1⟩²  = {fidelity_ab:.6}  (expected 1.0 — identical)");
    println!("  ⟨e1|e2⟩²  = {fidelity_ac:.6}  (expected 0.0 — orthogonal)");
    println!("  ⟨e1|d ⟩²  = {fidelity_ad:.6}  (expected 0.5 — 45°)");

    assert!((fidelity_ab - 1.0).abs() < 1e-10, "Identical vectors: fidelity=1");
    assert!(fidelity_ac < 1e-10, "Orthogonal vectors: fidelity=0");
    assert!((fidelity_ad - 0.5).abs() < 0.01, "45° angle: fidelity≈0.5");

    // ---- Example 4: Zero vector error handling ----
    println!("\n--- Example 4: Error handling (zero vector) ---");
    let zero = array![0.0f64, 0.0, 0.0, 0.0];
    match amplitude_encode(&zero) {
        Err(e) => println!("  Zero vector rejected: {e}  ✓"),
        Ok(_) => panic!("Should have rejected zero vector"),
    }

    // ---- Summary ----
    println!("\n=== Summary ===");
    println!("  d=4  → 2 qubits (log₂(4) = 2)");
    println!("  d=8  → 3 qubits (log₂(8) = 3)");
    println!("  d=2^n data encoded in n qubits — exponential compression");
    println!("  Quantum inner product computable in O(1) measurement shots");

    println!("\nAll checks passed — OK");

    Ok(())
}

/// Compute quantum fidelity |⟨ψ_a|ψ_b⟩|² between two encoded states
fn quantum_fidelity(psi_a: &Array1<Complex64>, psi_b: &Array1<Complex64>) -> f64 {
    // ⟨ψ_a|ψ_b⟩ = Σ_i conj(ψ_a[i]) * ψ_b[i]
    let inner: Complex64 = psi_a
        .iter()
        .zip(psi_b.iter())
        .map(|(&a, &b)| a.conj() * b)
        .sum();
    inner.norm_sqr()
}
