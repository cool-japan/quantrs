//! State-Vector Simulation Demo — 8-qubit uniform superposition
//!
//! Demonstrates:
//!   1. Building an 8-qubit circuit that creates a uniform superposition |ψ⟩ = H⊗8|0⟩
//!   2. Running it with the StateVectorSimulator
//!   3. Checking probabilities (all equal to 1/256)
//!   4. Computing ⟨Z_i⟩ expectation values (should be ≈ 0 for all qubits)
//!
//! Run with:
//!   cargo run --example state_vector_demo -p quantrs2-sim --all-features

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_sim::statevector::StateVectorSimulator;

const N: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 8-Qubit State-Vector Simulation Demo ===\n");

    // ---- Build circuit: H⊗8 ----
    let mut circuit = Circuit::<N>::new();
    for qubit in 0..N {
        circuit.h(qubit)?;
    }

    println!("Circuit: H applied to all {N} qubits");
    println!("Expected state: uniform superposition over {} basis states\n", 1 << N);

    // ---- Run simulation ----
    let sim = StateVectorSimulator::new();
    let register = sim.run(&circuit)?;

    // ---- Check probabilities ----
    let probs = register.probabilities();
    let dim = 1usize << N;
    let expected_prob = 1.0 / dim as f64;

    println!("Probability for each basis state (first 8 of {dim}):");
    for (i, &p) in probs.iter().enumerate().take(8) {
        println!("  |{i:08b}⟩ : {p:.8}");
    }
    println!("  ...");

    let max_deviation = probs
        .iter()
        .map(|&p| (p - expected_prob).abs())
        .fold(0.0f64, f64::max);
    println!("\nMax deviation from 1/{dim}: {max_deviation:.2e}");
    assert!(
        max_deviation < 1e-10,
        "Probabilities should all be 1/256, max deviation = {max_deviation}"
    );

    // ---- Expectation values ⟨Z_i⟩ ----
    // For uniform superposition: ⟨Z_i⟩ = 0 for all i
    println!("\nExpectation values ⟨Z_i⟩ (should all be ≈ 0):");
    let mut max_z = 0.0f64;
    for qubit in 0..N {
        let ez = register.expectation_z(qubit as u32)?;
        println!("  ⟨Z_{qubit}⟩ = {ez:+.8}");
        max_z = max_z.max(ez.abs());
    }
    assert!(
        max_z < 1e-10,
        "⟨Z_i⟩ should be 0 for uniform superposition, max = {max_z}"
    );

    // ---- Verify normalization ----
    let norm_sq: f64 = register.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    println!("\nNorm² = {norm_sq:.15}  (should be 1.0)");
    assert!((norm_sq - 1.0).abs() < 1e-12, "State not normalized");

    // ---- Simulate a GHZ state and check entanglement ----
    println!("\n--- Bonus: 8-qubit GHZ state ---");
    let mut ghz = Circuit::<N>::new();
    ghz.h(0)?;
    for i in 0..(N - 1) {
        ghz.cnot(i, i + 1)?;
    }

    let ghz_reg = sim.run(&ghz)?;
    let p_all_zero = ghz_reg.probability(&[0u8; N])?;
    let p_all_one = ghz_reg.probability(&[1u8; N])?;

    println!("GHZ |0⟩⊗8 amplitude² : {p_all_zero:.6}");
    println!("GHZ |1⟩⊗8 amplitude² : {p_all_one:.6}");
    println!("Sum (should be ≈ 1)   : {:.6}", p_all_zero + p_all_one);

    assert!(
        (p_all_zero - 0.5).abs() < 1e-10,
        "GHZ |0..0⟩ probability should be 0.5"
    );
    assert!(
        (p_all_one - 0.5).abs() < 1e-10,
        "GHZ |1..1⟩ probability should be 0.5"
    );

    println!("\nAll checks passed — OK");

    Ok(())
}
