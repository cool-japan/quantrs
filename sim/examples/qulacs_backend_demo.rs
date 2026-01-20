#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Qulacs-inspired backend demonstration
//!
//! This example shows how to use the Qulacs-style high-performance backend
//! for quantum simulation.

use quantrs2_sim::prelude::*;
use scirs2_core::Float;

fn main() -> Result<()> {
    println!("=== Qulacs-Style Backend Demo ===\n");

    // Create a 3-qubit quantum state
    let mut state = QulacsStateVector::new(3)?;
    println!("Created 3-qubit state: |000⟩");
    println!("Initial norm: {:.10}\n", state.norm_squared());

    // Apply Hadamard to all qubits to create superposition
    println!("Applying Hadamard gates to all qubits...");
    for qubit in 0..3 {
        qulacs_gates::hadamard(&mut state, qubit)?;
    }
    println!("State after Hadamards: Equal superposition");
    println!("Norm: {:.10}\n", state.norm_squared());

    // Create a GHZ state: (|000⟩ + |111⟩) / √2
    println!("Creating GHZ state...");
    let mut ghz_state = QulacsStateVector::new(3)?;
    qulacs_gates::hadamard(&mut ghz_state, 0)?;
    qulacs_gates::cnot(&mut ghz_state, 0, 1)?;
    qulacs_gates::cnot(&mut ghz_state, 1, 2)?;

    println!("GHZ state amplitudes:");
    for i in 0..8 {
        let amp = ghz_state.amplitudes()[i];
        if amp.norm() > 1e-10 {
            println!("  |{i:03b}⟩: {amp:.6}");
        }
    }
    println!("Norm: {:.10}\n", ghz_state.norm_squared());

    // Test Bell state
    println!("Creating Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2...");
    let mut bell_state = QulacsStateVector::new(2)?;
    qulacs_gates::hadamard(&mut bell_state, 0)?;
    qulacs_gates::cnot(&mut bell_state, 0, 1)?;

    println!("Bell state amplitudes:");
    for i in 0..4 {
        let amp = bell_state.amplitudes()[i];
        if amp.norm() > 1e-10 {
            println!("  |{i:02b}⟩: {amp:.6}");
        }
    }
    println!("Norm: {:.10}\n", bell_state.norm_squared());

    // Test inner product
    let mut state1 = QulacsStateVector::new(2)?;
    let mut state2 = QulacsStateVector::new(2)?;

    qulacs_gates::hadamard(&mut state1, 0)?;
    qulacs_gates::hadamard(&mut state2, 0)?;

    let inner = state1.inner_product(&state2)?;
    println!("Inner product ⟨H|0⟩|H|0⟩⟩ = {inner:.6}");
    println!("Expected: 1.0, Got: {:.10}\n", inner.norm());

    // Test Pauli gates
    println!("Testing Pauli gates...");
    let mut pauli_state = QulacsStateVector::new(1)?;

    qulacs_gates::pauli_x(&mut pauli_state, 0)?;
    println!("After X gate: |1⟩");
    println!("  Amplitude[1]: {:.6}", pauli_state.amplitudes()[1]);

    qulacs_gates::pauli_x(&mut pauli_state, 0)?;
    println!("After second X gate (back to |0⟩): |0⟩");
    println!("  Amplitude[0]: {:.6}", pauli_state.amplitudes()[0]);

    qulacs_gates::pauli_y(&mut pauli_state, 0)?;
    println!("After Y gate from |0⟩:");
    println!("  Amplitude[1]: {:.6}", pauli_state.amplitudes()[1]);

    qulacs_gates::pauli_z(&mut pauli_state, 0)?;
    println!("After Z gate:");
    println!("  Amplitude[1]: {:.6}\n", pauli_state.amplitudes()[1]);

    // Test rotation gates
    println!("=== Testing Rotation Gates ===\n");

    let mut rot_state = QulacsStateVector::new(1)?;

    // RY(π/2) creates equal superposition
    qulacs_gates::ry(&mut rot_state, 0, std::f64::consts::PI / 2.0)?;
    println!("After RY(π/2):");
    println!("  |0⟩: {:.6}", rot_state.amplitudes()[0]);
    println!("  |1⟩: {:.6}", rot_state.amplitudes()[1]);
    println!("  Norm: {:.10}\n", rot_state.norm_squared());

    // RX rotation
    let mut rx_state = QulacsStateVector::new(1)?;
    qulacs_gates::rx(&mut rx_state, 0, std::f64::consts::PI / 4.0)?;
    println!("After RX(π/4):");
    println!("  |0⟩: {:.6}", rx_state.amplitudes()[0]);
    println!("  |1⟩: {:.6}", rx_state.amplitudes()[1]);
    println!("  Norm: {:.10}\n", rx_state.norm_squared());

    // U3 universal gate
    let mut u3_state = QulacsStateVector::new(1)?;
    qulacs_gates::u3(
        &mut u3_state,
        0,
        std::f64::consts::PI / 3.0,
        std::f64::consts::PI / 6.0,
        std::f64::consts::PI / 4.0,
    )?;
    println!("After U3(π/3, π/6, π/4):");
    println!("  |0⟩: {:.6}", u3_state.amplitudes()[0]);
    println!("  |1⟩: {:.6}", u3_state.amplitudes()[1]);
    println!("  Norm: {:.10}\n", u3_state.norm_squared());

    // Test measurement operations
    println!("=== Testing Measurement Operations ===\n");

    // Probability calculation
    println!("Probability calculation on Bell state:");
    let mut bell = QulacsStateVector::new(2)?;
    qulacs_gates::hadamard(&mut bell, 0)?;
    qulacs_gates::cnot(&mut bell, 0, 1)?;

    let prob0 = bell.probability_zero(0)?;
    let prob1 = bell.probability_one(0)?;
    println!("  P(qubit 0 = |0⟩) = {prob0:.6}");
    println!("  P(qubit 0 = |1⟩) = {prob1:.6}");
    println!("  Sum = {:.6}\n", prob0 + prob1);

    // Measurement with collapse
    println!("Measurement with state collapse:");
    let mut measure_state = QulacsStateVector::new(1)?;
    qulacs_gates::hadamard(&mut measure_state, 0)?;
    println!("  Before measurement: |+⟩ = (|0⟩ + |1⟩) / √2");

    let outcome = measure_state.measure(0)?;
    println!(
        "  Measurement outcome: |{}⟩",
        if outcome { "1" } else { "0" }
    );
    println!(
        "  After collapse: amplitude[{}] = {:.6}\n",
        if outcome { "1" } else { "0" },
        measure_state.amplitudes()[usize::from(outcome)]
    );

    // Sampling without collapse
    println!("Sampling Bell state (100 shots, no collapse):");
    let mut bell_sample = QulacsStateVector::new(2)?;
    qulacs_gates::hadamard(&mut bell_sample, 0)?;
    qulacs_gates::cnot(&mut bell_sample, 0, 1)?;

    let samples = bell_sample.sample(100)?;
    let mut count_00 = 0;
    let mut count_11 = 0;
    for sample in &samples {
        if !sample[0] && !sample[1] {
            count_00 += 1;
        } else if sample[0] && sample[1] {
            count_11 += 1;
        }
    }
    println!(
        "  |00⟩: {} shots ({:.1}%)",
        count_00,
        100.0 * count_00 as f64 / 100.0
    );
    println!(
        "  |11⟩: {} shots ({:.1}%)",
        count_11,
        100.0 * count_11 as f64 / 100.0
    );
    println!("  State still intact: {:.6}\n", bell_sample.norm_squared());

    // Get counts (histogram)
    println!("Measurement histogram (1000 shots):");
    let mut hist_state = QulacsStateVector::new(2)?;
    qulacs_gates::hadamard(&mut hist_state, 0)?;
    qulacs_gates::cnot(&mut hist_state, 0, 1)?;

    let counts = hist_state.get_counts(1000)?;
    for (bitstring, count) in &counts {
        let binary_str: String = bitstring
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        println!(
            "  |{}⟩: {} counts ({:.1}%)",
            binary_str,
            count,
            100.0 * *count as f64 / 1000.0
        );
    }
    println!();

    // Sample specific qubits
    println!("Sampling specific qubits:");
    let mut three_qubit = QulacsStateVector::new(3)?;
    qulacs_gates::hadamard(&mut three_qubit, 0)?;
    qulacs_gates::hadamard(&mut three_qubit, 1)?;
    qulacs_gates::hadamard(&mut three_qubit, 2)?;

    let qubit_samples = three_qubit.sample_qubits(&[0, 2], 50)?;
    println!("  Sampling qubits [0, 2] from |+++⟩ state (50 shots):");
    let mut sample_counts = std::collections::HashMap::new();
    for sample in &qubit_samples {
        *sample_counts.entry(sample.clone()).or_insert(0) += 1;
    }
    for (bitstring, count) in &sample_counts {
        let binary_str: String = bitstring
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        println!("    |{binary_str}⟩: {count} times");
    }
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}
