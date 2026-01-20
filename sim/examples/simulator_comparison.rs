#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Comparison between different quantum simulators
//!
//! This example demonstrates the trade-offs between:
//! - Qulacs-style state-vector simulator (QulacsStateVector)
//! - Stabilizer/Clifford simulator (StabilizerSimulator)

use quantrs2_sim::prelude::*;
use quantrs2_sim::stabilizer::{StabilizerGate, StabilizerSimulator};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== QuantRS2 Simulator Comparison ===\n");

    // Test 1: Small Bell state - both simulators
    println!("Test 1: Bell State (2 qubits)");
    println!("─────────────────────────────────────");
    compare_bell_state()?;
    println!();

    // Test 2: Medium GHZ state
    println!("Test 2: GHZ State (10 qubits)");
    println!("─────────────────────────────────────");
    compare_ghz_state(10)?;
    println!();

    // Test 3: Large Clifford circuit - stabilizer only
    println!("Test 3: Large Clifford Circuit (1000 qubits)");
    println!("─────────────────────────────────────");
    println!("State-vector simulator: IMPOSSIBLE (would need ~10^301 bytes)");

    let start = Instant::now();
    let mut stab_sim = StabilizerSimulator::new(1000);
    // Apply Hadamard to all qubits
    for i in 0..1000 {
        stab_sim.apply_gate(StabilizerGate::H(i))?;
    }
    // Apply CNOT chain
    for i in 0..999 {
        stab_sim.apply_gate(StabilizerGate::CNOT(i, i + 1))?;
    }
    let duration = start.elapsed();
    println!("Stabilizer simulator: {duration:.2?}");
    println!("Memory usage: ~4 MB (stabilizer) vs IMPOSSIBLE (state-vector)");
    println!();

    // Test 4: Deep circuit comparison
    println!("Test 4: Deep Circuit (20 qubits, 100 layers)");
    println!("─────────────────────────────────────");
    compare_deep_circuit(20, 100)?;
    println!();

    // Summary
    println!("=== Summary ===");
    println!(
        "
When to use each simulator:

QulacsStateVector (State-vector):
  ✓ Universal quantum circuits (any gates)
  ✓ Need exact amplitudes
  ✓ Small systems (<25 qubits)
  ✓ Variational algorithms (VQE, QAOA)
  ✗ Limited to ~30 qubits max
  ✗ Exponential memory: O(2^n)
  ✗ Slow for large systems

StabilizerSimulator (Clifford):
  ✓ Clifford-only circuits
  ✓ Large systems (1M+ qubits possible)
  ✓ Error correction codes
  ✓ Randomized benchmarking
  ✓ Polynomial memory: O(n²)
  ✗ Clifford gates only (no T, RZ, etc.)
  ✗ No amplitude access
  ✗ Cannot simulate universal circuits

Recommendation:
- Use state-vector for <20 qubits or non-Clifford gates
- Use stabilizer for >20 qubits with Clifford-only circuits
- Use stabilizer for error correction simulations
    "
    );

    Ok(())
}

fn compare_bell_state() -> Result<()> {
    // State-vector version
    let start = Instant::now();
    let mut qulacs = QulacsStateVector::new(2)?;
    qulacs_gates::hadamard(&mut qulacs, 0)?;
    qulacs_gates::cnot(&mut qulacs, 0, 1)?;
    let sv_duration = start.elapsed();

    println!("State-vector (Qulacs): {sv_duration:.2?}");
    println!(
        "  Amplitudes: |00⟩={:.4}, |11⟩={:.4}",
        qulacs.amplitudes()[0].norm(),
        qulacs.amplitudes()[3].norm()
    );

    // Stabilizer version
    let start = Instant::now();
    let mut stabilizer = StabilizerSimulator::new(2);
    stabilizer.apply_gate(StabilizerGate::H(0))?;
    stabilizer.apply_gate(StabilizerGate::CNOT(0, 1))?;
    let stab_duration = start.elapsed();

    println!("Stabilizer (Clifford): {stab_duration:.2?}");
    let stabs = stabilizer.get_stabilizers();
    println!("  Stabilizers: {}, {}", stabs[0], stabs[1]);

    println!("  → Similar performance for small systems");

    Ok(())
}

fn compare_ghz_state(num_qubits: usize) -> Result<()> {
    // State-vector version
    let start = Instant::now();
    let mut qulacs = QulacsStateVector::new(num_qubits)?;
    qulacs_gates::hadamard(&mut qulacs, 0)?;
    for i in 0..num_qubits - 1 {
        qulacs_gates::cnot(&mut qulacs, i, i + 1)?;
    }
    let sv_duration = start.elapsed();
    let sv_memory = (1 << num_qubits) * 16; // 16 bytes per complex number

    println!("State-vector (Qulacs): {sv_duration:.2?}");
    println!("  Memory: {} KB", sv_memory / 1024);
    println!("  Norm: {:.6}", qulacs.norm_squared());

    // Stabilizer version
    let start = Instant::now();
    let mut stabilizer = StabilizerSimulator::new(num_qubits);
    stabilizer.apply_gate(StabilizerGate::H(0))?;
    for i in 0..num_qubits - 1 {
        stabilizer.apply_gate(StabilizerGate::CNOT(i, i + 1))?;
    }
    let stab_duration = start.elapsed();
    let stab_memory = num_qubits * num_qubits * 2; // 2 bits per entry (X and Z)

    println!("Stabilizer (Clifford): {stab_duration:.2?}");
    println!("  Memory: ~{} bytes", stab_memory / 8);

    let speedup = sv_duration.as_nanos() as f64 / stab_duration.as_nanos() as f64;
    println!("  → Stabilizer is {speedup:.1}x faster");
    println!(
        "  → Stabilizer uses {}x less memory",
        sv_memory / (stab_memory / 8)
    );

    Ok(())
}

fn compare_deep_circuit(num_qubits: usize, depth: usize) -> Result<()> {
    // State-vector version
    let start = Instant::now();
    let mut qulacs = QulacsStateVector::new(num_qubits)?;

    for layer in 0..depth {
        if layer % 2 == 0 {
            // Single-qubit layer
            for q in 0..num_qubits {
                qulacs_gates::hadamard(&mut qulacs, q)?;
            }
        } else {
            // Two-qubit layer
            for q in (layer % 2..num_qubits - 1).step_by(2) {
                qulacs_gates::cnot(&mut qulacs, q, q + 1)?;
            }
        }
    }
    let sv_duration = start.elapsed();

    println!("State-vector (Qulacs): {sv_duration:.2?}");

    // Stabilizer version
    let start = Instant::now();
    let mut stabilizer = StabilizerSimulator::new(num_qubits);

    for layer in 0..depth {
        if layer % 2 == 0 {
            // Single-qubit layer
            for q in 0..num_qubits {
                stabilizer.apply_gate(StabilizerGate::H(q))?;
            }
        } else {
            // Two-qubit layer
            for q in (layer % 2..num_qubits - 1).step_by(2) {
                stabilizer.apply_gate(StabilizerGate::CNOT(q, q + 1))?;
            }
        }
    }
    let stab_duration = start.elapsed();

    println!("Stabilizer (Clifford): {stab_duration:.2?}");

    let speedup = sv_duration.as_nanos() as f64 / stab_duration.as_nanos() as f64;
    println!("  → Stabilizer is {speedup:.1}x faster for deep circuits");

    Ok(())
}
