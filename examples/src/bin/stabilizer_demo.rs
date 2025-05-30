//! Demonstration of the stabilizer simulator for Clifford circuits

use quantrs2_core::prelude::*;
use quantrs2_sim::prelude::*;
use std::time::Instant;

fn main() {
    println!("=== Stabilizer Simulator Demo ===\n");

    // Example 1: Basic gates
    println!("Example 1: Basic Clifford Gates");
    demo_basic_gates();

    println!("\n{}\n", "=".repeat(50));

    // Example 2: Bell states
    println!("Example 2: Bell States");
    demo_bell_states();

    println!("\n{}\n", "=".repeat(50));

    // Example 3: GHZ states
    println!("Example 3: GHZ States");
    demo_ghz_states();

    println!("\n{}\n", "=".repeat(50));

    // Example 4: Quantum error correction
    println!("Example 4: Simple Error Detection");
    demo_error_detection();

    println!("\n{}\n", "=".repeat(50));

    // Example 5: Performance comparison
    println!("Example 5: Performance Comparison");
    demo_performance();
}

fn demo_basic_gates() {
    let mut sim = StabilizerSimulator::new(2);

    println!("Initial state |00⟩:");
    print_stabilizers(&sim);

    // Apply Hadamard to first qubit
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    println!("\nAfter H on qubit 0:");
    print_stabilizers(&sim);

    // Apply S gate
    sim.apply_gate(StabilizerGate::S(0)).unwrap();
    println!("\nAfter S on qubit 0:");
    print_stabilizers(&sim);

    // Apply Pauli gates
    sim.apply_gate(StabilizerGate::X(1)).unwrap();
    println!("\nAfter X on qubit 1:");
    print_stabilizers(&sim);
}

fn demo_bell_states() {
    println!("Creating all four Bell states:\n");

    // |Φ+⟩ = (|00⟩ + |11⟩)/√2
    let mut sim = StabilizerSimulator::new(2);
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    println!("|Φ+⟩ stabilizers:");
    print_stabilizers(&sim);

    // |Φ-⟩ = (|00⟩ - |11⟩)/√2
    sim.reset();
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    sim.apply_gate(StabilizerGate::Z(0)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    println!("\n|Φ-⟩ stabilizers:");
    print_stabilizers(&sim);

    // |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    sim.reset();
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    sim.apply_gate(StabilizerGate::X(1)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    println!("\n|Ψ+⟩ stabilizers:");
    print_stabilizers(&sim);

    // |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    sim.reset();
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    sim.apply_gate(StabilizerGate::X(1)).unwrap();
    sim.apply_gate(StabilizerGate::Z(0)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    println!("\n|Ψ-⟩ stabilizers:");
    print_stabilizers(&sim);
}

fn demo_ghz_states() {
    // Create GHZ states of various sizes
    for n in 3..=5 {
        let mut sim = StabilizerSimulator::new(n);

        // Create GHZ state: |000...0⟩ + |111...1⟩
        sim.apply_gate(StabilizerGate::H(0)).unwrap();
        for i in 0..(n - 1) {
            sim.apply_gate(StabilizerGate::CNOT(i, i + 1)).unwrap();
        }

        println!("{}-qubit GHZ state stabilizers:", n);
        print_stabilizers(&sim);
        println!();
    }
}

fn demo_error_detection() {
    // Simple 3-qubit repetition code for bit flip errors
    let mut sim = StabilizerSimulator::new(3);

    println!("3-qubit repetition code:");

    // Encode |0⟩ as |000⟩
    println!("Encoded |0⟩ state:");
    print_stabilizers(&sim);

    // Simulate error on qubit 1
    sim.apply_gate(StabilizerGate::X(1)).unwrap();
    println!("\nAfter bit flip on qubit 1:");
    print_stabilizers(&sim);

    // Create superposition and encode
    sim.reset();
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 2)).unwrap();
    println!("\nEncoded superposition state:");
    print_stabilizers(&sim);

    // Measure syndrome
    println!("\nMeasuring parity checks...");
    // In a real error correction scheme, we would measure
    // stabilizer generators to detect errors
}

fn demo_performance() {
    println!("Comparing stabilizer simulator performance:\n");

    // Test different circuit sizes
    for n in [10, 20, 30, 40, 50] {
        let mut sim = StabilizerSimulator::new(n);

        // Create a random Clifford circuit
        let num_gates = n * 10;
        let start = Instant::now();

        for _ in 0..num_gates {
            // Random gate selection (simplified)
            let gate_type = (start.elapsed().as_nanos() % 4) as usize;
            let qubit = ((start.elapsed().as_nanos() / 7) % n as u128) as usize;

            match gate_type {
                0 => sim.apply_gate(StabilizerGate::H(qubit)).unwrap(),
                1 => sim.apply_gate(StabilizerGate::S(qubit)).unwrap(),
                2 => {
                    let target = (qubit + 1) % n;
                    sim.apply_gate(StabilizerGate::CNOT(qubit, target)).unwrap();
                }
                _ => sim.apply_gate(StabilizerGate::X(qubit)).unwrap(),
            }
        }

        let elapsed = start.elapsed();
        println!("{} qubits, {} gates: {:?}", n, num_gates, elapsed);

        // Compare with theoretical scaling
        let ops_per_sec = (num_gates as f64) / elapsed.as_secs_f64();
        println!("  Operations per second: {:.0}", ops_per_sec);
        println!(
            "  Time per gate: {:.2} ns",
            elapsed.as_nanos() as f64 / num_gates as f64
        );
    }

    println!("\nNote: Stabilizer simulation scales as O(n²) per gate,");
    println!("compared to O(2^n) for full state vector simulation!");
}

fn print_stabilizers(sim: &StabilizerSimulator) {
    let stabs = sim.get_stabilizers();
    for (i, stab) in stabs.iter().enumerate() {
        println!("  S{}: {}", i, stab);
    }
}

// Demonstrate measurement
fn demo_measurement() {
    println!("\n=== Measurement Demo ===\n");

    let mut sim = StabilizerSimulator::new(3);

    // Create GHZ state
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(1, 2)).unwrap();

    println!("GHZ state before measurement:");
    print_stabilizers(&sim);

    // Measure first qubit
    let outcome = sim.measure(0).unwrap();
    println!("\nMeasured qubit 0: {}", if outcome { "1" } else { "0" });

    println!("\nState after measurement:");
    print_stabilizers(&sim);

    // Measure remaining qubits
    let outcome1 = sim.measure(1).unwrap();
    let outcome2 = sim.measure(2).unwrap();

    println!(
        "\nAll measurement outcomes: {}{}{}",
        if outcome { "1" } else { "0" },
        if outcome1 { "1" } else { "0" },
        if outcome2 { "1" } else { "0" }
    );
}
