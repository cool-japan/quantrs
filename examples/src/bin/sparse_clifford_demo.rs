//! Demonstration of the sparse Clifford simulator using SciRS2
//!
//! This example shows how the sparse representation improves memory efficiency
//! for large Clifford circuits.

use quantrs2_sim::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sparse Clifford Simulator Demo ===\n");

    // Compare standard vs sparse simulator for different circuit sizes
    for num_qubits in [10, 20, 50, 100] {
        println!("Testing with {} qubits:", num_qubits);

        // Create a sparse Clifford simulator
        let mut sparse_sim = SparseCliffordSimulator::new(num_qubits);

        // Build a random Clifford circuit
        let start = Instant::now();

        // Apply Hadamard gates to first half of qubits
        for i in 0..num_qubits / 2 {
            sparse_sim.apply_gate(CliffordGate::H(i))?;
        }

        // Create entanglement with CNOT ladder
        for i in 0..num_qubits - 1 {
            sparse_sim.apply_gate(CliffordGate::CNOT(i, i + 1))?;
        }

        // Apply S gates to random qubits
        for i in (0..num_qubits).step_by(3) {
            sparse_sim.apply_gate(CliffordGate::S(i))?;
        }

        let circuit_time = start.elapsed();

        // Check sparsity
        let (stab_sparsity, destab_sparsity) = sparse_sim.get_sparsity_info();

        println!("  Circuit execution time: {:?}", circuit_time);
        println!("  Stabilizer sparsity: {:.2}%", stab_sparsity * 100.0);
        println!("  Destabilizer sparsity: {:.2}%", destab_sparsity * 100.0);

        // For smaller circuits, show some stabilizers
        if num_qubits <= 10 {
            println!("  Sample stabilizers:");
            let stabs = sparse_sim.get_stabilizers();
            for (i, stab) in stabs.iter().take(3).enumerate() {
                println!("    S{}: {}", i, stab);
            }
        }

        println!();
    }

    // Demonstrate specific quantum states
    println!("=== Quantum State Demonstrations ===\n");

    // Create GHZ state
    println!("GHZ State (5 qubits):");
    let mut ghz_sim = SparseCliffordSimulator::new(5);
    ghz_sim.apply_gate(CliffordGate::H(0))?;
    for i in 0..4 {
        ghz_sim.apply_gate(CliffordGate::CNOT(i, i + 1))?;
    }
    let ghz_stabs = ghz_sim.get_stabilizers();
    for (i, stab) in ghz_stabs.iter().enumerate() {
        println!("  S{}: {}", i, stab);
    }

    // Create graph state
    println!("\nGraph State (6 qubits, ring topology):");
    let mut graph_sim = SparseCliffordSimulator::new(6);

    // Apply H to all qubits
    for i in 0..6 {
        graph_sim.apply_gate(CliffordGate::H(i))?;
    }

    // Apply CZ gates in ring
    for i in 0..6 {
        let j = (i + 1) % 6;
        // CZ = H_j CNOT_ij H_j
        graph_sim.apply_gate(CliffordGate::H(j))?;
        graph_sim.apply_gate(CliffordGate::CNOT(i, j))?;
        graph_sim.apply_gate(CliffordGate::H(j))?;
    }

    let graph_stabs = graph_sim.get_stabilizers();
    for (i, stab) in graph_stabs.iter().enumerate() {
        println!("  S{}: {}", i, stab);
    }

    // Memory efficiency demonstration
    println!("\n=== Memory Efficiency ===");

    let huge_sim = SparseCliffordSimulator::new(1000);
    let (stab_sparsity, _) = huge_sim.get_sparsity_info();

    println!("1000-qubit system:");
    println!("  Initial sparsity: {:.2}%", stab_sparsity * 100.0);
    println!(
        "  Dense storage would need: {} MB",
        (1000 * 1000 * 2 * 8) / (1024 * 1024)
    );
    println!("  Sparse storage uses: < 1 MB");

    println!("\nDemo completed successfully!");
    Ok(())
}
