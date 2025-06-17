//! Demonstration of optimized quantum circuit simulation with Quantrs
//!
//! This example shows how to use the optimized simulator implementations
//! for efficient simulation of quantum circuits, especially with large qubit counts.

use quantrs2_circuit::builder::Circuit;
use quantrs2_circuit::builder::Simulator;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::optimized_simulator::OptimizedSimulator;
use quantrs2_sim::statevector::StateVectorSimulator;
use std::time::Instant;

fn main() {
    println!("Quantrs Optimized Simulator Demonstration");
    println!("========================================\n");

    // Try different qubit counts
    run_demo(10, "Small Circuit");
    run_demo(20, "Medium Circuit");
    run_demo(25, "Large Circuit");
}

fn run_demo(qubits: usize, name: &str) {
    println!("=== {} ({} qubits) ===", name, qubits);

    match qubits {
        // Use compile-time known qubit counts to utilize const generics
        10 => run_with_qubits::<10>(),
        20 => run_with_qubits::<20>(),
        25 => run_with_qubits::<25>(),
        n => println!("Qubit count {} not supported in this demo", n),
    }

    println!();
}

fn run_with_qubits<const N: usize>() {
    // Create a quantum circuit with N qubits
    let mut circuit = Circuit::<N>::new();

    // Apply Hadamard to all qubits (creates uniform superposition)
    for i in 0..N {
        circuit.h(QubitId::new(i as u32)).unwrap();
    }

    // Apply CNOT gates between adjacent qubits (creates entanglement)
    for i in 0..(N - 1) {
        circuit
            .cnot(QubitId::new(i as u32), QubitId::new((i + 1) as u32))
            .unwrap();
    }

    // Apply some rotations
    for i in 0..N {
        circuit
            .rz(
                QubitId::new(i as u32),
                std::f64::consts::PI / (i + 1) as f64,
            )
            .unwrap();
    }

    // Run with standard simulator first (if it's a small enough circuit)
    if N <= 20 {
        let standard_sim = StateVectorSimulator::new();
        let start = Instant::now();
        let result = standard_sim.run(&circuit).unwrap();
        let duration = start.elapsed();

        println!("Standard simulator: {:.3} seconds", duration.as_secs_f64());
        println!("  State vector size: {}", result.amplitudes().len());

        // Print the first few amplitudes
        println!("  First few amplitudes:");
        for i in 0..std::cmp::min(5, result.amplitudes().len()) {
            println!(
                "    |{:0width$b}⟩: {}",
                i,
                result.amplitudes()[i],
                width = N
            );
        }
    } else {
        println!("Standard simulator: skipped (too many qubits)");
    }

    // Run with optimized simulator
    let optimized_sim = OptimizedSimulator::new();
    let start = Instant::now();
    let result = optimized_sim.run(&circuit).unwrap();
    let duration = start.elapsed();

    println!("Optimized simulator: {:.3} seconds", duration.as_secs_f64());
    println!("  State vector size: {}", result.amplitudes().len());

    // Print the first few amplitudes
    println!("  First few amplitudes:");
    for i in 0..std::cmp::min(5, result.amplitudes().len()) {
        println!(
            "    |{:0width$b}⟩: {}",
            i,
            result.amplitudes()[i],
            width = N
        );
    }

    // For large qubit counts, demonstrate the memory-efficient version
    if N >= 20 {
        let memory_efficient_sim = OptimizedSimulator::memory_efficient();
        let start = Instant::now();
        let result = memory_efficient_sim.run(&circuit).unwrap();
        let duration = start.elapsed();

        println!(
            "Memory-efficient simulator: {:.3} seconds",
            duration.as_secs_f64()
        );
        println!("  State vector size: {}", result.amplitudes().len());

        // Print the first few amplitudes
        println!("  First few amplitudes:");
        for i in 0..std::cmp::min(5, result.amplitudes().len()) {
            println!(
                "    |{:0width$b}⟩: {}",
                i,
                result.amplitudes()[i],
                width = N
            );
        }
    }
}
