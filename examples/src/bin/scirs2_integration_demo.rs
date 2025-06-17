//! Demonstration of SciRS2 integration in QuantRS2
//!
//! This example shows how QuantRS2 leverages SciRS2's features for:
//! - Enhanced complex number operations
//! - Memory-efficient state storage
//! - SIMD-accelerated operations
//! - Optimized linear algebra

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::prelude::*;
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QuantRS2 with SciRS2 Integration Demo ===\n");

    // Example 1: Using enhanced complex number operations
    println!("1. Enhanced Complex Number Operations:");
    demonstrate_complex_operations()?;

    // Example 2: Memory-efficient state storage
    println!("\n2. Memory-Efficient State Storage:");
    demonstrate_memory_efficient_storage()?;

    // Example 3: SIMD-accelerated operations
    println!("\n3. SIMD-Accelerated Operations:");
    demonstrate_simd_operations()?;

    // Example 4: State vector simulation
    println!("\n4. State Vector Simulation:");
    demonstrate_simulation()?;

    Ok(())
}

fn demonstrate_complex_operations() -> Result<(), Box<dyn std::error::Error>> {
    use num_complex::Complex64;
    use quantrs2_core::complex_ext::QuantumComplexExt;

    // Create quantum amplitudes
    let amp1 = Complex64::new(0.6, 0.8);
    let amp2 = Complex64::new(0.8, -0.6);

    // Use quantum-specific operations
    println!("  Amplitude 1: {}", amp1);
    println!("  Probability: {}", amp1.probability());
    println!("  Normalized: {}", amp1.normalize());

    println!("\n  Amplitude 2: {}", amp2);
    println!("  Fidelity between amp1 and amp2: {}", amp1.fidelity(&amp2));

    // Create phase factors
    let phase = quantum_states::phase_factor(std::f64::consts::PI / 4.0);
    println!("\n  Phase factor e^(iπ/4): {}", phase);

    Ok(())
}

fn demonstrate_memory_efficient_storage() -> Result<(), Box<dyn std::error::Error>> {
    // Create memory-efficient state vectors for different qubit counts
    for n_qubits in [5, 10, 15] {
        let state = EfficientStateVector::new(n_qubits)?;
        let stats = state.memory_stats();

        println!("  {} qubits:", n_qubits);
        println!("    - State size: {} amplitudes", stats.num_amplitudes);
        println!("    - Memory usage: {} KB", stats.memory_bytes / 1024);
    }

    // Demonstrate chunk processing
    let mut state = EfficientStateVector::new(8)?;
    println!("\n  Processing 8-qubit state in chunks:");

    state.process_chunks(64, |chunk, start_idx| {
        println!("    - Processing chunk starting at index {}", start_idx);
    })?;

    Ok(())
}

fn demonstrate_simd_operations() -> Result<(), Box<dyn std::error::Error>> {
    use num_complex::Complex64;

    // Create a quantum state
    let mut state = vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
    ];

    println!("  Initial state: {:?}", state);

    // Apply phase rotation using SIMD
    apply_phase_simd(&mut state, std::f64::consts::PI / 2.0);
    println!("  After phase rotation (π/2): {:?}", state);

    // Normalize using SIMD
    normalize_simd(&mut state)?;
    println!("  After normalization: {:?}", state);

    // Compute expectation value
    let expectation = expectation_z_simd(&state, 0, 1);
    println!("  Expectation value <Z>: {}", expectation);

    Ok(())
}

fn demonstrate_simulation() -> Result<(), Box<dyn std::error::Error>> {
    // Create a quantum circuit
    let mut circuit = Circuit::<3>::new();

    // Build a GHZ state
    circuit.h(QubitId(0));
    circuit.cnot(QubitId(0), QubitId(1));
    circuit.cnot(QubitId(1), QubitId(2));

    // Run with standard simulator
    let mut sim = StateVectorSimulator::new();
    let result = sim.run(&circuit)?;

    println!("  Simulator probabilities: {:?}", result.probabilities());

    // Should produce the GHZ state: |000⟩ + |111⟩
    println!("\n  GHZ state created successfully!");

    // Test with larger circuit
    let mut large_circuit = Circuit::<10>::new();
    for i in 0..10 {
        large_circuit.h(QubitId(i));
    }

    println!("\n  Testing with 10-qubit circuit:");
    let start = std::time::Instant::now();
    let _ = sim.run(&large_circuit)?;
    println!("    Simulation time: {:?}", start.elapsed());

    Ok(())
}
