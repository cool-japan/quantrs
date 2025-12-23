use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use quantrs2_sim::statevector::StateVectorSimulator;
use std::f64::consts::PI;

/// Example demonstrating gate decomposition and optimization
fn main() -> QuantRS2Result<()> {
    println!("Gate Decomposition and Optimization Example");
    println!("==========================================");

    // Create a circuit with common gates
    let mut original_circuit = Circuit::<3>::new();

    // Add a Toffoli gate (which is complex and decomposable)
    original_circuit.toffoli(0, 1, 2)?;

    // Add some single qubit gates
    original_circuit.h(0)?;
    original_circuit.h(0)?; // Two Hadamards in a row cancel out

    // Add a SWAP gate (which is also decomposable)
    original_circuit.swap(1, 2)?;

    // Print the original circuit info
    println!("\nOriginal Circuit:");
    println!("Number of qubits: {}", original_circuit.num_qubits());
    println!("Number of gates: {}", original_circuit.num_gates());
    print_gates(&original_circuit);

    // Simulate the circuit
    println!("\nSimulating original circuit...");
    let simulator = StateVectorSimulator::new();
    let result = simulator.run(&original_circuit)?;
    print_state(&result);

    // Decompose the circuit
    println!("\nDecomposing the circuit...");
    let decomposed_circuit = original_circuit.decompose()?;

    // Print the decomposed circuit info
    println!("\nDecomposed Circuit:");
    println!("Number of qubits: {}", decomposed_circuit.num_qubits());
    println!("Number of gates: {}", decomposed_circuit.num_gates());
    print_gates(&decomposed_circuit);

    // Simulate the decomposed circuit
    println!("\nSimulating decomposed circuit...");
    let decomposed_result = simulator.run(&decomposed_circuit)?;
    print_state(&decomposed_result);

    // Check if the results are the same (they should be)
    let original_probs = result.probabilities();
    let decomposed_probs = decomposed_result.probabilities();

    println!("\nComparing results:");
    let mut max_diff: f64 = 0.0;
    for i in 0..original_probs.len() {
        let diff = (original_probs[i] - decomposed_probs[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("Maximum probability difference: {max_diff:.10}");

    if max_diff < 1e-10 {
        println!("Results match! The decomposition preserves the circuit's behavior.");
    } else {
        println!("Results differ! The decomposition might have issues.");
    }

    // Demonstrate optimization
    println!("\n\nOptimization Example");
    println!("===================");

    // Create a circuit with redundant gates
    let mut redundant_circuit = Circuit::<2>::new();

    // Add gates that should cancel out
    redundant_circuit.h(0)?;
    redundant_circuit.h(0)?; // These two H gates cancel out

    redundant_circuit.x(1)?;
    redundant_circuit.x(1)?; // These two X gates cancel out

    // Add gates that can be combined
    redundant_circuit.rx(0, PI / 4.0)?;
    redundant_circuit.rx(0, PI / 4.0)?; // These combine to rx(0, PI/2)

    // Print the redundant circuit info
    println!("\nRedundant Circuit:");
    println!("Number of qubits: {}", redundant_circuit.num_qubits());
    println!("Number of gates: {}", redundant_circuit.num_gates());
    print_gates(&redundant_circuit);

    // Optimize the circuit
    println!("\nOptimizing the circuit...");
    let optimized_circuit = redundant_circuit.optimize()?;

    // Print the optimized circuit info
    println!("\nOptimized Circuit:");
    println!("Number of qubits: {}", optimized_circuit.num_qubits());
    println!("Number of gates: {}", optimized_circuit.num_gates());
    print_gates(&optimized_circuit);

    // Simulate both to verify they're equivalent
    println!("\nSimulating redundant circuit...");
    let redundant_result = simulator.run(&redundant_circuit)?;

    println!("\nSimulating optimized circuit...");
    let optimized_result = simulator.run(&optimized_circuit)?;

    // Compare the results
    let redundant_probs = redundant_result.probabilities();
    let optimized_probs = optimized_result.probabilities();

    println!("\nComparing results:");
    let mut max_diff: f64 = 0.0;
    for i in 0..redundant_probs.len() {
        let diff = (redundant_probs[i] - optimized_probs[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("Maximum probability difference: {max_diff:.10}");

    if max_diff < 1e-10 {
        println!("Results match! The optimization preserves the circuit's behavior.");
    } else {
        println!("Results differ! The optimization might have issues.");
    }

    Ok(())
}

/// Helper function to print gate information
fn print_gates<const N: usize>(circuit: &Circuit<N>) {
    let gates = circuit.gates();
    for (i, gate) in gates.iter().enumerate() {
        println!("  Gate {}: {}", i, gate.name());
    }
}

/// Helper function to print register state
fn print_state<const N: usize>(register: &Register<N>) {
    println!("State vector:");

    let probabilities = register.probabilities();
    for (i, prob) in probabilities.iter().enumerate() {
        if *prob > 1e-10 {
            println!("  |{i:0N$b}⟩: {prob:.6}");
        }
    }
}
