use quantrs_circuit::Circuit;
use quantrs_core::{QubitId, Register};
use quantrs_sim::StateVectorSimulator;
use std::f64::consts::PI;

/// Implements Quantum Phase Estimation to estimate the phase of a unitary operator
/// In this example, we estimate the phase of the Z gate, which has eigenvalues e^(i*PI) = -1
fn main() {
    println!("Quantum Phase Estimation Example");
    println!("Estimating the phase of the Z gate (expected: 1/2)\n");

    // Number of precision qubits (more qubits = higher precision)
    let precision_qubits = 4;
    // Total qubits: precision qubits + 1 target qubit
    let total_qubits = precision_qubits + 1;

    // Create a new circuit with precision_qubits + 1 qubits
    // The last qubit is the target qubit (eigenstate of the unitary)
    let mut circuit = Circuit::<5>::new(); // Using 5 = 4 precision + 1 target qubit

    // Step 1: Prepare the target qubit in the eigenstate of Z (|1⟩)
    let target_qubit = QubitId::new(precision_qubits);
    circuit.x(target_qubit);

    println!("Prepared target qubit in the |1⟩ state (eigenstate of Z)");

    // Step 2: Apply Hadamard gates to all precision qubits
    for i in 0..precision_qubits {
        circuit.h(QubitId::new(i));
    }

    println!("Applied Hadamard gates to precision qubits");

    // Step 3: Apply controlled unitary operations
    // For each precision qubit, apply the controlled-unitary operations
    // with the required number of repetitions (2^j)
    for i in 0..precision_qubits {
        let repetitions = 1 << i; // 2^i

        // Apply the controlled-Z gate the required number of times
        for _ in 0..repetitions {
            circuit.controlled_u(QubitId::new(i), target_qubit, |b| {
                b.z(target_qubit);
            });
        }
    }

    println!("Applied controlled-Z operations with appropriate repetitions");

    // Step 4: Apply inverse QFT to the precision qubits
    apply_inverse_qft(&mut circuit, precision_qubits);

    println!("Applied inverse QFT to precision qubits");

    // Initialize the simulator and register
    let mut simulator = StateVectorSimulator::new();
    let register = Register::<5>::new();

    // Run the circuit
    println!("\nExecuting circuit...");
    let result = simulator.run(&circuit, &register);

    // Measure the precision qubits to get the phase estimate
    let precision_qubit_ids: Vec<_> = (0..precision_qubits).map(|i| QubitId::new(i)).collect();
    let measured = simulator.measure(precision_qubit_ids, &result);

    // Print results
    println!("\nMeasurement results of precision qubits:");

    let mut max_prob = 0.0;
    let mut most_likely_outcome = 0;

    // The possible measurement outcomes correspond to binary fractions
    let max_outcome = 1 << precision_qubits;
    for i in 0..max_outcome {
        let prob = simulator.probability(i, &measured);

        if prob > 0.01 {
            // Only show states with significant probability
            let binary = format!("{:0width$b}", i, width = precision_qubits);
            let phase_estimate = i as f64 / max_outcome as f64;
            println!(
                "State |{}⟩: probability = {:.6}, phase estimate = {:.6}",
                binary, prob, phase_estimate
            );

            if prob > max_prob {
                max_prob = prob;
                most_likely_outcome = i;
            }
        }
    }

    // Calculate the estimated phase
    let estimated_phase = most_likely_outcome as f64 / max_outcome as f64;
    println!("\nMost likely phase estimate: {:.6}", estimated_phase);
    println!("Expected phase for Z gate: 0.5 (1/2)");

    // Calculate accuracy
    let expected_phase = 0.5; // For Z gate
    let error = (estimated_phase - expected_phase).abs();

    println!("\nError in phase estimation: {:.6}", error);
    if error < 0.1 {
        println!("Success! The estimated phase is close to the expected value.");
    } else {
        println!("The estimated phase has significant error.");
        println!("Try increasing the number of precision qubits for better accuracy.");
    }
}

/// Applies the inverse Quantum Fourier Transform to the first n qubits
fn apply_inverse_qft(circuit: &mut Circuit<5>, n: usize) {
    // QFT† circuit (inverse of QFT)

    // First, swap qubits
    for i in 0..n / 2 {
        circuit.swap(QubitId::new(i), QubitId::new(n - i - 1));
    }

    // Apply inverse QFT operations in reverse order
    for i in (0..n).rev() {
        // Apply inverse controlled rotations
        for j in (i + 1..n).rev() {
            let angle = -PI / (1 << (j - i)); // -PI/2^(j-i)
            circuit.controlled_u(QubitId::new(j), QubitId::new(i), |b| {
                b.rz(QubitId::new(i), angle);
            });
        }

        // Apply Hadamard to the current qubit
        circuit.h(QubitId::new(i));
    }
}
