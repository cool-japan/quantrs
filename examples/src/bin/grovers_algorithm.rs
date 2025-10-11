use quantrs2_circuit::prelude::{Circuit, Simulator};
use quantrs2_core::{qubit::QubitId, register::Register};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::f64::consts::PI;

/// Implements Grover's algorithm for a 3-qubit search space
/// Searching for the state |101⟩ (decimal 5)
fn main() {
    println!("Grover's Algorithm Example");
    println!("Searching for state |101⟩ in a 3-qubit system\n");

    // Create a new circuit with 3 qubits
    let mut circuit = Circuit::<3>::new();

    // Step 1: Apply Hadamard gates to all qubits to create superposition
    for i in 0..3 {
        circuit.h(QubitId::new(i));
    }

    println!("Applied Hadamard gates to create uniform superposition");

    // Step 2: Implement the oracle that marks the |101⟩ state
    implement_oracle(&mut circuit);

    println!("Applied oracle to mark the target state |101⟩");

    // Step 3: Apply the diffusion operator (Grover's diffusion)
    apply_diffusion(&mut circuit);

    println!("Applied diffusion operator to amplify the marked state");

    // For 3 qubits, one iteration is optimal
    // For larger qubit counts, we would need to repeat steps 2 and 3 approximately sqrt(N) times

    // Initialize the simulator
    let mut simulator = StateVectorSimulator::new();

    // Run the circuit
    println!("\nExecuting circuit...");
    let result = simulator.run(&circuit);

    // Print the results
    println!("\nFinal state probabilities:");
    let register = result.expect("Failed to run circuit");
    let probabilities = register.probabilities();
    for i in 0..8 {
        println!("State |{:03b}⟩: {:.6}", i, probabilities[i]);
    }

    // Find the most probable measurement outcome
    println!("\nAnalyzing measurement probabilities...");

    // Calculate the most probable state
    let mut max_prob = 0.0;
    let mut max_state = 0;
    for i in 0..8 {
        let prob = probabilities[i];
        if prob > max_prob {
            max_prob = prob;
            max_state = i;
        }
    }

    println!("Most probable state: |{max_state:03b}⟩ with probability {max_prob:.6}");

    // Check if we found the correct state
    if max_state == 5 {
        println!("\nSuccess! Found the target state |101⟩");
    } else {
        println!("\nFailed to find the target state |101⟩");
    }
}

/// Implements the oracle that marks the |101⟩ state (decimal 5)
fn implement_oracle(circuit: &mut Circuit<3>) {
    // For the state |101⟩, we need to negate the outputs from qubits 0 and 2 (as they are 1)
    // We leave qubit 1 alone (as it is 0)
    circuit.x(QubitId::new(1)); // Flip qubit 1 from 0 to 1 (to make them all 1's for the next step)

    // Apply a controlled-Z operation that acts when all qubits are 1
    // First, implement a multi-controlled Z gate using Toffoli gate
    circuit.toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2));
    circuit.z(QubitId::new(2));
    circuit.toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2));

    // Undo the flip on qubit 1
    circuit.x(QubitId::new(1));
}

/// Implements the diffusion operator (Grover's diffusion)
fn apply_diffusion(circuit: &mut Circuit<3>) {
    // Apply Hadamard to all qubits
    for i in 0..3 {
        circuit.h(QubitId::new(i));
    }

    // Apply X to all qubits
    for i in 0..3 {
        circuit.x(QubitId::new(i));
    }

    // Implement controlled-Z similar to the oracle
    circuit.toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2));
    circuit.z(QubitId::new(2));
    circuit.toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2));

    // Undo X gates
    for i in 0..3 {
        circuit.x(QubitId::new(i));
    }

    // Undo Hadamard gates
    for i in 0..3 {
        circuit.h(QubitId::new(i));
    }
}
