use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() {
    // Create a circuit with 2 qubits
    let mut circuit = Circuit::<2>::new();

    // Build a Bell state circuit: H(0) followed by CNOT(0, 1)
    circuit
        .h(0)
        .expect("Failed to apply H gate to qubit 0 in Bell state circuit");
    circuit
        .cnot(0, 1)
        .expect("Failed to apply CNOT from qubit 0 to qubit 1 in Bell state circuit");

    // Run the circuit on the state vector simulator
    let simulator = StateVectorSimulator::new();
    let result = simulator
        .run(&circuit)
        .expect("Failed to run Bell state circuit on StateVector simulator");

    // Print the resulting amplitudes
    println!("Bell state (|00⟩ + |11⟩)/√2 amplitudes:");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{i:02b}");
        println!("|{}⟩: {} + {}i", bits, amplitude.re, amplitude.im);
    }

    // Calculate probabilities
    println!("\nProbabilities:");
    for (i, prob) in result.probabilities().iter().enumerate() {
        let bits = format!("{i:02b}");
        println!("|{bits}⟩: {prob:.6}");
    }
}
