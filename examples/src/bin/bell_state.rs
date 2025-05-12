use quantrs_circuit::builder::Circuit;
use quantrs_sim::statevector::StateVectorSimulator;

fn main() {
    // Create a circuit with 2 qubits
    let mut circuit = Circuit::<2>::new();
    
    // Build a Bell state circuit: H(0) followed by CNOT(0, 1)
    circuit.h(0).unwrap()
           .cnot(0, 1).unwrap();
    
    // Run the circuit on the state vector simulator
    let simulator = StateVectorSimulator::new();
    let result = circuit.run(simulator).unwrap();
    
    // Print the resulting amplitudes
    println!("Bell state (|00⟩ + |11⟩)/√2 amplitudes:");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!("|{}⟩: {} + {}i", bits, amplitude.re, amplitude.im);
    }
    
    // Calculate probabilities
    println!("\nProbabilities:");
    for (i, prob) in result.probabilities().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!("|{}⟩: {:.6}", bits, prob);
    }
}