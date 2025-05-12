use quantrs_circuit::builder::Circuit;
use quantrs_sim::statevector::StateVectorSimulator;

fn main() {
    // Create a circuit with 3 qubits
    // Qubit 0: Alice's qubit to be teleported
    // Qubit 1: Alice's entangled qubit
    // Qubit 2: Bob's entangled qubit
    let mut circuit = Circuit::<3>::new();
    
    // Prepare a state to teleport (apply X, H to qubit 0)
    circuit.x(0).unwrap()
           .h(0).unwrap();
    
    println!("Prepared state to teleport:");
    let simulator = StateVectorSimulator::new();
    let state_prep = circuit.run(simulator.clone()).unwrap();
    
    // Print the resulting amplitudes
    for (i, amplitude) in state_prep.amplitudes().iter().enumerate() {
        let bits = format!("{:03b}", i);
        if amplitude.norm_sqr() > 1e-10 {
            println!("|{}⟩: {} + {}i", bits, amplitude.re, amplitude.im);
        }
    }
    
    // Create Bell pair between qubits 1 and 2
    circuit.h(1).unwrap()
           .cnot(1, 2).unwrap();
    
    // Teleportation protocol
    circuit.cnot(0, 1).unwrap()
           .h(0).unwrap();
    
    // Measurements and corrections would normally happen here,
    // but for simulation we continue with the quantum circuit
    
    // Apply corrections based on measurement outcomes
    circuit.cnot(0, 2).unwrap()  // Apply X correction if first qubit is 1
           .cz(1, 2).unwrap();   // Apply Z correction if second qubit is 1
    
    // Run the full circuit
    let result = circuit.run(simulator).unwrap();
    
    println!("\nFinal state after teleportation:");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:03b}", i);
        if amplitude.norm_sqr() > 1e-10 {
            println!("|{}⟩: {} + {}i", bits, amplitude.re, amplitude.im);
        }
    }
    
    // Verify that qubit 2 has the same state as qubit 0 had initially
    println!("\nVerification:");
    
    // For |+⟩ state, we expect Z-basis measurement to be random
    let prob_z0 = result.probability(&[0, 0, 0]).unwrap() + 
                  result.probability(&[0, 1, 0]).unwrap() +
                  result.probability(&[1, 0, 0]).unwrap() +
                  result.probability(&[1, 1, 0]).unwrap();
    
    let prob_z1 = result.probability(&[0, 0, 1]).unwrap() + 
                  result.probability(&[0, 1, 1]).unwrap() +
                  result.probability(&[1, 0, 1]).unwrap() +
                  result.probability(&[1, 1, 1]).unwrap();
    
    println!("Probability of measuring |0⟩ on teleported qubit: {:.6}", prob_z0);
    println!("Probability of measuring |1⟩ on teleported qubit: {:.6}", prob_z1);
}