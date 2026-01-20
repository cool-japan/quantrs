use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() {
    // Create a circuit with 3 qubits
    // Qubit 0: Alice's qubit to be teleported
    // Qubit 1: Alice's entangled qubit
    // Qubit 2: Bob's entangled qubit
    let mut circuit = Circuit::<3>::new();

    // Prepare a state to teleport (apply X, H to qubit 0)
    circuit
        .x(0)
        .expect("Failed to apply X gate to qubit 0")
        .h(0)
        .expect("Failed to apply H gate to qubit 0");

    println!("Prepared state to teleport:");
    let simulator = StateVectorSimulator::new();
    let state_prep = simulator
        .run(&circuit)
        .expect("Failed to run circuit for state preparation");

    // Print the resulting amplitudes
    for (i, amplitude) in state_prep.amplitudes().iter().enumerate() {
        let bits = format!("{i:03b}");
        if amplitude.norm_sqr() > 1e-10 {
            println!("|{}⟩: {} + {}i", bits, amplitude.re, amplitude.im);
        }
    }

    // Create Bell pair between qubits 1 and 2
    circuit
        .h(1)
        .expect("Failed to apply H gate to qubit 1 for Bell pair")
        .cnot(1, 2)
        .expect("Failed to apply CNOT for Bell pair creation");

    // Teleportation protocol
    circuit
        .cnot(0, 1)
        .expect("Failed to apply CNOT in teleportation protocol")
        .h(0)
        .expect("Failed to apply H gate in teleportation protocol");

    // Measurements and corrections would normally happen here,
    // but for simulation we continue with the quantum circuit

    // Apply corrections based on measurement outcomes
    circuit
        .cnot(0, 2)
        .expect("Failed to apply X correction (CNOT)") // Apply X correction if first qubit is 1
        .cz(1, 2)
        .expect("Failed to apply Z correction (CZ)"); // Apply Z correction if second qubit is 1

    // Run the full circuit
    let result = simulator
        .run(&circuit)
        .expect("Failed to run full teleportation circuit");

    println!("\nFinal state after teleportation:");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{i:03b}");
        if amplitude.norm_sqr() > 1e-10 {
            println!("|{}⟩: {} + {}i", bits, amplitude.re, amplitude.im);
        }
    }

    // Verify that qubit 2 has the same state as qubit 0 had initially
    println!("\nVerification:");

    // For |+⟩ state, we expect Z-basis measurement to be random
    let prob_z0 = result
        .probability(&[0, 0, 0])
        .expect("Failed to compute probability for state |000⟩")
        + result
            .probability(&[0, 1, 0])
            .expect("Failed to compute probability for state |010⟩")
        + result
            .probability(&[1, 0, 0])
            .expect("Failed to compute probability for state |100⟩")
        + result
            .probability(&[1, 1, 0])
            .expect("Failed to compute probability for state |110⟩");

    let prob_z1 = result
        .probability(&[0, 0, 1])
        .expect("Failed to compute probability for state |001⟩")
        + result
            .probability(&[0, 1, 1])
            .expect("Failed to compute probability for state |011⟩")
        + result
            .probability(&[1, 0, 1])
            .expect("Failed to compute probability for state |101⟩")
        + result
            .probability(&[1, 1, 1])
            .expect("Failed to compute probability for state |111⟩");

    println!("Probability of measuring |0⟩ on teleported qubit: {prob_z0:.6}");
    println!("Probability of measuring |1⟩ on teleported qubit: {prob_z1:.6}");
}
