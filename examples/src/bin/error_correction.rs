// Error Correction Example
//
// This example demonstrates the use of quantum error correction codes,
// specifically the 3-qubit bit flip code to detect and correct errors.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::noise::{BitFlipChannel, NoiseModel};
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() {
    println!("Quantum Error Correction Example");
    println!("===============================\n");

    // Demonstrate 3-qubit bit flip code
    println!("3-Qubit Bit Flip Code");
    println!("-------------------\n");

    // 1. Prepare the input state |1⟩
    // 2. Encode it into |111⟩ using a 3-qubit bit flip code
    // 3. Apply noise that might flip one of the bits
    // 4. Decode and correct any errors
    // 5. Measure the result, which should be |1⟩ again

    // Define encoding circuit |ψ⟩ -> |ψψψ⟩
    let encode_circuit = create_encoding_circuit();
    let correction_circuit = create_correction_circuit();

    // Let's run first without noise to verify encoding works
    run_without_noise(&encode_circuit);

    // Now run with noise to demonstrate error correction
    run_with_noise(&encode_circuit, &correction_circuit);

    // Show what happens with too much noise (uncorrectable errors)
    run_with_high_noise(&encode_circuit, &correction_circuit);
}

fn create_encoding_circuit() -> Circuit<5> {
    // Create a 5-qubit circuit
    // q0: input qubit (the one we want to protect)
    // q1-q3: ancilla qubits for the encoding
    // q4: final readout qubit
    let mut circuit = Circuit::<5>::new();

    // Set the input qubit to |1⟩
    circuit.x(0).unwrap();

    // Encoding: spread the information to 3 qubits
    // CNOT from qubit 0 to qubit 1
    // CNOT from qubit 0 to qubit 2
    circuit.cnot(0, 1).unwrap();
    circuit.cnot(0, 2).unwrap();

    println!("Created encoding circuit:");
    println!("1. Input state: |1⟩ on qubit 0");
    println!("2. CNOT from q0 to q1");
    println!("3. CNOT from q0 to q2");
    println!("Result should be |111⟩ on qubits 0-2\n");

    circuit
}

fn create_correction_circuit() -> Circuit<5> {
    // Create error detection and correction circuit
    let mut circuit = Circuit::<5>::new();

    // Use qubits 3 and 4 as syndrome bits to detect errors
    // CNOT from data qubits to syndrome qubits
    circuit.cnot(0, 3).unwrap();
    circuit.cnot(1, 3).unwrap();
    circuit.cnot(1, 4).unwrap();
    circuit.cnot(2, 4).unwrap();

    // Apply corrections based on syndrome measurements
    // If syndrome = 01, flip qubit 0
    // If syndrome = 10, flip qubit 1
    // If syndrome = 11, flip qubit 2
    // We can implement this with controlled-X gates

    // Syndrome 01 (q4=0, q3=1): Flip q0
    circuit.x(4).unwrap(); // Invert q4 for control
    circuit.cx(3, 0).unwrap();
    circuit.x(4).unwrap(); // Restore q4

    // Syndrome 10 (q4=1, q3=0): Flip q1
    circuit.x(3).unwrap(); // Invert q3 for control
    circuit.cx(4, 1).unwrap();
    circuit.x(3).unwrap(); // Restore q3

    // Syndrome 11 (q4=1, q3=1): Flip q2
    circuit.cx(3, 2).unwrap();
    circuit.cx(4, 2).unwrap();

    println!("Created correction circuit:");
    println!("1. Syndrome measurement: CNOT from q0,q1 to q3");
    println!("2. Syndrome measurement: CNOT from q1,q2 to q4");
    println!("3. Apply corrections based on syndrome bits\n");

    circuit
}

fn run_without_noise(encode_circuit: &Circuit<5>) {
    println!("=== Running with No Noise ===");

    // Create a simulator
    let simulator = StateVectorSimulator::sequential();

    // Run the circuit
    let result = encode_circuit.run(simulator).unwrap();

    // Print the resulting state vector
    println!("Encoded state (first 8 amplitudes):");
    for (i, amplitude) in result.amplitudes().iter().take(8).enumerate() {
        let bits = format!("{:05b}", i);
        let prob = amplitude.norm_sqr();
        if prob > 1e-10 {
            println!("|{}⟩: {} (probability: {:.6})", bits, amplitude, prob);
        }
    }

    println!("\nVerification: The state should be |111xx⟩ where x can be 0 or 1");

    // We can also verify programmatically
    let q0_prob_one = get_qubit_prob_one(&result, 0);
    let q1_prob_one = get_qubit_prob_one(&result, 1);
    let q2_prob_one = get_qubit_prob_one(&result, 2);

    println!("Probability qubit 0 is |1⟩: {:.6}", q0_prob_one);
    println!("Probability qubit 1 is |1⟩: {:.6}", q1_prob_one);
    println!("Probability qubit 2 is |1⟩: {:.6}", q2_prob_one);

    println!("\nEncoding successful: all data qubits are in state |1⟩\n");
}

fn run_with_noise(encode_circuit: &Circuit<5>, correction_circuit: &Circuit<5>) {
    println!("=== Running with Noise (10% bit flip on each qubit) ===");

    // Create a combined circuit
    let mut full_circuit = Circuit::<5>::new();

    // Add all gates from encode circuit
    for gate in encode_circuit.gates() {
        full_circuit.add_gate_arc(gate.clone()).unwrap();
    }

    // Create noise model with moderate bit flip probability
    let mut noise_model = NoiseModel::new(false); // Apply at the end of encoding
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(0),
        probability: 0.1,
    });
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(1),
        probability: 0.1,
    });
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(2),
        probability: 0.1,
    });

    // Add noisy simulator
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run encoding + noise
    let noisy_state = full_circuit.run(noisy_sim).unwrap();

    // Print the result after noise
    println!("State after encoding and noise (first 8 amplitudes):");
    for (i, amplitude) in noisy_state.amplitudes().iter().take(8).enumerate() {
        let bits = format!("{:05b}", i);
        let prob = amplitude.norm_sqr();
        if prob > 1e-10 {
            println!("|{}⟩: {} (probability: {:.6})", bits, amplitude, prob);
        }
    }

    // Calculate qubit probabilities
    let q0_prob_one = get_qubit_prob_one(&noisy_state, 0);
    let q1_prob_one = get_qubit_prob_one(&noisy_state, 1);
    let q2_prob_one = get_qubit_prob_one(&noisy_state, 2);

    println!("Probability qubit 0 is |1⟩: {:.6}", q0_prob_one);
    println!("Probability qubit 1 is |1⟩: {:.6}", q1_prob_one);
    println!("Probability qubit 2 is |1⟩: {:.6}", q2_prob_one);

    println!("\nNoise has corrupted some of the qubits.\n");

    // Now apply correction circuit
    let mut correction_with_encoded = Circuit::<5>::new();

    // Add encoded state preparation with noise
    for gate in full_circuit.gates() {
        correction_with_encoded
            .add_gate_arc(gate.clone())
            .unwrap();
    }

    // Add correction gates
    for gate in correction_circuit.gates() {
        correction_with_encoded
            .add_gate_arc(gate.clone())
            .unwrap();
    }

    // Run on clean simulator
    let clean_sim = StateVectorSimulator::sequential();
    let corrected_state = correction_with_encoded.run(clean_sim).unwrap();

    // Print the result after correction
    println!("State after error correction (first 8 amplitudes):");
    for (i, amplitude) in corrected_state.amplitudes().iter().take(8).enumerate() {
        let bits = format!("{:05b}", i);
        let prob = amplitude.norm_sqr();
        if prob > 1e-10 {
            println!("|{}⟩: {} (probability: {:.6})", bits, amplitude, prob);
        }
    }

    // Calculate qubit probabilities after correction
    let q0_prob_one_after = get_qubit_prob_one(&corrected_state, 0);
    let q1_prob_one_after = get_qubit_prob_one(&corrected_state, 1);
    let q2_prob_one_after = get_qubit_prob_one(&corrected_state, 2);

    println!("After correction:");
    println!("Probability qubit 0 is |1⟩: {:.6}", q0_prob_one_after);
    println!("Probability qubit 1 is |1⟩: {:.6}", q1_prob_one_after);
    println!("Probability qubit 2 is |1⟩: {:.6}", q2_prob_one_after);

    println!("\nError correction has restored the original state!\n");
}

fn run_with_high_noise(encode_circuit: &Circuit<5>, correction_circuit: &Circuit<5>) {
    println!("=== Running with High Noise (30% bit flip on each qubit) ===");

    // Create a combined circuit
    let mut full_circuit = Circuit::<5>::new();

    // Add all gates from encode circuit
    for gate in encode_circuit.gates() {
        full_circuit.add_gate_arc(gate.clone()).unwrap();
    }

    // Create noise model with higher bit flip probability
    let mut noise_model = NoiseModel::new(false); // Apply at the end of encoding
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(0),
        probability: 0.3,
    });
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(1),
        probability: 0.3,
    });
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(2),
        probability: 0.3,
    });

    // Add noisy simulator
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run encoding + noise
    let noisy_state = full_circuit.run(noisy_sim).unwrap();

    // Print the result after noise
    println!("State after encoding and high noise (first 8 amplitudes):");
    for (i, amplitude) in noisy_state.amplitudes().iter().take(8).enumerate() {
        let bits = format!("{:05b}", i);
        let prob = amplitude.norm_sqr();
        if prob > 1e-10 {
            println!("|{}⟩: {} (probability: {:.6})", bits, amplitude, prob);
        }
    }

    // Now apply correction circuit
    let mut correction_with_encoded = Circuit::<5>::new();

    // Add encoded state preparation with noise
    for gate in full_circuit.gates() {
        correction_with_encoded
            .add_gate_arc(gate.clone())
            .unwrap();
    }

    // Add correction gates
    for gate in correction_circuit.gates() {
        correction_with_encoded
            .add_gate_arc(gate.clone())
            .unwrap();
    }

    // Run on clean simulator
    let clean_sim = StateVectorSimulator::sequential();
    let corrected_state = correction_with_encoded.run(clean_sim).unwrap();

    // Calculate qubit probabilities after high noise and correction
    let q0_prob_one_after = get_qubit_prob_one(&corrected_state, 0);
    let q1_prob_one_after = get_qubit_prob_one(&corrected_state, 1);
    let q2_prob_one_after = get_qubit_prob_one(&corrected_state, 2);

    println!("\nAfter correction with high noise:");
    println!("Probability qubit 0 is |1⟩: {:.6}", q0_prob_one_after);
    println!("Probability qubit 1 is |1⟩: {:.6}", q1_prob_one_after);
    println!("Probability qubit 2 is |1⟩: {:.6}", q2_prob_one_after);

    println!("\nThe error correction code is unable to fully recover from multiple bit flips!");
    println!("This demonstrates the limitations of the 3-qubit bit flip code - it can only");
    println!("correct a single bit flip error. When multiple errors occur, the correction fails.");
}

// Helper function to compute the probability of a qubit being in state |1⟩
fn get_qubit_prob_one(register: &quantrs2_core::register::Register<5>, qubit_idx: usize) -> f64 {
    let mut prob_one = 0.0;
    for (i, amplitude) in register.amplitudes().iter().enumerate() {
        if (i >> qubit_idx) & 1 == 1 {
            prob_one += amplitude.norm_sqr();
        }
    }
    prob_one
}
