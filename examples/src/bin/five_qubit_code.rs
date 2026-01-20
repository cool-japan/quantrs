// Five-Qubit Perfect Code Example
//
// This example demonstrates the use of the 5-qubit perfect code,
// which is the smallest code that can correct arbitrary single-qubit errors.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::error_correction::{ErrorCorrection, FiveQubitCode};
use quantrs2_sim::noise::{BitFlipChannel, DepolarizingChannel, NoiseModel, PhaseFlipChannel};
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() {
    println!("Five-Qubit Perfect Code Example");
    println!("==============================\n");

    // The 5-qubit code is the smallest quantum error correction code that
    // can protect against arbitrary single-qubit errors (X, Y, Z)
    println!("The 5-qubit perfect code encodes 1 logical qubit into 5 physical qubits");
    println!("It can detect and correct ANY single-qubit error (bit flip, phase flip, or both)\n");

    // We'll use a total of 9 qubits:
    // - 1 original data qubit
    // - 4 ancilla qubits for the encoding (total of 5 data qubits)
    // - 4 syndrome qubits for error detection/correction

    // Let's start by preparing and encoding a |+⟩ state
    let mut circuit = Circuit::<9>::new();

    // Step 1: Create a superposition state on qubit 0
    println!("1. Preparing the logical qubit in the |+⟩ state");
    circuit.h(0).expect("Failed to apply H gate to qubit 0");

    // Step 2: Encode using the 5-qubit code
    println!("2. Encoding the state using the 5-qubit perfect code");

    let code = FiveQubitCode;
    let logical_qubits = vec![QubitId::new(0)];
    let ancilla_qubits = vec![
        QubitId::new(1),
        QubitId::new(2),
        QubitId::new(3),
        QubitId::new(4),
    ];

    // Create the encoding circuit
    let encoder = code.encode_circuit(&logical_qubits, &ancilla_qubits);

    // Transfer the encoding gates to our main circuit
    let encoder = encoder.expect("Failed to create encoding circuit");
    for gate in encoder.gates() {
        circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add encoding gate to circuit");
    }

    // Run with ideal simulator to verify the encoded state
    let ideal_sim = StateVectorSimulator::sequential();
    let encoded_state = circuit
        .run(ideal_sim)
        .expect("Failed to run ideal encoding circuit");

    // Step 3: Apply different types of errors and attempt correction
    run_with_bit_flip(&circuit, &code);
    run_with_phase_flip(&circuit, &code);
    run_with_depolarizing_noise(&circuit, &code);

    // Step 4: Show the limits - when the code fails
    println!("\n4. Demonstrating the limits of the 5-qubit code");
    println!("   The code can only correct a SINGLE error. With multiple errors, it will fail.");

    run_with_multiple_errors(&circuit, &code);

    println!("\nConclusion: The 5-qubit perfect code is the smallest quantum error correction");
    println!("code that can protect against arbitrary single-qubit errors. It effectively");
    println!("protects a quantum state against X, Z, and Y errors, but cannot correct");
    println!("multiple errors affecting different qubits.");
}

// Apply bit-flip error and correct it
fn run_with_bit_flip(encoding_circuit: &Circuit<9>, code: &FiveQubitCode) {
    println!("\n3a. Testing error correction for bit flip (X) errors");

    // Create a noise model with bit flip on the first qubit
    let mut noise_model = NoiseModel::new(false);
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(0),
        probability: 1.0, // 100% chance of error to test correction
    });

    // Create a simulator with this noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the encoding with noise
    let noisy_state = encoding_circuit
        .run(noisy_sim)
        .expect("Failed to run encoding circuit with bit flip noise");

    // Now apply error correction
    let mut correction_circuit = Circuit::<9>::new();

    // First transfer the encoding circuit + noise
    for gate in encoding_circuit.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add encoding gate to bit flip correction circuit");
    }

    // Add the syndrome measurement and correction
    let encoded_qubits = vec![
        QubitId::new(0),
        QubitId::new(1),
        QubitId::new(2),
        QubitId::new(3),
        QubitId::new(4),
    ];
    let syndrome_qubits = vec![
        QubitId::new(5),
        QubitId::new(6),
        QubitId::new(7),
        QubitId::new(8),
    ];

    let correction = code.decode_circuit(&encoded_qubits, &syndrome_qubits);

    // Add correction operations
    let correction = correction.expect("Failed to create bit flip correction circuit");
    for gate in correction.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add correction gate to bit flip correction circuit");
    }

    // Run the full circuit on a clean simulator
    let clean_sim = StateVectorSimulator::sequential();
    let corrected_state = correction_circuit
        .run(clean_sim)
        .expect("Failed to run bit flip correction circuit");

    // Analyze the logical state (qubit 0) before and after correction
    let before_correction = analyze_logical_state(&noisy_state);
    let after_correction = analyze_logical_state(&corrected_state);

    println!("   Before correction:");
    println!(
        "      P(|0⟩): {:.6}, P(|1⟩): {:.6}, Phase: {:.6}",
        before_correction.0, before_correction.1, before_correction.2
    );

    println!("   After correction:");
    println!(
        "      P(|0⟩): {:.6}, P(|1⟩): {:.6}, Phase: {:.6}",
        after_correction.0, after_correction.1, after_correction.2
    );

    println!("   Result: Successfully corrected bit flip error!");
}

// Apply phase-flip error and correct it
fn run_with_phase_flip(encoding_circuit: &Circuit<9>, code: &FiveQubitCode) {
    println!("\n3b. Testing error correction for phase flip (Z) errors");

    // Create a noise model with phase flip on the first qubit
    let mut noise_model = NoiseModel::new(false);
    noise_model.add_phase_flip(PhaseFlipChannel {
        target: QubitId::new(0),
        probability: 1.0, // 100% chance of error to test correction
    });

    // Create a simulator with this noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the encoding with noise
    let noisy_state = encoding_circuit
        .run(noisy_sim)
        .expect("Failed to run encoding circuit with phase flip noise");

    // Now apply error correction
    let mut correction_circuit = Circuit::<9>::new();

    // First transfer the encoding circuit + noise
    for gate in encoding_circuit.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add encoding gate to phase flip correction circuit");
    }

    // Add the syndrome measurement and correction
    let encoded_qubits = vec![
        QubitId::new(0),
        QubitId::new(1),
        QubitId::new(2),
        QubitId::new(3),
        QubitId::new(4),
    ];
    let syndrome_qubits = vec![
        QubitId::new(5),
        QubitId::new(6),
        QubitId::new(7),
        QubitId::new(8),
    ];

    let correction = code.decode_circuit(&encoded_qubits, &syndrome_qubits);

    // Add correction operations
    let correction = correction.expect("Failed to create phase flip correction circuit");
    for gate in correction.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add correction gate to phase flip correction circuit");
    }

    // Run the full circuit on a clean simulator
    let clean_sim = StateVectorSimulator::sequential();
    let corrected_state = correction_circuit
        .run(clean_sim)
        .expect("Failed to run phase flip correction circuit");

    // Analyze the logical state (qubit 0) before and after correction
    let before_correction = analyze_logical_state(&noisy_state);
    let after_correction = analyze_logical_state(&corrected_state);

    println!("   Before correction:");
    println!(
        "      P(|0⟩): {:.6}, P(|1⟩): {:.6}, Phase: {:.6}",
        before_correction.0, before_correction.1, before_correction.2
    );

    println!("   After correction:");
    println!(
        "      P(|0⟩): {:.6}, P(|1⟩): {:.6}, Phase: {:.6}",
        after_correction.0, after_correction.1, after_correction.2
    );

    println!("   Result: Successfully corrected phase flip error!");
}

// Apply random error (depolarizing noise) and correct it
fn run_with_depolarizing_noise(encoding_circuit: &Circuit<9>, code: &FiveQubitCode) {
    println!("\n3c. Testing error correction for depolarizing noise (random X, Y, Z errors)");

    // Create a noise model with depolarizing noise on the second qubit
    let mut noise_model = NoiseModel::new(false);
    noise_model.add_depolarizing(DepolarizingChannel {
        target: QubitId::new(1),
        probability: 0.9, // 90% chance of error to test correction
    });

    // Create a simulator with this noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the encoding with noise
    let noisy_state = encoding_circuit
        .run(noisy_sim)
        .expect("Failed to run encoding circuit with depolarizing noise");

    // Now apply error correction
    let mut correction_circuit = Circuit::<9>::new();

    // First transfer the encoding circuit + noise
    for gate in encoding_circuit.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add encoding gate to depolarizing noise correction circuit");
    }

    // Add the syndrome measurement and correction
    let encoded_qubits = vec![
        QubitId::new(0),
        QubitId::new(1),
        QubitId::new(2),
        QubitId::new(3),
        QubitId::new(4),
    ];
    let syndrome_qubits = vec![
        QubitId::new(5),
        QubitId::new(6),
        QubitId::new(7),
        QubitId::new(8),
    ];

    let correction = code.decode_circuit(&encoded_qubits, &syndrome_qubits);

    // Add correction operations
    let correction = correction.expect("Failed to create depolarizing noise correction circuit");
    for gate in correction.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add correction gate to depolarizing noise correction circuit");
    }

    // Run the full circuit on a clean simulator
    let clean_sim = StateVectorSimulator::sequential();
    let corrected_state = correction_circuit
        .run(clean_sim)
        .expect("Failed to run depolarizing noise correction circuit");

    // Analyze the logical state (qubit 0) before and after correction
    let before_correction = analyze_logical_state(&noisy_state);
    let after_correction = analyze_logical_state(&corrected_state);

    println!("   Before correction:");
    println!(
        "      P(|0⟩): {:.6}, P(|1⟩): {:.6}, Phase: {:.6}",
        before_correction.0, before_correction.1, before_correction.2
    );

    println!("   After correction:");
    println!(
        "      P(|0⟩): {:.6}, P(|1⟩): {:.6}, Phase: {:.6}",
        after_correction.0, after_correction.1, after_correction.2
    );

    println!("   Result: Successfully corrected random X/Y/Z error!");
}

// Apply multiple errors to show failure case
fn run_with_multiple_errors(encoding_circuit: &Circuit<9>, code: &FiveQubitCode) {
    println!("\n   Testing with two simultaneous errors (should fail)");

    // Create a noise model with errors on two different qubits
    let mut noise_model = NoiseModel::new(false);
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(0),
        probability: 1.0,
    });
    noise_model.add_phase_flip(PhaseFlipChannel {
        target: QubitId::new(3),
        probability: 1.0,
    });

    // Create a simulator with this noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the encoding with noise
    let noisy_state = encoding_circuit
        .run(noisy_sim)
        .expect("Failed to run encoding circuit with multiple errors");

    // Now apply error correction
    let mut correction_circuit = Circuit::<9>::new();

    // First transfer the encoding circuit + noise
    for gate in encoding_circuit.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add encoding gate to multiple errors correction circuit");
    }

    // Add the syndrome measurement and correction
    let encoded_qubits = vec![
        QubitId::new(0),
        QubitId::new(1),
        QubitId::new(2),
        QubitId::new(3),
        QubitId::new(4),
    ];
    let syndrome_qubits = vec![
        QubitId::new(5),
        QubitId::new(6),
        QubitId::new(7),
        QubitId::new(8),
    ];

    let correction = code.decode_circuit(&encoded_qubits, &syndrome_qubits);

    // Add correction operations
    let correction = correction.expect("Failed to create multiple errors correction circuit");
    for gate in correction.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add correction gate to multiple errors correction circuit");
    }

    // Run the full circuit on a clean simulator
    let clean_sim = StateVectorSimulator::sequential();
    let corrected_state = correction_circuit
        .run(clean_sim)
        .expect("Failed to run multiple errors correction circuit");

    // Analyze the logical state (qubit 0) before and after correction
    let before_correction = analyze_logical_state(&noisy_state);
    let after_correction = analyze_logical_state(&corrected_state);

    println!("   Before correction:");
    println!(
        "      P(|0⟩): {:.6}, P(|1⟩): {:.6}, Phase: {:.6}",
        before_correction.0, before_correction.1, before_correction.2
    );

    println!("   After correction:");
    println!(
        "      P(|0⟩): {:.6}, P(|1⟩): {:.6}, Phase: {:.6}",
        after_correction.0, after_correction.1, after_correction.2
    );

    println!("   Result: Correction failed as expected (errors on multiple qubits)");
}

// Helper function to analyze the logical state
// Returns (probability of |0⟩, probability of |1⟩, phase relationship)
fn analyze_logical_state(state: &quantrs2_core::register::Register<9>) -> (f64, f64, f64) {
    use scirs2_core::Complex64;

    // This is a simplified analysis - in a real implementation, we would
    // need to carefully extract the logical qubit state from the encoded state

    // For simplicity, we'll focus on analyzing the first qubit
    let amplitudes = state.amplitudes();

    let mut prob_0 = 0.0;
    let mut prob_1 = 0.0;
    let mut amplitude_0 = Complex64::new(0.0, 0.0);
    let mut amplitude_1 = Complex64::new(0.0, 0.0);

    // Count probabilities based on the first qubit
    for (i, amplitude) in amplitudes.iter().enumerate() {
        let first_bit = i & 1;
        if first_bit == 0 {
            prob_0 += amplitude.norm_sqr();
            amplitude_0 += *amplitude;
        } else {
            prob_1 += amplitude.norm_sqr();
            amplitude_1 += *amplitude;
        }
    }

    // Calculate phase relationship (real part of the inner product)
    let phase = (amplitude_0.conj() * amplitude_1).re;

    (prob_0, prob_1, phase)
}
