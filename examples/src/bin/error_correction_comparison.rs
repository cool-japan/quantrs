// Error Correction Comparison Example
//
// This example compares the effectiveness of different quantum error correction codes
// against various types of noise, showing their strengths and limitations.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::error_correction::{
    utils, BitFlipCode, ErrorCorrection, FiveQubitCode, PhaseFlipCode, ShorCode,
};
use quantrs2_sim::noise::{
    AmplitudeDampingChannel, BitFlipChannel, DepolarizingChannel, NoiseModel, PhaseFlipChannel,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::time::Instant;

fn main() {
    println!("Quantum Error Correction Code Comparison");
    println!("========================================\n");

    // We'll test each code against different types of errors
    // and compare their performance

    println!("This example compares the performance of various error correction codes:");
    println!("- 3-qubit bit flip code (protects against X errors)");
    println!("- 3-qubit phase flip code (protects against Z errors)");
    println!("- 9-qubit Shor code (protects against arbitrary single-qubit errors)");
    println!("- 5-qubit perfect code (protects against arbitrary single-qubit errors)");
    println!("\nWe'll test each code against different error types and compare effectiveness\n");

    // First, create a test state: |+⟩ = (|0⟩ + |1⟩)/√2
    let mut base_circuit = Circuit::<1>::new();
    base_circuit
        .h(0)
        .expect("Failed to apply H gate for base state");

    // Verify the base state is correct
    let ideal_sim = StateVectorSimulator::sequential();
    let base_state = base_circuit
        .run(ideal_sim)
        .expect("Failed to run base circuit");
    println!("Base state: |+⟩ = (|0⟩ + |1⟩)/√2");
    println!(
        "Probabilities: |0⟩ = {:.6}, |1⟩ = {:.6}\n",
        base_state.amplitudes()[0].norm_sqr(),
        base_state.amplitudes()[1].norm_sqr()
    );

    // Test against bit flip errors
    println!("TEST 1: BIT FLIP ERRORS (X ERRORS)");
    println!("-----------------------------------");
    test_with_bit_flip_error();

    // Test against phase flip errors
    println!("\nTEST 2: PHASE FLIP ERRORS (Z ERRORS)");
    println!("--------------------------------------");
    test_with_phase_flip_error();

    // Test against combined errors
    println!("\nTEST 3: ARBITRARY ERRORS (X, Y, Z ERRORS)");
    println!("------------------------------------------");
    test_with_arbitrary_errors();

    // Compare code sizes and encoding/decoding efficiency
    println!("\nCODE EFFICIENCY COMPARISON");
    println!("--------------------------");
    compare_efficiency();

    println!("\nSUMMARY");
    println!("-------");
    println!("- 3-qubit bit flip code: Efficient for X errors, fails for Z/Y errors");
    println!("- 3-qubit phase flip code: Efficient for Z errors, fails for X/Y errors");
    println!("- 9-qubit Shor code: Handles all error types but requires 9 physical qubits");
    println!("- 5-qubit perfect code: Handles all error types with optimal qubit count");
    println!("\nThe 5-qubit code is generally the best choice when arbitrary errors might occur");
    println!("and resources are limited. The 3-qubit codes are suitable when the error type");
    println!("is known in advance.");
}

// Test all codes against bit flip errors
fn test_with_bit_flip_error() {
    // Base test state
    let mut base_circuit = Circuit::<1>::new();
    base_circuit
        .h(0)
        .expect("Failed to apply H gate in bit flip test"); // |+⟩ state

    // Create a noise model with 100% bit flip probability
    let mut noise_model = NoiseModel::new(false);
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(0),
        probability: 1.0, // Guaranteed bit flip for testing
    });

    println!("Testing with 100% bit flip probability on logical qubit\n");

    // Test bit flip code
    println!("3-qubit bit flip code:");
    test_code_with_noise(&BitFlipCode, &base_circuit, &noise_model);

    // Test phase flip code
    println!("\n3-qubit phase flip code:");
    test_code_with_noise(&PhaseFlipCode, &base_circuit, &noise_model);

    // Test Shor code
    println!("\n9-qubit Shor code:");
    test_code_with_noise(&ShorCode, &base_circuit, &noise_model);

    // Test 5-qubit code
    println!("\n5-qubit perfect code:");
    test_code_with_noise(&FiveQubitCode, &base_circuit, &noise_model);
}

// Test all codes against phase flip errors
fn test_with_phase_flip_error() {
    // Base test state
    let mut base_circuit = Circuit::<1>::new();
    base_circuit
        .h(0)
        .expect("Failed to apply H gate in phase flip test"); // |+⟩ state

    // Create a noise model with 100% phase flip probability
    let mut noise_model = NoiseModel::new(false);
    noise_model.add_phase_flip(PhaseFlipChannel {
        target: QubitId::new(0),
        probability: 1.0, // Guaranteed phase flip for testing
    });

    println!("Testing with 100% phase flip probability on logical qubit\n");

    // Test bit flip code
    println!("3-qubit bit flip code:");
    test_code_with_noise(&BitFlipCode, &base_circuit, &noise_model);

    // Test phase flip code
    println!("\n3-qubit phase flip code:");
    test_code_with_noise(&PhaseFlipCode, &base_circuit, &noise_model);

    // Test Shor code
    println!("\n9-qubit Shor code:");
    test_code_with_noise(&ShorCode, &base_circuit, &noise_model);

    // Test 5-qubit code
    println!("\n5-qubit perfect code:");
    test_code_with_noise(&FiveQubitCode, &base_circuit, &noise_model);
}

// Test all codes against arbitrary errors (X, Y, Z)
fn test_with_arbitrary_errors() {
    // Base test state
    let mut base_circuit = Circuit::<1>::new();
    base_circuit
        .h(0)
        .expect("Failed to apply H gate in arbitrary errors test"); // |+⟩ state

    // Create a noise model with depolarizing noise (random X, Y, Z errors)
    let mut noise_model = NoiseModel::new(false);
    noise_model.add_depolarizing(DepolarizingChannel {
        target: QubitId::new(0),
        probability: 0.9, // High probability for testing
    });

    println!("Testing with 90% depolarizing noise on logical qubit\n");

    // Test bit flip code
    println!("3-qubit bit flip code:");
    test_code_with_noise(&BitFlipCode, &base_circuit, &noise_model);

    // Test phase flip code
    println!("\n3-qubit phase flip code:");
    test_code_with_noise(&PhaseFlipCode, &base_circuit, &noise_model);

    // Test Shor code
    println!("\n9-qubit Shor code:");
    test_code_with_noise(&ShorCode, &base_circuit, &noise_model);

    // Test 5-qubit code
    println!("\n5-qubit perfect code:");
    test_code_with_noise(&FiveQubitCode, &base_circuit, &noise_model);
}

// Compare efficiency of different codes
fn compare_efficiency() {
    // Display physical qubit requirements
    let bit_flip = BitFlipCode;
    let phase_flip = PhaseFlipCode;
    let shor = ShorCode;
    let five_qubit = FiveQubitCode;

    println!("Physical qubits required:");
    println!(
        "- 3-qubit bit flip code: {} qubits",
        bit_flip.physical_qubits()
    );
    println!(
        "- 3-qubit phase flip code: {} qubits",
        phase_flip.physical_qubits()
    );
    println!("- 9-qubit Shor code: {} qubits", shor.physical_qubits());
    println!(
        "- 5-qubit perfect code: {} qubits",
        five_qubit.physical_qubits()
    );

    // Compare encoding/decoding time
    println!("\nEncoding time comparison:");

    // Prepare test circuit
    let mut base_circuit = Circuit::<1>::new();
    base_circuit
        .h(0)
        .expect("Failed to apply H gate in efficiency comparison");

    // Measure encoding time for each code
    let base_qubits = vec![QubitId::new(0)];

    // Bit flip code
    let start = Instant::now();
    let _ = bit_flip.encode_circuit(&base_qubits, &(1..3).map(QubitId::new).collect::<Vec<_>>());
    let bit_flip_encode_time = start.elapsed();

    // Phase flip code
    let start = Instant::now();
    let _ = phase_flip.encode_circuit(&base_qubits, &(1..3).map(QubitId::new).collect::<Vec<_>>());
    let phase_flip_encode_time = start.elapsed();

    // Shor code
    let start = Instant::now();
    let _ = shor.encode_circuit(&base_qubits, &(1..9).map(QubitId::new).collect::<Vec<_>>());
    let shor_encode_time = start.elapsed();

    // 5-qubit code
    let start = Instant::now();
    let _ = five_qubit.encode_circuit(&base_qubits, &(1..5).map(QubitId::new).collect::<Vec<_>>());
    let five_qubit_encode_time = start.elapsed();

    println!("- 3-qubit bit flip code: {bit_flip_encode_time:?}");
    println!("- 3-qubit phase flip code: {phase_flip_encode_time:?}");
    println!("- 9-qubit Shor code: {shor_encode_time:?}");
    println!("- 5-qubit perfect code: {five_qubit_encode_time:?}");

    // Circuit depth (approximate based on gate count)
    println!("\nApproximate circuit complexity (gate count):");

    // Bit flip code
    let encoder =
        bit_flip.encode_circuit(&base_qubits, &(1..3).map(QubitId::new).collect::<Vec<_>>());
    let decoder = bit_flip.decode_circuit(
        &(0..3).map(QubitId::new).collect::<Vec<_>>(),
        &(3..5).map(QubitId::new).collect::<Vec<_>>(),
    );

    println!(
        "- 3-qubit bit flip code: {} encoding gates, {} correction gates",
        encoder.map(|c| c.num_gates()).unwrap_or(0),
        decoder.map(|c| c.num_gates()).unwrap_or(0)
    );

    // Phase flip code
    let encoder =
        phase_flip.encode_circuit(&base_qubits, &(1..3).map(QubitId::new).collect::<Vec<_>>());
    let decoder = phase_flip.decode_circuit(
        &(0..3).map(QubitId::new).collect::<Vec<_>>(),
        &(3..5).map(QubitId::new).collect::<Vec<_>>(),
    );

    println!(
        "- 3-qubit phase flip code: {} encoding gates, {} correction gates",
        encoder.map(|c| c.num_gates()).unwrap_or(0),
        decoder.map(|c| c.num_gates()).unwrap_or(0)
    );

    // Shor code
    let encoder = shor.encode_circuit(&base_qubits, &(1..9).map(QubitId::new).collect::<Vec<_>>());
    let decoder = shor.decode_circuit(
        &(0..9).map(QubitId::new).collect::<Vec<_>>(),
        &(9..17).map(QubitId::new).collect::<Vec<_>>(),
    );

    println!(
        "- 9-qubit Shor code: {} encoding gates, {} correction gates",
        encoder.map(|c| c.num_gates()).unwrap_or(0),
        decoder.map(|c| c.num_gates()).unwrap_or(0)
    );

    // 5-qubit code
    let encoder =
        five_qubit.encode_circuit(&base_qubits, &(1..5).map(QubitId::new).collect::<Vec<_>>());
    let decoder = five_qubit.decode_circuit(
        &(0..5).map(QubitId::new).collect::<Vec<_>>(),
        &(5..9).map(QubitId::new).collect::<Vec<_>>(),
    );

    println!(
        "- 5-qubit perfect code: {} encoding gates, {} correction gates",
        encoder
            .expect("Failed to create 5-qubit encoder circuit")
            .num_gates(),
        decoder
            .expect("Failed to create 5-qubit decoder circuit")
            .num_gates()
    );
}

// Helper function to test a code with specific noise
fn test_code_with_noise<T: ErrorCorrection>(
    code: &T,
    base_circuit: &Circuit<1>,
    noise_model: &NoiseModel,
) {
    // Step 1: Create ideal state for comparison
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_state = base_circuit
        .run(ideal_sim)
        .expect("Failed to run ideal state circuit");

    // Step 2: Run with noise but no error correction
    let noisy_sim = StateVectorSimulator::with_noise(noise_model.clone());
    let noisy_state = base_circuit
        .run(noisy_sim)
        .expect("Failed to run noisy state circuit");

    // Step 3: Setup error correction
    let num_qubits = 1 + code.physical_qubits() - 1 + code.physical_qubits(); // logical + ancilla + syndrome

    let base_qubit = QubitId::new(0);

    // Create a vector of ancilla qubits for encoding
    let ancilla_qubits = (1..code.physical_qubits())
        .map(|i| QubitId::new(i as u32))
        .collect::<Vec<_>>();

    // Create a vector of syndrome qubits for error detection/correction
    let syndrome_qubits = (code.physical_qubits()..2 * code.physical_qubits())
        .map(|i| QubitId::new(i as u32))
        .collect::<Vec<_>>();

    // Create encoded circuit
    let mut encoded_circuit = Circuit::<16>::new();

    // Add the base state preparation
    encoded_circuit
        .h(base_qubit)
        .expect("Failed to apply H gate to base qubit");

    // Add encoding operations
    let encoder = code
        .encode_circuit(&[base_qubit], &ancilla_qubits)
        .expect("Failed to create encoder circuit");
    for gate in encoder.gates() {
        // Convert gate reference to concrete gate type for circuit
        encoded_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add encoding gate to circuit");
    }

    // Step 4: Run the encoded circuit with noise
    let encoded_qubits = (0..code.physical_qubits())
        .map(|i| QubitId::new(i as u32))
        .collect::<Vec<_>>();
    let noisy_encoded_sim = StateVectorSimulator::with_noise(noise_model.clone());
    let noisy_encoded_state = encoded_circuit
        .run(noisy_encoded_sim)
        .expect("Failed to run encoded circuit with noise");

    // Step 5: Apply error correction
    let mut correction_circuit = Circuit::<16>::new();

    // Add the encoded circuit with noise
    for gate in encoded_circuit.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add encoded gate to correction circuit");
    }

    // Add error correction operations
    let correction = code
        .decode_circuit(&encoded_qubits, &syndrome_qubits)
        .expect("Failed to create correction decoder circuit");
    for gate in correction.gates() {
        correction_circuit
            .add_gate_arc(gate.clone())
            .expect("Failed to add correction gate to circuit");
    }

    // Run with error correction
    let corrected_sim = StateVectorSimulator::sequential();
    let corrected_state = correction_circuit
        .run(corrected_sim)
        .expect("Failed to run correction circuit");

    // Step 6: Analyze and compare results
    // Calculate fidelity before and after correction
    let fidelity_before =
        utils::calculate_fidelity(ideal_state.amplitudes(), noisy_state.amplitudes())
            .expect("Failed to calculate fidelity before correction");

    // For corrected state, we need to extract the logical qubit state,
    // but this is a simplified approach for demonstration
    let logical_state = extract_logical_state(&corrected_state);
    let fidelity_after = utils::calculate_fidelity(ideal_state.amplitudes(), &logical_state)
        .expect("Failed to calculate fidelity after correction");

    // Print results
    println!("Fidelity before correction: {fidelity_before:.6}");
    println!("Fidelity after correction: {fidelity_after:.6}");
    println!(
        "Improvement: {:.2}%",
        (fidelity_after - fidelity_before) * 100.0
    );

    // Determine if correction was successful
    if fidelity_after > 0.9 {
        println!("Status: SUCCESS ✓");
    } else if fidelity_after > fidelity_before * 1.5 {
        println!("Status: PARTIAL SUCCESS ⚠");
    } else {
        println!("Status: FAILED ✗");
    }
}

// Simplified function to extract logical state from encoded state
fn extract_logical_state(
    state: &quantrs2_core::register::Register<16>,
) -> Vec<scirs2_core::Complex64> {
    use scirs2_core::Complex64;

    let amplitudes = state.amplitudes();
    let mut logical_state = vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

    // For simplicity, check first qubit state (very simplified approach)
    let mut sum_0 = Complex64::new(0.0, 0.0);
    let mut sum_1 = Complex64::new(0.0, 0.0);
    let mut count_0 = 0;
    let mut count_1 = 0;

    for (i, &amp) in amplitudes.iter().enumerate().take(16) {
        if (i & 1) == 0 {
            sum_0 += amp;
            count_0 += 1;
        } else {
            sum_1 += amp;
            count_1 += 1;
        }
    }

    if count_0 > 0 {
        logical_state[0] = sum_0 / Complex64::new(f64::from(count_0), 0.0);
    }

    if count_1 > 0 {
        logical_state[1] = sum_1 / Complex64::new(f64::from(count_1), 0.0);
    }

    // Normalize
    let norm_squared = logical_state[0].norm_sqr() + logical_state[1].norm_sqr();
    let norm = norm_squared.sqrt();

    if norm > 1e-10 {
        logical_state[0] /= Complex64::new(norm, 0.0);
        logical_state[1] /= Complex64::new(norm, 0.0);
    }

    logical_state
}
