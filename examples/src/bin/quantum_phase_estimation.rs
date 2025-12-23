use quantrs2_circuit::prelude::{Circuit, Simulator};
use quantrs2_core::{gate::multi::CRZ, qubit::QubitId, register::Register};
use quantrs2_sim::noise_advanced::{RealisticNoiseModelBuilder, ThermalRelaxationChannel};
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::Complex64;
use std::f64::consts::PI;
use std::time::Duration;

/// Quantum Phase Estimation (QPE) Algorithm Example
///
/// This example demonstrates the Quantum Phase Estimation algorithm,
/// a fundamental quantum algorithm used in many applications including
/// Shor's algorithm for factoring and quantum chemistry simulations.
///
/// QPE estimates the eigenphase of a unitary operator for a given eigenstate.
/// In this example, we use a controlled-Z rotation as our unitary operator
/// with a known phase value, and show how QPE can estimate this phase.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QuantRS2 Quantum Phase Estimation Example");
    println!("=========================================");

    // Number of bits of precision for phase estimation
    let precision_bits = 5;

    // Total qubits needed: precision bits + 1 target qubit
    let total_qubits = precision_bits + 1;

    // The actual phase we want to estimate (phi = 0.25 = 1/4)
    let true_phase = 0.25;
    println!("True phase: {true_phase:.6} (= 1/4 = 0.25)");

    // Run the ideal QPE algorithm without noise
    run_ideal_qpe(precision_bits, true_phase)?;

    // Run with different levels of noise to show the effect on accuracy
    run_noisy_qpe(precision_bits, true_phase, 0.001)?; // Low noise
    run_noisy_qpe(precision_bits, true_phase, 0.01)?; // Medium noise
    run_noisy_qpe(precision_bits, true_phase, 0.05)?; // High noise

    Ok(())
}

/// Run the Quantum Phase Estimation algorithm without noise
fn run_ideal_qpe(precision_bits: usize, phase: f64) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nRunning ideal QPE with {precision_bits} precision qubits:");

    let total_qubits = precision_bits + 1;
    let target_qubit = total_qubits - 1;

    // Create QPE circuit
    let mut circuit = Circuit::<8>::new(); // Using a fixed size of 8 qubits for simplicity

    // Step 1: Prepare the target qubit in the eigenstate |1⟩
    // For our chosen unitary (controlled-Z rotation), |1⟩ is an eigenstate
    circuit.x(target_qubit)?;

    // Step 2: Apply Hadamard gates to all estimation qubits (0 to precision_bits-1)
    for i in 0..precision_bits {
        circuit.h(i)?;
    }

    // Step 3: Apply controlled-U operations with increasing powers
    // U is a phase rotation by 2π*phase
    // We use controlled-RZ gates as our U operator with appropriate angles
    for i in 0..precision_bits {
        let angle = 2.0 * PI * phase * f64::from(1 << (precision_bits - 1 - i));
        circuit.add_gate(CRZ {
            control: QubitId::new(i as u32),
            target: QubitId::new(target_qubit as u32),
            theta: angle,
        })?;
    }

    // Step 4: Apply inverse Quantum Fourier Transform to the estimation qubits
    inverse_qft(&mut circuit, 0, precision_bits)?;

    // Simulate the circuit
    let simulator = StateVectorSimulator::new();
    let result = simulator.run(&circuit)?;

    // Analyze the results
    analyze_qpe_result(&result, precision_bits, phase);

    Ok(())
}

/// Run the QPE algorithm with realistic noise
fn run_noisy_qpe(
    precision_bits: usize,
    phase: f64,
    noise_level: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\nRunning noisy QPE with {} precision qubits and {:.3}% noise:",
        precision_bits,
        noise_level * 100.0
    );

    let total_qubits = precision_bits + 1;
    let target_qubit = total_qubits - 1;

    // Create QPE circuit (same as in the ideal case)
    let mut circuit = Circuit::<8>::new(); // Using a fixed size of 8 qubits for simplicity

    // Prepare the target qubit in the eigenstate |1⟩
    circuit.x(target_qubit)?;

    // Apply Hadamard gates to all estimation qubits
    for i in 0..precision_bits {
        circuit.h(i)?;
    }

    // Apply controlled-U operations with increasing powers
    for i in 0..precision_bits {
        let angle = 2.0 * PI * phase * f64::from(1 << (precision_bits - 1 - i));
        circuit.add_gate(CRZ {
            control: QubitId::new(i as u32),
            target: QubitId::new(target_qubit as u32),
            theta: angle,
        })?;
    }

    // Apply inverse Quantum Fourier Transform
    inverse_qft(&mut circuit, 0, precision_bits)?;

    // Create a custom noise model with realistic parameters
    let qubits: Vec<QubitId> = (0..8).map(QubitId::new).collect();

    // Scale the noise model parameters based on the noise level
    let t1_us = 100.0 / noise_level.sqrt(); // Higher noise = lower T1
    let t2_us = 50.0 / noise_level.sqrt(); // Higher noise = lower T2
    let gate_error = noise_level;

    // Create the noise model
    let noise_model = RealisticNoiseModelBuilder::new(true)
        .with_custom_thermal_relaxation(
            &qubits,
            Duration::from_micros(t1_us as u64),
            Duration::from_micros(t2_us as u64),
            Duration::from_nanos(40),
        )
        .with_custom_two_qubit_noise(
            &[(QubitId::new(0), QubitId::new(1))],
            gate_error * 2.0, // Two-qubit gates have higher error rates
        )
        .build();

    // Create the noisy simulator
    let mut simulator = StateVectorSimulator::new();
    simulator.set_advanced_noise_model(noise_model);

    // Run the simulation
    let result = simulator.run(&circuit)?;

    // Analyze the results
    analyze_qpe_result(&result, precision_bits, phase);

    Ok(())
}

/// Apply inverse Quantum Fourier Transform to a subset of qubits
fn inverse_qft(
    circuit: &mut Circuit<8>,
    start: usize,
    count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Apply QFT† to qubits in reverse order
    for i in (start..start + count).rev() {
        // Apply Hadamard
        circuit.h(i)?;

        // Apply controlled phase rotations
        for j in (start..i).rev() {
            let angle = -PI / f64::from(1 << (i - j));
            circuit.add_gate(CRZ {
                control: QubitId::new(j as u32),
                target: QubitId::new(i as u32),
                theta: angle,
            })?;
        }
    }

    Ok(())
}

/// Analyze and display the results of the QPE algorithm
fn analyze_qpe_result(result: &Register<8>, precision_bits: usize, true_phase: f64) {
    // Get probabilities and select only the states where the target qubit is 1
    // (since we prepared the target qubit in |1⟩ state)
    let probabilities = result.probabilities();
    let target_qubit = precision_bits; // last qubit

    // Print the top measured states
    println!("Top measurement outcomes:");

    // Create a vector of (state, probability) pairs for sorting
    let mut state_probs: Vec<(usize, f64)> = Vec::new();
    for (i, &prob) in probabilities.iter().enumerate() {
        // Only consider states where the target qubit is 1
        if (i >> target_qubit) & 1 == 1 {
            state_probs.push((i, prob));
        }
    }

    // Sort by probability in descending order
    state_probs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("Failed to compare probabilities (NaN encountered in QPE result analysis)")
    });

    // Print the top 5 states (or fewer if there aren't 5)
    for (idx, (state, prob)) in state_probs.iter().take(5).enumerate() {
        // Extract just the estimation register bits (remove the target qubit)
        let est_bits = (*state & ((1 << precision_bits) - 1)) as u32;

        // Convert to binary string
        let binary = format!("{est_bits:0precision_bits$b}");

        // Convert binary measurement to phase estimate
        let measured_phase = f64::from(est_bits) / f64::from(1 << precision_bits);

        // Calculate error
        let error = (measured_phase - true_phase).abs();

        println!(
            "  {}. |{}⟩|1⟩: {:.6} (phase = {:.6}, error = {:.6})",
            idx + 1,
            binary,
            prob,
            measured_phase,
            error
        );
    }

    // Calculate the expected phase based on the measurement outcomes
    let mut expected_phase = 0.0;
    let mut total_prob = 0.0;

    for (state, prob) in &state_probs {
        // Only consider states with non-negligible probability
        if *prob > 1e-6 {
            let est_bits = (*state & ((1 << precision_bits) - 1)) as u32;
            let phase = f64::from(est_bits) / f64::from(1 << precision_bits);
            expected_phase += phase * prob;
            total_prob += prob;
        }
    }

    // Normalize
    if total_prob > 0.0 {
        expected_phase /= total_prob;
    }

    // Calculate error
    let overall_error = (expected_phase - true_phase).abs();

    println!("Expected phase from measurements: {expected_phase:.6}");
    println!("Error in phase estimation: {overall_error:.6}");
    println!("Success rate: {:.2}%", 100.0 * (1.0 - overall_error / 0.5));
}
