use num_complex::Complex64;
use quantrs2_circuit::builder::Simulator;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::noise_advanced::{AdvancedNoiseModel, RealisticNoiseModelBuilder};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::collections::HashMap;
use std::time::Duration;

/// This example demonstrates how to use realistic noise models for quantum simulation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QuantRS2 Realistic Noise Simulation Example");
    println!("===========================================");

    // Compare results for different noise models
    compare_bell_state_with_noise()?;

    // Show how to create and use IBM device noise model
    simulate_with_ibm_noise()?;

    // Demonstrate custom noise parameters
    simulate_with_custom_noise()?;

    Ok(())
}

/// Compare Bell state with different noise models
fn compare_bell_state_with_noise() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Bell State with Different Noise Models");
    println!("---------------------------------------");

    // Create a Bell state circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(0)?;
    circuit.cnot(0, 1)?;

    // Create simulators with different noise models
    let simulator_ideal = StateVectorSimulator::new();

    // Create a basic depolarizing noise model
    let mut builder = RealisticNoiseModelBuilder::new(true);
    let basic_noise = builder
        .with_custom_thermal_relaxation(
            &[QubitId::new(0), QubitId::new(1)],
            Duration::from_micros(50), // T1 = 50μs
            Duration::from_micros(30), // T2 = 30μs
            Duration::from_nanos(40),  // Gate time = 40ns
        )
        .build();

    let mut simulator_basic = StateVectorSimulator::new();
    simulator_basic.set_advanced_noise_model(basic_noise);

    // Create an IBM device noise model
    let ibm_noise = RealisticNoiseModelBuilder::new(true)
        .with_ibm_device_noise(&[QubitId::new(0), QubitId::new(1)], "ibmq_lima")
        .build();

    let mut simulator_ibm = StateVectorSimulator::new();
    simulator_ibm.set_advanced_noise_model(ibm_noise);

    // Run simulations
    println!("Ideal simulation (no noise):");
    let result_ideal = simulator_ideal.run(&circuit)?;
    print_state_probabilities(&result_ideal.probabilities(), 2);

    println!("\nSimulation with basic thermal relaxation noise:");
    let result_basic = simulator_basic.run(&circuit)?;
    print_state_probabilities(&result_basic.probabilities(), 2);

    println!("\nSimulation with realistic IBM noise model:");
    let result_ibm = simulator_ibm.run(&circuit)?;
    print_state_probabilities(&result_ibm.probabilities(), 2);

    Ok(())
}

/// Simulate a GHZ state with IBM device noise model
fn simulate_with_ibm_noise() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. GHZ State with IBM Device Noise Model");
    println!("--------------------------------------");

    // Create a 5-qubit GHZ state circuit
    let mut circuit = Circuit::<5>::new();
    circuit.h(0)?;
    circuit.cnot(0, 1)?;
    circuit.cnot(1, 2)?;
    circuit.cnot(2, 3)?;
    circuit.cnot(3, 4)?;

    println!("Created 5-qubit GHZ state circuit");

    // Create IBM device noise model
    let qubits = (0..5).map(|i| QubitId::new(i as u32)).collect::<Vec<_>>();

    // Try different device models
    let device_models = ["ibmq_lima", "ibmq_bogota", "ibm_cairo"];

    for device in device_models {
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_ibm_device_noise(&qubits, device)
            .build();

        // Create simulator with noise model
        let mut simulator = StateVectorSimulator::new();
        simulator.set_advanced_noise_model(noise_model);

        // Run simulation
        println!("\nSimulation with {} noise model:", device);
        let result = simulator.run(&circuit)?;

        // Count states with significant probability
        let significant_states = result
            .probabilities()
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.01)
            .collect::<Vec<_>>();

        // Print state statistics
        println!(
            "Number of significant states (p > 1%): {}",
            significant_states.len()
        );

        // Calculate fidelity to ideal state (|00000⟩ + |11111⟩)/√2
        let ideal_states = [0b00000, 0b11111];
        let fidelity = calculate_fidelity(&result.amplitudes(), &ideal_states);
        println!("Fidelity to ideal GHZ state: {:.4}", fidelity);

        // Print the top 5 most probable states
        println!("Top 5 most probable states:");
        let mut probs = result
            .probabilities()
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (i, (idx, prob)) in probs.iter().take(5).enumerate() {
            println!("  {}. |{:05b}⟩: {:.6}", i + 1, idx, prob);
        }
    }

    Ok(())
}

/// Simulate a custom circuit with custom noise parameters
fn simulate_with_custom_noise() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Custom Circuit with Custom Noise Parameters");
    println!("--------------------------------------------");

    // Create a custom circuit: Quantum Fourier Transform on 3 qubits
    let mut circuit = Circuit::<3>::new();

    // Apply Hadamard to all qubits
    circuit.h(0)?;
    circuit.h(1)?;
    circuit.h(2)?;

    // Apply controlled phase rotations
    circuit.cz(0, 1)?;
    circuit.cz(0, 2)?;
    circuit.cz(1, 2)?;

    // Apply more Hadamards to create superposition
    circuit.h(0)?;
    circuit.h(1)?;
    circuit.h(2)?;

    println!("Created 3-qubit Quantum Fourier Transform circuit");

    // Create custom noise model with different noise types
    let qubits = [QubitId::new(0), QubitId::new(1), QubitId::new(2)];
    let qubit_pairs = [
        (QubitId::new(0), QubitId::new(1)),
        (QubitId::new(1), QubitId::new(2)),
    ];

    let custom_noise = RealisticNoiseModelBuilder::new(true)
        .with_custom_thermal_relaxation(
            &qubits,
            Duration::from_micros(80), // T1 = 80μs
            Duration::from_micros(40), // T2 = 40μs
            Duration::from_nanos(50),  // Gate time = 50ns
        )
        .with_custom_two_qubit_noise(
            &qubit_pairs,
            0.015, // 1.5% error rate for two-qubit gates
        )
        .with_custom_crosstalk(
            &qubit_pairs,
            0.005, // 0.5% crosstalk between neighboring qubits
        )
        .build();

    // Create simulator with noise model
    let mut simulator = StateVectorSimulator::new();
    simulator.set_advanced_noise_model(custom_noise);

    // Run ideal simulation
    let simulator_ideal = StateVectorSimulator::new();
    let result_ideal = simulator_ideal.run(&circuit)?;

    println!("\nIdeal simulation (no noise):");
    print_state_probabilities(&result_ideal.probabilities(), 3);

    // Run noisy simulation
    println!("\nSimulation with custom noise model:");
    let result_noisy = simulator.run(&circuit)?;
    print_state_probabilities(&result_noisy.probabilities(), 3);

    // Calculate fidelity between ideal and noisy results
    let fidelity = calculate_state_fidelity(&result_ideal.amplitudes(), &result_noisy.amplitudes());
    println!("\nFidelity between ideal and noisy states: {:.6}", fidelity);

    Ok(())
}

/// Print state probabilities in a readable format
fn print_state_probabilities(probabilities: &[f64], num_qubits: usize) {
    // Create a map of states to probabilities
    let mut state_probs = HashMap::new();
    for (i, &prob) in probabilities.iter().enumerate() {
        if prob > 0.001 {
            // Only show states with probability > 0.1%
            let state = format!("{:0width$b}", i, width = num_qubits);
            state_probs.insert(state, prob);
        }
    }

    // Print the states
    for (state, prob) in state_probs.iter() {
        println!("  |{}⟩: {:.6} ({:.2}%)", state, prob, prob * 100.0);
    }
}

/// Calculate fidelity to target states
fn calculate_fidelity(amplitudes: &[Complex64], target_states: &[usize]) -> f64 {
    let mut fidelity = 0.0;
    let target_prob = 1.0 / (target_states.len() as f64);

    for &state in target_states {
        fidelity += amplitudes[state].norm_sqr();
    }

    // For pure target states, this is the fidelity
    // For mixed states like (|00000⟩ + |11111⟩)/√2, we need to normalize
    fidelity / target_states.len() as f64
}

/// Calculate state fidelity between two quantum states
fn calculate_state_fidelity(state1: &[Complex64], state2: &[Complex64]) -> f64 {
    assert_eq!(
        state1.len(),
        state2.len(),
        "States must have same dimension"
    );

    let mut fidelity = Complex64::new(0.0, 0.0);
    for i in 0..state1.len() {
        fidelity += state1[i].conj() * state2[i];
    }

    fidelity.norm_sqr()
}
