use quantrs2_circuit::builder::Simulator;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::{
    gate::multi::CNOT, gate::multi::CZ, gate::single::Hadamard, qubit::QubitId, register::Register,
};
use quantrs2_sim::noise_advanced::{
    AdvancedNoiseModel, CrosstalkChannel, RealisticNoiseModelBuilder,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::random::{thread_rng, Rng};
use scirs2_core::Complex64;
use std::time::Duration;

/// Grover's Algorithm with Realistic Noise Models
///
/// This example demonstrates Grover's search algorithm with realistic noise models
/// to show how quantum noise affects the performance of the algorithm.
///
/// Grover's algorithm allows searching an unstructured database of N elements
/// in approximately O(√N) time, which is quadratically faster than the classical O(N).
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QuantRS2 Grover's Algorithm with Realistic Noise Example");
    println!("=====================================================");

    // Run Grover's algorithm for different database sizes
    for n_qubits in 3..=6 {
        // The database size is 2^n_qubits
        let db_size = 1 << n_qubits;

        println!("\n===== Database size: {db_size} (using {n_qubits} qubits) =====");

        // Pick a random item to search for (the marked element)
        let marked_item = thread_rng().gen_range(0..db_size);
        println!("Searching for marked item: {marked_item} (binary: {marked_item:0n_qubits$b})");

        // Run without noise first for comparison
        run_grovers(n_qubits, marked_item, None)?;

        // Run with different noise levels to show degradation
        let noise_levels = [0.001, 0.01, 0.05];

        for &noise_level in &noise_levels {
            let noise_model = create_noise_model(n_qubits, noise_level);
            run_grovers(n_qubits, marked_item, Some(noise_model))?;
        }
    }

    Ok(())
}

/// Runs Grover's algorithm with optional noise
fn run_grovers(
    n_qubits: usize,
    marked_item: usize,
    noise_model: Option<AdvancedNoiseModel>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Max qubits we can handle with Circuit<T> (using 16 as a safe default)
    // In a production system, we'd use the dynamic circuit instead
    let max_supported_qubits = 16;

    if n_qubits > max_supported_qubits {
        return Err(format!(
            "Number of qubits ({n_qubits}) exceeds maximum supported ({max_supported_qubits})"
        )
        .into());
    }

    // Number of Grover iterations to apply
    let iterations = calculate_optimal_iterations(n_qubits);

    // Create the circuit - we need to handle different sizes
    // This is where templates in Rust shine
    match n_qubits {
        3 => run_grovers_with_size::<3>(marked_item, iterations, noise_model),
        4 => run_grovers_with_size::<4>(marked_item, iterations, noise_model),
        5 => run_grovers_with_size::<5>(marked_item, iterations, noise_model),
        6 => run_grovers_with_size::<6>(marked_item, iterations, noise_model),
        _ => Err(format!("Unsupported number of qubits: {n_qubits}").into()),
    }
}

/// Runs Grover's algorithm with a fixed circuit size
fn run_grovers_with_size<const N: usize>(
    marked_item: usize,
    iterations: usize,
    noise_model: Option<AdvancedNoiseModel>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create the circuit
    let mut circuit = Circuit::<N>::new();

    // Step 1: Apply Hadamard to all qubits to create equal superposition
    for i in 0..N {
        circuit.h(i)?;
    }

    // Step 2: Apply Grover iterations
    for _ in 0..iterations {
        // Apply oracle (marks the solution)
        apply_oracle(&mut circuit, marked_item)?;

        // Apply diffusion operator (amplifies the marked state)
        apply_diffusion(&mut circuit)?;
    }

    // Create simulator with appropriate noise model
    let mut simulator = StateVectorSimulator::new();
    if let Some(noise) = noise_model {
        let num_channels = noise.num_channels();
        simulator.set_advanced_noise_model(noise);
        println!("\nRunning with noise model ({num_channels} channels)");
    } else {
        println!("\nRunning without noise (ideal simulation)");
    }

    // Run the simulation
    let result = simulator.run(&circuit)?;

    // Analyze the results
    analyze_results(&result, marked_item);

    Ok(())
}

/// Apply the oracle that marks the solution state
fn apply_oracle<const N: usize>(
    circuit: &mut Circuit<N>,
    marked_item: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // For the given marked item, we need to apply a phase flip
    // to that specific computational basis state

    // First, we need to apply X gates to qubits that correspond to 0s
    // in the binary representation of the marked item
    for i in 0..N {
        // If the i-th bit of marked_item is 0, apply X gate
        if (marked_item >> i) & 1 == 0 {
            circuit.x(i)?;
        }
    }

    // Apply a multi-controlled Z gate to flip the phase of the marked state
    // We'll use a recursive approach similar to the Toffoli decomposition
    match N {
        1 => {
            // For 1 qubit, just apply Z
            circuit.z(0)?;
        }
        2 => {
            // For 2 qubits, use CZ
            circuit.add_gate(CZ {
                control: QubitId::new(0),
                target: QubitId::new(1),
            })?;
        }
        _ => {
            // For N > 2, we need to build a multi-controlled Z gate
            // We use a standard approach with an ancilla qubit
            // This is slightly simplified as a full implementation
            // would use multiple ancilla qubits for better depth

            // Using the last qubit as target for the phase flip
            let target = N - 1;

            // Apply multi-controlled X gate (to flip if all controls are 1)
            for ctrl in 0..target {
                circuit.add_gate(CNOT {
                    control: QubitId::new(ctrl as u32),
                    target: QubitId::new(target as u32),
                })?;
            }

            // Apply Z to the target
            circuit.z(target)?;

            // Undo the multi-controlled X gate
            for ctrl in (0..target).rev() {
                circuit.add_gate(CNOT {
                    control: QubitId::new(ctrl as u32),
                    target: QubitId::new(target as u32),
                })?;
            }
        }
    }

    // Undo the X gates
    for i in 0..N {
        if (marked_item >> i) & 1 == 0 {
            circuit.x(i)?;
        }
    }

    Ok(())
}

/// Apply the diffusion operator (Grover iteration)
fn apply_diffusion<const N: usize>(
    circuit: &mut Circuit<N>,
) -> Result<(), Box<dyn std::error::Error>> {
    // The diffusion operator is: 2|s⟩⟨s| - I
    // where |s⟩ is the equal superposition state

    // Apply Hadamard to all qubits
    for i in 0..N {
        circuit.h(i)?;
    }

    // Apply X to all qubits
    for i in 0..N {
        circuit.x(i)?;
    }

    // Apply the multi-controlled Z (same approach as in the oracle)
    match N {
        1 => {
            circuit.z(0)?;
        }
        2 => {
            circuit.add_gate(CZ {
                control: QubitId::new(0),
                target: QubitId::new(1),
            })?;
        }
        _ => {
            // Using the last qubit as target for the phase flip
            let target = N - 1;

            // Apply multi-controlled X gate
            for ctrl in 0..target {
                circuit.add_gate(CNOT {
                    control: QubitId::new(ctrl as u32),
                    target: QubitId::new(target as u32),
                })?;
            }

            // Apply Z to the target
            circuit.z(target)?;

            // Undo the multi-controlled X gate
            for ctrl in (0..target).rev() {
                circuit.add_gate(CNOT {
                    control: QubitId::new(ctrl as u32),
                    target: QubitId::new(target as u32),
                })?;
            }
        }
    }

    // Undo X gates on all qubits
    for i in 0..N {
        circuit.x(i)?;
    }

    // Undo Hadamard gates on all qubits
    for i in 0..N {
        circuit.h(i)?;
    }

    Ok(())
}

/// Calculate the optimal number of Grover iterations
fn calculate_optimal_iterations(n_qubits: usize) -> usize {
    // The optimal number of iterations is approximately (π/4)·√N
    // where N = 2^n_qubits

    let n = 1_usize << n_qubits; // 2^n_qubits
    let f_iterations = (std::f64::consts::PI / 4.0) * (n as f64).sqrt();

    // Round to nearest integer
    f_iterations.round() as usize
}

/// Analyze and display the results of Grover's algorithm
fn analyze_results<const N: usize>(result: &Register<N>, marked_item: usize) {
    // Get probabilities
    let probabilities = result.probabilities();

    // Create a vector of (state, probability) pairs for sorting
    let mut state_probs: Vec<(usize, f64)> = Vec::new();
    for (i, &prob) in probabilities.iter().enumerate() {
        if prob > 0.001 {
            // Only include states with non-negligible probability
            state_probs.push((i, prob));
        }
    }

    // Sort by probability in descending order
    state_probs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("Failed to compare probabilities (NaN encountered in Grover's algorithm result analysis)")
    });

    // Print the top 5 states
    println!("Top measured states:");
    for (idx, (state, prob)) in state_probs.iter().take(5).enumerate() {
        // Convert to binary string
        let binary = format!("{state:0N$b}");

        let marker = if *state == marked_item {
            " <<< CORRECT"
        } else {
            ""
        };

        println!(
            "  {}. |{}⟩: {:.6} ({:.1}%){}",
            idx + 1,
            binary,
            prob,
            prob * 100.0,
            marker
        );
    }

    // Find the marked item in the sorted list
    let marked_position = state_probs
        .iter()
        .position(|(state, _)| *state == marked_item);

    // Calculate success probability
    let marked_prob = match marked_position {
        Some(pos) => state_probs[pos].1,
        None => 0.0,
    };

    println!(
        "Probability of measuring the marked item: {:.6} ({:.1}%)",
        marked_prob,
        marked_prob * 100.0
    );

    // Calculate success rate compared to ideal
    let ideal_prob = 1.0; // Ideally, Grover's gives certain or near-certain success
    let success_rate = (marked_prob / ideal_prob) * 100.0;

    println!("Success rate: {success_rate:.1}%");
}

/// Create a realistic noise model with the given noise level
fn create_noise_model(n_qubits: usize, noise_level: f64) -> AdvancedNoiseModel {
    // Create qubit IDs
    let qubits: Vec<QubitId> = (0..n_qubits).map(|i| QubitId::new(i as u32)).collect();

    // Create pairs for two-qubit noise
    let mut qubit_pairs = Vec::new();
    for i in 0..n_qubits - 1 {
        qubit_pairs.push((QubitId::new(i as u32), QubitId::new((i + 1) as u32)));
    }

    // Scale the noise model parameters based on the noise level
    let t1_us = 150.0 / noise_level.sqrt(); // Higher noise = lower T1
    let t2_us = 100.0 / noise_level.sqrt(); // Higher noise = lower T2
    let gate_error_1q = noise_level * 0.5; // Single-qubit gate error
    let gate_error_2q = noise_level * 2.0; // Two-qubit gate error

    // Create a description of the noise level
    println!(
        "Creating noise model with level {:.3}%:",
        noise_level * 100.0
    );
    println!("  - T1 relaxation time: {t1_us:.1} μs");
    println!("  - T2 dephasing time: {t2_us:.1} μs");
    println!("  - 1-qubit gate error: {gate_error_1q:.5}");
    println!("  - 2-qubit gate error: {gate_error_2q:.5}");

    // Create the noise model

    RealisticNoiseModelBuilder::new(true)
        .with_custom_thermal_relaxation(
            &qubits,
            Duration::from_micros(t1_us as u64),
            Duration::from_micros(t2_us as u64),
            Duration::from_nanos(50), // 50ns gate time
        )
        .with_custom_two_qubit_noise(&qubit_pairs, gate_error_2q)
        .build()
}
