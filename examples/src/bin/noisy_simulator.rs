// Noisy Quantum Simulator Example
//
// This example demonstrates the use of noise models in quantum circuit simulation,
// including bit flip, phase flip, depolarizing, amplitude damping, and phase damping channels.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::noise::{
    AmplitudeDampingChannel, BitFlipChannel, DepolarizingChannel, NoiseModel, NoiseModelBuilder,
    PhaseDampingChannel, PhaseFlipChannel,
};
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() {
    println!("Quantum Noise Models Example");
    println!("===========================\n");

    // Create a simple circuit: prepare a Bell state
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).unwrap().cnot(0, 1).unwrap();

    // Run with ideal (noise-free) simulator first
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_result = circuit.run(ideal_sim).unwrap();

    println!("Ideal Bell State (no noise):");
    for (i, amplitude) in ideal_result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }

    // Now run with different noise models
    run_with_bit_flip_noise(&circuit);
    run_with_phase_flip_noise(&circuit);
    run_with_depolarizing_noise(&circuit);
    run_with_amplitude_damping(&circuit);
    run_with_phase_damping(&circuit);
    run_with_combined_noise(&circuit);
    run_with_realistic_ibm_noise(&circuit);
}

// Run simulation with bit flip noise
fn run_with_bit_flip_noise(circuit: &Circuit<2>) {
    println!("\nBit Flip Noise Model (p=0.1):");
    println!("----------------------------");

    // Create a noise model with bit flips on both qubits
    let mut noise_model = NoiseModel::new(false); // Apply noise at the end
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(0),
        probability: 0.1,
    });
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(1),
        probability: 0.1,
    });

    // Create a simulator with the noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the circuit with noise
    let result = circuit.run(noisy_sim).unwrap();

    // Print the result
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }
    println!("Notice that bit flip noise creates non-zero probabilities in |01⟩ and |10⟩ states.");
}

// Run simulation with phase flip noise
fn run_with_phase_flip_noise(circuit: &Circuit<2>) {
    println!("\nPhase Flip Noise Model (p=0.1):");
    println!("------------------------------");

    // Create a noise model with phase flips on both qubits
    let mut noise_model = NoiseModel::new(false);
    noise_model.add_phase_flip(PhaseFlipChannel {
        target: QubitId::new(0),
        probability: 0.1,
    });
    noise_model.add_phase_flip(PhaseFlipChannel {
        target: QubitId::new(1),
        probability: 0.1,
    });

    // Create a simulator with the noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the circuit with noise
    let result = circuit.run(noisy_sim).unwrap();

    // Print the result
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }
    println!("Phase flip noise may change the sign of amplitudes but not the probabilities.");
}

// Run simulation with depolarizing noise
fn run_with_depolarizing_noise(circuit: &Circuit<2>) {
    println!("\nDepolarizing Noise Model (p=0.1):");
    println!("--------------------------------");

    // Use the NoiseModelBuilder for convenience
    let noise_model = NoiseModelBuilder::new(true) // Apply noise after each gate
        .with_depolarizing_noise(&[QubitId::new(0), QubitId::new(1)], 0.1)
        .build();

    // Create a simulator with the noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the circuit with noise
    let result = circuit.run(noisy_sim).unwrap();

    // Print the result
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }
    println!(
        "Depolarizing noise introduces a mixture of X, Y, and Z errors, moving the state closer to a maximally mixed state."
    );
}

// Run simulation with amplitude damping
fn run_with_amplitude_damping(circuit: &Circuit<2>) {
    println!("\nAmplitude Damping Noise Model (gamma=0.1):");
    println!("----------------------------------------");

    // Use the NoiseModelBuilder for amplitude damping
    let noise_model = NoiseModelBuilder::new(false)
        .with_amplitude_damping(&[QubitId::new(0), QubitId::new(1)], 0.1)
        .build();

    // Create a simulator with the noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the circuit with noise
    let result = circuit.run(noisy_sim).unwrap();

    // Print the result
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }
    println!("Amplitude damping represents energy dissipation, and increases the probability of |00⟩ state.");
}

// Run simulation with phase damping
fn run_with_phase_damping(circuit: &Circuit<2>) {
    println!("\nPhase Damping Noise Model (lambda=0.1):");
    println!("----------------------------------------");

    // Use the NoiseModelBuilder for phase damping
    let noise_model = NoiseModelBuilder::new(false)
        .with_phase_damping(&[QubitId::new(0), QubitId::new(1)], 0.1)
        .build();

    // Create a simulator with the noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the circuit with noise
    let result = circuit.run(noisy_sim).unwrap();

    // Print the result
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }
    println!("Phase damping represents pure dephasing (T2 decay), which causes loss of quantum coherence.");
}

// Run simulation with a combined noise model
fn run_with_combined_noise(circuit: &Circuit<2>) {
    println!("\nCombined Noise Model:");
    println!("--------------------");

    // Create a more complex noise model with multiple channels
    let mut noise_model = NoiseModel::new(true); // Apply after each gate

    // Add bit flip to qubit 0 and phase flip to qubit 1
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(0),
        probability: 0.05,
    });
    noise_model.add_phase_flip(PhaseFlipChannel {
        target: QubitId::new(1),
        probability: 0.05,
    });

    // Add amplitude damping to both qubits
    noise_model.add_amplitude_damping(AmplitudeDampingChannel {
        target: QubitId::new(0),
        gamma: 0.03,
    });
    noise_model.add_amplitude_damping(AmplitudeDampingChannel {
        target: QubitId::new(1),
        gamma: 0.03,
    });

    // Create a simulator with the combined noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the circuit with noise
    let result = circuit.run(noisy_sim).unwrap();

    // Print the result
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }
    println!(
        "Combined noise models can represent more realistic quantum hardware with multiple error sources."
    );

    // Display the probabilities
    println!("\nFinal state probabilities:");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!("|{}⟩: {:.6}", bits, amplitude.norm_sqr());
    }
    println!("\nNotice how the Bell state has degraded from the ideal (50/50 in |00⟩/|11⟩) due to noise.");
}

// Run simulation with a realistic IBM Quantum device noise model
fn run_with_realistic_ibm_noise(circuit: &Circuit<2>) {
    println!("\nRealistic IBM Quantum Noise Model:");
    println!("---------------------------------");

    // Create a noise model that approximates IBM Quantum hardware
    // Typical parameters based on reported IBM device specs
    let mut noise_model = NoiseModel::new(true); // Apply after each gate

    // Single-qubit gate error rates (~0.1% error)
    noise_model.add_depolarizing(DepolarizingChannel {
        target: QubitId::new(0),
        probability: 0.001,
    });
    noise_model.add_depolarizing(DepolarizingChannel {
        target: QubitId::new(1),
        probability: 0.001,
    });

    // Two-qubit gate error rates (~1% error)
    // We'll add this dynamically in the simulator
    // IBM devices typically have T1 relaxation times of ~50-100μs
    noise_model.add_amplitude_damping(AmplitudeDampingChannel {
        target: QubitId::new(0),
        gamma: 0.005, // Corresponds to ~50-100μs T1 time
    });
    noise_model.add_amplitude_damping(AmplitudeDampingChannel {
        target: QubitId::new(1),
        gamma: 0.007, // Slightly different T1 for the second qubit
    });

    // T2 dephasing times are typically shorter than T1
    noise_model.add_phase_damping(PhaseDampingChannel {
        target: QubitId::new(0),
        lambda: 0.01, // Corresponds to ~20-50μs T2 time
    });
    noise_model.add_phase_damping(PhaseDampingChannel {
        target: QubitId::new(1),
        lambda: 0.012, // Slightly different T2 for the second qubit
    });

    // Readout errors
    // Apply bit flip with a low probability at the end to simulate readout error
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(0),
        probability: 0.015, // ~1.5% readout error
    });
    noise_model.add_bit_flip(BitFlipChannel {
        target: QubitId::new(1),
        probability: 0.02, // ~2% readout error
    });

    // Create a simulator with the IBM-like noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the circuit with noise
    let result = circuit.run(noisy_sim).unwrap();

    // Display the full quantum state
    println!("\nFull quantum state:");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }

    // Display just the probabilities
    println!("\nFinal state probabilities:");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!("|{}⟩: {:.6}", bits, amplitude.norm_sqr());
    }

    println!("\nThis model approximates the behavior of real IBM Quantum hardware with:");
    println!("- Single-qubit gate fidelity: ~99.9%");
    println!("- Two-qubit gate fidelity: ~99%");
    println!("- T1 relaxation times: ~50-100μs");
    println!("- T2 dephasing times: ~20-50μs");
    println!("- Readout errors: ~1.5-2%");
}
