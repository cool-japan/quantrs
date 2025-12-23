// Extended Gates with Noise Example
//
// This example demonstrates the use of various gates and noise models
// for more complex quantum circuits.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::noise::{
    AmplitudeDampingChannel, DepolarizingChannel, NoiseModel, PhaseDampingChannel, PhaseFlipChannel,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::f64::consts::PI;

fn main() {
    println!("Extended Quantum Gates with Noise Example");
    println!("=======================================\n");

    // Run different circuit examples
    run_grover_with_noise();
    run_qft_with_noise();
    run_error_correction_code();
    run_variational_circuit();
}

// Run Grover's algorithm with noise
fn run_grover_with_noise() {
    println!("\nGrover's Algorithm with Noise");
    println!("----------------------------");

    // Create a circuit for a 2-qubit Grover search
    // We're searching for the |11⟩ state
    let mut circuit = Circuit::<2>::new();

    // Initialize in superposition
    circuit
        .h(0)
        .expect("Failed to apply H gate to qubit 0")
        .h(1)
        .expect("Failed to apply H gate to qubit 1");

    // Oracle: Mark the |11⟩ state (using a Z gate controlled by both qubits)
    // For a 2-qubit circuit, we can implement this with a CZ gate
    circuit
        .x(0)
        .expect("Failed to apply X to qubit 0")
        .x(1)
        .expect("Failed to apply X to qubit 1"); // Flip to |11⟩
    circuit.h(1).expect("Failed to apply H to qubit 1 for CZ"); // Prepare for CZ
    circuit.cnot(0, 1).expect("Failed to apply CNOT for CZ"); // CNOT part of CZ
    circuit.h(1).expect("Failed to apply H to complete CZ"); // Complete CZ
    circuit
        .x(0)
        .expect("Failed to apply X to qubit 0 (flip back)")
        .x(1)
        .expect("Failed to apply X to qubit 1 (flip back)"); // Flip back

    // Amplitude amplification (diffusion operator)
    circuit
        .h(0)
        .expect("Failed to apply H to qubit 0 in diffusion")
        .h(1)
        .expect("Failed to apply H to qubit 1 in diffusion"); // H gates
    circuit
        .x(0)
        .expect("Failed to apply X to qubit 0 in diffusion")
        .x(1)
        .expect("Failed to apply X to qubit 1 in diffusion"); // X gates
    circuit.h(1).expect("Failed to apply H for diffusion CZ"); // H on target for CZ
    circuit
        .cnot(0, 1)
        .expect("Failed to apply CNOT in diffusion"); // CNOT part of CZ
    circuit.h(1).expect("Failed to complete diffusion CZ"); // Complete CZ
    circuit
        .x(0)
        .expect("Failed to apply final X to qubit 0")
        .x(1)
        .expect("Failed to apply final X to qubit 1"); // X gates
    circuit
        .h(0)
        .expect("Failed to apply final H to qubit 0")
        .h(1)
        .expect("Failed to apply final H to qubit 1"); // H gates

    // Run with ideal simulator first
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_result = circuit
        .run(ideal_sim)
        .expect("Failed to run ideal Grover circuit");

    println!("Ideal Grover result:");
    for (i, amplitude) in ideal_result.amplitudes().iter().enumerate() {
        let bits = format!("{i:02b}");
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }

    // Now run with a realistic noise model
    // Create a noise model with typical quantum hardware parameters
    let mut noise_model = NoiseModel::new(true); // Apply after each gate

    // Add depolarizing noise to represent gate errors
    for qubit in 0..2 {
        noise_model.add_depolarizing(DepolarizingChannel {
            target: QubitId::new(qubit),
            probability: 0.005, // 0.5% error rate per gate
        });
    }

    // Add amplitude damping for T1 relaxation
    for qubit in 0..2 {
        noise_model.add_amplitude_damping(AmplitudeDampingChannel {
            target: QubitId::new(qubit),
            gamma: 0.002, // Small T1 decay per gate
        });
    }

    // Create a simulator with the noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the circuit with noise
    let noisy_result = circuit
        .run(noisy_sim)
        .expect("Failed to run Grover circuit with noise");

    println!("\nGrover with realistic noise:");
    for (i, amplitude) in noisy_result.amplitudes().iter().enumerate() {
        let bits = format!("{i:02b}");
        println!(
            "|{}⟩: {} (probability: {:.6})",
            bits,
            amplitude,
            amplitude.norm_sqr()
        );
    }

    // Analyze the impact of noise
    println!("\nImpact of noise on Grover's algorithm:");
    println!(
        "- Ideal |11⟩ probability: {:.6}",
        ideal_result.amplitudes()[3].norm_sqr()
    );
    println!(
        "- Noisy |11⟩ probability: {:.6}",
        noisy_result.amplitudes()[3].norm_sqr()
    );
    println!(
        "- Reduction in success probability: {:.2}%",
        (1.0 - noisy_result.amplitudes()[3].norm_sqr() / ideal_result.amplitudes()[3].norm_sqr())
            * 100.0
    );
}

// Run Quantum Fourier Transform with noise
fn run_qft_with_noise() {
    println!("\nQuantum Fourier Transform with Noise");
    println!("---------------------------------");

    // Create a 3-qubit QFT circuit
    let mut circuit = Circuit::<3>::new();

    // Initialize in a non-trivial state |101⟩
    circuit.x(0).expect("Failed to initialize qubit 0 to |1⟩");
    circuit.x(2).expect("Failed to initialize qubit 2 to |1⟩");

    // Apply QFT
    // QFT on qubit 0
    circuit.h(0).expect("Failed to apply H to qubit 0 in QFT");
    circuit
        .crz(1, 0, PI / 2.0)
        .expect("Failed to apply CRZ(π/2) in QFT");
    circuit
        .crz(2, 0, PI / 4.0)
        .expect("Failed to apply CRZ(π/4) in QFT");

    // QFT on qubit 1
    circuit.h(1).expect("Failed to apply H to qubit 1 in QFT");
    circuit
        .crz(2, 1, PI / 2.0)
        .expect("Failed to apply CRZ(π/2) to qubit 1 in QFT");

    // QFT on qubit 2
    circuit.h(2).expect("Failed to apply H to qubit 2 in QFT");

    // Swap qubits for correct output order
    circuit
        .swap(0, 2)
        .expect("Failed to swap qubits for QFT output");

    // Run with ideal simulator first
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_result = circuit
        .run(ideal_sim)
        .expect("Failed to run ideal QFT circuit");

    println!("Ideal QFT result:");
    // Find states with significant probability
    let threshold = 0.01; // 1% probability threshold
    for (i, amplitude) in ideal_result.amplitudes().iter().enumerate() {
        let prob = amplitude.norm_sqr();
        if prob > threshold {
            let bits = format!("{i:03b}");
            println!("|{bits}⟩: {amplitude} (probability: {prob:.6})");
        }
    }

    // Now run with a realistic noise model
    let mut noise_model = NoiseModel::new(true); // Apply after each gate

    // Add phase damping noise (T2 decoherence)
    for qubit in 0..3 {
        noise_model.add_phase_damping(PhaseDampingChannel {
            target: QubitId::new(qubit),
            lambda: 0.01, // 1% phase damping per gate
        });
    }

    // Create a simulator with the noise model
    let noisy_sim = StateVectorSimulator::with_noise(noise_model);

    // Run the circuit with noise
    let noisy_result = circuit
        .run(noisy_sim)
        .expect("Failed to run QFT circuit with phase damping noise");

    println!("\nQFT with phase damping noise:");
    // Find states with significant probability
    for (i, amplitude) in noisy_result.amplitudes().iter().enumerate() {
        let prob = amplitude.norm_sqr();
        if prob > threshold {
            let bits = format!("{i:03b}");
            println!("|{bits}⟩: {amplitude} (probability: {prob:.6})");
        }
    }

    println!("\nImpact of noise on QFT:");
    println!("- Phase damping causes loss of the interference pattern");
    println!("- Results in a more uniform distribution instead of the sharp peaks expected in QFT");
}

// Run a simple quantum error correction code
fn run_error_correction_code() {
    println!("\nQuantum Error Correction with Bit-Flip Code");
    println!("------------------------------------------");

    // Create a circuit for a 3-qubit bit-flip code
    let mut circuit = Circuit::<3>::new();

    // Encode logical |1⟩ state into |111⟩
    circuit.x(0).expect("Failed to prepare qubit 0 in |1⟩"); // Prepare in |1⟩
    circuit
        .cnot(0, 1)
        .expect("Failed to spread to second qubit"); // Spread to second qubit
    circuit.cnot(0, 2).expect("Failed to spread to third qubit"); // Spread to third qubit

    // Introduce an X error on the second qubit
    println!("Introducing X error on qubit 1");
    circuit
        .x(1)
        .expect("Failed to introduce X error on qubit 1");

    // Error detection and correction
    // Use CNOTs to detect the error syndrome
    circuit
        .cnot(0, 1)
        .expect("Failed to apply CNOT for parity check"); // Check 0-1 parity
    circuit
        .cnot(0, 2)
        .expect("Failed to apply CNOT for 0-2 parity"); // Check 0-2 parity

    // Use Toffoli gate for correction (if both syndrome bits are 1, flip qubit 1)
    // Since Toffoli is not directly supported, we'll use a decomposition
    circuit
        .h(1)
        .expect("Failed to apply H in Toffoli decomposition");
    circuit
        .cnot(2, 1)
        .expect("Failed to apply CNOT in Toffoli decomposition");
    circuit
        .tdg(1)
        .expect("Failed to apply T† in Toffoli decomposition");
    circuit
        .cnot(0, 1)
        .expect("Failed to apply CNOT in Toffoli decomposition");
    circuit
        .t(1)
        .expect("Failed to apply T in Toffoli decomposition");
    circuit
        .cnot(2, 1)
        .expect("Failed to apply CNOT in Toffoli decomposition");
    circuit
        .tdg(1)
        .expect("Failed to apply T† in Toffoli decomposition");
    circuit
        .cnot(0, 1)
        .expect("Failed to apply CNOT in Toffoli decomposition");
    circuit
        .t(2)
        .expect("Failed to apply T to qubit 2 in Toffoli decomposition");
    circuit
        .t(1)
        .expect("Failed to apply T to qubit 1 in Toffoli decomposition");
    circuit
        .h(1)
        .expect("Failed to apply final H in Toffoli decomposition");
    circuit
        .cnot(2, 0)
        .expect("Failed to apply CNOT in Toffoli decomposition");
    circuit
        .t(0)
        .expect("Failed to apply T to qubit 0 in Toffoli decomposition");
    circuit
        .tdg(2)
        .expect("Failed to apply T† to qubit 2 in Toffoli decomposition");
    circuit
        .cnot(2, 0)
        .expect("Failed to complete Toffoli decomposition");

    // Run with ideal simulator
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_result = circuit
        .run(ideal_sim)
        .expect("Failed to run ideal error correction circuit");

    println!("Ideal error correction result:");
    for (i, amplitude) in ideal_result.amplitudes().iter().enumerate() {
        let prob = amplitude.norm_sqr();
        if prob > 0.01 {
            let bits = format!("{i:03b}");
            println!("|{bits}⟩: {amplitude} (probability: {prob:.6})");
        }
    }

    // Run the same circuit with phase-flip noise
    let mut noise_model = NoiseModel::new(true);

    // Add phase flip noise to all qubits
    for qubit in 0..3 {
        noise_model.add_phase_flip(PhaseFlipChannel {
            target: QubitId::new(qubit),
            probability: 0.05, // 5% phase flip probability per gate
        });
    }

    let noisy_sim = StateVectorSimulator::with_noise(noise_model);
    let noisy_result = circuit
        .run(noisy_sim)
        .expect("Failed to run error correction circuit with phase flip noise");

    println!("\nError correction with phase flip noise:");
    for (i, amplitude) in noisy_result.amplitudes().iter().enumerate() {
        let prob = amplitude.norm_sqr();
        if prob > 0.01 {
            let bits = format!("{i:03b}");
            println!("|{bits}⟩: {amplitude} (probability: {prob:.6})");
        }
    }

    println!("\nNote: The bit-flip code can correct X errors but not Z (phase) errors");
    println!("This demonstrates how different error types require different correction codes");
}

// Run a simple variational quantum circuit
fn run_variational_circuit() {
    println!("\nVariational Quantum Circuit with Noise");
    println!("------------------------------------");

    // Create a variational circuit similar to those used in QAOA or VQE
    let mut circuit = Circuit::<4>::new();

    // Initial state preparation (all qubits in superposition)
    for i in 0..4 {
        circuit
            .h(i)
            .expect(&format!("Failed to apply H to qubit {i}"));
    }

    // Variational ansatz layer 1 (problem Hamiltonian)
    // ZZ interactions between neighboring qubits
    for i in 0..3 {
        circuit.cnot(i, i + 1).expect(&format!(
            "Failed to apply CNOT between qubits {i} and {}",
            i + 1
        ));
        circuit
            .rz(i + 1, 0.1 * PI)
            .expect(&format!("Failed to apply RZ to qubit {}", i + 1)); // Interaction strength parameter
        circuit.cnot(i, i + 1).expect(&format!(
            "Failed to apply second CNOT between qubits {i} and {}",
            i + 1
        ));
    }

    // Variational ansatz layer 2 (mixer Hamiltonian)
    // X rotations on all qubits
    for i in 0..4 {
        circuit
            .rx(i, 0.2 * PI)
            .expect(&format!("Failed to apply RX to qubit {i}")); // Mixing parameter
    }

    // Repeat the ansatz (deeper circuit)
    // Layer 3 (problem Hamiltonian again)
    for i in 0..3 {
        circuit.cnot(i, i + 1).expect(&format!(
            "Failed to apply CNOT in layer 3 between qubits {i} and {}",
            i + 1
        ));
        circuit
            .rz(i + 1, 0.3 * PI)
            .expect(&format!("Failed to apply RZ in layer 3 to qubit {}", i + 1)); // Different parameter
        circuit.cnot(i, i + 1).expect(&format!(
            "Failed to apply second CNOT in layer 3 between qubits {i} and {}",
            i + 1
        ));
    }

    // Layer 4 (mixer Hamiltonian again)
    for i in 0..4 {
        circuit
            .rx(i, 0.4 * PI)
            .expect(&format!("Failed to apply RX in layer 4 to qubit {i}")); // Different parameter
    }

    // Run with ideal simulator
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_result = circuit
        .run(ideal_sim)
        .expect("Failed to run ideal variational circuit");

    println!("Ideal variational circuit result (top 5 states):");
    // Find top 5 states by probability
    let mut probs: Vec<(usize, f64)> = ideal_result
        .amplitudes()
        .iter()
        .enumerate()
        .map(|(i, amp)| (i, amp.norm_sqr()))
        .collect();

    probs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("Failed to compare probabilities")
    });

    for (i, prob) in probs.iter().take(5) {
        let bits = format!("{:04b}", *i);
        println!("|{bits}⟩: {prob:.6}");
    }

    // Run with realistic noise model for NISQ devices
    let mut noise_model = NoiseModel::new(true);

    // Add depolarizing noise to represent gate errors
    for qubit in 0..4 {
        noise_model.add_depolarizing(DepolarizingChannel {
            target: QubitId::new(qubit),
            probability: 0.01, // 1% error rate per gate
        });
    }

    // Add amplitude and phase damping for decoherence
    for qubit in 0..4 {
        noise_model.add_amplitude_damping(AmplitudeDampingChannel {
            target: QubitId::new(qubit),
            gamma: 0.005, // T1 relaxation
        });
        noise_model.add_phase_damping(PhaseDampingChannel {
            target: QubitId::new(qubit),
            lambda: 0.01, // T2 dephasing
        });
    }

    let noisy_sim = StateVectorSimulator::with_noise(noise_model);
    let noisy_result = circuit
        .run(noisy_sim)
        .expect("Failed to run variational circuit with NISQ noise");

    println!("\nVariational circuit with realistic NISQ noise (top 5 states):");
    let mut noisy_probs: Vec<(usize, f64)> = noisy_result
        .amplitudes()
        .iter()
        .enumerate()
        .map(|(i, amp)| (i, amp.norm_sqr()))
        .collect();

    noisy_probs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("Failed to compare noisy probabilities")
    });

    for (i, prob) in noisy_probs.iter().take(5) {
        let bits = format!("{:04b}", *i);
        println!("|{bits}⟩: {prob:.6}");
    }

    // Calculate the total variation distance between the ideal and noisy distributions
    let mut tvd = 0.0;
    for i in 0..ideal_result.amplitudes().len() {
        let ideal_prob = ideal_result.amplitudes()[i].norm_sqr();
        let noisy_prob = noisy_result.amplitudes()[i].norm_sqr();
        tvd += (ideal_prob - noisy_prob).abs();
    }
    tvd /= 2.0; // Normalize

    println!("\nImpact of noise on variational circuit:");
    println!("- Total variation distance: {tvd:.6}");
    println!("- Noise tends to 'flatten' the distribution toward a uniform mixture");
    println!("- Deeper circuits (more gates) are more susceptible to noise effects");
    println!("- Real NISQ devices would typically show even more degradation");
}
