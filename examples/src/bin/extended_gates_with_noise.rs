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
    circuit.h(0).unwrap().h(1).unwrap();

    // Oracle: Mark the |11⟩ state (using a Z gate controlled by both qubits)
    // For a 2-qubit circuit, we can implement this with a CZ gate
    circuit.x(0).unwrap().x(1).unwrap(); // Flip to |11⟩
    circuit.h(1).unwrap(); // Prepare for CZ
    circuit.cnot(0, 1).unwrap(); // CNOT part of CZ
    circuit.h(1).unwrap(); // Complete CZ
    circuit.x(0).unwrap().x(1).unwrap(); // Flip back

    // Amplitude amplification (diffusion operator)
    circuit.h(0).unwrap().h(1).unwrap(); // H gates
    circuit.x(0).unwrap().x(1).unwrap(); // X gates
    circuit.h(1).unwrap(); // H on target for CZ
    circuit.cnot(0, 1).unwrap(); // CNOT part of CZ
    circuit.h(1).unwrap(); // Complete CZ
    circuit.x(0).unwrap().x(1).unwrap(); // X gates
    circuit.h(0).unwrap().h(1).unwrap(); // H gates

    // Run with ideal simulator first
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_result = circuit.run(ideal_sim).unwrap();

    println!("Ideal Grover result:");
    for (i, amplitude) in ideal_result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
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
    let noisy_result = circuit.run(noisy_sim).unwrap();

    println!("\nGrover with realistic noise:");
    for (i, amplitude) in noisy_result.amplitudes().iter().enumerate() {
        let bits = format!("{:02b}", i);
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
    circuit.x(0).unwrap();
    circuit.x(2).unwrap();

    // Apply QFT
    // QFT on qubit 0
    circuit.h(0).unwrap();
    circuit.crz(1, 0, PI / 2.0).unwrap();
    circuit.crz(2, 0, PI / 4.0).unwrap();

    // QFT on qubit 1
    circuit.h(1).unwrap();
    circuit.crz(2, 1, PI / 2.0).unwrap();

    // QFT on qubit 2
    circuit.h(2).unwrap();

    // Swap qubits for correct output order
    circuit.swap(0, 2).unwrap();

    // Run with ideal simulator first
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_result = circuit.run(ideal_sim).unwrap();

    println!("Ideal QFT result:");
    // Find states with significant probability
    let threshold = 0.01; // 1% probability threshold
    for (i, amplitude) in ideal_result.amplitudes().iter().enumerate() {
        let prob = amplitude.norm_sqr();
        if prob > threshold {
            let bits = format!("{:03b}", i);
            println!("|{}⟩: {} (probability: {:.6})", bits, amplitude, prob);
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
    let noisy_result = circuit.run(noisy_sim).unwrap();

    println!("\nQFT with phase damping noise:");
    // Find states with significant probability
    for (i, amplitude) in noisy_result.amplitudes().iter().enumerate() {
        let prob = amplitude.norm_sqr();
        if prob > threshold {
            let bits = format!("{:03b}", i);
            println!("|{}⟩: {} (probability: {:.6})", bits, amplitude, prob);
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
    circuit.x(0).unwrap(); // Prepare in |1⟩
    circuit.cnot(0, 1).unwrap(); // Spread to second qubit
    circuit.cnot(0, 2).unwrap(); // Spread to third qubit

    // Introduce an X error on the second qubit
    println!("Introducing X error on qubit 1");
    circuit.x(1).unwrap();

    // Error detection and correction
    // Use CNOTs to detect the error syndrome
    circuit.cnot(0, 1).unwrap(); // Check 0-1 parity
    circuit.cnot(0, 2).unwrap(); // Check 0-2 parity

    // Use Toffoli gate for correction (if both syndrome bits are 1, flip qubit 1)
    // Since Toffoli is not directly supported, we'll use a decomposition
    circuit.h(1).unwrap();
    circuit.cnot(2, 1).unwrap();
    circuit.tdg(1).unwrap();
    circuit.cnot(0, 1).unwrap();
    circuit.t(1).unwrap();
    circuit.cnot(2, 1).unwrap();
    circuit.tdg(1).unwrap();
    circuit.cnot(0, 1).unwrap();
    circuit.t(2).unwrap();
    circuit.t(1).unwrap();
    circuit.h(1).unwrap();
    circuit.cnot(2, 0).unwrap();
    circuit.t(0).unwrap();
    circuit.tdg(2).unwrap();
    circuit.cnot(2, 0).unwrap();

    // Run with ideal simulator
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_result = circuit.run(ideal_sim).unwrap();

    println!("Ideal error correction result:");
    for (i, amplitude) in ideal_result.amplitudes().iter().enumerate() {
        let prob = amplitude.norm_sqr();
        if prob > 0.01 {
            let bits = format!("{:03b}", i);
            println!("|{}⟩: {} (probability: {:.6})", bits, amplitude, prob);
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
    let noisy_result = circuit.run(noisy_sim).unwrap();

    println!("\nError correction with phase flip noise:");
    for (i, amplitude) in noisy_result.amplitudes().iter().enumerate() {
        let prob = amplitude.norm_sqr();
        if prob > 0.01 {
            let bits = format!("{:03b}", i);
            println!("|{}⟩: {} (probability: {:.6})", bits, amplitude, prob);
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
        circuit.h(i).unwrap();
    }

    // Variational ansatz layer 1 (problem Hamiltonian)
    // ZZ interactions between neighboring qubits
    for i in 0..3 {
        circuit.cnot(i, i + 1).unwrap();
        circuit.rz(i + 1, 0.1 * PI).unwrap(); // Interaction strength parameter
        circuit.cnot(i, i + 1).unwrap();
    }

    // Variational ansatz layer 2 (mixer Hamiltonian)
    // X rotations on all qubits
    for i in 0..4 {
        circuit.rx(i, 0.2 * PI).unwrap(); // Mixing parameter
    }

    // Repeat the ansatz (deeper circuit)
    // Layer 3 (problem Hamiltonian again)
    for i in 0..3 {
        circuit.cnot(i, i + 1).unwrap();
        circuit.rz(i + 1, 0.3 * PI).unwrap(); // Different parameter
        circuit.cnot(i, i + 1).unwrap();
    }

    // Layer 4 (mixer Hamiltonian again)
    for i in 0..4 {
        circuit.rx(i, 0.4 * PI).unwrap(); // Different parameter
    }

    // Run with ideal simulator
    let ideal_sim = StateVectorSimulator::sequential();
    let ideal_result = circuit.run(ideal_sim).unwrap();

    println!("Ideal variational circuit result (top 5 states):");
    // Find top 5 states by probability
    let mut probs: Vec<(usize, f64)> = ideal_result
        .amplitudes()
        .iter()
        .enumerate()
        .map(|(i, amp)| (i, amp.norm_sqr()))
        .collect();

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, prob) in probs.iter().take(5) {
        let bits = format!("{:04b}", *i);
        println!("|{}⟩: {:.6}", bits, prob);
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
    let noisy_result = circuit.run(noisy_sim).unwrap();

    println!("\nVariational circuit with realistic NISQ noise (top 5 states):");
    let mut noisy_probs: Vec<(usize, f64)> = noisy_result
        .amplitudes()
        .iter()
        .enumerate()
        .map(|(i, amp)| (i, amp.norm_sqr()))
        .collect();

    noisy_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, prob) in noisy_probs.iter().take(5) {
        let bits = format!("{:04b}", *i);
        println!("|{}⟩: {:.6}", bits, prob);
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
    println!("- Total variation distance: {:.6}", tvd);
    println!("- Noise tends to 'flatten' the distribution toward a uniform mixture");
    println!("- Deeper circuits (more gates) are more susceptible to noise effects");
    println!("- Real NISQ devices would typically show even more degradation");
}
