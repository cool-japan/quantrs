use quantrs2_circuit::prelude::Circuit;
use quantrs2_circuit::builder::Simulator;
use quantrs2_core::prelude::{QubitId, Register};
use quantrs2_sim::prelude::StateVectorSimulator;
use std::f64::consts::PI;

/// Implements the Quantum Fourier Transform on a 4-qubit register
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Quantum Fourier Transform Example");
    println!("Performing QFT on a 4-qubit register\n");

    // Create a new circuit with 4 qubits
    let mut circuit = Circuit::<4>::new();

    // Step 1: Prepare an interesting initial state
    // Set the initial state to |0011⟩ (3 in decimal)
    circuit.x(QubitId::new(0))?;
    circuit.x(QubitId::new(1))?;

    println!("Prepared initial state |0011⟩");

    // Step 2: Apply QFT
    apply_qft(&mut circuit, 4)?;

    println!("Applied QFT circuit");

    // Initialize the simulator and register
    let mut simulator = StateVectorSimulator::new();
    let register = Register::<4>::new();

    // Run the circuit
    println!("\nExecuting circuit...");
    let result = simulator.run(&circuit)?;

    // Print the results
    println!("\nAmplitudes after QFT:");
    for i in 0..16 {
        let bits = [(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1].map(|x| x as u8);
        let amplitude = result.amplitude(&bits)?;
        let magnitude = (amplitude.re * amplitude.re + amplitude.im * amplitude.im).sqrt();
        let phase = amplitude.im.atan2(amplitude.re) * 180.0 / PI;
        println!(
            "State |{:04b}⟩: magnitude = {:.6}, phase = {:.2}°",
            i, magnitude, phase
        );
    }

    // Apply inverse QFT to verify we get back to |0011⟩
    let mut inverse_circuit = Circuit::<4>::new();

    // Set the initial state to the result of the QFT
    // In practice, we would just continue from the previous circuit,
    // but for clarity, we'll start with the same initial state and apply QFT†
    inverse_circuit.x(QubitId::new(0))?;
    inverse_circuit.x(QubitId::new(1))?;

    // Apply QFT
    apply_qft(&mut inverse_circuit, 4)?;

    // Apply QFT† (inverse QFT)
    apply_inverse_qft(&mut inverse_circuit, 4)?;

    println!("\nApplied inverse QFT circuit");

    // Run the inverse circuit
    let inverse_result = simulator.run(&inverse_circuit)?;

    // Print the results
    println!("\nAmplitudes after QFT followed by QFT†:");
    for i in 0..16 {
        let bits = [(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1].map(|x| x as u8);
        let prob = inverse_result.probability(&bits)?;
        if prob > 0.01 {
            // Only show states with significant probability
            println!("State |{:04b}⟩: {:.6}", i, prob);
        }
    }

    // Check if we recovered the original state
    let bits_3 = [0, 0, 1, 1]; // |0011⟩ = 3 in binary
    let prob_3 = inverse_result.probability(&bits_3)?;
    if prob_3 > 0.99 {
        println!(
            "\nSuccess! Recovered the original state |0011⟩ with probability {:.6}",
            prob_3
        );
    } else {
        println!("\nFailed to recover the original state |0011⟩");
    }
    Ok(())
}

/// Applies the Quantum Fourier Transform to the first n qubits
fn apply_qft(circuit: &mut Circuit<4>, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    // QFT circuit (we'll implement for up to 4 qubits)
    for i in 0..n {
        // Apply Hadamard to the current qubit
        circuit.h(QubitId::new(i as u32))?;

        // Apply controlled rotations
        for j in i + 1..n {
            let angle = PI / (1 << (j - i)) as f64; // PI/2^(j-i)
            circuit.crz(QubitId::new(j as u32), QubitId::new(i as u32), angle)?;
        }
    }

    // Swap qubits to match standard QFT output order
    for i in 0..n / 2 {
        circuit.swap(QubitId::new(i as u32), QubitId::new((n - i - 1) as u32))?;
    }
    Ok(())
}

/// Applies the inverse Quantum Fourier Transform to the first n qubits
fn apply_inverse_qft(circuit: &mut Circuit<4>, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    // QFT† circuit (inverse of QFT)

    // First, swap qubits (reverse of the last step in QFT)
    for i in 0..n / 2 {
        circuit.swap(QubitId::new(i as u32), QubitId::new((n - i - 1) as u32))?;
    }

    // Apply inverse QFT operations in reverse order
    for i in (0..n).rev() {
        // Apply inverse controlled rotations
        for j in (i + 1..n).rev() {
            let angle = -PI / (1 << (j - i)) as f64; // -PI/2^(j-i)
            circuit.crz(QubitId::new(j as u32), QubitId::new(i as u32), angle)?;
        }

        // Apply Hadamard to the current qubit
        circuit.h(QubitId::new(i as u32))?;
    }
    Ok(())
}
