use quantrs2_circuit::prelude::{Circuit, Simulator};
use quantrs2_core::prelude::{QubitId, Register};
use quantrs2_sim::prelude::StateVectorSimulator;
use std::f64::consts::PI;

/// Implements a simplified version of Shor's algorithm
/// This implementation demonstrates the quantum part of Shor's algorithm
/// for factoring N=15 with a=7 (coprime to 15)
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simplified Shor's Algorithm Example");
    println!("Finding the period of f(x) = 7^x mod 15\n");

    // For N=15, a=7, the period r=4 (since 7^4 mod 15 = 1)
    // We need enough qubits to represent values up to 2r = 8, so 3 counting qubits
    // Plus 4 qubits for the modular arithmetic (to represent values 0-15)

    // In a full implementation, we would:
    // 1. Create a circuit that computes f(x) = a^x mod N
    // 2. Use phase estimation to find the period
    // 3. Use the period to find the factors

    // For simplicity, we'll implement a pre-determined circuit that will
    // result in a period measurement for a=7, N=15

    // Here, we use 3 counting qubits to find a period r ≤ 4
    // The 4th qubit is a temporary workspace
    let mut circuit = Circuit::<4>::new();

    println!("Step 1: Creating superposition of input states");

    // Apply Hadamard to the counting qubits to create superposition
    for i in 0..3 {
        circuit.h(QubitId::new(i as u32));
    }

    println!("Step 2: Implementing the period-finding circuit for f(x) = 7^x mod 15");

    // This is a simplified implementation specific to a=7, N=15
    // We encode the following mapping based on the pattern of 7^x mod 15:
    // 7^0 mod 15 = 1
    // 7^1 mod 15 = 7
    // 7^2 mod 15 = 4
    // 7^3 mod 15 = 13
    // 7^4 mod 15 = 1 (period r=4)

    // For a real implementation, we would build a modular exponentiation circuit
    // Instead, we'll create a circuit that reproduces the expected output for
    // phase estimation of the period r=4

    // Implement a simplified circuit that will produce peaks at multiples of 1/r = 1/4
    // For period r=4, we expect to measure 0, 2, 4, or 6 (in binary) with equal probability,
    // corresponding to phases 0/4, 1/4, 2/4, and 3/4

    // Apply a phase shift to each computational basis state
    // Phase for |000⟩ = 0
    // Phase for |001⟩ = 2π/4 = π/2
    // Phase for |010⟩ = 4π/4 = π
    // Phase for |011⟩ = 6π/4 = 3π/2
    // And similarly for |100⟩ through |111⟩

    // To achieve this pattern, apply the following phase rotations:

    // For this simplified version, we'll apply controlled rotations manually using available gates
    // This is a workaround since controlled_u isn't available
    // Apply phase rotations to simulate the quantum period-finding
    circuit.cp(QubitId::new(0), QubitId::new(3), PI / 2.0);
    circuit.cp(QubitId::new(1), QubitId::new(3), PI);
    circuit.cp(QubitId::new(2), QubitId::new(3), 3.0 * PI / 2.0);

    println!("Step 3: Applying inverse QFT to extract the period");

    // Apply inverse QFT to the counting qubits
    apply_inverse_qft(&mut circuit, 3);

    // Initialize the simulator and register
    let mut simulator = StateVectorSimulator::new();
    let register = Register::<4>::new();

    // Run the circuit
    println!("\nExecuting circuit...");
    let result = simulator.run(&circuit)?;

    // For this simplified demonstration, we'll analyze the final state
    let amplitudes = result.amplitudes();

    // Print measurement results
    println!("\nState amplitudes:");
    for i in 0..8 {
        let prob = amplitudes[i].norm_sqr();
        if prob > 0.01 {
            // Only show states with significant probability
            println!("State |{:03b}⟩: probability = {:.6}", i, prob);

            // Calculate the phase and interpret it in terms of the period
            let phase = i as f64 / 8.0;
            let denominator = find_denominator(phase, 10);

            if denominator > 0 {
                println!(
                    "  Phase = {:.4} ≈ {}/{}",
                    phase,
                    (phase * denominator as f64).round() as i64,
                    denominator
                );

                if denominator % 4 == 0 {
                    println!("  This suggests period r = 4 (correct!)");
                } else if denominator % 2 == 0 {
                    println!("  This suggests period r = 2 (potential period or multiple of the true period)");
                } else {
                    println!("  This doesn't clearly indicate period r = 4, possibly due to measurement error");
                }
            }
        }
    }

    println!("\nClassical post-processing:");
    println!("In a complete Shor's algorithm implementation, we would:");
    println!("1. Use the measured period r = 4");
    println!("2. Compute gcd(7^(r/2) - 1, 15) = gcd(48, 15) = 3");
    println!("3. Compute gcd(7^(r/2) + 1, 15) = gcd(50, 15) = 5");
    println!("4. Conclude that the factors of 15 are 3 and 5");

    Ok(())
}

/// Applies the inverse Quantum Fourier Transform to the first n qubits
fn apply_inverse_qft(circuit: &mut Circuit<4>, n: usize) {
    // First, swap qubits
    for i in 0..n / 2 {
        circuit.swap(QubitId::new(i as u32), QubitId::new((n - i - 1) as u32));
    }

    // Apply inverse QFT operations in reverse order
    for i in (0..n).rev() {
        // Apply inverse controlled rotations using available gates
        for j in (i + 1..n).rev() {
            let angle = -PI / (1 << (j - i)) as f64; // -PI/2^(j-i)
            circuit.cp(QubitId::new(j as u32), QubitId::new(i as u32), angle);
        }

        // Apply Hadamard to the current qubit
        circuit.h(QubitId::new(i as u32));
    }
}

/// Finds the denominator of a fraction approximating the given value
/// Uses the continued fraction method to find a good approximation
fn find_denominator(x: f64, max_denominator: i64) -> i64 {
    if x < 1e-10 {
        return 1; // If x is very close to 0, denominator is 1 (x ≈ 0/1)
    }

    let mut a = x;
    let mut p1: i64 = 1;
    let mut q1: i64 = 0;
    let mut p2: i64 = 0;
    let mut q2: i64 = 1;

    while q1 + q2 <= max_denominator {
        let a_i = a.floor();
        let a_i_64 = a_i as i64;

        let p = a_i_64 * p1 + p2;
        let q = a_i_64 * q1 + q2;

        p2 = p1;
        q2 = q1;
        p1 = p;
        q1 = q;

        let remainder = a - a_i;
        if remainder < 1e-10 {
            break;
        }

        a = 1.0 / remainder;
    }

    q1
}
