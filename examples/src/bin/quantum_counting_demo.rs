//! Quantum Counting and Amplitude Estimation Demonstration
//!
//! This example demonstrates:
//! - Quantum Phase Estimation (QPE)
//! - Quantum Counting for search problems
//! - Quantum Amplitude Estimation

use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::f64::consts::PI;

fn main() {
    println!("=== Quantum Counting and Amplitude Estimation Demo ===\n");

    // Example 1: Quantum Phase Estimation
    phase_estimation_demo();

    // Example 2: Quantum Counting
    quantum_counting_demo();

    // Example 3: Amplitude Estimation
    amplitude_estimation_demo();

    // Example 4: Practical application - Database search
    database_search_demo();
}

fn phase_estimation_demo() {
    println!("1. Quantum Phase Estimation");
    println!("---------------------------");

    // Create a rotation gate with known phase
    let angle = PI / 3.0; // 60 degrees
    let phase = angle / (2.0 * PI); // Normalized phase

    println!(
        "Testing phase estimation with rotation angle: {:.2}π",
        2.0 * phase
    );

    // Create the unitary matrix (rotation around Z-axis)
    let u = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new((angle / 2.0).cos(), -(angle / 2.0).sin()),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new((angle / 2.0).cos(), (angle / 2.0).sin()),
        ],
    )
    .expect("Failed to create 2x2 unitary matrix for Z-axis rotation in phase estimation");

    // Test with different precision levels
    for precision in [3, 4, 5] {
        let qpe = QuantumPhaseEstimation::new(precision, u.clone());

        // Use eigenstate |0⟩
        let eigenstate = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let estimated_phase = qpe.estimate_phase(eigenstate);

        println!(
            "  Precision {} bits: estimated phase = {:.4}, error = {:.4}",
            precision,
            estimated_phase,
            (estimated_phase - phase).abs()
        );
    }
    println!();
}

fn quantum_counting_demo() {
    println!("2. Quantum Counting");
    println!("-------------------");

    // Example: Count prime numbers in range 0-31
    let is_prime = |n: usize| -> bool {
        if n < 2 {
            return false;
        }
        for i in 2..=((n as f64).sqrt() as usize) {
            if n % i == 0 {
                return false;
            }
        }
        true
    };

    let n_items = 32;
    let actual_primes: Vec<usize> = (0..n_items).filter(|&x| is_prime(x)).collect();

    println!("Counting prime numbers in range 0-31:");
    println!("Actual primes: {actual_primes:?}");
    println!("Actual count: {}", actual_primes.len());

    // Run quantum counting with different precision levels
    for precision in [4, 5, 6] {
        let counter = QuantumCounting::new(n_items, precision, Box::new(is_prime));
        let estimated_count = counter.count();

        println!(
            "  Precision {} bits: estimated count = {:.1}, error = {:.1}",
            precision,
            estimated_count,
            (estimated_count - actual_primes.len() as f64).abs()
        );
    }
    println!();
}

fn amplitude_estimation_demo() {
    println!("3. Quantum Amplitude Estimation");
    println!("-------------------------------");

    // Create a state that prepares unequal superposition
    // |ψ⟩ = 0.8|0⟩ + 0.6|1⟩
    let state_prep = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.8, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.6, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .expect("Failed to create 2x2 state preparation matrix for amplitude estimation");

    // Oracle marks state |1⟩
    let oracle = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .expect("Failed to create 2x2 oracle matrix for amplitude estimation (marking state |1⟩)");

    println!("State preparation: |ψ⟩ = 0.8|0⟩ + 0.6|1⟩");
    println!("Oracle marks state |1⟩");
    println!("True amplitude of marked state: 0.6");

    // Estimate amplitude with different precision levels
    for precision in [3, 4, 5] {
        let qae = QuantumAmplitudeEstimation::new(state_prep.clone(), oracle.clone(), precision);
        let estimated_amplitude = qae.estimate();

        println!(
            "  Precision {} bits: estimated amplitude = {:.3}, error = {:.3}",
            precision,
            estimated_amplitude,
            (estimated_amplitude - 0.6).abs()
        );
    }
    println!();
}

fn database_search_demo() {
    println!("4. Database Search Application");
    println!("------------------------------");

    // Simulate a database with 64 entries
    // We're looking for entries that match certain criteria
    let database_size = 64;

    // Define search criteria: entries whose binary representation has exactly 3 ones
    let search_criteria = |x: usize| -> bool { x.count_ones() == 3 };

    // Count matching entries classically
    let matches: Vec<usize> = (0..database_size).filter(|&x| search_criteria(x)).collect();

    println!("Database size: {database_size}");
    println!("Search criteria: binary representation has exactly 3 ones");
    println!("Matching entries: {:?}", &matches[..5.min(matches.len())]);
    if matches.len() > 5 {
        println!("... and {} more", matches.len() - 5);
    }
    println!("Total matches: {}", matches.len());

    // Use quantum counting to estimate the number of matches
    let counter = QuantumCounting::new(database_size, 6, Box::new(search_criteria));
    let quantum_estimate = counter.count();

    println!("\nQuantum counting result:");
    println!("  Estimated matches: {quantum_estimate:.1}");
    println!(
        "  Error: {:.1}",
        (quantum_estimate - matches.len() as f64).abs()
    );

    // Calculate success probability for Grover's algorithm
    let success_prob = matches.len() as f64 / database_size as f64;
    let grover_iterations = (PI / 4.0) * (database_size as f64 / matches.len() as f64).sqrt();

    println!("\nGrover's algorithm analysis:");
    println!("  Success probability: {success_prob:.3}");
    println!("  Optimal iterations: {grover_iterations:.1}");
    println!(
        "  Classical average searches: {:.1}",
        database_size as f64 / 2.0
    );
    println!(
        "  Quantum speedup: {:.1}x",
        (database_size as f64 / 2.0) / grover_iterations
    );
}

// Additional helper function for more complex amplitude estimation
fn multi_amplitude_estimation_demo() {
    println!("\n5. Multi-State Amplitude Estimation");
    println!("-----------------------------------");

    let n = 8;

    // Create a non-uniform superposition
    let mut state_prep = Array2::zeros((n, n));
    let amplitudes = [0.5, 0.3, 0.4, 0.2, 0.1, 0.3, 0.2, 0.4];

    // Normalize
    let norm: f64 = amplitudes.iter().map(|&a| a * a).sum::<f64>().sqrt();

    for i in 0..n {
        state_prep[[i, 0]] = Complex64::new(amplitudes[i] / norm, 0.0);
    }

    // Oracle marks states 1, 3, and 5
    let mut oracle = Array2::zeros((n, n));
    for &marked in &[1, 3, 5] {
        oracle[[marked, marked]] = Complex64::new(1.0, 0.0);
    }

    let marked_amplitude = (amplitudes[5].mul_add(
        amplitudes[5],
        amplitudes[3].mul_add(amplitudes[3], amplitudes[1].powi(2)),
    ) / norm.powi(2))
    .sqrt();

    println!("State amplitudes (normalized):");
    for i in 0..n {
        println!("  |{}⟩: {:.3}", i, amplitudes[i] / norm);
    }
    println!("Marked states: 1, 3, 5");
    println!("True total amplitude of marked states: {marked_amplitude:.3}");

    let qae = QuantumAmplitudeEstimation::new(state_prep, oracle, 5);
    let estimated = qae.estimate();

    println!("Estimated amplitude: {estimated:.3}");
    println!("Error: {:.3}", (estimated - marked_amplitude).abs());
}
