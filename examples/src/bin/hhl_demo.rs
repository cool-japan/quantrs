//! Demonstration of the HHL algorithm for solving linear systems

use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::time::Instant;

fn main() {
    println!("=== HHL Algorithm Demo ===\n");

    // Example 1: Simple 2x2 system
    println!("Example 1: Simple 2x2 Linear System");
    simple_2x2_example();

    println!("\n{}\n", "=".repeat(50));

    // Example 2: 4x4 system
    println!("Example 2: 4x4 Linear System");
    larger_system_example();

    println!("\n{}\n", "=".repeat(50));

    // Example 3: Comparison with classical solution
    println!("Example 3: Quantum vs Classical Comparison");
    quantum_vs_classical();
}

fn simple_2x2_example() {
    // Use the built-in example
    match hhl_example() {
        Ok(()) => println!("HHL example completed successfully"),
        Err(e) => println!("Error: {e}"),
    }
}

fn larger_system_example() {
    // Create a 4x4 Hermitian matrix
    let matrix = Array2::from_shape_vec(
        (4, 4),
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .expect("Failed to create 4x4 Hermitian matrix for HHL algorithm demonstration");

    // Vector b
    let vector_b = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ]);

    println!("Matrix A (4x4):");
    for i in 0..4 {
        for j in 0..4 {
            print!("{:.2} ", matrix[[i, j]].re);
        }
        println!();
    }

    println!("\nVector b: {vector_b:?}");

    // Set up HHL parameters
    let params = HHLParams {
        n_qubits: 2,     // 2^2 = 4 dimensional system
        clock_qubits: 4, // Good precision
        evolution_time: std::f64::consts::PI,
        condition_number: 10.0,
        eigenvalue_scale: 1.0,
    };

    // Run HHL algorithm
    let start = Instant::now();

    match HHLAlgorithm::new(matrix, vector_b, params) {
        Ok(hhl) => match hhl.run() {
            Ok((solution, success_prob)) => {
                let elapsed = start.elapsed();

                println!("\nHHL Algorithm Results:");
                println!("Time: {elapsed:?}");
                println!("Solution |x⟩:");
                for (i, val) in solution.iter().enumerate() {
                    println!("  x[{}] = {:.4} + {:.4}i", i, val.re, val.im);
                }
                println!("Success probability: {success_prob:.4}");
            }
            Err(e) => println!("Error running HHL: {e}"),
        },
        Err(e) => println!("Error creating HHL instance: {e}"),
    }
}

fn quantum_vs_classical() {
    // Create a well-conditioned Hermitian matrix
    let n = 4;
    let mut matrix = Array2::zeros((n, n));

    // Tridiagonal matrix (well-conditioned)
    for i in 0..n {
        matrix[[i, i]] = Complex64::new(2.0, 0.0);
        if i > 0 {
            matrix[[i, i - 1]] = Complex64::new(-0.5, 0.0);
            matrix[[i - 1, i]] = Complex64::new(-0.5, 0.0);
        }
    }

    // Create a known solution
    let true_solution = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);

    // Compute b = A * x_true
    let vector_b = matrix.dot(&true_solution);

    println!("Test system: A * x = b");
    println!("True solution x:");
    for (i, val) in true_solution.iter().enumerate() {
        println!("  x[{}] = {:.4}", i, val.re);
    }

    // Classical solution (using direct inversion - not recommended for large systems)
    println!("\nClassical solution (direct inversion):");
    let start_classical = Instant::now();

    // For demonstration, we'll use a simple iterative method
    let mut x_classical = Array1::from_elem(n, Complex64::new(0.0, 0.0));

    // Simple gradient descent (not optimal, just for demo)
    for _ in 0..100 {
        let grad = matrix.dot(&x_classical) - &vector_b;
        for i in 0..n {
            x_classical[i] -= grad[i] * 0.1;
        }
    }

    let classical_time = start_classical.elapsed();
    println!("Time: {classical_time:?}");
    for (i, val) in x_classical.iter().enumerate() {
        println!("  x[{}] = {:.4}", i, val.re);
    }

    // Quantum solution using HHL
    println!("\nQuantum solution (HHL):");
    let params = HHLParams::new(2);

    let start_quantum = Instant::now();
    match HHLAlgorithm::new(matrix.clone(), vector_b, params) {
        Ok(hhl) => {
            match hhl.run() {
                Ok((solution, success_prob)) => {
                    let quantum_time = start_quantum.elapsed();

                    println!("Time: {quantum_time:?}");

                    // Normalize to compare with true solution
                    let scale = true_solution[0] / solution[0];
                    println!("Scaled solution:");
                    for (i, val) in solution.iter().enumerate() {
                        let scaled = val * scale;
                        println!("  x[{}] = {:.4}", i, scaled.re);
                    }
                    println!("Success probability: {success_prob:.4}");

                    // Compute error
                    let mut error = 0.0;
                    for i in 0..n {
                        let scaled = solution[i] * scale;
                        error += (scaled - true_solution[i]).norm();
                    }
                    println!("Total error: {error:.6}");
                }
                Err(e) => println!("Error running HHL: {e}"),
            }
        }
        Err(e) => println!("Error creating HHL instance: {e}"),
    }
}

// Demonstrate the quantum advantage for specific types of systems
fn quantum_advantage_demo() {
    println!("\n=== Quantum Advantage Analysis ===\n");

    println!("HHL Algorithm Complexity:");
    println!("- Classical: O(n³) for general matrix inversion");
    println!("- Quantum: O(log n × κ²) where κ is the condition number");
    println!("\nQuantum advantage appears when:");
    println!("1. The matrix is sparse (few non-zero elements)");
    println!("2. The condition number κ is small");
    println!("3. We only need to know expectation values, not the full solution");
    println!("4. The system size n is very large");

    // Show scaling comparison
    println!("\nScaling comparison for sparse matrices:");
    println!("n\tClassical\tQuantum (κ=10)");
    for i in 3..=10 {
        let n = 1 << i;
        let classical = n * n * n;
        let quantum = i * 100; // log(n) × κ²
        println!("{n}\t{classical}\t\t{quantum}");
    }
}
