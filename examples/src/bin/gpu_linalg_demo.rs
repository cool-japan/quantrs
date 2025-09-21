//! Demonstration of GPU-accelerated linear algebra operations

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use ndarray::Array2;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use num_complex::Complex64;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use quantrs2_core::prelude::*;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use quantrs2_sim::gpu_linalg::GpuLinearAlgebra;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use quantrs2_sim::prelude::*;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use std::time::Instant;

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Linear Algebra Demo ===\n");

    // Check if GPU is available
    let gpu_linalg = match GpuLinearAlgebra::new().await {
        Ok(gpu) => {
            println!("GPU device found and initialized!");
            gpu
        }
        Err(e) => {
            println!("GPU not available: {}", e);
            println!("Please ensure you have a compatible GPU and drivers installed.");
            return Ok(());
        }
    };

    // Example 1: Simple matrix multiplication
    println!("\nExample 1: Matrix Multiplication");
    demo_matrix_multiplication(&gpu_linalg).await?;

    println!("\n{}\n", "=".repeat(50));

    // Example 2: Quantum gate application
    println!("Example 2: Quantum Gate Application");
    demo_quantum_gates(&gpu_linalg).await?;

    println!("\n{}\n", "=".repeat(50));

    // Example 3: Performance benchmark
    println!("Example 3: Performance Benchmark");
    benchmark_gpu_linalg(&gpu_linalg).await?;

    Ok(())
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
async fn demo_matrix_multiplication(
    gpu: &GpuLinearAlgebra,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create Pauli matrices
    let pauli_x = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )?;

    let pauli_y = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )?;

    let pauli_z = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )?;

    // Compute XY on GPU
    println!("Computing Pauli X × Pauli Y:");
    let xy = gpu.matmul(&pauli_x, &pauli_y).await?;
    println!("Result:\n{:?}", xy);
    println!("Expected: i × Pauli Z");

    // Verify: XY = iZ
    let expected = &pauli_z * Complex64::new(0.0, 1.0);
    let diff: f64 = xy
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).norm())
        .sum();
    println!("Verification error: {:.2e}", diff);

    Ok(())
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
async fn demo_quantum_gates(
    gpu: &GpuLinearAlgebra,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create Hadamard gate
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    let hadamard = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(-sqrt2_inv, 0.0),
        ],
    )?;

    // Compute H² (should be identity)
    println!("Computing H² (Hadamard squared):");
    let h_squared = gpu.matmul(&hadamard, &hadamard).await?;
    println!("Result:\n{:?}", h_squared);

    // Check if it's identity
    let identity = Array2::eye(2);
    let diff: f64 = h_squared
        .iter()
        .zip(identity.iter())
        .map(|(a, b)| (a - Complex64::new(*b, 0.0)).norm())
        .sum();
    println!("Distance from identity: {:.2e}", diff);

    // Create CNOT gate (4x4)
    let cnot = Array2::from_shape_vec(
        (4, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )?;

    // CNOT² should also be identity
    println!("\nComputing CNOT²:");
    let cnot_squared = gpu.matmul(&cnot, &cnot).await?;
    let identity_4 = Array2::eye(4);
    let diff: f64 = cnot_squared
        .iter()
        .zip(identity_4.iter())
        .map(|(a, b)| (a - Complex64::new(*b, 0.0)).norm())
        .sum();
    println!("Distance from identity: {:.2e}", diff);

    Ok(())
}

// Performance comparison for different matrix sizes
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
async fn benchmark_gpu_linalg(
    gpu_linalg: &GpuLinearAlgebra,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Performance Comparison ===\n");

    for size in [16, 32, 64, 128, 256] {
        println!("Matrix size: {}x{}", size, size);

        // Create random unitary matrix (simplified - just random complex)
        let matrix = Array2::from_shape_fn((size, size), |_| {
            let angle = fastrand::f64() * 2.0 * std::f64::consts::PI;
            Complex64::from_polar(1.0, angle)
        });

        // Time CPU multiplication
        let cpu_start = Instant::now();
        let _cpu_result = matrix.dot(&matrix);
        let cpu_time = cpu_start.elapsed();

        // Time GPU multiplication
        let gpu_start = Instant::now();
        let _gpu_result = gpu_linalg.matmul(&matrix, &matrix).await?;
        let gpu_time = gpu_start.elapsed();

        println!("  CPU: {:?}", cpu_time);
        println!("  GPU: {:?}", gpu_time);

        if gpu_time < cpu_time {
            println!(
                "  GPU speedup: {:.2}x",
                cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
            );
        } else {
            println!(
                "  CPU is faster by: {:.2}x",
                gpu_time.as_secs_f64() / cpu_time.as_secs_f64()
            );
        }
        println!();
    }

    Ok(())
}

// Stub main function for platforms without GPU support
#[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
fn main() {
    println!("=== GPU Linear Algebra Demo ===\n");
    println!("GPU acceleration is not available on this platform.");
    println!("This demo requires a non-macOS system with GPU support enabled.");
}
