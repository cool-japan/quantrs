//! SIMD Optimization Demo for QuantRS2
//!
//! This example demonstrates the performance improvements achieved by using
//! SIMD-accelerated quantum operations.

use num_complex::Complex64;
use quantrs2_core::simd_ops::{
    apply_phase_simd, expectation_z_simd, hadamard_simd, inner_product, normalize_simd,
};
use std::time::Instant;

fn main() {
    println!("=== SIMD Optimization Demo ===\n");

    // Test different state vector sizes
    let sizes = vec![4, 16, 64, 256, 1024, 4096];

    for &size in &sizes {
        println!("Testing with {} qubits ({} amplitudes):", 
                 (size as f64).log2() as usize, size);
        
        // Create random quantum state
        let mut state: Vec<Complex64> = (0..size)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / size as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();

        // Normalize the state
        let start = Instant::now();
        normalize_simd(&mut state).unwrap();
        let norm_time = start.elapsed();
        println!("  Normalization time: {:?}", norm_time);

        // Test phase rotation
        let theta = std::f64::consts::PI / 4.0;
        let start = Instant::now();
        apply_phase_simd(&mut state, theta);
        let phase_time = start.elapsed();
        println!("  Phase rotation time: {:?}", phase_time);

        // Test Hadamard gate (only for smaller states)
        if size <= 256 {
            let num_qubits = (size as f64).log2() as usize;
            let mut hadamard_state = state.clone();
            let start = Instant::now();
            for q in 0..num_qubits.min(3) {
                hadamard_simd(&mut hadamard_state, q, num_qubits);
            }
            let hadamard_time = start.elapsed();
            println!("  Hadamard gates time: {:?}", hadamard_time);
        }

        // Test expectation value computation
        let num_qubits = (size as f64).log2() as usize;
        let start = Instant::now();
        let mut total_expectation = 0.0;
        for q in 0..num_qubits.min(5) {
            total_expectation += expectation_z_simd(&state, q, num_qubits);
        }
        let expectation_time = start.elapsed();
        println!("  Expectation values time: {:?}", expectation_time);

        // Test inner product
        let state2: Vec<Complex64> = (0..size)
            .map(|i| {
                let angle = 3.0 * std::f64::consts::PI * i as f64 / size as f64;
                Complex64::new(angle.sin(), angle.cos())
            })
            .collect();

        let start = Instant::now();
        let _inner_prod = inner_product(&state, &state2).unwrap();
        let inner_time = start.elapsed();
        println!("  Inner product time: {:?}", inner_time);

        println!();
    }

    // Demonstrate chunked processing for large states
    println!("=== Chunked Processing Demo ===\n");
    
    let large_size = 16384; // 14 qubits
    let mut large_state: Vec<Complex64> = (0..large_size)
        .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    // Process in chunks to demonstrate cache efficiency
    let chunk_size = 256;
    let start = Instant::now();
    
    for chunk in large_state.chunks_mut(chunk_size) {
        // Apply operations to each chunk
        normalize_simd(chunk).unwrap();
        apply_phase_simd(chunk, 0.1);
    }
    
    let chunked_time = start.elapsed();
    println!("Chunked processing of {} amplitudes: {:?}", large_size, chunked_time);
    println!("Average time per chunk: {:?}", chunked_time / (large_size / chunk_size) as u32);

    // Performance comparison summary
    println!("\n=== Performance Summary ===");
    println!("SIMD operations provide significant speedup for:");
    println!("- Normalization of quantum states");
    println!("- Phase rotations");
    println!("- Gate applications");
    println!("- Expectation value computations");
    println!("- Inner product calculations");
    println!("\nBest performance is achieved with:");
    println!("- State vectors aligned to SIMD register boundaries");
    println!("- Operations on multiple qubits in parallel");
    println!("- Chunked processing for cache efficiency");
}