//! Utility functions example for QuantRS2
//!
//! This example demonstrates the utility functions provided by QuantRS2
//! for memory estimation, validation, and formatting.
//!
//! Run with: cargo run --example utility_functions

use quantrs2::utils::*;
use std::time::Duration;

fn main() {
    println!("=== QuantRS2 Utility Functions Example ===\n");

    // 1. Memory estimation for quantum simulations
    println!("1. Memory Estimation:");
    for num_qubits in [10, 20, 25, 30, 35] {
        let mem_bytes = estimate_statevector_memory(num_qubits);
        println!(
            "   {} qubits requires: {}",
            num_qubits,
            format_memory(mem_bytes)
        );
    }
    println!();

    // 2. Check if qubit count is valid for available memory
    println!("2. Memory Validation:");
    let available_memory = 8 * 1024 * 1024 * 1024; // 8 GB
    println!("   Available memory: {}", format_memory(available_memory));

    for num_qubits in [20, 25, 28, 30, 32] {
        let is_valid = is_valid_qubit_count(num_qubits, available_memory);
        let status = if is_valid { "✓ CAN" } else { "✗ CANNOT" };
        println!("   {} simulate {} qubits", status, num_qubits);
    }
    println!();

    // 3. Calculate maximum qubits for given memory
    println!("3. Maximum Qubit Calculations:");
    for mem_gb in [1, 4, 8, 16, 32, 64] {
        let mem_bytes = mem_gb * 1024 * 1024 * 1024;
        let max_qubits = max_qubits_for_memory(mem_bytes);
        println!("   {} GB → max {} qubits", mem_gb, max_qubits);
    }
    println!();

    // 4. Memory formatting
    println!("4. Memory Formatting:");
    let sizes = [
        512,
        1024,
        1024 * 1024,
        1536 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
    ];
    for size in sizes {
        println!("   {} bytes = {}", size, format_memory(size));
    }
    println!();

    // 5. Duration formatting
    println!("5. Duration Formatting:");
    let durations = [
        Duration::from_micros(500),
        Duration::from_millis(250),
        Duration::from_millis(1500),
        Duration::from_secs(5),
        Duration::from_secs(90),
    ];
    for duration in durations {
        println!("   {:?} = {}", duration, format_duration(duration));
    }
    println!();

    // 6. Range validation
    println!("6. Range Validation:");
    let test_values = [0, 5, 10, 15, 20];
    let (min, max) = (5, 15);
    println!("   Valid range: [{}, {}]", min, max);
    for value in test_values {
        let in_range = is_in_range(&value, &min, &max);
        let status = if in_range { "✓" } else { "✗" };
        println!(
            "   {} Value {} is {}",
            status,
            value,
            if in_range { "valid" } else { "invalid" }
        );
    }
    println!();

    // 7. Binomial coefficients
    println!("7. Binomial Coefficients:");
    println!("   Pascal's Triangle (first 6 rows):");
    for n in 0..6 {
        print!("   n={}: ", n);
        for k in 0..=n {
            print!("{:4} ", binomial(n, k));
        }
        println!();
    }
    println!();

    // 8. Factorials
    println!("8. Factorials:");
    for n in 0..=10 {
        println!("   {}! = {}", n, factorial(n));
    }
    println!();

    // 9. Practical example: Circuit complexity estimation
    println!("9. Practical Example - Circuit Complexity:");
    let num_qubits = 10;
    let circuit_depth = 20;
    let gates_per_layer = 5;

    println!("   Quantum Circuit:");
    println!("     - Qubits: {}", num_qubits);
    println!("     - Depth: {}", circuit_depth);
    println!("     - Gates per layer: {}", gates_per_layer);
    println!();
    println!("   Resource Requirements:");
    println!(
        "     - State vector memory: {}",
        format_memory(estimate_statevector_memory(num_qubits))
    );
    println!(
        "     - Minimum RAM needed: {}",
        format_memory(estimate_statevector_memory(num_qubits) * 2)
    ); // 2x for operations
    println!("     - Total gates: {}", circuit_depth * gates_per_layer);
    println!();

    // Estimate execution time (rough approximation)
    let microseconds_per_gate = 10;
    let total_time_us = (circuit_depth * gates_per_layer) as u64 * microseconds_per_gate;
    let estimated_duration = Duration::from_micros(total_time_us);
    println!(
        "     - Estimated execution time: {}",
        format_duration(estimated_duration)
    );
    println!();

    println!("=== Example Complete ===");
}
