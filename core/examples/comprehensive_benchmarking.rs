//! Comprehensive Quantum Benchmarking Example
//!
//! This example demonstrates the full capabilities of the QuantRS2 benchmarking suite,
//! including noise characterization, error mitigation, and comprehensive reporting.
//!
//! Run with: cargo run --example comprehensive_benchmarking

use quantrs2_core::benchmarking_integration::ComprehensiveBenchmarkSuite;
use quantrs2_core::noise_characterization::NoiseModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   QuantRS2 Comprehensive Benchmarking Suite                  ║");
    println!("║   Demonstrating Advanced Noise Mitigation & Analysis        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Example 1: Default noise model benchmarking
    println!("═══ Example 1: Default Noise Model ═══\n");
    run_example_with_default_noise()?;

    println!("\n\n");

    // Example 2: High noise environment
    println!("═══ Example 2: High Noise Environment ═══\n");
    run_example_with_high_noise()?;

    println!("\n\n");

    // Example 3: Low coherence time system
    println!("═══ Example 3: Low Coherence Time System ═══\n");
    run_example_with_low_coherence()?;

    println!("\n\n");

    // Example 4: Near-ideal system
    println!("═══ Example 4: Near-Ideal System ═══\n");
    run_example_with_near_ideal()?;

    println!("\n\n═══ Benchmarking Complete ═══\n");

    Ok(())
}

/// Example 1: Default noise model
fn run_example_with_default_noise() -> Result<(), Box<dyn std::error::Error>> {
    let suite = ComprehensiveBenchmarkSuite::new();

    // 4-qubit MaxCut problem with triangle graph
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];

    let report = suite.benchmark_qaoa_comprehensive(4, edges, 2)?;

    report.print_detailed_report();

    Ok(())
}

/// Example 2: High noise environment (representing early NISQ devices)
fn run_example_with_high_noise() -> Result<(), Box<dyn std::error::Error>> {
    let noise_model = NoiseModel::new(
        0.01, // 1% single-qubit error
        0.05, // 5% two-qubit error
        30.0, // 30 μs T1
        40.0, // 40 μs T2
        0.05, // 5% readout error
    );

    let suite = ComprehensiveBenchmarkSuite::with_noise_model(noise_model);

    // Simple 3-qubit problem
    let edges = vec![(0, 1), (1, 2)];

    let report = suite.benchmark_qaoa_comprehensive(3, edges, 1)?;

    report.print_detailed_report();

    println!("\nJSON Export:");
    println!("{}", report.to_json());

    Ok(())
}

/// Example 3: Low coherence time (requires dynamical decoupling)
fn run_example_with_low_coherence() -> Result<(), Box<dyn std::error::Error>> {
    let noise_model = NoiseModel::new(
        0.005, // 0.5% single-qubit error
        0.02,  // 2% two-qubit error
        20.0,  // 20 μs T1 (low)
        25.0,  // 25 μs T2 (low)
        0.02,  // 2% readout error
    );

    let suite = ComprehensiveBenchmarkSuite::with_noise_model(noise_model);

    // 5-qubit pentagon graph
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];

    let report = suite.benchmark_qaoa_comprehensive(5, edges, 2)?;

    report.print_detailed_report();

    Ok(())
}

/// Example 4: Near-ideal system (future fault-tolerant devices)
fn run_example_with_near_ideal() -> Result<(), Box<dyn std::error::Error>> {
    let noise_model = NoiseModel::new(
        0.0001, // 0.01% single-qubit error
        0.0005, // 0.05% two-qubit error
        1000.0, // 1 ms T1
        1500.0, // 1.5 ms T2
        0.001,  // 0.1% readout error
    );

    let mut suite = ComprehensiveBenchmarkSuite::with_noise_model(noise_model);

    // Enable all features for near-ideal system
    suite.enable_all_features();

    // Larger 6-qubit problem
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];

    let report = suite.benchmark_qaoa_comprehensive(6, edges, 3)?;

    report.print_detailed_report();

    Ok(())
}
