#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
//! Quantum Error Correction Annealing Example
//!
//! This example demonstrates the use of quantum error correction (QEC) in quantum annealing.
//! It shows how to:
//! - Configure and use the QuantumErrorCorrectionAnnealer
//! - Compare different error correction codes
//! - Analyze the benefits of error correction for solution quality
//! - Track performance statistics and overhead
//!
//! The example solves a Max-Cut problem on a small graph using various QEC configurations
//! and compares the results.

use quantrs2_anneal::ising::IsingModel;
use quantrs2_anneal::quantum_error_correction::{
    CodeParameters, ErrorCorrectionCode, QECConfig, QuantumErrorCorrectionAnnealer,
};
use quantrs2_anneal::simulator::AnnealingParams;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Error Correction Annealing Example ===\n");

    // Create a Max-Cut problem on a 6-node graph
    let problem = create_max_cut_problem();
    println!("Problem: Max-Cut on 6-node graph");
    println!("Number of qubits: {}\n", problem.num_qubits);

    // Configure annealing parameters
    let mut params = AnnealingParams::new();
    params.num_sweeps = 1000;
    params.num_repetitions = 10;

    // Example 1: Basic QEC with default configuration
    println!("--- Example 1: Basic QEC with Default Configuration ---");
    run_basic_qec_example(&problem, params.clone())?;

    // Example 2: Comparison of different error correction codes
    println!("\n--- Example 2: Comparing Different Error Correction Codes ---");
    compare_error_correction_codes(&problem, params.clone())?;

    // Example 3: Performance analysis
    println!("\n--- Example 3: Performance Analysis ---");
    analyze_qec_performance(&problem, params.clone())?;

    // Example 4: Statistics tracking
    println!("\n--- Example 4: Statistics Tracking ---");
    demonstrate_statistics_tracking(&problem, params)?;

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Create a Max-Cut problem on a small graph
fn create_max_cut_problem() -> IsingModel {
    let n = 6; // 6 nodes
    let mut model = IsingModel::new(n);

    // Define edges (couplings) for a complete graph with varying weights
    let edges = vec![
        (0, 1, -1.0),
        (0, 2, -1.5),
        (1, 2, -1.0),
        (1, 3, -2.0),
        (2, 3, -1.0),
        (2, 4, -1.5),
        (3, 4, -1.0),
        (3, 5, -1.0),
        (4, 5, -1.5),
    ];

    for (i, j, weight) in edges {
        model.set_coupling(i, j, weight).unwrap();
    }

    model
}

/// Example 1: Basic usage of QEC annealer
fn run_basic_qec_example(
    problem: &IsingModel,
    params: AnnealingParams,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create QEC annealer with default configuration
    let mut annealer = QuantumErrorCorrectionAnnealer::new(QECConfig::default())?;

    println!("Running QEC annealing with default configuration...");
    let result = annealer.solve(problem, Some(params))?;

    println!("Results:");
    println!("  Logical solution: {:?}", result.logical_solution);
    println!("  Logical energy: {:.4}", result.logical_energy);
    println!("  Logical fidelity: {:.4}", result.logical_fidelity);
    println!("  Annealing time: {:?}", result.annealing_time);
    println!("  Correction overhead: {:?}", result.correction_overhead);
    println!("  Total time: {:?}", result.total_time);

    println!("\nStatistics:");
    println!("  Physical qubits: {}", result.stats.num_physical_qubits);
    println!("  Logical qubits: {}", result.stats.num_logical_qubits);
    println!("  Code distance: {}", result.stats.code_distance);
    println!(
        "  Physical error rate: {:.6}",
        result.stats.physical_error_rate
    );
    println!(
        "  Logical error rate: {:.6}",
        result.stats.logical_error_rate
    );

    Ok(())
}

/// Example 2: Compare different error correction codes
fn compare_error_correction_codes(
    problem: &IsingModel,
    params: AnnealingParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let codes = vec![
        (ErrorCorrectionCode::RepetitionCode, "Repetition Code"),
        (ErrorCorrectionCode::SteaneCode, "Steane Code"),
        (ErrorCorrectionCode::ShorCode, "Shor Code"),
    ];

    println!("Comparing error correction codes:\n");
    println!(
        "{:<20} {:>12} {:>12} {:>15}",
        "Code", "Energy", "Fidelity", "Overhead (ms)"
    );
    println!("{}", "-".repeat(60));

    for (code, name) in codes {
        let config = QECConfig {
            code_type: code,
            code_parameters: CodeParameters::default(),
            error_threshold: 0.01,
            correction_frequency: 1000.0,
            logical_operations: Vec::new(),
            fault_tolerance_level: 1,
            resource_constraints: Default::default(),
            annealing_integration: Default::default(),
        };

        let mut annealer = QuantumErrorCorrectionAnnealer::new(config)?;
        let result = annealer.solve(problem, Some(params.clone()))?;

        println!(
            "{:<20} {:>12.4} {:>12.4} {:>15.2}",
            name,
            result.logical_energy,
            result.logical_fidelity,
            result.correction_overhead.as_secs_f64() * 1000.0
        );
    }

    Ok(())
}

/// Example 3: Analyze QEC performance benefits
fn analyze_qec_performance(
    problem: &IsingModel,
    params: AnnealingParams,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Analyzing QEC performance benefits:\n");

    // Run multiple times to get statistics
    let num_runs = 5;
    let mut qec_energies = Vec::new();
    let mut qec_fidelities = Vec::new();
    let mut qec_times = Vec::new();

    let config = QECConfig::default();
    let mut annealer = QuantumErrorCorrectionAnnealer::new(config)?;

    println!("Running {num_runs} annealing iterations...");
    for i in 0..num_runs {
        let result = annealer.solve(problem, Some(params.clone()))?;
        qec_energies.push(result.logical_energy);
        qec_fidelities.push(result.logical_fidelity);
        qec_times.push(result.total_time.as_secs_f64());

        println!(
            "  Run {}: Energy = {:.4}, Fidelity = {:.4}",
            i + 1,
            result.logical_energy,
            result.logical_fidelity
        );
    }

    // Calculate statistics
    let avg_energy = qec_energies.iter().sum::<f64>() / num_runs as f64;
    let avg_fidelity = qec_fidelities.iter().sum::<f64>() / num_runs as f64;
    let avg_time = qec_times.iter().sum::<f64>() / num_runs as f64;

    let min_energy = qec_energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_energy = qec_energies
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\nPerformance Summary:");
    println!("  Average energy: {avg_energy:.4}");
    println!("  Best energy: {min_energy:.4}");
    println!("  Worst energy: {max_energy:.4}");
    println!("  Energy variance: {:.4}", max_energy - min_energy);
    println!("  Average fidelity: {avg_fidelity:.4}");
    println!("  Average time: {avg_time:.4}s");

    Ok(())
}

/// Example 4: Demonstrate statistics tracking
fn demonstrate_statistics_tracking(
    problem: &IsingModel,
    params: AnnealingParams,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating statistics tracking:\n");

    let mut annealer = QuantumErrorCorrectionAnnealer::new(QECConfig::default())?;

    // Run multiple problems
    println!("Running multiple annealing problems...");
    for i in 0..3 {
        let _ = annealer.solve(problem, Some(params.clone()))?;
        let stats = annealer.get_statistics();

        println!("\nAfter run {}:", i + 1);
        println!("  Total runs: {}", stats.total_runs);
        println!("  Successful runs: {}", stats.successful_runs);
        println!(
            "  Success rate: {:.2}%",
            100.0 * stats.successful_runs as f64 / stats.total_runs as f64
        );
        println!(
            "  Avg logical error rate: {:.6}",
            stats.avg_logical_error_rate
        );
        println!(
            "  Avg physical error rate: {:.6}",
            stats.avg_physical_error_rate
        );
        println!(
            "  Avg correction overhead: {:?}",
            stats.avg_correction_overhead
        );
        println!(
            "  Best fidelity achieved: {:.4}",
            stats.best_logical_fidelity
        );
    }

    // Reset statistics
    println!("\nResetting statistics...");
    annealer.reset_statistics();
    let stats = annealer.get_statistics();
    println!("Total runs after reset: {}", stats.total_runs);

    Ok(())
}

/// Example 5: Advanced configuration with custom parameters
#[allow(dead_code)]
fn advanced_qec_configuration_example(
    problem: &IsingModel,
    params: AnnealingParams,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Advanced QEC Configuration ---\n");

    // Create custom code parameters
    let code_params = CodeParameters {
        distance: 5, // Higher distance for better error correction
        num_logical_qubits: 1,
        num_physical_qubits: 25, // 5x5 for distance-5 code
        num_ancilla_qubits: 12,
        code_rate: 0.04,
        threshold_probability: 0.001, // Lower threshold for better performance
    };

    let config = QECConfig {
        code_type: ErrorCorrectionCode::SurfaceCode,
        code_parameters: code_params,
        error_threshold: 0.005,       // Stricter error threshold
        correction_frequency: 2000.0, // Higher correction frequency
        logical_operations: Vec::new(),
        fault_tolerance_level: 2, // Higher fault tolerance level
        resource_constraints: Default::default(),
        annealing_integration: Default::default(),
    };

    let mut annealer = QuantumErrorCorrectionAnnealer::new(config)?;
    let result = annealer.solve(problem, Some(params))?;

    println!("Advanced QEC Results:");
    println!("  Code: Surface Code (distance 5)");
    println!("  Physical qubits: {}", result.stats.num_physical_qubits);
    println!("  Logical energy: {:.4}", result.logical_energy);
    println!("  Logical fidelity: {:.4}", result.logical_fidelity);
    println!(
        "  Error suppression: {:.2}x",
        result.stats.physical_error_rate / result.stats.logical_error_rate.max(1e-10)
    );

    Ok(())
}

/// Helper function to demonstrate error mitigation benefits
#[allow(dead_code)]
fn compare_with_without_qec(
    problem: &IsingModel,
    params: AnnealingParams,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Comparison: With vs Without QEC ---\n");

    // Without QEC (using standard classical annealing)
    use quantrs2_anneal::simulator::ClassicalAnnealingSimulator;
    let classical_simulator = ClassicalAnnealingSimulator::new(params.clone())?;
    let classical_result = classical_simulator.solve(problem)?;

    // Convert i8 to i32 for energy calculation
    let classical_spins_i8 = classical_result.best_spins;
    let classical_energy = problem.energy(&classical_spins_i8)?;

    println!("Without QEC (Classical):");
    println!("  Energy: {classical_energy:.4}");
    println!("  Runtime: {:?}", classical_result.runtime);

    // With QEC
    let mut qec_annealer = QuantumErrorCorrectionAnnealer::new(QECConfig::default())?;
    let qec_result = qec_annealer.solve(problem, Some(params))?;

    println!("\nWith QEC:");
    println!("  Logical energy: {:.4}", qec_result.logical_energy);
    println!("  Logical fidelity: {:.4}", qec_result.logical_fidelity);
    println!("  Total runtime: {:?}", qec_result.total_time);
    println!("  QEC overhead: {:?}", qec_result.correction_overhead);

    // Analysis
    let energy_improvement = classical_energy - qec_result.logical_energy;
    let overhead_percentage = 100.0 * qec_result.correction_overhead.as_secs_f64()
        / qec_result.annealing_time.as_secs_f64();

    println!("\nAnalysis:");
    if energy_improvement > 0.0 {
        println!("  Energy improvement: {energy_improvement:.4} (better)");
    } else {
        println!(
            "  Energy difference: {:.4} (comparable)",
            energy_improvement.abs()
        );
    }
    println!("  QEC overhead: {overhead_percentage:.2}%");
    println!(
        "  Error suppression factor: {:.2}x",
        qec_result.stats.physical_error_rate / qec_result.stats.logical_error_rate.max(1e-10)
    );

    Ok(())
}
