//! IBM Quantum Example: Maximum Cut Problem
//!
//! This example demonstrates how to use the IBM Quantum sampler
//! to solve a Maximum Cut problem using QAOA on IBM's quantum hardware.
//!
//! Maximum Cut (MaxCut) is a classic graph optimization problem:
//! Given a graph, partition its vertices into two sets to maximize
//! the number of edges between the sets.

use quantrs2_tytan::sampler::hardware::{IBMBackend, IBMQuantumConfig, IBMQuantumSampler};
use quantrs2_tytan::sampler::Sampler;
use scirs2_core::ndarray::{Array, Array2};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBM Quantum Maximum Cut Example ===\n");

    // Define a simple graph with 4 vertices
    // Graph edges: (0,1), (0,2), (1,3), (2,3)
    let graph = vec![(0, 1), (0, 2), (1, 3), (2, 3)];
    let num_vertices = 4;

    println!("Graph vertices: {num_vertices}");
    println!("Graph edges: {graph:?}\n");

    // Convert MaxCut to QUBO formulation
    // For MaxCut, we want to maximize: sum of x_i(1-x_j) + x_j(1-x_i) for each edge (i,j)
    // This is equivalent to minimizing: -sum of (x_i + x_j - 2*x_i*x_j) for each edge

    let mut qubo_matrix = Array2::<f64>::zeros((num_vertices, num_vertices));

    for (i, j) in &graph {
        // Add linear terms
        qubo_matrix[[*i, *i]] -= 1.0;
        qubo_matrix[[*j, *j]] -= 1.0;

        // Add quadratic terms (interaction)
        qubo_matrix[[*i, *j]] += 2.0;
    }

    // Create variable mapping
    let var_map: HashMap<String, usize> = (0..num_vertices).map(|i| (format!("v{i}"), i)).collect();

    println!("QUBO Matrix:");
    println!("{qubo_matrix:.2}\n");

    // Example 1: Using IBM Quantum Simulator
    println!("--- Example 1: IBM Quantum Simulator ---");
    let simulator_sampler = IBMQuantumSampler::with_token("YOUR_IBM_QUANTUM_TOKEN")
        .with_backend(IBMBackend::Simulator)
        .with_optimization_level(2)
        .with_error_mitigation(true);

    println!("Running on IBM Quantum Simulator with QAOA...");
    let results = simulator_sampler.run_qubo(&(qubo_matrix.clone(), var_map.clone()), 1000)?;

    println!("Top 5 solutions from simulator:");
    for (idx, result) in results.iter().take(5).enumerate() {
        println!(
            "  {}. Energy: {:.4}, Occurrences: {}, Partition: {:?}",
            idx + 1,
            result.energy,
            result.occurrences,
            result.assignments
        );
    }
    println!();

    // Example 2: Using IBM Quantum Hardware (specific backend)
    println!("--- Example 2: IBM Quantum Hardware (ibm_osaka) ---");
    let hardware_sampler = IBMQuantumSampler::with_token("YOUR_IBM_QUANTUM_TOKEN")
        .with_backend(IBMBackend::Hardware("ibm_osaka".to_string()))
        .with_optimization_level(3)
        .with_error_mitigation(true);

    println!("Running on IBM Quantum Hardware (ibm_osaka)...");
    println!("Note: This would require a valid IBM Quantum API token and available hardware.");
    println!("In this example, we're showing the API usage pattern.\n");

    // Example 3: Using any available IBM Quantum hardware
    println!("--- Example 3: IBM Quantum Hardware (Any Available) ---");
    let any_hardware_sampler = IBMQuantumSampler::with_token("YOUR_IBM_QUANTUM_TOKEN")
        .with_backend(IBMBackend::AnyHardware)
        .with_optimization_level(1)
        .with_error_mitigation(false);

    println!("Configuration:");
    println!("  - Backend: Any available hardware");
    println!("  - Optimization Level: 1 (faster compilation)");
    println!("  - Error Mitigation: Disabled (faster execution)");
    println!();

    // Analyze the results
    println!("--- Result Analysis ---");
    if let Some(best_solution) = results.first() {
        println!("Best solution found:");
        println!("  Energy: {:.4}", best_solution.energy);
        println!(
            "  Cut size: {}",
            calculate_cut_size(&best_solution.assignments, &graph, &var_map)
        );

        let partition_0: Vec<_> = best_solution
            .assignments
            .iter()
            .filter(|(_, &val)| !val)
            .map(|(name, _)| name)
            .collect();
        let partition_1: Vec<_> = best_solution
            .assignments
            .iter()
            .filter(|(_, &val)| val)
            .map(|(name, _)| name)
            .collect();

        println!("  Partition 0: {partition_0:?}");
        println!("  Partition 1: {partition_1:?}");
    }

    // Tips for using IBM Quantum
    println!("\n--- Tips for IBM Quantum ---");
    println!("1. Get a free API token from: https://quantum-computing.ibm.com/");
    println!("2. Use the simulator for testing and development");
    println!("3. Enable error mitigation for better results on real hardware");
    println!("4. Higher optimization levels may take longer but produce better circuits");
    println!("5. Check queue times before submitting to real hardware");
    println!("6. For QAOA, typical circuit depth is O(p) where p is the number of layers");

    Ok(())
}

/// Calculate the cut size for a given partition
fn calculate_cut_size(
    assignments: &HashMap<String, bool>,
    graph: &[(usize, usize)],
    var_map: &HashMap<String, usize>,
) -> usize {
    let idx_to_var: HashMap<usize, String> = var_map
        .iter()
        .map(|(name, &idx)| (idx, name.clone()))
        .collect();

    graph
        .iter()
        .filter(|(i, j)| {
            let vi = idx_to_var.get(i).unwrap();
            let vj = idx_to_var.get(j).unwrap();
            assignments.get(vi) != assignments.get(vj)
        })
        .count()
}
