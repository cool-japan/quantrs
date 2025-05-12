//! Max-Cut Problem Solver with SciRS2 Integration
//!
//! This example shows how to use quantrs-tytan with SciRS2 integration
//! to solve the max-cut problem on a graph.

use quantrs_tytan::{
    calculate_diversity, cluster_solutions, optimize_qubo, symbols_list, Auto_array, Compile,
    GASampler, SASampler,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Max-Cut Problem Solver with SciRS2 Integration");
    println!("==============================================");

    // Define the graph as an adjacency matrix
    let graph = vec![
        vec![0, 1, 1, 0, 0, 0], // Node 0 is connected to 1, 2
        vec![1, 0, 1, 1, 0, 0], // Node 1 is connected to 0, 2, 3
        vec![1, 1, 0, 1, 1, 0], // Node 2 is connected to 0, 1, 3, 4
        vec![0, 1, 1, 0, 1, 1], // Node 3 is connected to 1, 2, 4, 5
        vec![0, 0, 1, 1, 0, 1], // Node 4 is connected to 2, 3, 5
        vec![0, 0, 0, 1, 1, 0], // Node 5 is connected to 3, 4
    ];

    let n_nodes = graph.len();

    println!("Graph with {} nodes:", n_nodes);
    for (i, row) in graph.iter().enumerate() {
        println!(
            "Node {}: connected to nodes {:?}",
            i,
            row.iter()
                .enumerate()
                .filter(|(_, &v)| v == 1)
                .map(|(j, _)| j)
                .collect::<Vec<_>>()
        );
    }

    // Create a binary variable for each node
    // x_i = 0 means node i is in partition 0
    // x_i = 1 means node i is in partition 1
    let x = symbols_list(n_nodes, "x{}").expect("Failed to create symbols");

    // The objective is to maximize the number of edges between partitions
    // For QUBO, we need to minimize, so we use the negative of the objective

    // Each edge (i,j) contributes weight * (x_i * (1-x_j) + (1-x_i) * x_j)
    // = weight * (x_i - 2*x_i*x_j + x_j)

    // Start with empty expression
    let mut h = 0.into();

    // Add terms for each edge
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            if graph[i][j] == 1 {
                // Edge weight is 1 for this example
                // For each edge (i,j), add -1 * (x_i + x_j - 2*x_i*x_j)
                // The negative sign is because we want to maximize cuts
                h = h - (x[[i]].clone() + x[[j]].clone() - 2 * x[[i]].clone() * x[[j]].clone());
            }
        }
    }

    println!("\nCreated QUBO expression for max-cut");

    // Compile to QUBO
    let (qubo, offset) = Compile::new(&h).get_qubo()?;
    println!("Compiled to QUBO with offset: {}", offset);

    // Solve using different methods and compare
    println!("\nSolving with multiple methods:");

    // 1. Simulated Annealing
    println!("\n1. Simulated Annealing:");
    let start = Instant::now();
    let solver = SASampler::new(Some(42)); // Fixed seed for reproducibility
    let sa_results = solver.run_qubo(&qubo, 100)?;
    let sa_duration = start.elapsed();

    // 2. Genetic Algorithm
    println!("\n2. Genetic Algorithm:");
    let start = Instant::now();
    let solver = GASampler::with_params(Some(42), 1000, 100);
    let ga_results = solver.run_qubo(&qubo, 100)?;
    let ga_duration = start.elapsed();

    // 3. Advanced Optimization (with SciRS2 if available)
    println!("\n3. Advanced Optimization:");
    let start = Instant::now();
    let adv_results = optimize_qubo(&qubo.0, &qubo.1, None, 1000);
    let adv_duration = start.elapsed();

    // Compare results
    println!("\nPerformance Comparison:");
    println!("----------------------");
    println!("Method               | Best Energy | Time (ms)");
    println!("----------------------|------------|----------");
    println!(
        "Simulated Annealing  | {:10.2} | {:10.2}",
        sa_results[0].energy,
        sa_duration.as_millis()
    );
    println!(
        "Genetic Algorithm    | {:10.2} | {:10.2}",
        ga_results[0].energy,
        ga_duration.as_millis()
    );
    println!(
        "Advanced Optimization| {:10.2} | {:10.2}",
        adv_results[0].energy,
        adv_duration.as_millis()
    );

    // Print best solution
    let best_result = &sa_results[0];
    println!("\nBest Solution (Simulated Annealing):");
    println!("Energy: {}", best_result.energy);

    // Convert to more readable format
    let (arr, _) = Auto_array::new(best_result).get_ndarray("x{}")?;

    // Display partition
    let mut partition0 = Vec::new();
    let mut partition1 = Vec::new();

    for i in 0..n_nodes {
        if arr[[i]] == 1 {
            partition1.push(i);
        } else {
            partition0.push(i);
        }
    }

    println!("Partition 0: {:?}", partition0);
    println!("Partition 1: {:?}", partition1);

    // Count the number of cut edges
    let mut cut_edges = 0;
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            if graph[i][j] == 1 {
                let val_i = arr[[i]];
                let val_j = arr[[j]];
                if val_i != val_j {
                    cut_edges += 1;
                }
            }
        }
    }

    println!("Number of cut edges: {}", cut_edges);

    // Solution analysis
    println!("\nSolution Analysis:");

    // Cluster similar solutions
    let clusters = cluster_solutions(&sa_results, 5)?;
    println!("\nFound {} solution clusters:", clusters.len());

    for (i, (indices, avg_energy)) in clusters.iter().enumerate() {
        println!(
            "Cluster {}: {} solutions, Avg Energy: {:.4}",
            i + 1,
            indices.len(),
            avg_energy
        );

        // Show first solution in each cluster
        if !indices.is_empty() {
            let example = &sa_results[indices[0]];
            let (arr, _) = Auto_array::new(example).get_ndarray("x{}")?;

            let mut partition0 = Vec::new();
            let mut partition1 = Vec::new();

            for i in 0..n_nodes {
                if arr[[i]] == 1 {
                    partition1.push(i);
                } else {
                    partition0.push(i);
                }
            }

            println!("  Example: Partition 0: {:?}", partition0);
            println!("           Partition 1: {:?}", partition1);
        }
    }

    // Diversity metrics
    let diversity = calculate_diversity(&sa_results)?;
    println!("\nDiversity Metrics:");

    if let Some(&avg_distance) = diversity.get("avg_distance") {
        println!("Average Solution Distance: {:.4}", avg_distance);
    }

    if let Some(&energy_range) = diversity.get("energy_range") {
        println!("Energy Range: {:.4}", energy_range);
    }

    Ok(())
}
