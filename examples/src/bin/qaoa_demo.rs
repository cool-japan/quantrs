//! QAOA demonstration for solving Max-Cut problems

use quantrs2_core::prelude::*;
use std::time::Instant;

fn main() {
    println!("=== QAOA Max-Cut Solver Demo ===\n");

    // Example 1: Simple triangle graph
    println!("Example 1: Triangle Graph (3 nodes)");
    solve_triangle_maxcut();

    println!("\n{}\n", "=".repeat(50));

    // Example 2: Square graph
    println!("Example 2: Square Graph (4 nodes)");
    solve_square_maxcut();

    println!("\n{}\n", "=".repeat(50));

    // Example 3: Complete graph K5
    println!("Example 3: Complete Graph K5 (5 nodes)");
    solve_k5_maxcut();

    println!("\n{}\n", "=".repeat(50));

    // Example 4: Weighted graph
    println!("Example 4: Weighted Graph");
    solve_weighted_maxcut();
}

fn solve_triangle_maxcut() {
    // Triangle graph: 3 nodes, 3 edges
    let edges = vec![(0, 1), (1, 2), (2, 0)];
    let num_qubits = 3;

    println!("Graph edges: {edges:?}");
    println!("Optimal Max-Cut value: 2");

    // Try different number of QAOA layers
    for p in 1..=3 {
        println!("\nRunning QAOA with p = {p} layers");

        let start = Instant::now();

        // Initialize QAOA
        let params = QAOAParams::random(p);
        let circuit = QAOACircuit::new(
            num_qubits,
            CostHamiltonian::MaxCut(edges.clone()),
            MixerHamiltonian::TransverseField,
            params,
        );

        // Optimize
        let mut optimizer = QAOAOptimizer::new(circuit, 200, 0.001);
        let (optimized_params, _, final_cost) = optimizer.optimize();

        let elapsed = start.elapsed();

        // Get the solution
        let state = optimizer.execute_circuit();
        let solution = optimizer.get_solution(&state);

        println!("  Optimization time: {elapsed:?}");
        println!("  Final cost: {final_cost:.4}");
        println!("  Solution: {solution:?}");
        println!("  Cut size: {}", calculate_cut_size(&edges, &solution));
        println!("  Optimized params: {optimized_params:?}");
    }
}

fn solve_square_maxcut() {
    // Square graph: 4 nodes, 4 edges
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    let num_qubits = 4;

    println!("Graph edges: {edges:?}");
    println!("Optimal Max-Cut value: 4");

    let p = 3; // Use 3 layers for better results
    println!("\nRunning QAOA with p = {p} layers");

    let start = Instant::now();

    let params = QAOAParams::random(p);
    let circuit = QAOACircuit::new(
        num_qubits,
        CostHamiltonian::MaxCut(edges.clone()),
        MixerHamiltonian::TransverseField,
        params,
    );

    let mut optimizer = QAOAOptimizer::new(circuit, 300, 0.001);
    let (optimized_params, _, final_cost) = optimizer.optimize();

    let elapsed = start.elapsed();

    let state = optimizer.execute_circuit();
    let solution = optimizer.get_solution(&state);

    println!("  Optimization time: {elapsed:?}");
    println!("  Final cost: {final_cost:.4}");
    println!("  Solution: {solution:?}");
    println!("  Cut size: {}", calculate_cut_size(&edges, &solution));
}

fn solve_k5_maxcut() {
    // Complete graph K5: 5 nodes, 10 edges
    let mut edges = Vec::new();
    for i in 0..5 {
        for j in (i + 1)..5 {
            edges.push((i, j));
        }
    }
    let num_qubits = 5;

    println!("Complete graph K5 with {} edges", edges.len());
    println!("Optimal Max-Cut value: 6");

    let p = 4; // Use 4 layers for this harder problem
    println!("\nRunning QAOA with p = {p} layers");

    let start = Instant::now();

    let params = QAOAParams::random(p);
    let circuit = QAOACircuit::new(
        num_qubits,
        CostHamiltonian::MaxCut(edges.clone()),
        MixerHamiltonian::TransverseField,
        params,
    );

    let mut optimizer = QAOAOptimizer::new(circuit, 500, 0.001);
    let (optimized_params, _, final_cost) = optimizer.optimize();

    let elapsed = start.elapsed();

    let state = optimizer.execute_circuit();
    let solution = optimizer.get_solution(&state);

    println!("  Optimization time: {elapsed:?}");
    println!("  Final cost: {final_cost:.4}");
    println!("  Solution: {solution:?}");
    println!("  Cut size: {}", calculate_cut_size(&edges, &solution));
}

fn solve_weighted_maxcut() {
    // Weighted graph: different edge weights
    let weighted_edges = vec![
        (0, 1, 1.0),
        (1, 2, 2.0),
        (2, 3, 1.5),
        (3, 0, 0.5),
        (0, 2, 3.0),
    ];
    let num_qubits = 4;

    println!("Weighted graph edges: {weighted_edges:?}");

    let p = 3;
    println!("\nRunning QAOA with p = {p} layers");

    let start = Instant::now();

    let params = QAOAParams::random(p);
    let circuit = QAOACircuit::new(
        num_qubits,
        CostHamiltonian::WeightedMaxCut(weighted_edges.clone()),
        MixerHamiltonian::TransverseField,
        params,
    );

    let mut optimizer = QAOAOptimizer::new(circuit, 300, 0.001);
    let (optimized_params, _, final_cost) = optimizer.optimize();

    let elapsed = start.elapsed();

    let state = optimizer.execute_circuit();
    let solution = optimizer.get_solution(&state);

    println!("  Optimization time: {elapsed:?}");
    println!("  Final cost: {final_cost:.4}");
    println!("  Solution: {solution:?}");
    println!(
        "  Weighted cut value: {:.2}",
        calculate_weighted_cut(&weighted_edges, &solution)
    );
}

fn calculate_cut_size(edges: &[(usize, usize)], solution: &[bool]) -> usize {
    edges
        .iter()
        .filter(|(i, j)| solution[*i] != solution[*j])
        .count()
}

fn calculate_weighted_cut(edges: &[(usize, usize, f64)], solution: &[bool]) -> f64 {
    edges
        .iter()
        .filter(|(i, j, _)| solution[*i] != solution[*j])
        .map(|(_, _, w)| w)
        .sum()
}
