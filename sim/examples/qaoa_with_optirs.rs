//! QAOA Example with OptiRS Optimization
//!
//! This example demonstrates how to use the Quantum Approximate Optimization Algorithm (QAOA)
//! with OptiRS's state-of-the-art optimizers (Adam, SGD, RMSprop, etc.) for solving
//! combinatorial optimization problems like MaxCut.
//!
//! Run with: cargo run --example qaoa_with_optirs --features optimize

use quantrs2_sim::error::Result;
use quantrs2_sim::qaoa_optimization::{
    QAOAConfig, QAOAGraph, QAOAInitializationStrategy, QAOAMixerType, QAOAOptimizationStrategy,
    QAOAOptimizer, QAOAProblemType,
};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== QAOA with OptiRS Optimization Demo ===\n");

    // Create a simple MaxCut problem on a triangle graph
    // Vertices: 0, 1, 2
    // Edges: (0,1), (1,2), (0,2) with weight 1.0
    let num_vertices = 4;
    let mut adjacency_matrix = Array2::zeros((num_vertices, num_vertices));
    let mut edge_weights = HashMap::new();

    // Create a square graph
    adjacency_matrix[[0, 1]] = 1.0;
    adjacency_matrix[[1, 0]] = 1.0;
    edge_weights.insert((0, 1), 1.0);
    edge_weights.insert((1, 0), 1.0);

    adjacency_matrix[[1, 2]] = 1.0;
    adjacency_matrix[[2, 1]] = 1.0;
    edge_weights.insert((1, 2), 1.0);
    edge_weights.insert((2, 1), 1.0);

    adjacency_matrix[[2, 3]] = 1.0;
    adjacency_matrix[[3, 2]] = 1.0;
    edge_weights.insert((2, 3), 1.0);
    edge_weights.insert((3, 2), 1.0);

    adjacency_matrix[[3, 0]] = 1.0;
    adjacency_matrix[[0, 3]] = 1.0;
    edge_weights.insert((3, 0), 1.0);
    edge_weights.insert((0, 3), 1.0);

    // Add diagonal edges
    adjacency_matrix[[0, 2]] = 1.0;
    adjacency_matrix[[2, 0]] = 1.0;
    edge_weights.insert((0, 2), 1.0);
    edge_weights.insert((2, 0), 1.0);

    let graph = QAOAGraph {
        num_vertices,
        adjacency_matrix,
        vertex_weights: vec![1.0; num_vertices],
        edge_weights,
        constraints: vec![],
    };

    println!(
        "Problem: MaxCut on a square graph with {} vertices",
        num_vertices
    );
    println!("Edges: (0,1), (1,2), (2,3), (3,0), (0,2)");
    println!("Expected max cut value: 4 (optimal: {{0,2}} vs {{1,3}})\n");

    // Test different optimization strategies
    let strategies = vec![
        (
            "Classical Gradient Descent",
            QAOAOptimizationStrategy::Classical,
        ),
        #[cfg(feature = "optimize")]
        ("OptiRS (Adam)", QAOAOptimizationStrategy::OptiRS),
    ];

    for (strategy_name, strategy) in strategies {
        println!("\n=== Testing {} ===", strategy_name);

        // Configure QAOA
        let config = QAOAConfig {
            num_layers: 2,
            mixer_type: QAOAMixerType::Standard,
            initialization: QAOAInitializationStrategy::UniformSuperposition,
            optimization_strategy: strategy,
            max_iterations: 50,
            convergence_tolerance: 1e-5,
            learning_rate: 0.1,
            multi_angle: false,
            parameter_transfer: false,
            hardware_aware: false,
            shots: None,
            adaptive_layers: false,
            max_adaptive_layers: 5,
        };

        // Create QAOA optimizer
        let mut qaoa = QAOAOptimizer::new(config, graph.clone(), QAOAProblemType::MaxCut)?;

        // Run optimization
        let start = Instant::now();
        let result = qaoa.optimize()?;
        let elapsed = start.elapsed();

        // Print results
        println!("  Converged: {}", result.converged);
        println!("  Best cost: {:.4}", result.best_cost);
        println!("  Approximation ratio: {:.4}", result.approximation_ratio);
        println!("  Iterations: {}", result.function_evaluations);
        println!("  Time: {:.2?}", elapsed);
        println!("  Best solution: {}", result.best_solution);

        // Print final parameters
        println!("\n  Final parameters:");
        println!("    Gammas: {:?}", result.optimal_gammas);
        println!("    Betas:  {:?}", result.optimal_betas);

        // Print top solutions
        println!("\n  Top 5 solutions:");
        let mut sorted_probs: Vec<_> = result.final_probabilities.iter().collect();
        sorted_probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (i, (bitstring, prob)) in sorted_probs.iter().take(5).enumerate() {
            let cut_value = calculate_cut_value(bitstring, &graph);
            println!(
                "    {}: {} (prob: {:.4}, cut value: {})",
                i + 1,
                bitstring,
                prob,
                cut_value
            );
        }

        // Print solution quality
        println!("\n  Solution quality:");
        println!("    Feasible: {}", result.solution_quality.feasible);
        println!("    Confidence: {:.4}", result.solution_quality.confidence);
        if let Some(gap) = result.solution_quality.optimality_gap {
            println!("    Optimality gap: {:.4}", gap);
        }

        // Print cost history (last 10 iterations)
        if result.cost_history.len() > 10 {
            println!("\n  Last 10 cost values:");
            for (i, cost) in result.cost_history.iter().rev().take(10).rev().enumerate() {
                let iter_num = result.cost_history.len() - 10 + i;
                println!("    Iter {}: {:.6}", iter_num, cost);
            }
        }
    }

    println!("\n=== Summary ===");
    println!("OptiRS optimization (especially Adam) typically finds better solutions");
    println!("faster than classical gradient descent for QAOA problems.");
    println!("The adaptive learning rates help navigate the complex optimization landscape.");

    Ok(())
}

/// Helper function to calculate cut value for a bitstring
fn calculate_cut_value(bitstring: &str, graph: &QAOAGraph) -> usize {
    let assignment: Vec<bool> = bitstring.chars().map(|c| c == '1').collect();
    let mut cut_value = 0;

    for i in 0..graph.num_vertices {
        for j in (i + 1)..graph.num_vertices {
            if assignment[i] != assignment[j] {
                if let Some(&weight) = graph.edge_weights.get(&(i, j)) {
                    cut_value += weight as usize;
                }
            }
        }
    }

    cut_value
}
