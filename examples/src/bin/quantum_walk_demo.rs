//! Quantum Walk Algorithms Demonstration
//!
//! This example demonstrates various quantum walk algorithms implemented in QuantRS2:
//! - Discrete-time quantum walk on a line graph
//! - Continuous-time quantum walk on a complete graph
//! - Quantum walk search algorithm

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use quantrs2_core::prelude::*;
use quantrs2_core::quantum_walk::{CoinOperator, SearchOracle};
use std::f64::consts::PI;

fn main() {
    println!("=== Quantum Walk Algorithms Demonstration ===\n");

    // Example 1: Discrete-time quantum walk on a line graph
    discrete_walk_example();

    // Example 2: Continuous-time quantum walk on a complete graph
    continuous_walk_example();

    // Example 3: Quantum walk search algorithm
    quantum_search_example();
}

fn discrete_walk_example() {
    println!("1. Discrete-Time Quantum Walk on Line Graph");
    println!("-------------------------------------------");

    // Create a line graph with 8 vertices
    let graph = Graph::new(GraphType::Line, 8);

    // Create discrete quantum walk with Hadamard coin
    let mut walk = DiscreteQuantumWalk::new(graph, CoinOperator::Hadamard);

    // Initialize walker at position 0
    walk.initialize_position(0);

    println!("Initial position: 0");
    println!("Evolution steps:");

    // Evolve for 10 time steps
    for step in 1..=10 {
        walk.step();

        // Get position probabilities
        let probs = walk.position_probabilities();

        // Find the most probable position
        let (max_pos, max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        println!(
            "  Step {}: Most probable position = {} (p = {:.3})",
            step, max_pos, max_prob
        );
    }

    // Show final probability distribution
    println!("\nFinal probability distribution:");
    let final_probs = walk.position_probabilities();
    for (pos, prob) in final_probs.iter().enumerate() {
        if *prob > 0.01 {
            // Only show significant probabilities
            println!("  Position {}: {:.3}", pos, prob);
        }
    }
    println!();
}

fn continuous_walk_example() {
    println!("2. Continuous-Time Quantum Walk on Complete Graph");
    println!("-------------------------------------------------");

    // Create a complete graph with 5 vertices
    let graph = Graph::new(GraphType::Complete, 5);

    // Create continuous quantum walk
    let mut walk = ContinuousQuantumWalk::new(graph);

    // Initialize walker at vertex 0
    walk.initialize_vertex(0);

    println!("Initial vertex: 0");
    println!("Time evolution:");

    // Evolve for different time values
    let times = vec![0.5, 1.0, 2.0, 3.0, 5.0];

    for t in times {
        walk.evolve(t);

        // Get vertex probabilities
        let probs = walk.vertex_probabilities();

        println!("\n  Time t = {}:", t);
        for (vertex, prob) in probs.iter().enumerate() {
            println!("    Vertex {}: {:.3}", vertex, prob);
        }
    }

    // Demonstrate transport probability
    let transport_prob = walk.transport_probability(0, 4, PI);
    println!(
        "\nTransport probability from vertex 0 to 4 at t=π: {:.3}",
        transport_prob
    );
    println!();
}

fn quantum_search_example() {
    println!("3. Quantum Walk Search Algorithm");
    println!("--------------------------------");

    // Create a complete graph with 16 vertices
    let graph = Graph::new(GraphType::Complete, 16);

    // Define search oracle: marked vertices are 7 and 11
    let oracle = SearchOracle::new(vec![7, 11]);

    // Create quantum walk search
    let mut search = QuantumWalkSearch::new(graph, oracle);

    println!("Graph: Complete graph with 16 vertices");
    println!("Marked vertices: 7, 11");
    println!("Running quantum walk search...\n");

    // Run the search algorithm
    let (found_vertex, success_prob, steps) = search.run(100); // max 100 steps

    println!("Search results:");
    println!("  Found vertex: {}", found_vertex);
    println!("  Success probability: {:.3}", success_prob);
    println!("  Number of steps: {}", steps);

    // Compare with classical random walk
    let classical_expected_steps = 16.0 / 2.0; // Expected steps for classical search
    let speedup = classical_expected_steps / (steps as f64);

    println!("\nComparison with classical search:");
    println!(
        "  Classical expected steps: {:.1}",
        classical_expected_steps
    );
    println!("  Quantum speedup: {:.2}x", speedup);

    // Show probability distribution at marked vertices
    println!("\nFinal probability distribution at marked vertices:");
    let probs = search.vertex_probabilities();
    for &marked in &[7, 11] {
        println!("  Vertex {}: {:.3}", marked, probs[marked]);
    }
    println!();
}

// Additional example: Custom coin operator
fn custom_coin_example() {
    println!("4. Custom Coin Operator Example");
    println!("-------------------------------");

    // Create a line graph
    let graph = Graph::new(GraphType::Line, 10);

    // Create custom coin operator (biased coin)
    let theta = PI / 6.0; // 30 degree rotation
    let coin_matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(theta.cos(), 0.0),
            Complex64::new(theta.sin(), 0.0),
            Complex64::new(theta.sin(), 0.0),
            Complex64::new(-theta.cos(), 0.0),
        ],
    )
    .unwrap();

    let custom_coin = CoinOperator::Custom(coin_matrix);

    // Create walk with custom coin
    let mut walk = DiscreteQuantumWalk::new(graph, custom_coin);
    walk.initialize_position(5); // Start in the middle

    println!("Line graph with 10 vertices");
    println!("Starting position: 5");
    println!("Using biased coin (θ = π/6)");

    // Evolve and show bias
    for _ in 0..20 {
        walk.step();
    }

    let probs = walk.position_probabilities();
    let left_prob: f64 = probs[0..5].iter().sum();
    let right_prob: f64 = probs[5..10].iter().sum();

    println!("\nAfter 20 steps:");
    println!("  Probability on left side (0-4): {:.3}", left_prob);
    println!("  Probability on right side (5-9): {:.3}", right_prob);
    println!(
        "  Bias direction: {}",
        if right_prob > left_prob {
            "right"
        } else {
            "left"
        }
    );
}
