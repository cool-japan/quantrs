//! Quantum Walk Algorithms
//!
//! This module implements various quantum walk algorithms, including:
//! - Discrete-time quantum walks on graphs
//! - Continuous-time quantum walks
//! - Szegedy quantum walks
//!
//! Quantum walks are the quantum analog of classical random walks and form
//! the basis for many quantum algorithms.

mod continuous;
mod discrete;
mod eigensolvers;
mod graph;
mod multi;
mod search;

#[cfg(test)]
mod tests;

pub use continuous::{ContinuousQuantumWalk, SzegedyQuantumWalk};
pub use discrete::DiscreteQuantumWalk;
pub(crate) use eigensolvers::{compute_laplacian_eigenvalues_impl, estimate_fiedler_value_impl};
pub use graph::{CoinOperator, Graph, GraphType, SearchOracle};
pub use multi::{DecoherentQuantumWalk, MultiWalkerQuantumWalk};
pub use search::QuantumWalkSearch;

/// Example: Quantum walk on a line
pub fn quantum_walk_line_example() {
    println!("Quantum Walk on a Line (10 vertices)");

    let graph = Graph::new(GraphType::Line, 10);
    let walk = DiscreteQuantumWalk::new(graph, CoinOperator::Hadamard);

    // Start at vertex 5 (middle)
    let mut walk = walk;
    walk.initialize_position(5);

    // Evolve for different time steps
    for steps in [0, 5, 10, 20, 30] {
        // Reset and evolve
        walk.initialize_position(5);
        for _ in 0..steps {
            walk.step();
        }
        let probs = walk.position_probabilities();

        println!("\nAfter {steps} steps:");
        print!("Probabilities: ");
        for (v, p) in probs.iter().enumerate() {
            if *p > 0.01 {
                print!("v{v}: {p:.3} ");
            }
        }
        println!();
    }
}

/// Example: Search on a complete graph
pub fn quantum_walk_search_example() {
    println!("\nQuantum Walk Search on Complete Graph (8 vertices)");

    let graph = Graph::new(GraphType::Complete, 8);
    let marked = vec![3, 5]; // Mark vertices 3 and 5
    let oracle = SearchOracle::new(marked.clone());

    let mut search = QuantumWalkSearch::new(graph, oracle);

    println!("Marked vertices: {marked:?}");

    // Run search
    let (found, prob, steps) = search.run(50);

    println!("\nFound vertex {found} with probability {prob:.3} after {steps} steps");
}
