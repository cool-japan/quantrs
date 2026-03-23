//! Quantum walk search algorithm.

use super::discrete::DiscreteQuantumWalk;
use super::graph::{CoinOperator, Graph, SearchOracle};
use scirs2_core::Complex64;

/// Search algorithm using quantum walks
pub struct QuantumWalkSearch {
    #[allow(dead_code)]
    graph: Graph,
    oracle: SearchOracle,
    walk: DiscreteQuantumWalk,
}

impl QuantumWalkSearch {
    /// Create a new quantum walk search
    pub fn new(graph: Graph, oracle: SearchOracle) -> Self {
        let walk = DiscreteQuantumWalk::new(graph.clone(), CoinOperator::Grover);
        Self {
            graph,
            oracle,
            walk,
        }
    }

    /// Apply the oracle that marks vertices
    fn apply_oracle(&mut self) {
        for &vertex in &self.oracle.marked {
            for coin in 0..self.walk.coin_dimension {
                let idx = self.walk.state_index(vertex, coin);
                if idx < self.walk.state.len() {
                    self.walk.state[idx] = -self.walk.state[idx]; // Phase flip
                }
            }
        }
    }

    /// Run the search algorithm
    pub fn run(&mut self, max_steps: usize) -> (usize, f64, usize) {
        // Start in uniform superposition
        let amplitude = Complex64::new(1.0 / (self.walk.hilbert_dim as f64).sqrt(), 0.0);
        self.walk.state.fill(amplitude);

        let mut best_vertex = 0;
        let mut best_prob = 0.0;
        let mut best_step = 0;

        // Alternate between walk and oracle
        for step in 1..=max_steps {
            self.walk.step();
            self.apply_oracle();

            // Check probabilities at marked vertices
            let probs = self.walk.position_probabilities();
            for &marked in &self.oracle.marked {
                if probs[marked] > best_prob {
                    best_prob = probs[marked];
                    best_vertex = marked;
                    best_step = step;
                }
            }

            // Early stopping if we have high probability
            if best_prob > 0.5 {
                break;
            }
        }

        (best_vertex, best_prob, best_step)
    }

    /// Get vertex probabilities
    pub fn vertex_probabilities(&self) -> Vec<f64> {
        self.walk.position_probabilities()
    }
}
