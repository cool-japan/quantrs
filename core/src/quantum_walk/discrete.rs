//! Discrete-time quantum walk implementation.

use super::graph::{CoinOperator, Graph};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

/// Discrete-time quantum walk
pub struct DiscreteQuantumWalk {
    pub(crate) graph: Graph,
    coin_operator: CoinOperator,
    pub(crate) coin_dimension: usize,
    /// Total Hilbert space dimension: coin_dimension * num_vertices
    pub(crate) hilbert_dim: usize,
    /// Current state vector
    pub(crate) state: Vec<Complex64>,
}

impl DiscreteQuantumWalk {
    /// Create a new discrete quantum walk with specified coin operator
    pub fn new(graph: Graph, coin_operator: CoinOperator) -> Self {
        // Coin dimension is the maximum degree for standard walks
        // For hypercube, it's the dimension
        let coin_dimension = match graph.num_vertices {
            n if n > 0 => {
                (0..graph.num_vertices)
                    .map(|v| graph.degree(v))
                    .max()
                    .unwrap_or(2)
                    .max(2) // At least 2-dimensional coin
            }
            _ => 2,
        };

        let hilbert_dim = coin_dimension * graph.num_vertices;

        Self {
            graph,
            coin_operator,
            coin_dimension,
            hilbert_dim,
            state: vec![Complex64::new(0.0, 0.0); hilbert_dim],
        }
    }

    /// Initialize walker at a specific position
    pub fn initialize_position(&mut self, position: usize) {
        self.state = vec![Complex64::new(0.0, 0.0); self.hilbert_dim];

        // Equal superposition over all coin states at the position
        let degree = self.graph.degree(position) as f64;
        if degree > 0.0 {
            let amplitude = Complex64::new(1.0 / degree.sqrt(), 0.0);

            for coin in 0..self.coin_dimension.min(self.graph.degree(position)) {
                let index = self.state_index(position, coin);
                if index < self.state.len() {
                    self.state[index] = amplitude;
                }
            }
        }
    }

    /// Perform one step of the quantum walk
    pub fn step(&mut self) {
        // Apply coin operator
        self.apply_coin();

        // Apply shift operator
        self.apply_shift();
    }

    /// Get position probabilities
    pub fn position_probabilities(&self) -> Vec<f64> {
        let mut probs = vec![0.0; self.graph.num_vertices];

        for (vertex, prob) in probs.iter_mut().enumerate() {
            for coin in 0..self.coin_dimension {
                let idx = self.state_index(vertex, coin);
                if idx < self.state.len() {
                    *prob += self.state[idx].norm_sqr();
                }
            }
        }

        probs
    }

    /// Get the index in the state vector for (vertex, coin) pair
    pub(crate) const fn state_index(&self, vertex: usize, coin: usize) -> usize {
        vertex * self.coin_dimension + coin
    }

    /// Apply the coin operator
    fn apply_coin(&mut self) {
        match &self.coin_operator {
            CoinOperator::Hadamard => self.apply_hadamard_coin(),
            CoinOperator::Grover => self.apply_grover_coin(),
            CoinOperator::DFT => self.apply_dft_coin(),
            CoinOperator::Custom(matrix) => self.apply_custom_coin(matrix.clone()),
        }
    }

    /// Apply Hadamard coin
    fn apply_hadamard_coin(&mut self) {
        let h = 1.0 / std::f64::consts::SQRT_2;

        for vertex in 0..self.graph.num_vertices {
            if self.coin_dimension == 2 {
                let idx0 = self.state_index(vertex, 0);
                let idx1 = self.state_index(vertex, 1);

                if idx1 < self.state.len() {
                    let a0 = self.state[idx0];
                    let a1 = self.state[idx1];

                    self.state[idx0] = h * (a0 + a1);
                    self.state[idx1] = h * (a0 - a1);
                }
            }
        }
    }

    /// Apply Grover coin
    fn apply_grover_coin(&mut self) {
        // Grover coin: 2|s><s| - I, where |s> is uniform superposition
        for vertex in 0..self.graph.num_vertices {
            let degree = self.graph.degree(vertex);
            if degree <= 1 {
                continue; // No coin needed for degree 0 or 1
            }

            // Calculate sum of amplitudes for this vertex
            let mut sum = Complex64::new(0.0, 0.0);
            for coin in 0..degree.min(self.coin_dimension) {
                let idx = self.state_index(vertex, coin);
                if idx < self.state.len() {
                    sum += self.state[idx];
                }
            }

            // Apply Grover coin
            let factor = Complex64::new(2.0 / degree as f64, 0.0);
            for coin in 0..degree.min(self.coin_dimension) {
                let idx = self.state_index(vertex, coin);
                if idx < self.state.len() {
                    let old_amp = self.state[idx];
                    self.state[idx] = factor * sum - old_amp;
                }
            }
        }
    }

    /// Apply DFT coin
    fn apply_dft_coin(&mut self) {
        // DFT coin for 2-dimensional coin space
        if self.coin_dimension == 2 {
            self.apply_hadamard_coin(); // DFT is same as Hadamard for 2D
        }
        // For higher dimensions, would implement full DFT
    }

    /// Apply custom coin operator
    fn apply_custom_coin(&mut self, matrix: Array2<Complex64>) {
        if matrix.shape() != [self.coin_dimension, self.coin_dimension] {
            return; // Matrix size mismatch
        }

        for vertex in 0..self.graph.num_vertices {
            let mut coin_state = vec![Complex64::new(0.0, 0.0); self.coin_dimension];

            // Extract coin state for this vertex
            for (coin, cs) in coin_state.iter_mut().enumerate() {
                let idx = self.state_index(vertex, coin);
                if idx < self.state.len() {
                    *cs = self.state[idx];
                }
            }

            // Apply coin operator
            let new_coin_state = matrix.dot(&Array1::from(coin_state));

            // Write back
            for coin in 0..self.coin_dimension {
                let idx = self.state_index(vertex, coin);
                if idx < self.state.len() {
                    self.state[idx] = new_coin_state[coin];
                }
            }
        }
    }

    /// Apply the shift operator
    fn apply_shift(&mut self) {
        let mut new_state = vec![Complex64::new(0.0, 0.0); self.hilbert_dim];

        for vertex in 0..self.graph.num_vertices {
            for (coin, &neighbor) in self.graph.edges[vertex].iter().enumerate() {
                if coin < self.coin_dimension {
                    let from_idx = self.state_index(vertex, coin);

                    // Find which coin state corresponds to coming from 'vertex' at 'neighbor'
                    let to_coin = self.graph.edges[neighbor]
                        .iter()
                        .position(|&v| v == vertex)
                        .unwrap_or(0);

                    if to_coin < self.coin_dimension && from_idx < self.state.len() {
                        let to_idx = self.state_index(neighbor, to_coin);
                        if to_idx < new_state.len() {
                            new_state[to_idx] = self.state[from_idx];
                        }
                    }
                }
            }
        }

        self.state.copy_from_slice(&new_state);
    }
}
