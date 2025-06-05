//! Quantum Walk Algorithms
//!
//! This module implements various quantum walk algorithms, including:
//! - Discrete-time quantum walks on graphs
//! - Continuous-time quantum walks
//! - Szegedy quantum walks
//!
//! Quantum walks are the quantum analog of classical random walks and form
//! the basis for many quantum algorithms.

use crate::complex_ext::QuantumComplexExt;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Types of graphs for quantum walks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphType {
    /// Line graph (path graph)
    Line,
    /// Cycle graph
    Cycle,
    /// Complete graph
    Complete,
    /// Hypercube graph
    Hypercube,
    /// Grid graph (2D lattice)
    Grid2D,
    /// Custom graph
    Custom,
}

/// Coin operators for discrete quantum walks
#[derive(Debug, Clone)]
pub enum CoinOperator {
    /// Hadamard coin
    Hadamard,
    /// Grover coin
    Grover,
    /// DFT (Discrete Fourier Transform) coin
    DFT,
    /// Custom coin operator
    Custom(Array2<Complex64>),
}

/// Search oracle for quantum walk search
#[derive(Debug, Clone)]
pub struct SearchOracle {
    /// Marked vertices
    pub marked: Vec<usize>,
}

impl SearchOracle {
    /// Create a new search oracle with marked vertices
    pub fn new(marked: Vec<usize>) -> Self {
        Self { marked }
    }

    /// Check if a vertex is marked
    pub fn is_marked(&self, vertex: usize) -> bool {
        self.marked.contains(&vertex)
    }
}

/// Graph representation for quantum walks
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices
    pub num_vertices: usize,
    /// Adjacency list representation
    pub edges: Vec<Vec<usize>>,
    /// Optional edge weights
    pub weights: Option<Vec<Vec<f64>>>,
}

impl Graph {
    /// Create a new graph of a specific type
    pub fn new(graph_type: GraphType, size: usize) -> Self {
        let mut graph = Self {
            num_vertices: match graph_type {
                GraphType::Hypercube => 1 << size, // 2^size vertices
                GraphType::Grid2D => size * size,  // size x size grid
                _ => size,
            },
            edges: vec![],
            weights: None,
        };

        // Initialize edges based on graph type
        graph.edges = vec![Vec::new(); graph.num_vertices];

        match graph_type {
            GraphType::Line => {
                for i in 0..size.saturating_sub(1) {
                    graph.add_edge(i, i + 1);
                }
            }
            GraphType::Cycle => {
                for i in 0..size {
                    graph.add_edge(i, (i + 1) % size);
                }
            }
            GraphType::Complete => {
                for i in 0..size {
                    for j in i + 1..size {
                        graph.add_edge(i, j);
                    }
                }
            }
            GraphType::Hypercube => {
                let n = size; // dimension
                for i in 0..(1 << n) {
                    for j in 0..n {
                        let neighbor = i ^ (1 << j);
                        if neighbor > i {
                            graph.add_edge(i, neighbor);
                        }
                    }
                }
            }
            GraphType::Grid2D => {
                for i in 0..size {
                    for j in 0..size {
                        let idx = i * size + j;
                        // Right neighbor
                        if j < size - 1 {
                            graph.add_edge(idx, idx + 1);
                        }
                        // Bottom neighbor
                        if i < size - 1 {
                            graph.add_edge(idx, idx + size);
                        }
                    }
                }
            }
            GraphType::Custom => {
                // Empty graph, user will add edges manually
            }
        }

        graph
    }

    /// Create an empty graph with given number of vertices
    pub fn new_empty(num_vertices: usize) -> Self {
        Self {
            num_vertices,
            edges: vec![Vec::new(); num_vertices],
            weights: None,
        }
    }

    /// Add an undirected edge
    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u < self.num_vertices && v < self.num_vertices && u != v && !self.edges[u].contains(&v) {
            self.edges[u].push(v);
            self.edges[v].push(u);
        }
    }

    /// Add a weighted edge
    pub fn add_weighted_edge(&mut self, u: usize, v: usize, weight: f64) {
        if self.weights.is_none() {
            self.weights = Some(vec![vec![0.0; self.num_vertices]; self.num_vertices]);
        }

        self.add_edge(u, v);

        if let Some(ref mut weights) = self.weights {
            weights[u][v] = weight;
            weights[v][u] = weight;
        }
    }

    /// Get the degree of a vertex
    pub fn degree(&self, vertex: usize) -> usize {
        if vertex < self.num_vertices {
            self.edges[vertex].len()
        } else {
            0
        }
    }

    /// Get the adjacency matrix
    pub fn adjacency_matrix(&self) -> Array2<f64> {
        let mut matrix = Array2::zeros((self.num_vertices, self.num_vertices));

        for (u, neighbors) in self.edges.iter().enumerate() {
            for &v in neighbors {
                if let Some(ref weights) = self.weights {
                    matrix[[u, v]] = weights[u][v];
                } else {
                    matrix[[u, v]] = 1.0;
                }
            }
        }

        matrix
    }
}

/// Discrete-time quantum walk
pub struct DiscreteQuantumWalk {
    graph: Graph,
    coin_operator: CoinOperator,
    coin_dimension: usize,
    /// Total Hilbert space dimension: coin_dimension * num_vertices
    hilbert_dim: usize,
    /// Current state vector
    state: Vec<Complex64>,
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
    fn state_index(&self, vertex: usize, coin: usize) -> usize {
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

/// Continuous-time quantum walk
pub struct ContinuousQuantumWalk {
    graph: Graph,
    hamiltonian: Array2<Complex64>,
    state: Vec<Complex64>,
}

impl ContinuousQuantumWalk {
    /// Create a new continuous quantum walk
    pub fn new(graph: Graph) -> Self {
        let adj_matrix = graph.adjacency_matrix();
        let hamiltonian = adj_matrix.mapv(|x| Complex64::new(x, 0.0));
        let num_vertices = graph.num_vertices;

        Self {
            graph,
            hamiltonian,
            state: vec![Complex64::new(0.0, 0.0); num_vertices],
        }
    }

    /// Initialize walker at a specific vertex
    pub fn initialize_vertex(&mut self, vertex: usize) {
        self.state = vec![Complex64::new(0.0, 0.0); self.graph.num_vertices];
        if vertex < self.graph.num_vertices {
            self.state[vertex] = Complex64::new(1.0, 0.0);
        }
    }

    /// Evolve the quantum walk for time t
    pub fn evolve(&mut self, time: f64) {
        // This is a simplified version using first-order approximation
        // For a full implementation, we would diagonalize the Hamiltonian

        let dt = 0.01; // Time step
        let steps = (time / dt) as usize;

        for _ in 0..steps {
            let mut new_state = self.state.clone();

            // Apply exp(-iHt) ≈ I - iHdt for small dt
            for (i, ns) in new_state
                .iter_mut()
                .enumerate()
                .take(self.graph.num_vertices)
            {
                let mut sum = Complex64::new(0.0, 0.0);
                for j in 0..self.graph.num_vertices {
                    sum += self.hamiltonian[[i, j]] * self.state[j];
                }
                *ns = self.state[i] - Complex64::new(0.0, dt) * sum;
            }

            // Normalize
            let norm: f64 = new_state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

            if norm > 0.0 {
                for amp in new_state.iter_mut() {
                    *amp /= norm;
                }
            }

            self.state = new_state;
        }
    }

    /// Get vertex probabilities
    pub fn vertex_probabilities(&self) -> Vec<f64> {
        self.state.iter().map(|c| c.probability()).collect()
    }

    /// Calculate transport probability between two vertices at time t
    pub fn transport_probability(&mut self, from: usize, to: usize, time: f64) -> f64 {
        // Initialize at 'from' vertex
        self.initialize_vertex(from);

        // Evolve for time t
        self.evolve(time);

        // Return probability at 'to' vertex
        if to < self.state.len() {
            self.state[to].probability()
        } else {
            0.0
        }
    }

    /// Get the probability distribution
    pub fn get_probabilities(&self, state: &[Complex64]) -> Vec<f64> {
        state.iter().map(|c| c.probability()).collect()
    }
}

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

        println!("\nAfter {} steps:", steps);
        print!("Probabilities: ");
        for (v, p) in probs.iter().enumerate() {
            if *p > 0.01 {
                print!("v{}: {:.3} ", v, p);
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

    println!("Marked vertices: {:?}", marked);

    // Run search
    let (found, prob, steps) = search.run(50);

    println!(
        "\nFound vertex {} with probability {:.3} after {} steps",
        found, prob, steps
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = Graph::new(GraphType::Cycle, 4);
        assert_eq!(graph.num_vertices, 4);
        assert_eq!(graph.degree(0), 2);

        let complete = Graph::new(GraphType::Complete, 5);
        assert_eq!(complete.degree(0), 4);
    }

    #[test]
    fn test_discrete_walk_initialization() {
        let graph = Graph::new(GraphType::Line, 5);
        let mut walk = DiscreteQuantumWalk::new(graph, CoinOperator::Hadamard);

        walk.initialize_position(2);
        let probs = walk.position_probabilities();

        assert!((probs[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_walk() {
        let graph = Graph::new(GraphType::Cycle, 4);
        let mut walk = ContinuousQuantumWalk::new(graph);

        walk.initialize_vertex(0);
        walk.evolve(1.0);

        let probs = walk.vertex_probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
}
