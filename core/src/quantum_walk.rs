//! Quantum Walk Algorithms
//!
//! This module implements various quantum walk algorithms, including:
//! - Discrete-time quantum walks on graphs
//! - Continuous-time quantum walks
//! - Szegedy quantum walks
//!
//! Quantum walks are the quantum analog of classical random walks and form
//! the basis for many quantum algorithms.

use crate::{
    complex_ext::QuantumComplexExt,
    error::{QuantRS2Error, QuantRS2Result},
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::{HashMap, VecDeque};

/// Types of graphs for quantum walks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub const fn new(marked: Vec<usize>) -> Self {
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

    /// Get the Laplacian matrix
    pub fn laplacian_matrix(&self) -> Array2<f64> {
        let mut laplacian = Array2::zeros((self.num_vertices, self.num_vertices));

        for v in 0..self.num_vertices {
            let degree = self.degree(v) as f64;
            laplacian[[v, v]] = degree;

            for &neighbor in &self.edges[v] {
                if let Some(ref weights) = self.weights {
                    laplacian[[v, neighbor]] = -weights[v][neighbor];
                } else {
                    laplacian[[v, neighbor]] = -1.0;
                }
            }
        }

        laplacian
    }

    /// Get the normalized Laplacian matrix
    pub fn normalized_laplacian_matrix(&self) -> Array2<f64> {
        let mut norm_laplacian = Array2::zeros((self.num_vertices, self.num_vertices));

        for v in 0..self.num_vertices {
            let degree_v = self.degree(v) as f64;
            if degree_v == 0.0 {
                continue;
            }

            norm_laplacian[[v, v]] = 1.0;

            for &neighbor in &self.edges[v] {
                let degree_n = self.degree(neighbor) as f64;
                if degree_n == 0.0 {
                    continue;
                }

                let weight = if let Some(ref weights) = self.weights {
                    weights[v][neighbor]
                } else {
                    1.0
                };

                norm_laplacian[[v, neighbor]] = -weight / (degree_v * degree_n).sqrt();
            }
        }

        norm_laplacian
    }

    /// Get the transition matrix for random walks
    pub fn transition_matrix(&self) -> Array2<f64> {
        let mut transition = Array2::zeros((self.num_vertices, self.num_vertices));

        for v in 0..self.num_vertices {
            let degree = self.degree(v) as f64;
            if degree == 0.0 {
                continue;
            }

            for &neighbor in &self.edges[v] {
                let weight = if let Some(ref weights) = self.weights {
                    weights[v][neighbor]
                } else {
                    1.0
                };

                transition[[v, neighbor]] = weight / degree;
            }
        }

        transition
    }

    /// Check if the graph is bipartite
    pub fn is_bipartite(&self) -> bool {
        let mut colors = vec![-1; self.num_vertices];

        for start in 0..self.num_vertices {
            if colors[start] != -1 {
                continue;
            }

            let mut queue = VecDeque::new();
            queue.push_back(start);
            colors[start] = 0;

            while let Some(vertex) = queue.pop_front() {
                for &neighbor in &self.edges[vertex] {
                    if colors[neighbor] == -1 {
                        colors[neighbor] = 1 - colors[vertex];
                        queue.push_back(neighbor);
                    } else if colors[neighbor] == colors[vertex] {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Calculate the algebraic connectivity (second smallest eigenvalue of Laplacian)
    pub fn algebraic_connectivity(&self) -> f64 {
        let laplacian = self.laplacian_matrix();

        // For small graphs, we can compute eigenvalues directly
        // In practice, you'd use more sophisticated numerical methods
        if self.num_vertices <= 10 {
            self.compute_laplacian_eigenvalues(&laplacian)
                .get(1)
                .copied()
                .unwrap_or(0.0)
        } else {
            // Approximate using power iteration for larger graphs
            self.estimate_fiedler_value(&laplacian)
        }
    }

    /// Compute eigenvalues of Laplacian (simplified implementation)
    fn compute_laplacian_eigenvalues(&self, _laplacian: &Array2<f64>) -> Vec<f64> {
        // This is a placeholder - in practice you'd use a proper eigenvalue solver
        // For now, return approximate values based on graph structure
        let mut eigenvalues = vec![0.0]; // Always has 0 eigenvalue

        // Rough approximation based on degree sequence
        for v in 0..self.num_vertices {
            let degree = self.degree(v) as f64;
            if degree > 0.0 {
                eigenvalues.push(degree);
            }
        }

        eigenvalues.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("Failed to compare eigenvalues in Graph::compute_laplacian_eigenvalues")
        });
        eigenvalues.truncate(self.num_vertices);
        eigenvalues
    }

    /// Estimate Fiedler value using power iteration
    fn estimate_fiedler_value(&self, _laplacian: &Array2<f64>) -> f64 {
        // Simplified estimation - in practice use inverse power iteration
        let max_degree = (0..self.num_vertices)
            .map(|v| self.degree(v))
            .max()
            .unwrap_or(0);
        2.0 / max_degree as f64 // Rough approximation
    }

    /// Get shortest path distances between all pairs of vertices
    pub fn all_pairs_shortest_paths(&self) -> Array2<f64> {
        let mut distances =
            Array2::from_elem((self.num_vertices, self.num_vertices), f64::INFINITY);

        // Initialize distances
        for v in 0..self.num_vertices {
            distances[[v, v]] = 0.0;
            for &neighbor in &self.edges[v] {
                let weight = if let Some(ref weights) = self.weights {
                    weights[v][neighbor]
                } else {
                    1.0
                };
                distances[[v, neighbor]] = weight;
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..self.num_vertices {
            for i in 0..self.num_vertices {
                for j in 0..self.num_vertices {
                    let via_k = distances[[i, k]] + distances[[k, j]];
                    if via_k < distances[[i, j]] {
                        distances[[i, j]] = via_k;
                    }
                }
            }
        }

        distances
    }

    /// Create a graph from an adjacency matrix
    pub fn from_adjacency_matrix(matrix: &Array2<f64>) -> QuantRS2Result<Self> {
        let (rows, cols) = matrix.dim();
        if rows != cols {
            return Err(QuantRS2Error::InvalidInput(
                "Adjacency matrix must be square".to_string(),
            ));
        }

        let mut graph = Self::new_empty(rows);
        let mut has_weights = false;

        for i in 0..rows {
            for j in i + 1..cols {
                let weight = matrix[[i, j]];
                if weight != 0.0 {
                    if weight != 1.0 {
                        has_weights = true;
                    }
                    if has_weights {
                        graph.add_weighted_edge(i, j, weight);
                    } else {
                        graph.add_edge(i, j);
                    }
                }
            }
        }

        Ok(graph)
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
    const fn state_index(&self, vertex: usize, coin: usize) -> usize {
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
                for amp in &mut new_state {
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

/// Szegedy quantum walk for arbitrary graphs
/// This provides better mixing properties on irregular graphs
pub struct SzegedyQuantumWalk {
    graph: Graph,
    /// State lives on edges: |u,v> where edge (u,v) exists
    state: HashMap<(usize, usize), Complex64>,
    num_edges: usize,
}

impl SzegedyQuantumWalk {
    /// Create a new Szegedy quantum walk
    pub fn new(graph: Graph) -> Self {
        let mut num_edges = 0;
        for v in 0..graph.num_vertices {
            num_edges += graph.edges[v].len();
        }

        Self {
            graph,
            state: HashMap::new(),
            num_edges,
        }
    }

    /// Initialize in uniform superposition over all edges
    pub fn initialize_uniform(&mut self) {
        self.state.clear();

        if self.num_edges == 0 {
            return;
        }

        let amplitude = Complex64::new(1.0 / (self.num_edges as f64).sqrt(), 0.0);

        for u in 0..self.graph.num_vertices {
            for &v in &self.graph.edges[u] {
                self.state.insert((u, v), amplitude);
            }
        }
    }

    /// Initialize at a specific edge
    pub fn initialize_edge(&mut self, u: usize, v: usize) {
        self.state.clear();

        if u < self.graph.num_vertices && self.graph.edges[u].contains(&v) {
            self.state.insert((u, v), Complex64::new(1.0, 0.0));
        }
    }

    /// Perform one step of Szegedy walk
    pub fn step(&mut self) {
        // Szegedy walk: (2P - I)(2Q - I) where P and Q are projections

        // Apply reflection around vertex-uniform states
        self.reflect_vertex_uniform();

        // Apply reflection around edge-uniform states
        self.reflect_edge_uniform();
    }

    /// Reflect around vertex-uniform subspaces
    fn reflect_vertex_uniform(&mut self) {
        let mut vertex_sums: Vec<Complex64> =
            vec![Complex64::new(0.0, 0.0); self.graph.num_vertices];

        // Calculate sum of amplitudes for each vertex
        for (&(u, _), &amplitude) in &self.state {
            vertex_sums[u] += amplitude;
        }

        // Apply reflection: 2|psi_u><psi_u| - I
        let mut new_state = HashMap::new();

        for (&(u, v), &old_amp) in &self.state {
            let degree = self.graph.degree(u) as f64;
            if degree > 0.0 {
                let vertex_avg = vertex_sums[u] / degree;
                let new_amp = 2.0 * vertex_avg - old_amp;
                new_state.insert((u, v), new_amp);
            }
        }

        self.state = new_state;
    }

    /// Reflect around edge-uniform subspace
    fn reflect_edge_uniform(&mut self) {
        if self.num_edges == 0 {
            return;
        }

        // Calculate total amplitude
        let total_amp: Complex64 = self.state.values().sum();
        let uniform_amp = total_amp / self.num_edges as f64;

        // Apply reflection: 2|uniform><uniform| - I
        for amplitude in self.state.values_mut() {
            *amplitude = 2.0 * uniform_amp - *amplitude;
        }
    }

    /// Get vertex probabilities by summing over outgoing edges
    pub fn vertex_probabilities(&self) -> Vec<f64> {
        let mut probs = vec![0.0; self.graph.num_vertices];

        for (&(u, _), &amplitude) in &self.state {
            probs[u] += amplitude.norm_sqr();
        }

        probs
    }

    /// Get edge probabilities
    pub fn edge_probabilities(&self) -> Vec<((usize, usize), f64)> {
        self.state
            .iter()
            .map(|(&edge, &amplitude)| (edge, amplitude.norm_sqr()))
            .collect()
    }

    /// Calculate mixing time to epsilon-close to uniform distribution
    pub fn estimate_mixing_time(&mut self, epsilon: f64) -> usize {
        let uniform_prob = 1.0 / self.graph.num_vertices as f64;

        // Reset to uniform
        self.initialize_uniform();

        for steps in 1..1000 {
            self.step();

            let probs = self.vertex_probabilities();
            let max_deviation = probs
                .iter()
                .map(|&p| (p - uniform_prob).abs())
                .fold(0.0, f64::max);

            if max_deviation < epsilon {
                return steps;
            }
        }

        1000 // Return max if not converged
    }
}

/// Multi-walker quantum walk for studying entanglement and correlations
pub struct MultiWalkerQuantumWalk {
    graph: Graph,
    num_walkers: usize,
    /// State tensor product space: walker1_pos ⊗ walker2_pos ⊗ ...
    state: Array1<Complex64>,
    /// Dimension of single walker space
    single_walker_dim: usize,
}

impl MultiWalkerQuantumWalk {
    /// Create a new multi-walker quantum walk
    pub fn new(graph: Graph, num_walkers: usize) -> Self {
        let single_walker_dim = graph.num_vertices;
        let total_dim = single_walker_dim.pow(num_walkers as u32);

        Self {
            graph,
            num_walkers,
            state: Array1::zeros(total_dim),
            single_walker_dim,
        }
    }

    /// Initialize walkers at specific positions
    pub fn initialize_positions(&mut self, positions: &[usize]) -> QuantRS2Result<()> {
        if positions.len() != self.num_walkers {
            return Err(QuantRS2Error::InvalidInput(
                "Number of positions must match number of walkers".to_string(),
            ));
        }

        for &pos in positions {
            if pos >= self.single_walker_dim {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Position {pos} out of bounds"
                )));
            }
        }

        // Reset state
        self.state.fill(Complex64::new(0.0, 0.0));

        // Set amplitude for initial configuration
        let index = self.positions_to_index(positions);
        self.state[index] = Complex64::new(1.0, 0.0);

        Ok(())
    }

    /// Initialize in entangled superposition
    pub fn initialize_entangled_bell_state(
        &mut self,
        pos1: usize,
        pos2: usize,
    ) -> QuantRS2Result<()> {
        if self.num_walkers != 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Bell state initialization only works for 2 walkers".to_string(),
            ));
        }

        self.state.fill(Complex64::new(0.0, 0.0));

        let amplitude = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

        // |pos1,pos2> + |pos2,pos1>
        let idx1 = self.positions_to_index(&[pos1, pos2]);
        let idx2 = self.positions_to_index(&[pos2, pos1]);

        self.state[idx1] = amplitude;
        self.state[idx2] = amplitude;

        Ok(())
    }

    /// Convert walker positions to state vector index
    fn positions_to_index(&self, positions: &[usize]) -> usize {
        let mut index = 0;
        let mut multiplier = 1;

        for &pos in positions.iter().rev() {
            index += pos * multiplier;
            multiplier *= self.single_walker_dim;
        }

        index
    }

    /// Convert state vector index to walker positions
    fn index_to_positions(&self, mut index: usize) -> Vec<usize> {
        let mut positions = Vec::with_capacity(self.num_walkers);

        for _ in 0..self.num_walkers {
            positions.push(index % self.single_walker_dim);
            index /= self.single_walker_dim;
        }

        positions.reverse();
        positions
    }

    /// Perform one step (simplified version - each walker evolves independently)
    pub fn step_independent(&mut self) {
        let mut new_state = Array1::zeros(self.state.len());

        for (index, &amplitude) in self.state.iter().enumerate() {
            if amplitude.norm_sqr() < 1e-15 {
                continue;
            }

            let positions = self.index_to_positions(index);

            // Each walker moves to neighboring vertices with equal probability
            let mut total_neighbors = 1;
            for &pos in &positions {
                total_neighbors *= self.graph.degree(pos).max(1);
            }

            let neighbor_amplitude = amplitude / (total_neighbors as f64).sqrt();

            // Generate all possible neighbor configurations
            self.add_neighbor_amplitudes(
                &positions,
                0,
                &mut Vec::new(),
                neighbor_amplitude,
                &mut new_state,
            );
        }

        self.state = new_state;
    }

    /// Recursively add amplitudes for all neighbor configurations
    fn add_neighbor_amplitudes(
        &self,
        original_positions: &[usize],
        walker_idx: usize,
        current_positions: &mut Vec<usize>,
        amplitude: Complex64,
        new_state: &mut Array1<Complex64>,
    ) {
        if walker_idx >= self.num_walkers {
            let index = self.positions_to_index(current_positions);
            new_state[index] += amplitude;
            return;
        }

        let pos = original_positions[walker_idx];
        let neighbors = &self.graph.edges[pos];

        if neighbors.is_empty() {
            // Stay at same position if no neighbors
            current_positions.push(pos);
            self.add_neighbor_amplitudes(
                original_positions,
                walker_idx + 1,
                current_positions,
                amplitude,
                new_state,
            );
            current_positions.pop();
        } else {
            for &neighbor in neighbors {
                current_positions.push(neighbor);
                self.add_neighbor_amplitudes(
                    original_positions,
                    walker_idx + 1,
                    current_positions,
                    amplitude,
                    new_state,
                );
                current_positions.pop();
            }
        }
    }

    /// Get marginal probability distribution for a specific walker
    pub fn marginal_probabilities(&self, walker_idx: usize) -> Vec<f64> {
        let mut probs = vec![0.0; self.single_walker_dim];

        for (index, &amplitude) in self.state.iter().enumerate() {
            let positions = self.index_to_positions(index);
            probs[positions[walker_idx]] += amplitude.norm_sqr();
        }

        probs
    }

    /// Calculate entanglement entropy between walkers
    pub fn entanglement_entropy(&self) -> f64 {
        if self.num_walkers != 2 {
            return 0.0; // Only implemented for 2 walkers
        }

        // Compute reduced density matrix for walker 1
        let mut reduced_dm = Array2::zeros((self.single_walker_dim, self.single_walker_dim));

        for i in 0..self.single_walker_dim {
            for j in 0..self.single_walker_dim {
                for k in 0..self.single_walker_dim {
                    let idx1 = self.positions_to_index(&[i, k]);
                    let idx2 = self.positions_to_index(&[j, k]);

                    reduced_dm[[i, j]] += self.state[idx1].conj() * self.state[idx2];
                }
            }
        }

        // Calculate von Neumann entropy (simplified - would use eigenvalues in practice)
        let trace = reduced_dm.diag().mapv(|x: Complex64| x.re).sum();
        -trace * trace.ln() // Simplified approximation
    }
}

/// Quantum walk with environmental decoherence
pub struct DecoherentQuantumWalk {
    base_walk: DiscreteQuantumWalk,
    decoherence_rate: f64,
    measurement_probability: f64,
}

impl DecoherentQuantumWalk {
    /// Create a new decoherent quantum walk
    pub fn new(graph: Graph, coin_operator: CoinOperator, decoherence_rate: f64) -> Self {
        Self {
            base_walk: DiscreteQuantumWalk::new(graph, coin_operator),
            decoherence_rate,
            measurement_probability: 0.0,
        }
    }

    /// Initialize walker position
    pub fn initialize_position(&mut self, position: usize) {
        self.base_walk.initialize_position(position);
    }

    /// Perform one step with decoherence
    pub fn step(&mut self) {
        // Apply unitary evolution
        self.base_walk.step();

        // Apply decoherence
        self.apply_decoherence();
    }

    /// Apply decoherence by mixing with classical random walk
    fn apply_decoherence(&mut self) {
        if self.decoherence_rate <= 0.0 {
            return;
        }

        // Get current probabilities
        let quantum_probs = self.base_walk.position_probabilities();

        // Classical random walk step
        let mut classical_probs = vec![0.0; quantum_probs.len()];
        for (v, &prob) in quantum_probs.iter().enumerate() {
            if prob > 0.0 {
                let degree = self.base_walk.graph.degree(v) as f64;
                if degree > 0.0 {
                    for &neighbor in &self.base_walk.graph.edges[v] {
                        classical_probs[neighbor] += prob / degree;
                    }
                } else {
                    classical_probs[v] += prob; // Stay if isolated
                }
            }
        }

        // Mix quantum and classical
        let quantum_weight = 1.0 - self.decoherence_rate;
        let classical_weight = self.decoherence_rate;

        // Update quantum state to match mixed probabilities (approximate)
        for v in 0..quantum_probs.len() {
            let mixed_prob =
                quantum_weight * quantum_probs[v] + classical_weight * classical_probs[v];

            // Scale amplitudes to match mixed probabilities (simplified)
            if quantum_probs[v] > 0.0 {
                let scale_factor = (mixed_prob / quantum_probs[v]).sqrt();

                for coin in 0..self.base_walk.coin_dimension {
                    let idx = self.base_walk.state_index(v, coin);
                    if idx < self.base_walk.state.len() {
                        self.base_walk.state[idx] *= scale_factor;
                    }
                }
            }
        }

        // Renormalize
        let total_norm: f64 = self.base_walk.state.iter().map(|c| c.norm_sqr()).sum();
        if total_norm > 0.0 {
            let norm_factor = 1.0 / total_norm.sqrt();
            for amplitude in &mut self.base_walk.state {
                *amplitude *= norm_factor;
            }
        }
    }

    /// Get position probabilities
    pub fn position_probabilities(&self) -> Vec<f64> {
        self.base_walk.position_probabilities()
    }

    /// Set decoherence rate
    pub const fn set_decoherence_rate(&mut self, rate: f64) {
        self.decoherence_rate = rate.clamp(0.0, 1.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_relative_eq;

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

    #[test]
    fn test_weighted_graph() {
        let mut graph = Graph::new_empty(3);
        graph.add_weighted_edge(0, 1, 2.0);
        graph.add_weighted_edge(1, 2, 3.0);

        let adj_matrix = graph.adjacency_matrix();
        assert_eq!(adj_matrix[[0, 1]], 2.0);
        assert_eq!(adj_matrix[[1, 2]], 3.0);
        assert_eq!(adj_matrix[[0, 2]], 0.0);
    }

    #[test]
    fn test_graph_from_adjacency_matrix() {
        let mut matrix = Array2::zeros((3, 3));
        matrix[[0, 1]] = 1.0;
        matrix[[1, 0]] = 1.0;
        matrix[[1, 2]] = 2.0;
        matrix[[2, 1]] = 2.0;

        let graph = Graph::from_adjacency_matrix(&matrix).expect(
            "Failed to create graph from adjacency matrix in test_graph_from_adjacency_matrix",
        );
        assert_eq!(graph.num_vertices, 3);
        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(1), 2);
        assert_eq!(graph.degree(2), 1);
    }

    #[test]
    fn test_laplacian_matrix() {
        let graph = Graph::new(GraphType::Cycle, 3);
        let laplacian = graph.laplacian_matrix();

        // Each vertex in a 3-cycle has degree 2
        assert_eq!(laplacian[[0, 0]], 2.0);
        assert_eq!(laplacian[[1, 1]], 2.0);
        assert_eq!(laplacian[[2, 2]], 2.0);

        // Adjacent vertices have -1
        assert_eq!(laplacian[[0, 1]], -1.0);
        assert_eq!(laplacian[[1, 2]], -1.0);
        assert_eq!(laplacian[[2, 0]], -1.0);
    }

    #[test]
    fn test_bipartite_detection() {
        let bipartite = Graph::new(GraphType::Cycle, 4); // Even cycle is bipartite
        assert!(bipartite.is_bipartite());

        let non_bipartite = Graph::new(GraphType::Cycle, 3); // Odd cycle is not bipartite
        assert!(!non_bipartite.is_bipartite());

        let complete = Graph::new(GraphType::Complete, 3); // Complete graph with >2 vertices is not bipartite
        assert!(!complete.is_bipartite());
    }

    #[test]
    fn test_shortest_paths() {
        let graph = Graph::new(GraphType::Line, 4); // 0-1-2-3
        let distances = graph.all_pairs_shortest_paths();

        assert_eq!(distances[[0, 0]], 0.0);
        assert_eq!(distances[[0, 1]], 1.0);
        assert_eq!(distances[[0, 2]], 2.0);
        assert_eq!(distances[[0, 3]], 3.0);
        assert_eq!(distances[[1, 3]], 2.0);
    }

    #[test]
    fn test_szegedy_walk() {
        let graph = Graph::new(GraphType::Cycle, 4);
        let mut szegedy = SzegedyQuantumWalk::new(graph);

        szegedy.initialize_uniform();
        let initial_probs = szegedy.vertex_probabilities();

        // Should have some probability on each vertex
        for &prob in &initial_probs {
            assert!(prob > 0.0);
        }

        // Take a few steps
        for _ in 0..5 {
            szegedy.step();
        }

        let final_probs = szegedy.vertex_probabilities();
        let total: f64 = final_probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_szegedy_edge_initialization() {
        let mut graph = Graph::new_empty(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let mut szegedy = SzegedyQuantumWalk::new(graph);
        szegedy.initialize_edge(0, 1);

        let edge_probs = szegedy.edge_probabilities();
        assert_eq!(edge_probs.len(), 1);
        assert_eq!(edge_probs[0].0, (0, 1));
        assert!((edge_probs[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_walker_quantum_walk() {
        let graph = Graph::new(GraphType::Cycle, 3);
        let mut multi_walk = MultiWalkerQuantumWalk::new(graph, 2);

        // Initialize two walkers at positions 0 and 1
        multi_walk
            .initialize_positions(&[0, 1])
            .expect("Failed to initialize positions in test_multi_walker_quantum_walk");

        let marginal_0 = multi_walk.marginal_probabilities(0);
        let marginal_1 = multi_walk.marginal_probabilities(1);

        assert!((marginal_0[0] - 1.0).abs() < 1e-10);
        assert!((marginal_1[1] - 1.0).abs() < 1e-10);

        // Take a step
        multi_walk.step_independent();

        // Probabilities should have spread
        let new_marginal_0 = multi_walk.marginal_probabilities(0);
        let new_marginal_1 = multi_walk.marginal_probabilities(1);

        assert!(new_marginal_0[0] < 1.0);
        assert!(new_marginal_1[1] < 1.0);
    }

    #[test]
    fn test_multi_walker_bell_state() {
        let graph = Graph::new(GraphType::Cycle, 4);
        let mut multi_walk = MultiWalkerQuantumWalk::new(graph, 2);

        multi_walk
            .initialize_entangled_bell_state(0, 1)
            .expect("Failed to initialize entangled Bell state in test_multi_walker_bell_state");

        let marginal_0 = multi_walk.marginal_probabilities(0);
        let marginal_1 = multi_walk.marginal_probabilities(1);

        // Each walker should have 50% probability at each of their initial positions
        assert!((marginal_0[0] - 0.5).abs() < 1e-10);
        assert!((marginal_0[1] - 0.5).abs() < 1e-10);
        assert!((marginal_1[0] - 0.5).abs() < 1e-10);
        assert!((marginal_1[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_multi_walker_error_handling() {
        let graph = Graph::new(GraphType::Line, 3);
        let mut multi_walk = MultiWalkerQuantumWalk::new(graph.clone(), 2);

        // Wrong number of positions
        assert!(multi_walk.initialize_positions(&[0]).is_err());

        // Position out of bounds
        assert!(multi_walk.initialize_positions(&[0, 5]).is_err());

        // Bell state with wrong number of walkers
        let mut single_walk = MultiWalkerQuantumWalk::new(graph, 1);
        assert!(single_walk.initialize_entangled_bell_state(0, 1).is_err());
    }

    #[test]
    fn test_decoherent_quantum_walk() {
        let graph = Graph::new(GraphType::Line, 5);
        let mut decoherent = DecoherentQuantumWalk::new(graph, CoinOperator::Hadamard, 0.1);

        decoherent.initialize_position(2);
        let initial_probs = decoherent.position_probabilities();
        assert!((initial_probs[2] - 1.0).abs() < 1e-10);

        // Take steps with decoherence
        for _ in 0..10 {
            decoherent.step();
        }

        let final_probs = decoherent.position_probabilities();
        let total: f64 = final_probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);

        // Should have spread from initial position
        assert!(final_probs[2] < 1.0);
    }

    #[test]
    fn test_decoherence_rate_bounds() {
        let graph = Graph::new(GraphType::Cycle, 4);
        let mut decoherent = DecoherentQuantumWalk::new(graph, CoinOperator::Grover, 0.5);

        // Test clamping
        decoherent.set_decoherence_rate(-0.1);
        decoherent.initialize_position(0);
        decoherent.step(); // Should work without panicking

        decoherent.set_decoherence_rate(1.5);
        decoherent.step(); // Should work without panicking
    }

    #[test]
    fn test_transition_matrix() {
        let graph = Graph::new(GraphType::Cycle, 3);
        let transition = graph.transition_matrix();

        // Each vertex has degree 2, so each transition probability is 1/2
        for i in 0..3 {
            let mut row_sum = 0.0;
            for j in 0..3 {
                row_sum += transition[[i, j]];
            }
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalized_laplacian() {
        let graph = Graph::new(GraphType::Complete, 3);
        let norm_laplacian = graph.normalized_laplacian_matrix();

        // Diagonal entries should be 1
        for i in 0..3 {
            assert!((norm_laplacian[[i, i]] - 1.0).abs() < 1e-10);
        }

        // Off-diagonal entries for complete graph K_3
        let expected_off_diag = -1.0 / 2.0; // -1/sqrt(2*2)
        assert!((norm_laplacian[[0, 1]] - expected_off_diag).abs() < 1e-10);
        assert!((norm_laplacian[[1, 2]] - expected_off_diag).abs() < 1e-10);
        assert!((norm_laplacian[[0, 2]] - expected_off_diag).abs() < 1e-10);
    }

    #[test]
    fn test_algebraic_connectivity() {
        let complete_3 = Graph::new(GraphType::Complete, 3);
        let connectivity = complete_3.algebraic_connectivity();
        assert!(connectivity > 0.0); // Complete graphs have positive algebraic connectivity

        let line_5 = Graph::new(GraphType::Line, 5);
        let line_connectivity = line_5.algebraic_connectivity();
        assert!(line_connectivity > 0.0);
    }

    #[test]
    fn test_mixing_time_estimation() {
        let graph = Graph::new(GraphType::Complete, 4);
        let mut szegedy = SzegedyQuantumWalk::new(graph);

        let mixing_time = szegedy.estimate_mixing_time(0.1);
        assert!(mixing_time > 0);
        assert!(mixing_time <= 1000); // Should converge within max steps
    }

    #[test]
    fn test_quantum_walk_search_on_custom_graph() {
        // Create a star graph: central vertex connected to all others
        let mut graph = Graph::new_empty(5);
        for i in 1..5 {
            graph.add_edge(0, i);
        }

        let oracle = SearchOracle::new(vec![3]); // Mark vertex 3
        let mut search = QuantumWalkSearch::new(graph, oracle);

        let (found_vertex, prob, steps) = search.run(20);
        assert_eq!(found_vertex, 3);
        assert!(prob > 0.0);
        assert!(steps <= 20);
    }

    #[test]
    fn test_custom_coin_operator() {
        let graph = Graph::new(GraphType::Line, 3);

        // Create a custom 2x2 coin (Pauli-X)
        let mut coin_matrix = Array2::zeros((2, 2));
        coin_matrix[[0, 1]] = Complex64::new(1.0, 0.0);
        coin_matrix[[1, 0]] = Complex64::new(1.0, 0.0);

        let custom_coin = CoinOperator::Custom(coin_matrix);
        let mut walk = DiscreteQuantumWalk::new(graph, custom_coin);

        walk.initialize_position(1);
        walk.step();

        let probs = walk.position_probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_graph_edge_cases() {
        let empty_graph = Graph::new_empty(3);
        let mut szegedy = SzegedyQuantumWalk::new(empty_graph);

        szegedy.initialize_uniform();
        let probs = szegedy.vertex_probabilities();

        // No edges means no probability distribution
        for &prob in &probs {
            assert_eq!(prob, 0.0);
        }
    }

    #[test]
    fn test_hypercube_graph() {
        let hypercube = Graph::new(GraphType::Hypercube, 3); // 2^3 = 8 vertices
        assert_eq!(hypercube.num_vertices, 8);

        // Each vertex in a 3D hypercube has degree 3
        for i in 0..8 {
            assert_eq!(hypercube.degree(i), 3);
        }
    }

    #[test]
    fn test_grid_2d_graph() {
        let grid = Graph::new(GraphType::Grid2D, 3); // 3x3 grid
        assert_eq!(grid.num_vertices, 9);

        // Corner vertices have degree 2
        assert_eq!(grid.degree(0), 2); // Top-left
        assert_eq!(grid.degree(2), 2); // Top-right
        assert_eq!(grid.degree(6), 2); // Bottom-left
        assert_eq!(grid.degree(8), 2); // Bottom-right

        // Center vertex has degree 4
        assert_eq!(grid.degree(4), 4);
    }
}
