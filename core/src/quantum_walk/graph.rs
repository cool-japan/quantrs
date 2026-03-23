//! Graph types and implementations for quantum walks.

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use std::collections::VecDeque;

use super::eigensolvers::{compute_laplacian_eigenvalues_impl, estimate_fiedler_value_impl};

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

    /// Compute eigenvalues of the Laplacian using Householder tridiagonalization
    /// followed by Golub-Reinsch QR iteration with Wilkinson shifts.
    fn compute_laplacian_eigenvalues(&self, laplacian: &Array2<f64>) -> Vec<f64> {
        compute_laplacian_eigenvalues_impl(laplacian)
            .unwrap_or_else(|_| vec![0.0; self.num_vertices])
    }

    /// Estimate Fiedler value (second smallest Laplacian eigenvalue) using
    /// Rayleigh quotient power iteration restricted to the subspace
    /// orthogonal to the all-ones vector.
    fn estimate_fiedler_value(&self, laplacian: &Array2<f64>) -> f64 {
        estimate_fiedler_value_impl(laplacian)
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
