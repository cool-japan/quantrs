//! Continuous-time and Szegedy quantum walk implementations.

use super::graph::Graph;
use crate::complex_ext::QuantumComplexExt;
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use std::collections::HashMap;

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
