//! Coupling map representation for quantum devices
//!
//! This module defines the physical connectivity graph of quantum devices
//! and provides utilities for working with device topologies.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Distance type for coupling maps
pub type Distance = usize;

/// Coupling map representing the connectivity graph of a quantum device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMap {
    /// Number of physical qubits
    num_qubits: usize,
    /// Adjacency list representation
    edges: Vec<Vec<usize>>,
    /// Pre-computed distance matrix
    distances: Option<Vec<Vec<Distance>>>,
}

impl CouplingMap {
    /// Create a new coupling map with the specified number of qubits
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            edges: vec![Vec::new(); num_qubits],
            distances: None,
        }
    }

    /// Add a bidirectional edge between two qubits
    pub fn add_edge(&mut self, qubit1: usize, qubit2: usize) {
        if qubit1 < self.num_qubits && qubit2 < self.num_qubits && qubit1 != qubit2 {
            if !self.edges[qubit1].contains(&qubit2) {
                self.edges[qubit1].push(qubit2);
            }
            if !self.edges[qubit2].contains(&qubit1) {
                self.edges[qubit2].push(qubit1);
            }
            // Invalidate distance cache
            self.distances = None;
        }
    }

    /// Check if two qubits are directly connected
    pub fn are_connected(&self, qubit1: usize, qubit2: usize) -> bool {
        if qubit1 < self.num_qubits && qubit2 < self.num_qubits {
            self.edges[qubit1].contains(&qubit2)
        } else {
            false
        }
    }

    /// Get the neighbors of a qubit
    pub fn neighbors(&self, qubit: usize) -> &[usize] {
        if qubit < self.num_qubits {
            &self.edges[qubit]
        } else {
            &[]
        }
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get all edges as pairs
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for (i, neighbors) in self.edges.iter().enumerate() {
            for &j in neighbors {
                if i < j {
                    // Avoid duplicates
                    edges.push((i, j));
                }
            }
        }
        edges
    }

    /// Compute the distance between two qubits using BFS
    pub fn distance(&self, qubit1: usize, qubit2: usize) -> Distance {
        if qubit1 == qubit2 {
            return 0;
        }

        if qubit1 >= self.num_qubits || qubit2 >= self.num_qubits {
            return Distance::MAX;
        }

        // Use pre-computed distances if available
        if let Some(ref distances) = self.distances {
            return distances[qubit1][qubit2];
        }

        // BFS to find shortest path
        let mut queue = VecDeque::new();
        let mut visited = vec![false; self.num_qubits];
        let mut distances = vec![Distance::MAX; self.num_qubits];

        queue.push_back(qubit1);
        visited[qubit1] = true;
        distances[qubit1] = 0;

        while let Some(current) = queue.pop_front() {
            if current == qubit2 {
                return distances[current];
            }

            for &neighbor in &self.edges[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    distances[neighbor] = distances[current] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        Distance::MAX
    }

    /// Pre-compute all-pairs shortest distances
    pub fn compute_distances(&mut self) {
        let mut distances = vec![vec![Distance::MAX; self.num_qubits]; self.num_qubits];

        // Initialize diagonal and direct edges
        for i in 0..self.num_qubits {
            distances[i][i] = 0;
            for &j in &self.edges[i] {
                distances[i][j] = 1;
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..self.num_qubits {
            for i in 0..self.num_qubits {
                for j in 0..self.num_qubits {
                    if distances[i][k] != Distance::MAX && distances[k][j] != Distance::MAX {
                        let new_dist = distances[i][k] + distances[k][j];
                        if new_dist < distances[i][j] {
                            distances[i][j] = new_dist;
                        }
                    }
                }
            }
        }

        self.distances = Some(distances);
    }

    /// Get the shortest path between two qubits
    pub fn shortest_path(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        if start == end {
            return Some(vec![start]);
        }

        if start >= self.num_qubits || end >= self.num_qubits {
            return None;
        }

        // BFS to find path
        let mut queue = VecDeque::new();
        let mut visited = vec![false; self.num_qubits];
        let mut parent = vec![None; self.num_qubits];

        queue.push_back(start);
        visited[start] = true;

        while let Some(current) = queue.pop_front() {
            if current == end {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = Some(end);

                while let Some(n) = node {
                    path.push(n);
                    node = parent[n];
                }

                path.reverse();
                return Some(path);
            }

            for &neighbor in &self.edges[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = Some(current);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Check if the graph is connected
    pub fn is_connected(&self) -> bool {
        if self.num_qubits <= 1 {
            return true;
        }

        let mut visited = vec![false; self.num_qubits];
        let mut queue = VecDeque::new();

        // Start from qubit 0
        queue.push_back(0);
        visited[0] = true;
        let mut count = 1;

        while let Some(current) = queue.pop_front() {
            for &neighbor in &self.edges[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                    count += 1;
                }
            }
        }

        count == self.num_qubits
    }

    /// Get the diameter of the graph (maximum distance between any two nodes)
    pub fn diameter(&self) -> Distance {
        let mut max_distance = 0;

        for i in 0..self.num_qubits {
            for j in (i + 1)..self.num_qubits {
                let dist = self.distance(i, j);
                if dist != Distance::MAX {
                    max_distance = max_distance.max(dist);
                }
            }
        }

        max_distance
    }

    /// Create common device topologies

    /// Linear topology (1D chain)
    pub fn linear(num_qubits: usize) -> Self {
        let mut coupling_map = Self::new(num_qubits);
        for i in 0..(num_qubits.saturating_sub(1)) {
            coupling_map.add_edge(i, i + 1);
        }
        coupling_map.compute_distances();
        coupling_map
    }

    /// Ring topology (circular)
    pub fn ring(num_qubits: usize) -> Self {
        let mut coupling_map = Self::linear(num_qubits);
        if num_qubits > 2 {
            coupling_map.add_edge(0, num_qubits - 1);
        }
        coupling_map.compute_distances();
        coupling_map
    }

    /// Grid topology (2D)
    pub fn grid(rows: usize, cols: usize) -> Self {
        let num_qubits = rows * cols;
        let mut coupling_map = Self::new(num_qubits);

        for row in 0..rows {
            for col in 0..cols {
                let qubit = row * cols + col;

                // Connect to right neighbor
                if col + 1 < cols {
                    coupling_map.add_edge(qubit, qubit + 1);
                }

                // Connect to bottom neighbor
                if row + 1 < rows {
                    coupling_map.add_edge(qubit, qubit + cols);
                }
            }
        }

        coupling_map.compute_distances();
        coupling_map
    }

    /// All-to-all topology (complete graph)
    pub fn all_to_all(num_qubits: usize) -> Self {
        let mut coupling_map = Self::new(num_qubits);
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                coupling_map.add_edge(i, j);
            }
        }
        coupling_map.compute_distances();
        coupling_map
    }

    /// IBM Lagos device topology
    pub fn ibm_lagos() -> Self {
        let mut coupling_map = Self::new(7);

        // Lagos connectivity: heavy-hex layout
        let edges = [(0, 1), (1, 2), (1, 4), (2, 3), (3, 6), (4, 5), (5, 6)];

        for (q1, q2) in edges {
            coupling_map.add_edge(q1, q2);
        }

        coupling_map.compute_distances();
        coupling_map
    }

    /// IBM Nairobi device topology
    pub fn ibm_nairobi() -> Self {
        let mut coupling_map = Self::new(7);

        // Nairobi connectivity
        let edges = [(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)];

        for (q1, q2) in edges {
            coupling_map.add_edge(q1, q2);
        }

        coupling_map.compute_distances();
        coupling_map
    }

    /// Google Sycamore-like device topology
    pub fn google_sycamore() -> Self {
        let mut coupling_map = Self::new(12);

        // Simplified Sycamore-like 2D grid with some missing connections
        let edges = [
            (0, 1),
            (0, 4),
            (1, 2),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 8),
            (5, 6),
            (5, 9),
            (6, 7),
            (6, 10),
            (7, 11),
            (8, 9),
            (9, 10),
            (10, 11),
        ];

        for (q1, q2) in edges {
            coupling_map.add_edge(q1, q2);
        }

        coupling_map.compute_distances();
        coupling_map
    }

    /// Load coupling map from adjacency list
    pub fn from_edges(num_qubits: usize, edges: &[(usize, usize)]) -> Self {
        let mut coupling_map = Self::new(num_qubits);
        for &(q1, q2) in edges {
            coupling_map.add_edge(q1, q2);
        }
        coupling_map.compute_distances();
        coupling_map
    }
}

impl Default for CouplingMap {
    fn default() -> Self {
        Self::linear(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_coupling_map() {
        let coupling_map = CouplingMap::linear(5);

        assert_eq!(coupling_map.num_qubits(), 5);
        assert!(coupling_map.are_connected(0, 1));
        assert!(coupling_map.are_connected(1, 2));
        assert!(!coupling_map.are_connected(0, 2));

        assert_eq!(coupling_map.distance(0, 4), 4);
        assert_eq!(coupling_map.distance(1, 3), 2);
    }

    #[test]
    fn test_grid_coupling_map() {
        let coupling_map = CouplingMap::grid(2, 3);

        assert_eq!(coupling_map.num_qubits(), 6);
        assert!(coupling_map.are_connected(0, 1));
        assert!(coupling_map.are_connected(0, 3));
        assert!(!coupling_map.are_connected(0, 2));
    }

    #[test]
    fn test_shortest_path() {
        let coupling_map = CouplingMap::linear(5);

        let path = coupling_map.shortest_path(0, 4).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3, 4]);

        let path = coupling_map.shortest_path(2, 2).unwrap();
        assert_eq!(path, vec![2]);
    }

    #[test]
    fn test_connectivity() {
        let connected = CouplingMap::linear(5);
        assert!(connected.is_connected());

        let mut disconnected = CouplingMap::new(5);
        disconnected.add_edge(0, 1);
        disconnected.add_edge(2, 3);
        assert!(!disconnected.is_connected());
    }

    #[test]
    fn test_ibm_lagos() {
        let coupling_map = CouplingMap::ibm_lagos();
        assert_eq!(coupling_map.num_qubits(), 7);
        assert!(coupling_map.is_connected());
    }
}
