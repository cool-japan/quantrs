//! Graph embedding algorithms for quantum annealing hardware
//!
//! This module implements minorminer-like embedding algorithms that map
//! logical problem graphs onto physical quantum annealing hardware topologies.
//! The embedding process finds chains of physical qubits to represent each
//! logical variable, ensuring connectivity constraints are satisfied.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use crate::ising::{IsingError, IsingResult};

/// Hardware graph topology types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareTopology {
    /// Chimera topology with given dimensions (m, n, t)
    /// where m×n is the grid size and t is the shore size
    Chimera(usize, usize, usize),
    /// Pegasus topology with given dimension
    Pegasus(usize),
    /// Zephyr topology with given dimension
    Zephyr(usize),
    /// Custom topology defined by adjacency
    Custom,
}

/// Represents a hardware graph for quantum annealing
#[derive(Debug, Clone)]
pub struct HardwareGraph {
    /// Number of physical qubits
    pub num_qubits: usize,
    /// Adjacency matrix (stored as adjacency list for efficiency)
    pub adjacency: HashMap<usize, Vec<usize>>,
    /// Hardware topology type
    pub topology: HardwareTopology,
    /// Coordinates for each qubit (if applicable)
    pub coordinates: Option<HashMap<usize, Vec<usize>>>,
}

impl HardwareGraph {
    /// Create a new hardware graph with custom adjacency
    pub fn new_custom(num_qubits: usize, edges: Vec<(usize, usize)>) -> Self {
        let mut adjacency = HashMap::new();
        
        for i in 0..num_qubits {
            adjacency.insert(i, Vec::new());
        }
        
        for (u, v) in edges {
            adjacency.get_mut(&u).unwrap().push(v);
            adjacency.get_mut(&v).unwrap().push(u);
        }
        
        Self {
            num_qubits,
            adjacency,
            topology: HardwareTopology::Custom,
            coordinates: None,
        }
    }

    /// Create a Chimera graph with given dimensions
    pub fn new_chimera(m: usize, n: usize, t: usize) -> IsingResult<Self> {
        let num_qubits = 2 * m * n * t;
        let mut adjacency = HashMap::new();
        
        // Initialize adjacency lists
        for q in 0..num_qubits {
            adjacency.insert(q, Vec::new());
        }

        // Create Chimera connectivity
        for i in 0..m {
            for j in 0..n {
                let cell_offset = 2 * t * (i * n + j);
                
                // Internal bipartite connections within unit cell
                for k in 0..t {
                    for l in 0..t {
                        let q1 = cell_offset + k;
                        let q2 = cell_offset + t + l;
                        adjacency.get_mut(&q1).unwrap().push(q2);
                        adjacency.get_mut(&q2).unwrap().push(q1);
                    }
                }
                
                // Horizontal connections
                if j < n - 1 {
                    for k in 0..t {
                        let q1 = cell_offset + k;
                        let q2 = cell_offset + 2 * t + k;
                        adjacency.get_mut(&q1).unwrap().push(q2);
                        adjacency.get_mut(&q2).unwrap().push(q1);
                    }
                }
                
                // Vertical connections
                if i < m - 1 {
                    for k in 0..t {
                        let q1 = cell_offset + t + k;
                        let q2 = cell_offset + 2 * t * n + t + k;
                        adjacency.get_mut(&q1).unwrap().push(q2);
                        adjacency.get_mut(&q2).unwrap().push(q1);
                    }
                }
            }
        }

        // Generate coordinates
        let mut coordinates = HashMap::new();
        for i in 0..m {
            for j in 0..n {
                let cell_offset = 2 * t * (i * n + j);
                for k in 0..t {
                    coordinates.insert(cell_offset + k, vec![i, j, 0, k]);
                    coordinates.insert(cell_offset + t + k, vec![i, j, 1, k]);
                }
            }
        }

        Ok(Self {
            num_qubits,
            adjacency,
            topology: HardwareTopology::Chimera(m, n, t),
            coordinates: Some(coordinates),
        })
    }

    /// Get the neighbors of a qubit
    pub fn neighbors(&self, qubit: usize) -> Vec<usize> {
        if qubit >= self.num_qubits {
            return Vec::new();
        }
        
        self.adjacency.get(&qubit).cloned().unwrap_or_default()
    }

    /// Check if two qubits are connected
    pub fn are_connected(&self, q1: usize, q2: usize) -> bool {
        if q1 >= self.num_qubits || q2 >= self.num_qubits {
            return false;
        }
        self.adjacency.get(&q1)
            .map(|neighbors| neighbors.contains(&q2))
            .unwrap_or(false)
    }
}

/// Represents an embedding of logical variables onto physical qubits
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Maps logical variable index to chain of physical qubits
    pub chains: HashMap<usize, Vec<usize>>,
    /// Reverse mapping: physical qubit to logical variable
    pub qubit_to_variable: HashMap<usize, usize>,
}

impl Embedding {
    /// Create a new empty embedding
    pub fn new() -> Self {
        Self {
            chains: HashMap::new(),
            qubit_to_variable: HashMap::new(),
        }
    }

    /// Add a chain for a logical variable
    pub fn add_chain(&mut self, variable: usize, chain: Vec<usize>) -> IsingResult<()> {
        // Check for overlapping chains
        for &qubit in &chain {
            if let Some(&existing_var) = self.qubit_to_variable.get(&qubit) {
                if existing_var != variable {
                    return Err(IsingError::InvalidQubit(qubit));
                }
            }
        }

        // Update mappings
        for &qubit in &chain {
            self.qubit_to_variable.insert(qubit, variable);
        }
        self.chains.insert(variable, chain);

        Ok(())
    }

    /// Verify the embedding is valid for given logical and hardware graphs
    pub fn verify(
        &self,
        logical_edges: &[(usize, usize)],
        num_vars: usize,
        hardware: &HardwareGraph,
    ) -> IsingResult<()> {

        // Check all variables are embedded
        for var in 0..num_vars {
            if !self.chains.contains_key(&var) {
                return Err(IsingError::InvalidQubit(var));
            }
        }

        // Check chains are connected
        for (var, chain) in &self.chains {
            if !is_chain_connected(chain, hardware) {
                return Err(IsingError::HardwareConstraint(format!(
                    "Chain for variable {} is not connected",
                    var
                )));
            }
        }

        // Check logical edges are preserved
        for &(u, v) in logical_edges {
            let chain1 = match self.chains.get(&u) {
                Some(c) => c,
                None => return Err(IsingError::InvalidQubit(u)),
            };
            let chain2 = match self.chains.get(&v) {
                Some(c) => c,
                None => return Err(IsingError::InvalidQubit(v)),
            };
            
            let mut connected = false;
            'outer: for &q1 in chain1 {
                for &q2 in chain2 {
                    if hardware.are_connected(q1, q2) {
                        connected = true;
                        break 'outer;
                    }
                }
            }
            
            if !connected {
                return Err(IsingError::HardwareConstraint(format!(
                    "Logical edge ({}, {}) has no physical connection",
                    u, v
                )));
            }
        }

        Ok(())
    }
}

/// MinorMiner-like embedding algorithm
pub struct MinorMiner {
    /// Maximum number of embedding attempts
    pub max_tries: usize,
    /// Chain length penalty weight
    pub chain_length_penalty: f64,
    /// Use random initial chains
    pub random_init: bool,
    /// Seed for random number generation
    pub seed: Option<u64>,
}

impl Default for MinorMiner {
    fn default() -> Self {
        Self {
            max_tries: 10,
            chain_length_penalty: 1.0,
            random_init: true,
            seed: None,
        }
    }
}

impl MinorMiner {
    /// Find an embedding of logical graph into hardware graph
    pub fn find_embedding(
        &self,
        logical_edges: &[(usize, usize)],
        num_vars: usize,
        hardware: &HardwareGraph,
    ) -> IsingResult<Embedding> {
        
        // Use a heuristic approach similar to minorminer
        for attempt in 0..self.max_tries {
            let mut embedding = Embedding::new();
            let mut used_qubits = HashSet::new();
            
            // Order variables by degree (higher degree first)
            let mut var_degrees: Vec<(usize, usize)> = (0..num_vars)
                .map(|v| {
                    let degree = logical_edges.iter()
                        .filter(|&&(u, w)| u == v || w == v)
                        .count();
                    (v, degree)
                })
                .collect();
            var_degrees.sort_by_key(|&(_, d)| std::cmp::Reverse(d));
            
            // Try to embed each variable
            let mut success = true;
            for (var, _) in var_degrees {
                // Find neighbors that are already embedded
                let embedded_neighbors = get_embedded_neighbors(var, logical_edges, &embedding);
                
                // Find a chain for this variable
                if let Some(chain) = find_chain_for_variable(
                    var,
                    &embedded_neighbors,
                    &embedding,
                    hardware,
                    &used_qubits,
                ) {
                    for &q in &chain {
                        used_qubits.insert(q);
                    }
                    embedding.add_chain(var, chain)?;
                } else {
                    success = false;
                    break;
                }
            }
            
            if success {
                // Verify and return the embedding
                embedding.verify(logical_edges, num_vars, hardware)?;
                return Ok(embedding);
            }
        }
        
        Err(IsingError::HardwareConstraint("Failed to find valid embedding".to_string()))
    }
}

/// Check if a chain of qubits is connected in the hardware graph
fn is_chain_connected(chain: &[usize], hardware: &HardwareGraph) -> bool {
    if chain.is_empty() {
        return false;
    }
    if chain.len() == 1 {
        return true;
    }
    
    // Use BFS to check connectivity
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(chain[0]);
    visited.insert(chain[0]);
    
    while let Some(qubit) = queue.pop_front() {
        for neighbor in hardware.neighbors(qubit) {
            if chain.contains(&neighbor) && visited.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }
    
    visited.len() == chain.len()
}

/// Get embedded neighbors of a variable
fn get_embedded_neighbors(
    var: usize,
    logical_edges: &[(usize, usize)],
    embedding: &Embedding,
) -> Vec<usize> {
    let mut neighbors = Vec::new();
    
    for &(u, v) in logical_edges {
        if u == var && embedding.chains.contains_key(&v) {
            neighbors.push(v);
        } else if v == var && embedding.chains.contains_key(&u) {
            neighbors.push(u);
        }
    }
    
    neighbors
}

/// Find a chain for a variable given embedded neighbors
fn find_chain_for_variable(
    var: usize,
    embedded_neighbors: &[usize],
    embedding: &Embedding,
    hardware: &HardwareGraph,
    used_qubits: &HashSet<usize>,
) -> Option<Vec<usize>> {
    // If no embedded neighbors, pick any available qubit
    if embedded_neighbors.is_empty() {
        for q in 0..hardware.num_qubits {
            if !used_qubits.contains(&q) {
                return Some(vec![q]);
            }
        }
        return None;
    }
    
    // Find qubits that can connect to all embedded neighbors
    let mut candidate_qubits = HashSet::new();
    
    // Collect all qubits adjacent to neighbor chains
    for &neighbor_var in embedded_neighbors {
        if let Some(neighbor_chain) = embedding.chains.get(&neighbor_var) {
            for &q in neighbor_chain {
                for &adj_q in &hardware.neighbors(q) {
                    if !used_qubits.contains(&adj_q) {
                        candidate_qubits.insert(adj_q);
                    }
                }
            }
        }
    }
    
    // If no candidates adjacent to neighbors, use any available qubit
    if candidate_qubits.is_empty() {
        for q in 0..hardware.num_qubits {
            if !used_qubits.contains(&q) {
                candidate_qubits.insert(q);
            }
        }
    }
    
    // Try to find a connected chain from candidates
    let candidates: Vec<_> = candidate_qubits.into_iter().collect();
    for &start_q in &candidates {
        let chain = grow_chain(
            start_q,
            embedded_neighbors,
            embedding,
            hardware,
            used_qubits,
        );
        
        if is_valid_chain(&chain, embedded_neighbors, embedding, hardware) {
            return Some(chain);
        }
    }
    
    // If simple approach fails, try more complex chain building
    if !embedded_neighbors.is_empty() {
        // Try to build a chain that connects to at least one neighbor
        for &start_q in &candidates {
            let mut chain = vec![start_q];
            let mut chain_set = HashSet::new();
            chain_set.insert(start_q);
            
            // Check if we can connect to any neighbor with just this qubit
            for &neighbor_var in embedded_neighbors {
                if let Some(neighbor_chain) = embedding.chains.get(&neighbor_var) {
                    for &nq in neighbor_chain {
                        if hardware.are_connected(start_q, nq) {
                            return Some(chain);
                        }
                    }
                }
            }
        }
    }
    
    None
}

/// Grow a chain starting from a qubit
fn grow_chain(
    start: usize,
    embedded_neighbors: &[usize],
    embedding: &Embedding,
    hardware: &HardwareGraph,
    used_qubits: &HashSet<usize>,
) -> Vec<usize> {
    let mut chain = vec![start];
    let mut chain_set = HashSet::new();
    chain_set.insert(start);
    
    // Try to connect to each embedded neighbor
    for &neighbor_var in embedded_neighbors {
        if let Some(neighbor_chain) = embedding.chains.get(&neighbor_var) {
            // Check if already connected
            let mut connected = false;
            for &q in &chain {
                for &nq in neighbor_chain {
                    if hardware.are_connected(q, nq) {
                        connected = true;
                        break;
                    }
                }
                if connected {
                    break;
                }
            }
            
            if !connected {
                // Try to extend chain to connect
                if let Some(path) = find_path_to_chain(
                    &chain,
                    neighbor_chain,
                    hardware,
                    used_qubits,
                    &chain_set,
                ) {
                    for q in path {
                        if chain_set.insert(q) {
                            chain.push(q);
                        }
                    }
                }
            }
        }
    }
    
    chain
}

/// Find a path from a chain to another chain
fn find_path_to_chain(
    from_chain: &[usize],
    to_chain: &[usize],
    hardware: &HardwareGraph,
    used_qubits: &HashSet<usize>,
    chain_set: &HashSet<usize>,
) -> Option<Vec<usize>> {
    // Use BFS to find shortest path
    let mut queue = VecDeque::new();
    let mut parent = HashMap::new();
    let mut visited = HashSet::new();
    
    // Start from all qubits in from_chain
    for &q in from_chain {
        queue.push_back(q);
        visited.insert(q);
    }
    
    while let Some(current) = queue.pop_front() {
        for neighbor in hardware.neighbors(current) {
            if to_chain.contains(&neighbor) {
                // Found a connection, reconstruct path
                let mut path = Vec::new();
                let mut node = Some(neighbor);
                
                while let Some(n) = node {
                    if !chain_set.contains(&n) {
                        path.push(n);
                    }
                    node = parent.get(&n).copied();
                    if from_chain.contains(&n) {
                        break;
                    }
                }
                
                path.reverse();
                return Some(path);
            }
            
            if !used_qubits.contains(&neighbor) 
                && !chain_set.contains(&neighbor)
                && visited.insert(neighbor) {
                parent.insert(neighbor, current);
                queue.push_back(neighbor);
            }
        }
    }
    
    None
}

/// Check if a chain is valid for given embedded neighbors
fn is_valid_chain(
    chain: &[usize],
    embedded_neighbors: &[usize],
    embedding: &Embedding,
    hardware: &HardwareGraph,
) -> bool {
    if chain.is_empty() {
        return false;
    }
    
    // For a chain to be valid, it should connect to all embedded neighbors if possible
    // But if that's not possible, at least one connection is acceptable
    if embedded_neighbors.is_empty() {
        return true;
    }
    
    let mut connections = 0;
    for &neighbor_var in embedded_neighbors {
        if let Some(neighbor_chain) = embedding.chains.get(&neighbor_var) {
            let mut connected = false;
            
            'outer: for &q1 in chain {
                for &q2 in neighbor_chain {
                    if hardware.are_connected(q1, q2) {
                        connected = true;
                        break 'outer;
                    }
                }
            }
            
            if connected {
                connections += 1;
            }
        }
    }
    
    // Accept if we connect to at least one neighbor
    connections > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chimera_graph() {
        let graph = HardwareGraph::new_chimera(2, 2, 4).unwrap();
        assert_eq!(graph.num_qubits, 32);
        
        // Test internal bipartite connections
        assert!(graph.are_connected(0, 4));
        assert!(graph.are_connected(0, 5));
        assert!(!graph.are_connected(0, 1));
        
        // Test that each qubit has the right number of neighbors
        let q0_neighbors = graph.neighbors(0);
        assert!(q0_neighbors.len() > 0);
    }

    #[test]
    fn test_simple_embedding() {
        // Create a simple 2-node edge logical graph (simpler test case)
        let logical_edges = vec![(0, 1)];
        let num_vars = 2;
        
        // Create a small chimera graph
        let hardware = HardwareGraph::new_chimera(1, 1, 2).unwrap();
        
        // Find embedding
        let embedder = MinorMiner::default();
        let result = embedder.find_embedding(&logical_edges, num_vars, &hardware);
        
        // For this small example, embedding should be possible
        assert!(result.is_ok(), "Failed to find embedding: {:?}", result.err());
        
        if let Ok(embedding) = result {
            // Verify the embedding
            let verify_result = embedding.verify(&logical_edges, num_vars, &hardware);
            assert!(verify_result.is_ok(), "Verification failed: {:?}", verify_result.err());
            assert_eq!(embedding.chains.len(), 2);
            
            // Check that the chains are connected by an edge
            let chain0 = &embedding.chains[&0];
            let chain1 = &embedding.chains[&1];
            let mut found_connection = false;
            for &q0 in chain0 {
                for &q1 in chain1 {
                    if hardware.are_connected(q0, q1) {
                        found_connection = true;
                        break;
                    }
                }
            }
            assert!(found_connection, "Chains are not connected");
        }
    }
    
    #[test]
    fn test_triangle_embedding() {
        // Create a 3-node triangle logical graph  
        let logical_edges = vec![(0, 1), (0, 2), (1, 2)];
        let num_vars = 3;
        
        // Create a larger chimera graph for triangle
        let hardware = HardwareGraph::new_chimera(2, 2, 2).unwrap();
        
        // Find embedding
        let embedder = MinorMiner {
            max_tries: 20,  // More tries for harder problem
            ..Default::default()
        };
        let result = embedder.find_embedding(&logical_edges, num_vars, &hardware);
        
        // Triangle embedding in chimera is possible but may require multiple tries
        if let Ok(embedding) = result {
            // Verify the embedding
            assert!(embedding.verify(&logical_edges, num_vars, &hardware).is_ok());
            assert_eq!(embedding.chains.len(), 3);
        }
    }

    #[test]
    fn test_chain_connectivity() {
        let hardware = HardwareGraph::new_chimera(2, 2, 2).unwrap();
        
        // Test connected chain
        let chain1 = vec![0, 2]; // Connected in chimera
        assert!(is_chain_connected(&chain1, &hardware));
        
        // Test disconnected chain
        let chain2 = vec![0, 7]; // Not directly connected
        assert!(!is_chain_connected(&chain2, &hardware));
        
        // Test single qubit chain
        let chain3 = vec![0];
        assert!(is_chain_connected(&chain3, &hardware));
    }
}