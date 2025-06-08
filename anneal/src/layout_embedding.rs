//! Layout-aware graph embedding algorithms
//!
//! This module implements advanced graph embedding techniques that take into
//! account the physical layout of quantum annealing hardware to optimize
//! embedding quality, minimize chain lengths, and improve solution quality.

use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;

use crate::embedding::{Embedding, HardwareGraph, HardwareTopology};
use crate::ising::{IsingError, IsingResult};

/// Configuration for layout-aware embedding
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Weight for physical distance in cost function
    pub distance_weight: f64,
    /// Weight for chain length in cost function
    pub chain_length_weight: f64,
    /// Weight for chain degree (number of connections) in cost function
    pub chain_degree_weight: f64,
    /// Maximum allowed chain length
    pub max_chain_length: usize,
    /// Use spectral placement for initial embedding
    pub use_spectral_placement: bool,
    /// Number of refinement iterations
    pub refinement_iterations: usize,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            distance_weight: 1.0,
            chain_length_weight: 2.0,
            chain_degree_weight: 0.5,
            max_chain_length: 5,
            use_spectral_placement: true,
            refinement_iterations: 10,
        }
    }
}

/// Statistics from layout-aware embedding
#[derive(Debug, Clone)]
pub struct LayoutStats {
    /// Average chain length
    pub avg_chain_length: f64,
    /// Maximum chain length
    pub max_chain_length: usize,
    /// Total chain length
    pub total_chain_length: usize,
    /// Number of chains exceeding threshold
    pub long_chains: usize,
    /// Embedding quality score
    pub quality_score: f64,
}

/// Layout-aware embedder that considers hardware topology
pub struct LayoutAwareEmbedder {
    config: LayoutConfig,
    /// Cache of shortest paths in hardware graph
    shortest_paths: HashMap<(usize, usize), Vec<usize>>,
    /// Physical coordinates of qubits (if available)
    qubit_coordinates: Option<HashMap<usize, Vec<f64>>>,
}

impl LayoutAwareEmbedder {
    /// Create a new layout-aware embedder
    pub fn new(config: LayoutConfig) -> Self {
        Self {
            config,
            shortest_paths: HashMap::new(),
            qubit_coordinates: None,
        }
    }

    /// Set physical coordinates for qubits
    pub fn set_coordinates(&mut self, coordinates: HashMap<usize, Vec<f64>>) {
        self.qubit_coordinates = Some(coordinates);
    }

    /// Find a layout-aware embedding
    pub fn find_embedding(
        &mut self,
        logical_edges: &[(usize, usize)],
        num_vars: usize,
        hardware: &HardwareGraph,
    ) -> IsingResult<(Embedding, LayoutStats)> {
        // Initialize coordinates from hardware if available
        if self.qubit_coordinates.is_none() && hardware.coordinates.is_some() {
            self.qubit_coordinates = hardware.coordinates.as_ref().map(|coords| {
                coords.iter()
                    .map(|(q, coord)| (*q, coord.iter().map(|&x| x as f64).collect()))
                    .collect()
            });
        }

        // Phase 1: Initial placement
        let initial_placement = if self.config.use_spectral_placement {
            self.spectral_placement(logical_edges, num_vars, hardware)?
        } else {
            self.greedy_placement(logical_edges, num_vars, hardware)?
        };

        // Phase 2: Chain building
        let mut embedding = self.build_chains(&initial_placement, logical_edges, hardware)?;

        // Phase 3: Iterative refinement
        for _ in 0..self.config.refinement_iterations {
            let improved = self.refine_embedding(&mut embedding, logical_edges, hardware)?;
            if !improved {
                break;
            }
        }

        // Compute statistics
        let stats = self.compute_stats(&embedding);

        Ok((embedding, stats))
    }

    /// Spectral placement using eigenvectors of the logical graph Laplacian
    fn spectral_placement(
        &self,
        logical_edges: &[(usize, usize)],
        num_vars: usize,
        hardware: &HardwareGraph,
    ) -> IsingResult<HashMap<usize, usize>> {
        // Simplified spectral placement
        // In practice, would compute eigenvectors of graph Laplacian
        
        let mut placement = HashMap::new();
        let mut available_qubits: Vec<_> = (0..hardware.num_qubits).collect();
        
        // Sort variables by degree
        let mut var_degrees: Vec<_> = (0..num_vars)
            .map(|v| {
                let degree = logical_edges.iter()
                    .filter(|&&(u, w)| u == v || w == v)
                    .count();
                (v, degree)
            })
            .collect();
        var_degrees.sort_by_key(|&(_, d)| std::cmp::Reverse(d));

        // Place high-degree variables first
        for (var, _) in var_degrees {
            if let Some(best_qubit) = self.find_best_qubit(var, &placement, logical_edges, &available_qubits, hardware) {
                placement.insert(var, best_qubit);
                available_qubits.retain(|&q| q != best_qubit);
            }
        }

        Ok(placement)
    }

    /// Greedy placement algorithm
    fn greedy_placement(
        &self,
        logical_edges: &[(usize, usize)],
        num_vars: usize,
        hardware: &HardwareGraph,
    ) -> IsingResult<HashMap<usize, usize>> {
        let mut placement = HashMap::new();
        let mut available_qubits: HashSet<_> = (0..hardware.num_qubits).collect();

        // Order variables by connectivity
        let mut var_order = self.order_variables_by_connectivity(logical_edges, num_vars);

        for var in var_order {
            let neighbors = self.get_logical_neighbors(var, logical_edges);
            let placed_neighbors: Vec<_> = neighbors.iter()
                .filter_map(|&n| placement.get(&n).copied())
                .collect();

            let best_qubit = if placed_neighbors.is_empty() {
                // No placed neighbors, choose central qubit
                self.find_central_qubit(&available_qubits, hardware)
            } else {
                // Find qubit closest to placed neighbors
                self.find_closest_qubit(&placed_neighbors, &available_qubits, hardware)?
            };

            placement.insert(var, best_qubit);
            available_qubits.remove(&best_qubit);
        }

        Ok(placement)
    }

    /// Build chains from initial placement
    fn build_chains(
        &mut self,
        placement: &HashMap<usize, usize>,
        logical_edges: &[(usize, usize)],
        hardware: &HardwareGraph,
    ) -> IsingResult<Embedding> {
        let mut embedding = Embedding::new();
        let mut used_qubits = HashSet::new();

        // First, assign single-qubit chains for placed variables
        for (&var, &qubit) in placement {
            embedding.add_chain(var, vec![qubit])?;
            used_qubits.insert(qubit);
        }

        // Extend chains to ensure connectivity
        for &(u, v) in logical_edges {
            if let (Some(chain_u), Some(chain_v)) = (embedding.chains.get(&u), embedding.chains.get(&v)) {
                if !self.chains_connected(chain_u, chain_v, hardware) {
                    // Need to extend one or both chains
                    let path = self.find_connection_path(chain_u, chain_v, hardware, &used_qubits)?;
                    
                    // Decide which chain to extend based on current lengths
                    if chain_u.len() <= chain_v.len() {
                        self.extend_chain(&mut embedding, u, &path, &mut used_qubits)?;
                    } else {
                        self.extend_chain(&mut embedding, v, &path, &mut used_qubits)?;
                    }
                }
            }
        }

        Ok(embedding)
    }

    /// Refine embedding to improve quality
    fn refine_embedding(
        &self,
        embedding: &mut Embedding,
        logical_edges: &[(usize, usize)],
        hardware: &HardwareGraph,
    ) -> IsingResult<bool> {
        let mut improved = false;

        // Try to shorten long chains
        for (var, chain) in embedding.chains.clone() {
            if chain.len() > self.config.max_chain_length {
                if let Some(shorter_chain) = self.find_shorter_chain(
                    var,
                    &chain,
                    embedding,
                    logical_edges,
                    hardware,
                ) {
                    embedding.chains.insert(var, shorter_chain);
                    improved = true;
                }
            }
        }

        // Try local swaps to improve layout
        for var in 0..embedding.chains.len() {
            if self.try_local_swap(embedding, var, logical_edges, hardware)? {
                improved = true;
            }
        }

        Ok(improved)
    }

    /// Find the best qubit for a variable during placement
    fn find_best_qubit(
        &self,
        var: usize,
        placement: &HashMap<usize, usize>,
        logical_edges: &[(usize, usize)],
        available: &[usize],
        hardware: &HardwareGraph,
    ) -> Option<usize> {
        let neighbors = self.get_logical_neighbors(var, logical_edges);
        let placed_neighbors: Vec<_> = neighbors.iter()
            .filter_map(|&n| placement.get(&n))
            .collect();

        if placed_neighbors.is_empty() {
            // No constraints, pick any available qubit
            available.first().copied()
        } else {
            // Find qubit minimizing distance to placed neighbors
            available.iter()
                .min_by_key(|&&q| {
                    let total_dist: usize = placed_neighbors.iter()
                        .map(|&&nq| self.qubit_distance(q, nq, hardware))
                        .sum();
                    total_dist
                })
                .copied()
        }
    }

    /// Find central qubit in available set
    fn find_central_qubit(&self, available: &HashSet<usize>, hardware: &HardwareGraph) -> usize {
        // Choose qubit with maximum connectivity
        available.iter()
            .max_by_key(|&&q| hardware.neighbors(q).len())
            .copied()
            .unwrap_or(0)
    }

    /// Find closest qubit to a set of target qubits
    fn find_closest_qubit(
        &self,
        targets: &[usize],
        available: &HashSet<usize>,
        hardware: &HardwareGraph,
    ) -> IsingResult<usize> {
        available.iter()
            .min_by_key(|&&q| {
                targets.iter()
                    .map(|&t| self.qubit_distance(q, t, hardware))
                    .sum::<usize>()
            })
            .copied()
            .ok_or_else(|| IsingError::HardwareConstraint("No available qubits".to_string()))
    }

    /// Compute distance between two qubits
    fn qubit_distance(&self, q1: usize, q2: usize, hardware: &HardwareGraph) -> usize {
        if let Some(coords) = &self.qubit_coordinates {
            // Use Euclidean distance if coordinates available
            if let (Some(c1), Some(c2)) = (coords.get(&q1), coords.get(&q2)) {
                let dist_sq: f64 = c1.iter().zip(c2.iter())
                    .map(|(x1, x2)| (x1 - x2).powi(2))
                    .sum();
                return dist_sq.sqrt() as usize;
            }
        }

        // Fall back to graph distance
        self.graph_distance(q1, q2, hardware)
    }

    /// Compute graph distance between qubits
    fn graph_distance(&self, q1: usize, q2: usize, hardware: &HardwareGraph) -> usize {
        if q1 == q2 {
            return 0;
        }

        // BFS to find shortest path
        let mut visited = HashSet::new();
        let mut queue = vec![(q1, 0)];
        visited.insert(q1);

        while let Some((current, dist)) = queue.pop() {
            for neighbor in hardware.neighbors(current) {
                if neighbor == q2 {
                    return dist + 1;
                }
                if visited.insert(neighbor) {
                    queue.push((neighbor, dist + 1));
                }
            }
        }

        usize::MAX // No path found
    }

    /// Check if two chains are connected
    fn chains_connected(&self, chain1: &[usize], chain2: &[usize], hardware: &HardwareGraph) -> bool {
        for &q1 in chain1 {
            for &q2 in chain2 {
                if hardware.are_connected(q1, q2) {
                    return true;
                }
            }
        }
        false
    }

    /// Find a path to connect two chains
    fn find_connection_path(
        &mut self,
        chain1: &[usize],
        chain2: &[usize],
        hardware: &HardwareGraph,
        used_qubits: &HashSet<usize>,
    ) -> IsingResult<Vec<usize>> {
        // A* search for shortest path
        let mut best_path = None;
        let mut best_cost = usize::MAX;

        for &start in chain1 {
            for &goal in chain2 {
                if let Some(path) = self.astar_path(start, goal, hardware, used_qubits) {
                    if path.len() < best_cost {
                        best_cost = path.len();
                        best_path = Some(path);
                    }
                }
            }
        }

        best_path.ok_or_else(|| IsingError::HardwareConstraint("Cannot connect chains".to_string()))
    }

    /// A* pathfinding algorithm
    fn astar_path(
        &self,
        start: usize,
        goal: usize,
        hardware: &HardwareGraph,
        used_qubits: &HashSet<usize>,
    ) -> Option<Vec<usize>> {
        #[derive(Eq, PartialEq)]
        struct State {
            cost: usize,
            position: usize,
        }

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other.cost.cmp(&self.cost)
            }
        }

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        let mut dist = HashMap::new();
        let mut parent = HashMap::new();

        heap.push(State { cost: 0, position: start });
        dist.insert(start, 0);

        while let Some(State { cost, position }) = heap.pop() {
            if position == goal {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current = goal;
                while current != start {
                    if !used_qubits.contains(&current) {
                        path.push(current);
                    }
                    current = parent[&current];
                }
                path.reverse();
                return Some(path);
            }

            if cost > dist[&position] {
                continue;
            }

            for neighbor in hardware.neighbors(position) {
                let next_cost = cost + 1;
                
                if next_cost < *dist.get(&neighbor).unwrap_or(&usize::MAX) {
                    dist.insert(neighbor, next_cost);
                    parent.insert(neighbor, position);
                    let heuristic = self.qubit_distance(neighbor, goal, hardware);
                    heap.push(State {
                        cost: next_cost + heuristic,
                        position: neighbor,
                    });
                }
            }
        }

        None
    }

    /// Extend a chain with additional qubits
    fn extend_chain(
        &self,
        embedding: &mut Embedding,
        var: usize,
        extension: &[usize],
        used_qubits: &mut HashSet<usize>,
    ) -> IsingResult<()> {
        if let Some(chain) = embedding.chains.get_mut(&var) {
            for &qubit in extension {
                if !used_qubits.contains(&qubit) {
                    chain.push(qubit);
                    used_qubits.insert(qubit);
                    embedding.qubit_to_variable.insert(qubit, var);
                }
            }
        }
        Ok(())
    }

    /// Find a shorter chain for a variable
    fn find_shorter_chain(
        &self,
        var: usize,
        current_chain: &[usize],
        embedding: &Embedding,
        logical_edges: &[(usize, usize)],
        hardware: &HardwareGraph,
    ) -> Option<Vec<usize>> {
        // Find required connections
        let neighbors = self.get_logical_neighbors(var, logical_edges);
        let neighbor_chains: Vec<_> = neighbors.iter()
            .filter_map(|&n| embedding.chains.get(&n))
            .collect();

        // Try to find a shorter path that maintains all connections
        // This is a simplified version - full implementation would be more sophisticated
        None
    }

    /// Try a local swap to improve embedding
    fn try_local_swap(
        &self,
        embedding: &mut Embedding,
        var: usize,
        logical_edges: &[(usize, usize)],
        hardware: &HardwareGraph,
    ) -> IsingResult<bool> {
        // Simplified local search - in practice would try various swaps
        Ok(false)
    }

    /// Get logical neighbors of a variable
    fn get_logical_neighbors(&self, var: usize, logical_edges: &[(usize, usize)]) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for &(u, v) in logical_edges {
            if u == var {
                neighbors.push(v);
            } else if v == var {
                neighbors.push(u);
            }
        }
        neighbors
    }

    /// Order variables by connectivity for placement
    fn order_variables_by_connectivity(
        &self,
        logical_edges: &[(usize, usize)],
        num_vars: usize,
    ) -> Vec<usize> {
        let mut degrees: Vec<_> = (0..num_vars)
            .map(|v| {
                let degree = logical_edges.iter()
                    .filter(|&&(u, w)| u == v || w == v)
                    .count();
                (v, degree)
            })
            .collect();
        
        degrees.sort_by_key(|&(_, d)| std::cmp::Reverse(d));
        degrees.into_iter().map(|(v, _)| v).collect()
    }

    /// Compute embedding statistics
    fn compute_stats(&self, embedding: &Embedding) -> LayoutStats {
        let chain_lengths: Vec<_> = embedding.chains.values()
            .map(|chain| chain.len())
            .collect();

        let total_length: usize = chain_lengths.iter().sum();
        let max_length = chain_lengths.iter().copied().max().unwrap_or(0);
        let avg_length = if chain_lengths.is_empty() {
            0.0
        } else {
            total_length as f64 / chain_lengths.len() as f64
        };

        let long_chains = chain_lengths.iter()
            .filter(|&&len| len > self.config.max_chain_length)
            .count();

        // Simple quality score (lower is better)
        let quality_score = avg_length * self.config.chain_length_weight
            + long_chains as f64 * 10.0;

        LayoutStats {
            avg_chain_length: avg_length,
            max_chain_length: max_length,
            total_chain_length: total_length,
            long_chains,
            quality_score,
        }
    }
}

/// Multi-level embedding for hierarchical problems
pub struct MultiLevelEmbedder {
    /// Base embedder for each level
    base_embedder: LayoutAwareEmbedder,
    /// Number of coarsening levels
    num_levels: usize,
    /// Coarsening ratio at each level
    coarsening_ratio: f64,
}

impl MultiLevelEmbedder {
    /// Create a new multi-level embedder
    pub fn new(config: LayoutConfig, num_levels: usize) -> Self {
        Self {
            base_embedder: LayoutAwareEmbedder::new(config),
            num_levels,
            coarsening_ratio: 0.5,
        }
    }

    /// Find embedding using multi-level approach
    pub fn find_embedding(
        &mut self,
        logical_edges: &[(usize, usize)],
        num_vars: usize,
        hardware: &HardwareGraph,
    ) -> IsingResult<(Embedding, LayoutStats)> {
        // Coarsen the logical graph
        let coarsened_graphs = self.coarsen_graph(logical_edges, num_vars);
        
        // Embed coarsest graph
        let (mut embedding, _) = self.base_embedder.find_embedding(
            &coarsened_graphs.last().unwrap().0,
            coarsened_graphs.last().unwrap().1,
            hardware,
        )?;

        // Refine through levels
        for level in (0..coarsened_graphs.len() - 1).rev() {
            embedding = self.refine_level(
                embedding,
                &coarsened_graphs[level],
                &coarsened_graphs[level + 1],
                hardware,
            )?;
        }

        let stats = self.base_embedder.compute_stats(&embedding);
        Ok((embedding, stats))
    }

    /// Coarsen the logical graph into multiple levels
    fn coarsen_graph(
        &self,
        logical_edges: &[(usize, usize)],
        num_vars: usize,
    ) -> Vec<(Vec<(usize, usize)>, usize)> {
        // Simplified coarsening - in practice would use graph clustering
        vec![(logical_edges.to_vec(), num_vars)]
    }

    /// Refine embedding from coarse to fine level
    fn refine_level(
        &self,
        coarse_embedding: Embedding,
        fine_graph: &(Vec<(usize, usize)>, usize),
        coarse_graph: &(Vec<(usize, usize)>, usize),
        hardware: &HardwareGraph,
    ) -> IsingResult<Embedding> {
        // Simplified refinement - in practice would uncoarsen properly
        Ok(coarse_embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_embedder_creation() {
        let config = LayoutConfig::default();
        let embedder = LayoutAwareEmbedder::new(config);
        assert!(embedder.shortest_paths.is_empty());
    }

    #[test]
    fn test_qubit_distance() {
        let config = LayoutConfig::default();
        let embedder = LayoutAwareEmbedder::new(config);
        
        let hardware = HardwareGraph::new_chimera(2, 2, 2).unwrap();
        
        // Same qubit
        assert_eq!(embedder.qubit_distance(0, 0, &hardware), 0);
        
        // Adjacent qubits
        let dist = embedder.qubit_distance(0, 2, &hardware);
        assert!(dist > 0);
    }
}