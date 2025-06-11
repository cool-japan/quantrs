//! Qubit Routing Algorithms for Quantum Hardware
//!
//! This module implements advanced qubit routing algorithms using SciRS2
//! optimization techniques to map logical qubits to physical qubits.

use crate::topology::HardwareTopology;
use crate::{DeviceError, DeviceResult};
use petgraph::algo::astar;
use petgraph::graph::{NodeIndex, UnGraph};
use quantrs2_circuit::prelude::*;
use std::collections::{HashMap, HashSet};

/// Routing strategy for qubit mapping
#[derive(Debug, Clone, Copy)]
pub enum RoutingStrategy {
    /// Basic nearest-neighbor mapping
    NearestNeighbor,
    /// Steiner tree based routing
    SteinerTree,
    /// Lookahead-based routing
    Lookahead { depth: usize },
    /// Stochastic routing with simulated annealing
    StochasticAnnealing,
}

/// Result of qubit routing
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Initial qubit mapping (logical -> physical)
    pub initial_mapping: HashMap<usize, usize>,
    /// Final qubit mapping after all swaps
    pub final_mapping: HashMap<usize, usize>,
    /// List of SWAP gates to insert
    pub swap_gates: Vec<SwapGate>,
    /// Total routing cost (number of SWAPs)
    pub cost: usize,
    /// Circuit depth increase
    pub depth_overhead: usize,
}

/// SWAP gate information
#[derive(Debug, Clone)]
pub struct SwapGate {
    /// Physical qubit indices
    pub qubit1: usize,
    pub qubit2: usize,
    /// Position in circuit where SWAP should be inserted
    pub position: usize,
}

/// Qubit router using SciRS2 optimization
pub struct QubitRouter {
    /// Hardware topology
    topology: HardwareTopology,
    /// Routing strategy
    strategy: RoutingStrategy,
    /// Random seed for stochastic methods
    seed: u64,
}

impl QubitRouter {
    /// Create a new qubit router
    pub fn new(topology: HardwareTopology, strategy: RoutingStrategy) -> Self {
        Self {
            topology,
            strategy,
            seed: 42,
        }
    }

    /// Route a quantum circuit to hardware topology
    pub fn route_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<RoutingResult> {
        // Extract two-qubit gate interactions
        let interactions = self.extract_interactions(circuit)?;

        // Find initial mapping
        let initial_mapping = self.find_initial_mapping(&interactions, N)?;

        // Route based on strategy
        match self.strategy {
            RoutingStrategy::NearestNeighbor => {
                self.route_nearest_neighbor(circuit, initial_mapping, interactions)
            }
            RoutingStrategy::SteinerTree => {
                self.route_steiner_tree(circuit, initial_mapping, interactions)
            }
            RoutingStrategy::Lookahead { depth } => {
                self.route_lookahead(circuit, initial_mapping, interactions, depth)
            }
            RoutingStrategy::StochasticAnnealing => {
                self.route_simulated_annealing(circuit, initial_mapping, interactions)
            }
        }
    }

    /// Extract two-qubit interactions from circuit
    fn extract_interactions<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<Vec<(usize, usize)>> {
        let mut interactions = Vec::new();

        for gate in circuit.gates() {
            if gate.qubits().len() == 2 {
                let q0 = gate.qubits()[0].id() as usize;
                let q1 = gate.qubits()[1].id() as usize;
                interactions.push((q0, q1));
            }
        }

        Ok(interactions)
    }

    /// Find initial qubit mapping using graph algorithms
    fn find_initial_mapping(
        &self,
        interactions: &[(usize, usize)],
        num_logical_qubits: usize,
    ) -> DeviceResult<HashMap<usize, usize>> {
        if num_logical_qubits > self.topology.num_qubits {
            return Err(DeviceError::InsufficientQubits {
                required: num_logical_qubits,
                available: self.topology.num_qubits,
            });
        }

        // Build interaction graph
        let mut interaction_graph = UnGraph::<(), ()>::new_undirected();
        let mut nodes = HashMap::new();

        for i in 0..num_logical_qubits {
            let node = interaction_graph.add_node(());
            nodes.insert(i, node);
        }

        for &(q0, q1) in interactions {
            if let (Some(&n0), Some(&n1)) = (nodes.get(&q0), nodes.get(&q1)) {
                interaction_graph.add_edge(n0, n1, ());
            }
        }

        // Use spectral graph partitioning for initial mapping
        // For now, use a simple heuristic
        let mut mapping = HashMap::new();
        let _available_physical: Vec<usize> = (0..self.topology.num_qubits).collect();

        // Map most connected logical qubits to most connected physical qubits
        let mut logical_degrees: Vec<(usize, usize)> = nodes
            .iter()
            .map(|(&log_q, &node)| {
                let degree = interaction_graph.edges(node).count();
                (log_q, degree)
            })
            .collect();
        logical_degrees.sort_by_key(|&(_, deg)| std::cmp::Reverse(deg));

        let mut physical_degrees: Vec<(usize, usize)> = (0..self.topology.num_qubits)
            .map(|p| {
                let node = NodeIndex::new(p);
                let degree = self.topology.connectivity.edges(node).count();
                (p, degree)
            })
            .collect();
        physical_degrees.sort_by_key(|&(_, deg)| std::cmp::Reverse(deg));

        for (i, &(log_q, _)) in logical_degrees.iter().enumerate() {
            if i < physical_degrees.len() {
                mapping.insert(log_q, physical_degrees[i].0);
            }
        }

        Ok(mapping)
    }

    /// Nearest-neighbor routing algorithm
    fn route_nearest_neighbor<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
        initial_mapping: HashMap<usize, usize>,
        interactions: Vec<(usize, usize)>,
    ) -> DeviceResult<RoutingResult> {
        let mut current_mapping = initial_mapping.clone();
        let mut swap_gates = Vec::new();
        let mut position = 0;

        for (log_q0, log_q1) in interactions {
            let phys_q0 = current_mapping[&log_q0];
            let phys_q1 = current_mapping[&log_q1];

            // Check if qubits are connected
            if !self.are_connected(phys_q0, phys_q1)? {
                // Find shortest path and insert SWAPs
                let path = self.find_shortest_path(phys_q0, phys_q1)?;

                // Insert SWAPs along the path
                for i in 0..path.len() - 2 {
                    swap_gates.push(SwapGate {
                        qubit1: path[i],
                        qubit2: path[i + 1],
                        position,
                    });

                    // Update mapping
                    self.apply_swap(&mut current_mapping, path[i], path[i + 1]);
                }
            }

            position += 1;
        }

        Ok(RoutingResult {
            initial_mapping,
            final_mapping: current_mapping,
            cost: swap_gates.len(),
            depth_overhead: swap_gates.len() * 3, // Each SWAP is ~3 CNOTs
            swap_gates,
        })
    }

    /// Steiner tree based routing
    fn route_steiner_tree<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
        initial_mapping: HashMap<usize, usize>,
        interactions: Vec<(usize, usize)>,
    ) -> DeviceResult<RoutingResult> {
        // Implement Steiner tree approximation for multi-qubit gates
        // For now, fallback to nearest neighbor
        self.route_nearest_neighbor(_circuit, initial_mapping, interactions)
    }

    /// Lookahead routing with configurable depth
    fn route_lookahead<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        initial_mapping: HashMap<usize, usize>,
        interactions: Vec<(usize, usize)>,
        lookahead_depth: usize,
    ) -> DeviceResult<RoutingResult> {
        let mut current_mapping = initial_mapping.clone();
        let mut swap_gates = Vec::new();
        let mut position = 0;

        for i in 0..interactions.len() {
            let (log_q0, log_q1) = interactions[i];
            let phys_q0 = current_mapping[&log_q0];
            let phys_q1 = current_mapping[&log_q1];

            if !self.are_connected(phys_q0, phys_q1)? {
                // Look ahead at future gates
                let future_gates =
                    &interactions[i..std::cmp::min(i + lookahead_depth, interactions.len())];

                // Find best SWAP considering future gates
                let best_swap = self.find_best_swap_lookahead(
                    &current_mapping,
                    phys_q0,
                    phys_q1,
                    future_gates,
                )?;

                if let Some((swap_q0, swap_q1)) = best_swap {
                    swap_gates.push(SwapGate {
                        qubit1: swap_q0,
                        qubit2: swap_q1,
                        position,
                    });

                    self.apply_swap(&mut current_mapping, swap_q0, swap_q1);
                }
            }

            position += 1;
        }

        Ok(RoutingResult {
            initial_mapping,
            final_mapping: current_mapping,
            cost: swap_gates.len(),
            depth_overhead: swap_gates.len() * 3,
            swap_gates,
        })
    }

    /// Simulated annealing based routing
    fn route_simulated_annealing<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        initial_mapping: HashMap<usize, usize>,
        interactions: Vec<(usize, usize)>,
    ) -> DeviceResult<RoutingResult> {
        use fastrand::Rng;
        let mut rng = Rng::with_seed(self.seed);

        let mut best_mapping = initial_mapping.clone();
        let mut best_cost = self.evaluate_mapping(&best_mapping, &interactions)?;

        let mut current_mapping = best_mapping.clone();
        let mut current_cost = best_cost;

        // Annealing parameters
        let mut temperature = 100.0;
        let cooling_rate = 0.95;
        let min_temperature = 0.01;

        while temperature > min_temperature {
            // Generate neighbor by random swap
            let mut neighbor_mapping = current_mapping.clone();
            let logical_qubits: Vec<usize> = neighbor_mapping.keys().copied().collect();

            if logical_qubits.len() >= 2 {
                let idx1 = rng.usize(0..logical_qubits.len());
                let idx2 = rng.usize(0..logical_qubits.len());

                if idx1 != idx2 {
                    let log_q1 = logical_qubits[idx1];
                    let log_q2 = logical_qubits[idx2];

                    let phys_q1 = neighbor_mapping[&log_q1];
                    let phys_q2 = neighbor_mapping[&log_q2];

                    neighbor_mapping.insert(log_q1, phys_q2);
                    neighbor_mapping.insert(log_q2, phys_q1);
                }
            }

            let neighbor_cost = self.evaluate_mapping(&neighbor_mapping, &interactions)?;
            let delta = neighbor_cost as f64 - current_cost as f64;

            // Accept or reject
            if delta < 0.0 || rng.f64() < (-delta / temperature).exp() {
                current_mapping = neighbor_mapping;
                current_cost = neighbor_cost;

                if current_cost < best_cost {
                    best_mapping = current_mapping.clone();
                    best_cost = current_cost;
                }
            }

            temperature *= cooling_rate;
        }

        // Now route with the best mapping found
        self.route_nearest_neighbor(circuit, best_mapping, interactions)
    }

    /// Check if two physical qubits are connected
    fn are_connected(&self, phys_q0: usize, phys_q1: usize) -> DeviceResult<bool> {
        let n0 = NodeIndex::new(phys_q0);
        let n1 = NodeIndex::new(phys_q1);
        Ok(self.topology.connectivity.find_edge(n0, n1).is_some())
    }

    /// Find shortest path between two physical qubits
    fn find_shortest_path(&self, start: usize, end: usize) -> DeviceResult<Vec<usize>> {
        let start_node = NodeIndex::new(start);
        let end_node = NodeIndex::new(end);

        let result = astar(
            &self.topology.connectivity,
            start_node,
            |n| n == end_node,
            |e| *e.weight(),
            |_| 0.0,
        );

        match result {
            Some((_, path)) => Ok(path.into_iter().map(|n| n.index() as usize).collect()),
            None => Err(DeviceError::RoutingError(format!(
                "No path found between qubits {} and {}",
                start, end
            ))),
        }
    }

    /// Apply a SWAP to the mapping
    fn apply_swap(&self, mapping: &mut HashMap<usize, usize>, phys_q0: usize, phys_q1: usize) {
        // Find logical qubits mapped to these physical qubits
        let mut log_q0 = None;
        let mut log_q1 = None;

        for (&log_q, &phys_q) in mapping.iter() {
            if phys_q == phys_q0 {
                log_q0 = Some(log_q);
            } else if phys_q == phys_q1 {
                log_q1 = Some(log_q);
            }
        }

        // Swap the mappings
        if let (Some(l0), Some(l1)) = (log_q0, log_q1) {
            mapping.insert(l0, phys_q1);
            mapping.insert(l1, phys_q0);
        }
    }

    /// Find best swap considering future gates
    fn find_best_swap_lookahead(
        &self,
        mapping: &HashMap<usize, usize>,
        target_phys_q0: usize,
        target_phys_q1: usize,
        future_gates: &[(usize, usize)],
    ) -> DeviceResult<Option<(usize, usize)>> {
        let mut best_swap = None;
        let mut best_score = f64::MAX;

        // Try all possible swaps
        for edge in self.topology.connectivity.edge_indices() {
            if let Some((n0, n1)) = self.topology.connectivity.edge_endpoints(edge) {
                let phys_q0 = n0.index() as usize;
                let phys_q1 = n1.index() as usize;

                // Simulate this swap
                let mut test_mapping = mapping.clone();
                self.apply_swap(&mut test_mapping, phys_q0, phys_q1);

                // Evaluate the swap
                let score = self.evaluate_swap_lookahead(
                    &test_mapping,
                    target_phys_q0,
                    target_phys_q1,
                    future_gates,
                )?;

                if score < best_score {
                    best_score = score;
                    best_swap = Some((phys_q0, phys_q1));
                }
            }
        }

        Ok(best_swap)
    }

    /// Evaluate a swap considering future gates
    fn evaluate_swap_lookahead(
        &self,
        mapping: &HashMap<usize, usize>,
        target_phys_q0: usize,
        target_phys_q1: usize,
        future_gates: &[(usize, usize)],
    ) -> DeviceResult<f64> {
        let mut score = 0.0;

        // Check if target gate is now executable
        if self.are_connected(target_phys_q0, target_phys_q1)? {
            score -= 10.0; // Bonus for making target executable
        }

        // Evaluate future gates
        for (i, &(log_q0, log_q1)) in future_gates.iter().enumerate() {
            if let (Some(&phys_q0), Some(&phys_q1)) = (mapping.get(&log_q0), mapping.get(&log_q1)) {
                if self.are_connected(phys_q0, phys_q1)? {
                    score -= 1.0 / (i + 1) as f64; // Decreasing weight for future gates
                } else {
                    let path = self.find_shortest_path(phys_q0, phys_q1)?;
                    score += path.len() as f64 / (i + 1) as f64;
                }
            }
        }

        Ok(score)
    }

    /// Evaluate total cost of a mapping
    fn evaluate_mapping(
        &self,
        mapping: &HashMap<usize, usize>,
        interactions: &[(usize, usize)],
    ) -> DeviceResult<usize> {
        let mut total_cost = 0;

        for &(log_q0, log_q1) in interactions {
            if let (Some(&phys_q0), Some(&phys_q1)) = (mapping.get(&log_q0), mapping.get(&log_q1)) {
                if !self.are_connected(phys_q0, phys_q1)? {
                    let path = self.find_shortest_path(phys_q0, phys_q1)?;
                    total_cost += path.len() - 1; // Number of swaps needed
                }
            }
        }

        Ok(total_cost)
    }
}

/// Layout synthesis for initial qubit placement
pub struct LayoutSynthesis {
    /// Hardware topology
    topology: HardwareTopology,
}

impl LayoutSynthesis {
    /// Create a new layout synthesizer
    pub fn new(topology: HardwareTopology) -> Self {
        Self { topology }
    }

    /// Synthesize optimal initial layout using SciRS2 optimization
    pub fn synthesize_layout(
        &self,
        interaction_graph: &UnGraph<(), f64>,
    ) -> DeviceResult<HashMap<usize, usize>> {
        // Use spectral placement algorithm
        // For now, use degree-based heuristic
        let mut mapping = HashMap::new();

        // Sort logical qubits by degree
        let mut logical_degrees: Vec<(usize, usize)> = interaction_graph
            .node_indices()
            .map(|n| {
                let degree = interaction_graph.edges(n).count();
                (n.index(), degree)
            })
            .collect();
        logical_degrees.sort_by_key(|&(_, deg)| std::cmp::Reverse(deg));

        // Sort physical qubits by connectivity
        let mut physical_degrees: Vec<(usize, usize)> = (0..self.topology.num_qubits)
            .map(|p| {
                let node = NodeIndex::new(p);
                let degree = self.topology.connectivity.edges(node).count();
                (p, degree)
            })
            .collect();
        physical_degrees.sort_by_key(|&(_, deg)| std::cmp::Reverse(deg));

        // Map high-degree logical to high-degree physical
        for (i, &(log_q, _)) in logical_degrees.iter().enumerate() {
            if i < physical_degrees.len() {
                mapping.insert(log_q, physical_degrees[i].0);
            }
        }

        Ok(mapping)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::prelude::QubitId;

    #[test]
    fn test_qubit_router_creation() {
        let topology = HardwareTopology::linear_topology(5);
        let router = QubitRouter::new(topology, RoutingStrategy::NearestNeighbor);
        assert_eq!(router.topology.num_qubits, 5);
    }

    #[test]
    fn test_routing_strategies() {
        let topology = HardwareTopology::grid_topology(3, 3);

        // Test different strategies
        let strategies = vec![
            RoutingStrategy::NearestNeighbor,
            RoutingStrategy::SteinerTree,
            RoutingStrategy::Lookahead { depth: 3 },
            RoutingStrategy::StochasticAnnealing,
        ];

        for strategy in strategies {
            let router = QubitRouter::new(topology.clone(), strategy);
            assert_eq!(router.topology.num_qubits, 9);
        }
    }

    #[test]
    fn test_swap_application() {
        let topology = HardwareTopology::linear_topology(4);
        let router = QubitRouter::new(topology, RoutingStrategy::NearestNeighbor);

        let mut mapping = HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]);

        router.apply_swap(&mut mapping, 1, 2);

        assert_eq!(mapping[&1], 2);
        assert_eq!(mapping[&2], 1);
    }

    #[test]
    fn test_linear_topology_routing() {
        // Create a linear topology: 0-1-2-3-4
        let topology = HardwareTopology::linear_topology(5);
        let router = QubitRouter::new(topology, RoutingStrategy::NearestNeighbor);

        // Create a circuit that requires routing
        let mut circuit = Circuit::<5>::new();
        circuit.h(QubitId::new(0)).unwrap();
        circuit.cnot(QubitId::new(0), QubitId::new(4)).unwrap(); // This requires routing
        circuit.cnot(QubitId::new(1), QubitId::new(3)).unwrap(); // This also requires routing

        let result = router.route_circuit(&circuit).unwrap();

        // Check that we have a valid result
        // The initial mapping might be optimal, so swaps might not be needed
        // Note: cost is usize, so it's always >= 0

        // Verify initial and final mappings exist
        assert_eq!(result.initial_mapping.len(), 5);
        assert_eq!(result.final_mapping.len(), 5);
    }

    #[test]
    fn test_grid_topology_routing() {
        // Create a 3x3 grid topology
        let topology = HardwareTopology::grid_topology(3, 3);
        let router = QubitRouter::new(topology, RoutingStrategy::Lookahead { depth: 3 });

        // Create a circuit with non-local interactions
        let mut circuit = Circuit::<9>::new();
        circuit.h(QubitId::new(0)).unwrap();
        circuit.cnot(QubitId::new(0), QubitId::new(8)).unwrap(); // Corner to corner
        circuit.cnot(QubitId::new(4), QubitId::new(2)).unwrap(); // Center to edge

        let result = router.route_circuit(&circuit).unwrap();

        // Should require swaps for non-adjacent qubits
        assert!(result.swap_gates.len() > 0);
    }

    #[test]
    fn test_heavy_hex_routing() {
        // Test on IBM's heavy-hex topology
        let topology = HardwareTopology::from_heavy_hex(27);
        let router = QubitRouter::new(topology, RoutingStrategy::StochasticAnnealing);

        let mut circuit = Circuit::<10>::new();
        // Create a random circuit
        circuit.h(QubitId::new(0)).unwrap();
        circuit.cnot(QubitId::new(0), QubitId::new(5)).unwrap();
        circuit.cnot(QubitId::new(2), QubitId::new(7)).unwrap();
        circuit.cnot(QubitId::new(3), QubitId::new(9)).unwrap();

        let result = router.route_circuit(&circuit).unwrap();

        // Should successfully route
        assert!(result.initial_mapping.len() <= 27);
    }

    #[test]
    fn test_insufficient_qubits() {
        let topology = HardwareTopology::linear_topology(3);
        let router = QubitRouter::new(topology, RoutingStrategy::NearestNeighbor);

        // Try to route a circuit with more logical qubits than physical
        let circuit = Circuit::<5>::new();

        let result = router.route_circuit(&circuit);
        assert!(result.is_err());
    }

    #[test]
    fn test_different_strategies_comparison() {
        let topology = HardwareTopology::grid_topology(4, 4);

        // Create a challenging circuit
        let mut circuit = Circuit::<16>::new();
        for i in 0..8 {
            circuit.h(QubitId::new(i as u32)).unwrap();
            circuit
                .cnot(QubitId::new(i as u32), QubitId::new((i + 8) as u32))
                .unwrap();
        }

        // Test all strategies
        let strategies = vec![
            RoutingStrategy::NearestNeighbor,
            RoutingStrategy::Lookahead { depth: 3 },
            RoutingStrategy::Lookahead { depth: 5 },
            RoutingStrategy::StochasticAnnealing,
        ];

        let mut results = Vec::new();
        for strategy in strategies {
            let router = QubitRouter::new(topology.clone(), strategy);
            let result = router.route_circuit(&circuit).unwrap();
            results.push((strategy, result.cost));
        }

        // All strategies should produce valid results
        // Note: cost is usize, so it's always >= 0
    }

    #[test]
    fn test_already_connected_qubits() {
        let topology = HardwareTopology::linear_topology(3);
        let router = QubitRouter::new(topology, RoutingStrategy::NearestNeighbor);

        // Circuit with already connected qubits
        let mut circuit = Circuit::<3>::new();
        circuit.cnot(QubitId::new(0), QubitId::new(1)).unwrap(); // Adjacent
        circuit.cnot(QubitId::new(1), QubitId::new(2)).unwrap(); // Adjacent

        let result = router.route_circuit(&circuit).unwrap();

        // Should require no swaps
        assert_eq!(result.swap_gates.len(), 0);
        assert_eq!(result.cost, 0);
    }

    #[test]
    fn test_layout_synthesis() {
        let topology = HardwareTopology::grid_topology(3, 3);
        let synthesizer = LayoutSynthesis::new(topology);

        // Create interaction graph
        let mut graph = UnGraph::<(), f64>::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());

        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n1, n2, 1.0);
        graph.add_edge(n2, n3, 1.0);
        graph.add_edge(n3, n0, 1.0); // Square

        let layout = synthesizer.synthesize_layout(&graph).unwrap();

        // Should map all 4 qubits
        assert_eq!(layout.len(), 4);

        // All mappings should be to different physical qubits
        let physical_qubits: HashSet<usize> = layout.values().copied().collect();
        assert_eq!(physical_qubits.len(), 4);
    }

    #[test]
    fn test_path_finding() {
        let topology = HardwareTopology::linear_topology(5);
        let router = QubitRouter::new(topology, RoutingStrategy::NearestNeighbor);

        // Test path from 0 to 4
        let path = router.find_shortest_path(0, 4).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3, 4]);

        // Test path from 2 to 0
        let path = router.find_shortest_path(2, 0).unwrap();
        assert_eq!(path, vec![2, 1, 0]);
    }

    #[test]
    fn test_connectivity_check() {
        let topology = HardwareTopology::grid_topology(3, 3);
        let router = QubitRouter::new(topology, RoutingStrategy::NearestNeighbor);

        // Adjacent qubits in grid should be connected
        assert!(router.are_connected(0, 1).unwrap()); // Horizontal
        assert!(router.are_connected(0, 3).unwrap()); // Vertical

        // Diagonal qubits should not be connected
        assert!(!router.are_connected(0, 4).unwrap());

        // Far qubits should not be connected
        assert!(!router.are_connected(0, 8).unwrap());
    }
}
