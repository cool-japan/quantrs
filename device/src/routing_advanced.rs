//! Advanced qubit routing algorithms with SciRS2-style optimization.
//!
//! This module implements state-of-the-art routing algorithms including
//! SABRE, lookahead heuristics, and machine learning-guided routing.

use petgraph::visit::EdgeRef;
use scirs2_core::random::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::topology::HardwareTopology;
use crate::topology_analysis::{AllocationStrategy, TopologyAnalyzer};
use crate::{DeviceError, DeviceResult};
use quantrs2_circuit::prelude::*;

/// Advanced routing algorithms
#[derive(Debug, Clone, Copy)]
pub enum AdvancedRoutingStrategy {
    /// SABRE algorithm (Swap-based BidiREctional)
    SABRE { heuristic_weight: f64 },
    /// A* search with lookahead
    AStarLookahead { lookahead_depth: usize },
    /// Token swapping algorithm
    TokenSwapping,
    /// Hybrid approach combining multiple strategies
    Hybrid,
    /// Machine learning guided (placeholder)
    MLGuided,
}

/// Gate dependency information
#[derive(Debug, Clone)]
struct GateDependency {
    gate_id: usize,
    gate_type: String,
    qubits: Vec<usize>,
    predecessors: HashSet<usize>,
    successors: HashSet<usize>,
    scheduled: bool,
}

/// Routing state during algorithm execution
#[derive(Debug, Clone)]
struct RoutingState {
    /// Current qubit mapping (logical -> physical)
    mapping: HashMap<usize, usize>,
    /// Reverse mapping (physical -> logical)
    reverse_mapping: HashMap<usize, usize>,
    /// Scheduled gates
    scheduled_gates: HashSet<usize>,
    /// Current front layer of gates
    front_layer: HashSet<usize>,
    /// Total cost (number of swaps)
    cost: usize,
    /// Swap sequence
    swaps: Vec<SwapOperation>,
}

/// Swap operation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapOperation {
    /// Physical qubits to swap
    pub phys_qubit1: usize,
    pub phys_qubit2: usize,
    /// Logical qubits being swapped
    pub log_qubit1: Option<usize>,
    pub log_qubit2: Option<usize>,
    /// Position in the circuit
    pub position: usize,
    /// Cost of this swap
    pub cost: f64,
}

/// Extended routing result with detailed metrics
#[derive(Debug, Clone)]
pub struct AdvancedRoutingResult {
    /// Initial mapping
    pub initial_mapping: HashMap<usize, usize>,
    /// Final mapping
    pub final_mapping: HashMap<usize, usize>,
    /// Sequence of swap operations
    pub swap_sequence: Vec<SwapOperation>,
    /// Total routing cost
    pub total_cost: f64,
    /// Circuit depth overhead
    pub depth_overhead: usize,
    /// Number of additional gates
    pub gate_overhead: usize,
    /// Routing time (milliseconds)
    pub routing_time: u128,
    /// Detailed metrics
    pub metrics: RoutingMetrics,
}

/// Detailed routing metrics
#[derive(Debug, Clone)]
pub struct RoutingMetrics {
    /// Number of search iterations
    pub iterations: usize,
    /// States explored
    pub states_explored: usize,
    /// Average lookahead depth
    pub avg_lookahead: f64,
    /// Swap chain lengths
    pub swap_chain_lengths: Vec<usize>,
    /// Critical path length increase
    pub critical_path_increase: f64,
}

/// Advanced qubit router
pub struct AdvancedQubitRouter {
    topology: HardwareTopology,
    analyzer: TopologyAnalyzer,
    strategy: AdvancedRoutingStrategy,
    rng: StdRng,
}

impl AdvancedQubitRouter {
    /// Create a new advanced router
    pub fn new(topology: HardwareTopology, strategy: AdvancedRoutingStrategy, seed: u64) -> Self {
        let analyzer = TopologyAnalyzer::new(topology.clone());
        Self {
            topology,
            analyzer,
            strategy,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Route a circuit using the selected strategy
    pub fn route_circuit<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<AdvancedRoutingResult> {
        let start_time = std::time::Instant::now();

        // Build gate dependency graph
        let dependencies = self.build_dependency_graph(circuit)?;

        // Get initial mapping
        let initial_mapping = self.find_optimal_initial_mapping(&dependencies, N)?;

        // Route based on strategy
        let result = match self.strategy {
            AdvancedRoutingStrategy::SABRE { heuristic_weight } => {
                self.route_sabre(dependencies, initial_mapping.clone(), heuristic_weight)?
            }
            AdvancedRoutingStrategy::AStarLookahead { lookahead_depth } => {
                self.route_astar(dependencies, initial_mapping.clone(), lookahead_depth)?
            }
            AdvancedRoutingStrategy::TokenSwapping => {
                self.route_token_swapping(dependencies, initial_mapping.clone())?
            }
            AdvancedRoutingStrategy::Hybrid => {
                self.route_hybrid(dependencies, initial_mapping.clone())?
            }
            AdvancedRoutingStrategy::MLGuided => {
                // Placeholder - would use trained model
                self.route_sabre(dependencies, initial_mapping.clone(), 0.5)?
            }
        };

        let routing_time = start_time.elapsed().as_millis();

        Ok(AdvancedRoutingResult {
            initial_mapping,
            final_mapping: result.final_mapping,
            swap_sequence: result.swap_sequence,
            total_cost: result.total_cost,
            depth_overhead: result.depth_overhead,
            gate_overhead: result.gate_overhead,
            routing_time,
            metrics: result.metrics,
        })
    }

    /// Build dependency graph from circuit
    fn build_dependency_graph<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<Vec<GateDependency>> {
        let mut dependencies: Vec<GateDependency> = Vec::new();
        let mut last_gate_on_qubit: HashMap<usize, usize> = HashMap::new();

        for (gate_id, gate) in circuit.gates().iter().enumerate() {
            let qubits: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();
            let mut predecessors = HashSet::new();
            let mut successors = HashSet::new();

            // Find dependencies
            for &qubit in &qubits {
                if let Some(&pred_id) = last_gate_on_qubit.get(&qubit) {
                    predecessors.insert(pred_id);
                    dependencies[pred_id].successors.insert(gate_id);
                }
                last_gate_on_qubit.insert(qubit, gate_id);
            }

            dependencies.push(GateDependency {
                gate_id,
                gate_type: gate.name().to_string(),
                qubits,
                predecessors,
                successors,
                scheduled: false,
            });
        }

        Ok(dependencies)
    }

    /// Find optimal initial mapping using topology analysis
    fn find_optimal_initial_mapping(
        &mut self,
        dependencies: &[GateDependency],
        num_qubits: usize,
    ) -> DeviceResult<HashMap<usize, usize>> {
        // Count interactions between logical qubits
        let mut interaction_counts: HashMap<(usize, usize), usize> = HashMap::new();

        for dep in dependencies {
            if dep.qubits.len() == 2 {
                let (q1, q2) = (dep.qubits[0], dep.qubits[1]);
                let key = if q1 < q2 { (q1, q2) } else { (q2, q1) };
                *interaction_counts.entry(key).or_insert(0) += 1;
            }
        }

        // Use topology analyzer to allocate physical qubits
        let physical_qubits = self
            .analyzer
            .allocate_qubits(num_qubits, AllocationStrategy::Balanced)?;

        // Create interaction graph for logical qubits
        let mut logical_graph = petgraph::graph::UnGraph::<usize, usize>::new_undirected();
        let mut node_map = HashMap::new();

        for i in 0..num_qubits {
            let node = logical_graph.add_node(i);
            node_map.insert(i, node);
        }

        for ((q1, q2), &count) in &interaction_counts {
            if let (Some(&n1), Some(&n2)) = (node_map.get(q1), node_map.get(q2)) {
                logical_graph.add_edge(n1, n2, count);
            }
        }

        // Use graph matching to find good initial mapping
        self.graph_matching_mapping(&logical_graph, &physical_qubits)
    }

    /// Graph matching for initial mapping
    fn graph_matching_mapping(
        &self,
        logical_graph: &petgraph::graph::UnGraph<usize, usize>,
        physical_qubits: &[u32],
    ) -> DeviceResult<HashMap<usize, usize>> {
        let mut mapping = HashMap::new();
        let mut used_physical = HashSet::new();

        // Simple greedy matching - in practice, use more sophisticated algorithm
        for node in logical_graph.node_indices() {
            let logical_qubit = logical_graph[node];

            // Find best physical qubit
            let mut best_physical = None;
            let mut best_score = f64::NEG_INFINITY;

            for &phys in physical_qubits {
                if used_physical.contains(&phys) {
                    continue;
                }

                // Score based on matching neighborhoods
                let score = self.calculate_mapping_score(
                    logical_qubit,
                    phys as usize,
                    &mapping,
                    logical_graph,
                );

                if score > best_score {
                    best_score = score;
                    best_physical = Some(phys as usize);
                }
            }

            if let Some(phys) = best_physical {
                mapping.insert(logical_qubit, phys);
                used_physical.insert(phys as u32);
            }
        }

        Ok(mapping)
    }

    /// Calculate score for a potential mapping
    fn calculate_mapping_score(
        &self,
        logical_qubit: usize,
        physical_qubit: usize,
        current_mapping: &HashMap<usize, usize>,
        logical_graph: &petgraph::graph::UnGraph<usize, usize>,
    ) -> f64 {
        let mut score = 0.0;

        // Find logical neighbors
        if let Some(log_node) = logical_graph
            .node_indices()
            .find(|&n| logical_graph[n] == logical_qubit)
        {
            for neighbor in logical_graph.neighbors(log_node) {
                let neighbor_logical = logical_graph[neighbor];

                if let Some(&neighbor_physical) = current_mapping.get(&neighbor_logical) {
                    // Check if physical qubits are connected
                    if self.are_connected(physical_qubit, neighbor_physical) {
                        score += 10.0;
                    } else {
                        // Penalize by distance
                        let dist = self.get_distance(physical_qubit, neighbor_physical);
                        score -= dist as f64;
                    }
                }
            }
        }

        score
    }

    /// Check if two physical qubits are connected
    fn are_connected(&self, q1: usize, q2: usize) -> bool {
        self.topology
            .gate_properties
            .contains_key(&(q1 as u32, q2 as u32))
            || self
                .topology
                .gate_properties
                .contains_key(&(q2 as u32, q1 as u32))
    }

    /// Get distance between physical qubits
    fn get_distance(&self, q1: usize, q2: usize) -> usize {
        // Use shortest path - simplified version
        use petgraph::algo::dijkstra;

        if let (Some(n1), Some(n2)) = (
            self.topology
                .connectivity
                .node_indices()
                .find(|&n| self.topology.connectivity[n] == q1 as u32),
            self.topology
                .connectivity
                .node_indices()
                .find(|&n| self.topology.connectivity[n] == q2 as u32),
        ) {
            let result = dijkstra(&self.topology.connectivity, n1, Some(n2), |_| 1);

            result.get(&n2).copied().unwrap_or(usize::MAX)
        } else {
            usize::MAX
        }
    }

    /// SABRE routing algorithm
    fn route_sabre(
        &mut self,
        mut dependencies: Vec<GateDependency>,
        initial_mapping: HashMap<usize, usize>,
        heuristic_weight: f64,
    ) -> DeviceResult<InternalRoutingResult> {
        let mut state = RoutingState {
            mapping: initial_mapping.clone(),
            reverse_mapping: initial_mapping.iter().map(|(&k, &v)| (v, k)).collect(),
            scheduled_gates: HashSet::new(),
            front_layer: self.get_front_layer(&dependencies),
            cost: 0,
            swaps: Vec::new(),
        };

        let mut metrics = RoutingMetrics {
            iterations: 0,
            states_explored: 0,
            avg_lookahead: 0.0,
            swap_chain_lengths: Vec::new(),
            critical_path_increase: 0.0,
        };

        let mut position = 0;

        while state.scheduled_gates.len() < dependencies.len() {
            metrics.iterations += 1;

            // Get executable gates in front layer
            let executable = self.get_executable_gates(&state, &dependencies);

            if !executable.is_empty() {
                // Schedule executable gates
                for gate_id in executable {
                    state.scheduled_gates.insert(gate_id);
                    dependencies[gate_id].scheduled = true;
                }

                // Update front layer
                state.front_layer = self.update_front_layer(&state, &dependencies);
                position += 1;
            } else {
                // Need to insert swaps
                let best_swap = self.find_best_swap_sabre(
                    &state,
                    &dependencies,
                    heuristic_weight,
                    &mut metrics,
                )?;

                // Apply swap
                self.apply_swap(&mut state, &best_swap, position);
                metrics.swap_chain_lengths.push(1);
            }
        }

        Ok(InternalRoutingResult {
            final_mapping: state.mapping,
            swap_sequence: state.swaps,
            total_cost: state.cost as f64,
            depth_overhead: state.cost * 3, // Each swap is ~3 gates
            gate_overhead: state.cost * 3,
            metrics,
        })
    }

    /// Get front layer of gates
    fn get_front_layer(&self, dependencies: &[GateDependency]) -> HashSet<usize> {
        dependencies
            .iter()
            .filter(|dep| dep.predecessors.is_empty() && !dep.scheduled)
            .map(|dep| dep.gate_id)
            .collect()
    }

    /// Get executable gates from front layer
    fn get_executable_gates(
        &self,
        state: &RoutingState,
        dependencies: &[GateDependency],
    ) -> Vec<usize> {
        let mut executable = Vec::new();

        for &gate_id in &state.front_layer {
            let dep = &dependencies[gate_id];

            // Check if gate can be executed with current mapping
            if self.can_execute_gate(dep, &state.mapping) {
                executable.push(gate_id);
            }
        }

        executable
    }

    /// Check if a gate can be executed
    fn can_execute_gate(&self, dep: &GateDependency, mapping: &HashMap<usize, usize>) -> bool {
        if dep.qubits.len() == 1 {
            return true; // Single-qubit gates always executable
        }

        if dep.qubits.len() == 2 {
            let phys1 = mapping.get(&dep.qubits[0]).copied().unwrap_or(usize::MAX);
            let phys2 = mapping.get(&dep.qubits[1]).copied().unwrap_or(usize::MAX);

            return self.are_connected(phys1, phys2);
        }

        false // Multi-qubit gates need decomposition
    }

    /// Update front layer after scheduling gates
    fn update_front_layer(
        &self,
        state: &RoutingState,
        dependencies: &[GateDependency],
    ) -> HashSet<usize> {
        let mut new_front = HashSet::new();

        for (gate_id, dep) in dependencies.iter().enumerate() {
            if !dep.scheduled {
                // Check if all predecessors are scheduled
                if dep
                    .predecessors
                    .iter()
                    .all(|&pred| state.scheduled_gates.contains(&pred))
                {
                    new_front.insert(gate_id);
                }
            }
        }

        new_front
    }

    /// Find best swap using SABRE heuristic
    fn find_best_swap_sabre(
        &self,
        state: &RoutingState,
        dependencies: &[GateDependency],
        heuristic_weight: f64,
        metrics: &mut RoutingMetrics,
    ) -> DeviceResult<SwapOperation> {
        let mut best_swap = None;
        let mut best_score = f64::INFINITY;

        // Consider all possible swaps
        for edge in self.topology.connectivity.edge_references() {
            let phys1 = self.topology.connectivity[edge.source()] as usize;
            let phys2 = self.topology.connectivity[edge.target()] as usize;

            // Create temporary state with swap
            let mut temp_state = state.clone();
            let swap = SwapOperation {
                phys_qubit1: phys1,
                phys_qubit2: phys2,
                log_qubit1: temp_state.reverse_mapping.get(&phys1).copied(),
                log_qubit2: temp_state.reverse_mapping.get(&phys2).copied(),
                position: 0,
                cost: 1.0,
            };

            self.apply_swap(&mut temp_state, &swap, 0);

            // Calculate heuristic score
            let score = self.sabre_heuristic(
                &temp_state,
                dependencies,
                &state.front_layer,
                heuristic_weight,
            );

            if score < best_score {
                best_score = score;
                best_swap = Some(swap);
            }

            metrics.states_explored += 1;
        }

        best_swap.ok_or(DeviceError::RoutingError("No valid swap found".to_string()))
    }

    /// SABRE heuristic function
    fn sabre_heuristic(
        &self,
        state: &RoutingState,
        dependencies: &[GateDependency],
        front_layer: &HashSet<usize>,
        weight: f64,
    ) -> f64 {
        let mut score = 0.0;

        // H1: Number of gates in front layer that can be executed
        let executable_count = front_layer
            .iter()
            .filter(|&&gate_id| self.can_execute_gate(&dependencies[gate_id], &state.mapping))
            .count();
        score -= executable_count as f64;

        // H2: Sum of distances for gates in extended front layer
        let extended_layer = self.get_extended_front_layer(front_layer, dependencies);

        for &gate_id in &extended_layer {
            let dep = &dependencies[gate_id];
            if dep.qubits.len() == 2 {
                let phys1 = state.mapping.get(&dep.qubits[0]).copied().unwrap_or(0);
                let phys2 = state.mapping.get(&dep.qubits[1]).copied().unwrap_or(0);
                let dist = self.get_distance(phys1, phys2);
                score += weight * dist as f64;
            }
        }

        score
    }

    /// Get extended front layer for lookahead
    fn get_extended_front_layer(
        &self,
        front_layer: &HashSet<usize>,
        dependencies: &[GateDependency],
    ) -> HashSet<usize> {
        let mut extended = front_layer.clone();

        // Add immediate successors
        for &gate_id in front_layer {
            for &succ in &dependencies[gate_id].successors {
                extended.insert(succ);
            }
        }

        extended
    }

    /// Apply a swap to the state
    fn apply_swap(&self, state: &mut RoutingState, swap: &SwapOperation, position: usize) {
        // Update mappings
        if let (Some(log1), Some(log2)) = (swap.log_qubit1, swap.log_qubit2) {
            state.mapping.insert(log1, swap.phys_qubit2);
            state.mapping.insert(log2, swap.phys_qubit1);
            state.reverse_mapping.insert(swap.phys_qubit1, log2);
            state.reverse_mapping.insert(swap.phys_qubit2, log1);
        } else if let Some(log1) = swap.log_qubit1 {
            state.mapping.insert(log1, swap.phys_qubit2);
            state.reverse_mapping.remove(&swap.phys_qubit1);
            state.reverse_mapping.insert(swap.phys_qubit2, log1);
        } else if let Some(log2) = swap.log_qubit2 {
            state.mapping.insert(log2, swap.phys_qubit1);
            state.reverse_mapping.remove(&swap.phys_qubit2);
            state.reverse_mapping.insert(swap.phys_qubit1, log2);
        }

        // Update cost and swap list
        state.cost += 1;
        let mut swap_with_position = swap.clone();
        swap_with_position.position = position;
        state.swaps.push(swap_with_position);
    }

    /// A* routing with lookahead
    fn route_astar(
        &mut self,
        dependencies: Vec<GateDependency>,
        initial_mapping: HashMap<usize, usize>,
        lookahead_depth: usize,
    ) -> DeviceResult<InternalRoutingResult> {
        // A* state for priority queue
        #[derive(Clone)]
        struct AStarState {
            routing_state: RoutingState,
            f_score: f64,
            g_score: f64,
            h_score: f64,
        }

        impl PartialEq for AStarState {
            fn eq(&self, other: &Self) -> bool {
                self.f_score == other.f_score
            }
        }

        impl Eq for AStarState {}

        impl Ord for AStarState {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .f_score
                    .partial_cmp(&self.f_score)
                    .unwrap_or(Ordering::Equal)
            }
        }

        impl PartialOrd for AStarState {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        // Initialize
        let initial_state = RoutingState {
            mapping: initial_mapping.clone(),
            reverse_mapping: initial_mapping.iter().map(|(&k, &v)| (v, k)).collect(),
            scheduled_gates: HashSet::new(),
            front_layer: self.get_front_layer(&dependencies),
            cost: 0,
            swaps: Vec::new(),
        };

        let h_score = self.astar_heuristic(&initial_state, &dependencies, lookahead_depth);

        let mut open_set = BinaryHeap::new();
        open_set.push(AStarState {
            routing_state: initial_state,
            f_score: h_score,
            g_score: 0.0,
            h_score,
        });

        let mut metrics = RoutingMetrics {
            iterations: 0,
            states_explored: 0,
            avg_lookahead: 0.0,
            swap_chain_lengths: Vec::new(),
            critical_path_increase: 0.0,
        };

        // A* search
        while let Some(current) = open_set.pop() {
            metrics.iterations += 1;

            // Check if we're done
            if current.routing_state.scheduled_gates.len() == dependencies.len() {
                return Ok(InternalRoutingResult {
                    final_mapping: current.routing_state.mapping,
                    swap_sequence: current.routing_state.swaps,
                    total_cost: current.g_score,
                    depth_overhead: current.routing_state.cost * 3,
                    gate_overhead: current.routing_state.cost * 3,
                    metrics,
                });
            }

            // Generate neighbors
            let neighbors =
                self.generate_astar_neighbors(&current.routing_state, &dependencies, &mut metrics);

            for (neighbor_state, action_cost) in neighbors {
                let g_score = current.g_score + action_cost;
                let h_score = self.astar_heuristic(&neighbor_state, &dependencies, lookahead_depth);
                let f_score = g_score + h_score;

                open_set.push(AStarState {
                    routing_state: neighbor_state,
                    f_score,
                    g_score,
                    h_score,
                });

                metrics.states_explored += 1;
            }
        }

        Err(DeviceError::RoutingError(
            "A* search failed to find solution".to_string(),
        ))
    }

    /// Generate neighbor states for A*
    fn generate_astar_neighbors(
        &self,
        state: &RoutingState,
        dependencies: &[GateDependency],
        metrics: &RoutingMetrics,
    ) -> Vec<(RoutingState, f64)> {
        let mut neighbors = Vec::new();

        // Try executing gates
        let executable = self.get_executable_gates(state, dependencies);
        if !executable.is_empty() {
            let mut new_state = state.clone();
            for gate_id in executable {
                new_state.scheduled_gates.insert(gate_id);
            }
            new_state.front_layer = self.update_front_layer(&new_state, dependencies);
            neighbors.push((new_state, 0.0)); // No cost for executing gates
        } else {
            // Try all possible swaps
            for edge in self.topology.connectivity.edge_references() {
                let phys1 = self.topology.connectivity[edge.source()] as usize;
                let phys2 = self.topology.connectivity[edge.target()] as usize;

                let mut new_state = state.clone();
                let swap = SwapOperation {
                    phys_qubit1: phys1,
                    phys_qubit2: phys2,
                    log_qubit1: new_state.reverse_mapping.get(&phys1).copied(),
                    log_qubit2: new_state.reverse_mapping.get(&phys2).copied(),
                    position: 0,
                    cost: 1.0,
                };

                self.apply_swap(&mut new_state, &swap, 0);
                neighbors.push((new_state, 1.0)); // Cost of 1 for swap
            }
        }

        neighbors
    }

    /// A* heuristic with lookahead
    fn astar_heuristic(
        &self,
        state: &RoutingState,
        dependencies: &[GateDependency],
        lookahead_depth: usize,
    ) -> f64 {
        let mut score = 0.0;
        let mut current_layer = state.front_layer.clone();

        // Lookahead simulation
        for depth in 0..lookahead_depth {
            let mut min_swaps_needed = 0;

            for &gate_id in &current_layer {
                if state.scheduled_gates.contains(&gate_id) {
                    continue;
                }

                let dep = &dependencies[gate_id];
                if dep.qubits.len() == 2 {
                    let phys1 = state.mapping.get(&dep.qubits[0]).copied().unwrap_or(0);
                    let phys2 = state.mapping.get(&dep.qubits[1]).copied().unwrap_or(0);

                    if !self.are_connected(phys1, phys2) {
                        let dist = self.get_distance(phys1, phys2);
                        min_swaps_needed += (dist - 1).max(0);
                    }
                }
            }

            score += min_swaps_needed as f64 * (0.9_f64).powi(depth as i32);

            // Get next layer
            let mut next_layer = HashSet::new();
            for &gate_id in &current_layer {
                for &succ in &dependencies[gate_id].successors {
                    if !state.scheduled_gates.contains(&succ) {
                        next_layer.insert(succ);
                    }
                }
            }

            if next_layer.is_empty() {
                break;
            }
            current_layer = next_layer;
        }

        score
    }

    /// Token swapping algorithm
    fn route_token_swapping(
        &mut self,
        dependencies: Vec<GateDependency>,
        initial_mapping: HashMap<usize, usize>,
    ) -> DeviceResult<InternalRoutingResult> {
        // Simplified token swapping - find permutation to satisfy all gates
        let mut state = RoutingState {
            mapping: initial_mapping.clone(),
            reverse_mapping: initial_mapping.iter().map(|(&k, &v)| (v, k)).collect(),
            scheduled_gates: HashSet::new(),
            front_layer: HashSet::new(),
            cost: 0,
            swaps: Vec::new(),
        };

        let mut metrics = RoutingMetrics {
            iterations: 0,
            states_explored: 0,
            avg_lookahead: 0.0,
            swap_chain_lengths: Vec::new(),
            critical_path_increase: 0.0,
        };

        // For each two-qubit gate, ensure qubits are adjacent
        for (position, dep) in dependencies.iter().enumerate() {
            if dep.qubits.len() == 2 {
                let log1 = dep.qubits[0];
                let log2 = dep.qubits[1];
                let phys1 = state.mapping[&log1];
                let phys2 = state.mapping[&log2];

                if !self.are_connected(phys1, phys2) {
                    // Find shortest path and apply swaps
                    let swap_path = self.find_swap_path(phys1, phys2)?;

                    for i in 0..swap_path.len() - 1 {
                        let swap = SwapOperation {
                            phys_qubit1: swap_path[i],
                            phys_qubit2: swap_path[i + 1],
                            log_qubit1: state.reverse_mapping.get(&swap_path[i]).copied(),
                            log_qubit2: state.reverse_mapping.get(&swap_path[i + 1]).copied(),
                            position,
                            cost: 1.0,
                        };

                        self.apply_swap(&mut state, &swap, position);
                        metrics.iterations += 1;
                    }

                    metrics.swap_chain_lengths.push(swap_path.len() - 1);
                }
            }

            state.scheduled_gates.insert(dep.gate_id);
        }

        Ok(InternalRoutingResult {
            final_mapping: state.mapping,
            swap_sequence: state.swaps,
            total_cost: state.cost as f64,
            depth_overhead: state.cost * 3,
            gate_overhead: state.cost * 3,
            metrics,
        })
    }

    /// Find swap path between two physical qubits
    fn find_swap_path(&self, start: usize, target: usize) -> DeviceResult<Vec<usize>> {
        use petgraph::algo::astar;

        let start_node = self
            .topology
            .connectivity
            .node_indices()
            .find(|&n| self.topology.connectivity[n] == start as u32)
            .ok_or(DeviceError::RoutingError(
                "Start qubit not found".to_string(),
            ))?;

        let target_node = self
            .topology
            .connectivity
            .node_indices()
            .find(|&n| self.topology.connectivity[n] == target as u32)
            .ok_or(DeviceError::RoutingError(
                "Target qubit not found".to_string(),
            ))?;

        let result = astar(
            &self.topology.connectivity,
            start_node,
            |n| n == target_node,
            |_| 1,
            |_| 0,
        );

        if let Some((_, path)) = result {
            Ok(path
                .into_iter()
                .map(|n| self.topology.connectivity[n] as usize)
                .collect())
        } else {
            Err(DeviceError::RoutingError(
                "No path found between qubits".to_string(),
            ))
        }
    }

    /// Hybrid routing combining multiple strategies
    fn route_hybrid(
        &mut self,
        dependencies: Vec<GateDependency>,
        initial_mapping: HashMap<usize, usize>,
    ) -> DeviceResult<InternalRoutingResult> {
        // Try multiple strategies and pick the best
        let strategies = [
            (
                AdvancedRoutingStrategy::SABRE {
                    heuristic_weight: 0.5,
                },
                0.4,
            ),
            (
                AdvancedRoutingStrategy::AStarLookahead { lookahead_depth: 3 },
                0.4,
            ),
            (AdvancedRoutingStrategy::TokenSwapping, 0.2),
        ];

        let mut best_result = None;
        let mut best_cost = f64::INFINITY;

        for (strategy, weight) in strategies {
            let mut temp_router =
                AdvancedQubitRouter::new(self.topology.clone(), strategy, thread_rng().gen());

            if let Ok(result) = match strategy {
                AdvancedRoutingStrategy::SABRE { heuristic_weight } => temp_router.route_sabre(
                    dependencies.clone(),
                    initial_mapping.clone(),
                    heuristic_weight,
                ),
                AdvancedRoutingStrategy::AStarLookahead { lookahead_depth } => temp_router
                    .route_astar(
                        dependencies.clone(),
                        initial_mapping.clone(),
                        lookahead_depth,
                    ),
                AdvancedRoutingStrategy::TokenSwapping => {
                    temp_router.route_token_swapping(dependencies.clone(), initial_mapping.clone())
                }
                _ => continue,
            } {
                let weighted_cost = result.total_cost * weight;
                if weighted_cost < best_cost {
                    best_cost = weighted_cost;
                    best_result = Some(result);
                }
            }
        }

        best_result.ok_or(DeviceError::RoutingError(
            "Hybrid routing failed".to_string(),
        ))
    }
}

/// Internal routing result
struct InternalRoutingResult {
    final_mapping: HashMap<usize, usize>,
    swap_sequence: Vec<SwapOperation>,
    total_cost: f64,
    depth_overhead: usize,
    gate_overhead: usize,
    metrics: RoutingMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology_analysis::create_standard_topology;

    #[test]
    fn test_sabre_routing() {
        let topology = create_standard_topology("grid", 9).unwrap();
        let mut router = AdvancedQubitRouter::new(
            topology,
            AdvancedRoutingStrategy::SABRE {
                heuristic_weight: 0.5,
            },
            42,
        );

        // Create a simple circuit
        let mut circuit = Circuit::<4>::new();
        circuit
            .add_gate(quantrs2_core::gate::multi::CNOT {
                control: quantrs2_core::qubit::QubitId(0),
                target: quantrs2_core::qubit::QubitId(2),
            })
            .unwrap();
        circuit
            .add_gate(quantrs2_core::gate::multi::CNOT {
                control: quantrs2_core::qubit::QubitId(1),
                target: quantrs2_core::qubit::QubitId(3),
            })
            .unwrap();

        let result = router.route_circuit(&circuit).unwrap();

        assert!(!result.swap_sequence.is_empty());
        assert!(result.total_cost > 0.0);
    }

    #[test]
    fn test_astar_routing() {
        let topology = create_standard_topology("linear", 5).unwrap();
        let mut router = AdvancedQubitRouter::new(
            topology,
            AdvancedRoutingStrategy::AStarLookahead { lookahead_depth: 2 },
            42,
        );

        let mut circuit = Circuit::<3>::new();
        circuit
            .add_gate(quantrs2_core::gate::multi::CNOT {
                control: quantrs2_core::qubit::QubitId(0),
                target: quantrs2_core::qubit::QubitId(2),
            })
            .unwrap();

        let result = router.route_circuit(&circuit).unwrap();

        assert!(result.metrics.iterations > 0);
        assert!(result.metrics.states_explored > 0);
    }

    #[test]
    fn test_token_swapping() {
        let topology = create_standard_topology("linear", 4).unwrap();
        let mut router =
            AdvancedQubitRouter::new(topology, AdvancedRoutingStrategy::TokenSwapping, 42);

        let mut circuit = Circuit::<4>::new();
        circuit
            .add_gate(quantrs2_core::gate::multi::CNOT {
                control: quantrs2_core::qubit::QubitId(0),
                target: quantrs2_core::qubit::QubitId(3),
            })
            .unwrap();

        let result = router.route_circuit(&circuit).unwrap();

        // Should require swaps for distant qubits
        assert!(!result.swap_sequence.is_empty());
    }

    #[test]
    fn test_hybrid_routing() {
        let topology = create_standard_topology("grid", 9).unwrap();
        let mut router = AdvancedQubitRouter::new(topology, AdvancedRoutingStrategy::Hybrid, 42);

        let mut circuit = Circuit::<4>::new();
        circuit
            .add_gate(quantrs2_core::gate::single::Hadamard {
                target: quantrs2_core::qubit::QubitId(0),
            })
            .unwrap();
        circuit
            .add_gate(quantrs2_core::gate::multi::CNOT {
                control: quantrs2_core::qubit::QubitId(0),
                target: quantrs2_core::qubit::QubitId(1),
            })
            .unwrap();

        let result = router.route_circuit(&circuit).unwrap();

        // Routing time might be 0 on very fast systems or simple circuits
        // Note: routing_time is u128, so it's always >= 0
        assert_eq!(result.initial_mapping.len(), 4);
    }
}
