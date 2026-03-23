//! Enhanced Circuit Transpiler with SciRS2 Graph Optimization
//!
//! This module extends the basic transpiler with advanced graph-based circuit optimization
//! leveraging SciRS2's graph algorithms for:
//! - Gate dependency analysis
//! - Circuit topology optimization
//! - Optimal qubit routing
//! - Gate commutation and reordering
//! - Critical path analysis

use crate::{DeviceError, DeviceResult};
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// Graph structure for gate dependencies
#[derive(Debug, Clone)]
pub struct DirectedGraph<T> {
    nodes: Vec<T>,
    edges: HashMap<usize, Vec<usize>>,
}

/// Undirected weighted graph for hardware topology and routing
#[derive(Debug, Clone)]
pub struct UndirectedGraph {
    num_nodes: usize,
    /// Adjacency list: node -> Vec<(neighbor, weight)>
    adjacency: HashMap<usize, Vec<(usize, f64)>>,
}

impl UndirectedGraph {
    /// Create a new empty undirected graph with `num_nodes` nodes
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            adjacency: HashMap::new(),
        }
    }

    /// Add an undirected edge between `u` and `v` with the given weight
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        self.adjacency
            .entry(u)
            .or_insert_with(Vec::new)
            .push((v, weight));
        self.adjacency
            .entry(v)
            .or_insert_with(Vec::new)
            .push((u, weight));
    }

    /// Return the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// BFS traversal from `start`.  Returns visited nodes in BFS order.
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited = vec![false; self.num_nodes];
        let mut order = Vec::new();
        let mut queue = VecDeque::new();

        if start >= self.num_nodes {
            return order;
        }

        visited[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            order.push(node);
            if let Some(neighbors) = self.adjacency.get(&node) {
                let mut sorted_neighbors = neighbors.clone();
                sorted_neighbors.sort_by_key(|(n, _)| *n);
                for (neighbor, _) in sorted_neighbors {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        order
    }

    /// DFS traversal from `start`.  Returns visited nodes in DFS order.
    pub fn dfs(&self, start: usize) -> Vec<usize> {
        let mut visited = vec![false; self.num_nodes];
        let mut order = Vec::new();
        self.dfs_recursive(start, &mut visited, &mut order);
        order
    }

    fn dfs_recursive(&self, node: usize, visited: &mut Vec<bool>, order: &mut Vec<usize>) {
        if node >= self.num_nodes || visited[node] {
            return;
        }
        visited[node] = true;
        order.push(node);
        if let Some(neighbors) = self.adjacency.get(&node) {
            let mut sorted_neighbors = neighbors.clone();
            sorted_neighbors.sort_by_key(|(n, _)| *n);
            for (neighbor, _) in sorted_neighbors {
                self.dfs_recursive(neighbor, visited, order);
            }
        }
    }

    /// Dijkstra shortest path from `source` to all other nodes.
    ///
    /// Returns a `HashMap<usize, f64>` mapping node index to minimum distance.
    /// Unreachable nodes are absent from the map.
    ///
    /// Weights are scaled to `u64` (multiplied by 1e9) for use in the binary
    /// heap, which requires `Ord`.  This gives nanosecond precision and avoids
    /// pulling in an `ordered_float` dependency.
    pub fn dijkstra_distances(&self, source: usize) -> HashMap<usize, f64> {
        const SCALE: f64 = 1_000_000_000.0;
        // Min-heap stores (scaled_distance, node)
        let mut heap: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
        let mut dist: HashMap<usize, u64> = HashMap::new();

        dist.insert(source, 0);
        heap.push((Reverse(0), source));

        while let Some((Reverse(d), u)) = heap.pop() {
            if dist.get(&u).map_or(true, |&best| d > best) {
                continue;
            }
            if let Some(neighbors) = self.adjacency.get(&u) {
                for &(v, w) in neighbors {
                    let w_scaled = (w * SCALE) as u64;
                    let next_d = d.saturating_add(w_scaled);
                    let better = dist.get(&v).map_or(true, |&cur| next_d < cur);
                    if better {
                        dist.insert(v, next_d);
                        heap.push((Reverse(next_d), v));
                    }
                }
            }
        }
        dist.into_iter()
            .map(|(k, v)| (k, v as f64 / SCALE))
            .collect()
    }

    /// Dijkstra shortest path from `source` to `target`.
    ///
    /// Returns `Some((distance, path))` or `None` if unreachable.
    pub fn dijkstra_path(&self, source: usize, target: usize) -> Option<(f64, Vec<usize>)> {
        const SCALE: f64 = 1_000_000_000.0;
        let mut heap: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
        let mut dist: HashMap<usize, u64> = HashMap::new();
        let mut prev: HashMap<usize, usize> = HashMap::new();

        dist.insert(source, 0);
        heap.push((Reverse(0), source));

        while let Some((Reverse(d), u)) = heap.pop() {
            if u == target {
                break;
            }
            if dist.get(&u).map_or(true, |&best| d > best) {
                continue;
            }
            if let Some(neighbors) = self.adjacency.get(&u) {
                for &(v, w) in neighbors {
                    let w_scaled = (w * SCALE) as u64;
                    let next_d = d.saturating_add(w_scaled);
                    let better = dist.get(&v).map_or(true, |&cur| next_d < cur);
                    if better {
                        dist.insert(v, next_d);
                        prev.insert(v, u);
                        heap.push((Reverse(next_d), v));
                    }
                }
            }
        }

        let total_scaled = *dist.get(&target)?;
        let total = total_scaled as f64 / SCALE;

        // Reconstruct path by walking `prev` backwards
        let mut path = Vec::new();
        let mut cur = target;
        loop {
            path.push(cur);
            if cur == source {
                break;
            }
            match prev.get(&cur) {
                Some(&p) => cur = p,
                None => return None, // disconnected
            }
        }
        path.reverse();
        Some((total, path))
    }
}

/// A labeled pattern graph used for subgraph isomorphism matching.
///
/// Nodes carry a `String` label (e.g., gate type) and edges represent
/// dependencies or connections between them.
#[derive(Debug, Clone)]
pub struct PatternGraph {
    /// Node labels
    labels: Vec<String>,
    /// Adjacency set: (from, to) pairs
    edges: HashSet<(usize, usize)>,
}

impl PatternGraph {
    /// Create an empty pattern graph
    pub fn new() -> Self {
        Self {
            labels: Vec::new(),
            edges: HashSet::new(),
        }
    }

    /// Add a labeled node; returns its index
    pub fn add_node(&mut self, label: impl Into<String>) -> usize {
        let idx = self.labels.len();
        self.labels.push(label.into());
        idx
    }

    /// Add a directed edge
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.insert((from, to));
    }

    /// Number of nodes
    pub fn num_nodes(&self) -> usize {
        self.labels.len()
    }

    /// Label of node `i`
    pub fn label(&self, i: usize) -> Option<&str> {
        self.labels.get(i).map(String::as_str)
    }

    /// Whether there is an edge (from, to)
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.edges.contains(&(from, to))
    }
}

/// Result of a subgraph isomorphism search.
///
/// Each entry is a mapping `pattern_node -> target_node`.
#[derive(Debug, Clone)]
pub struct IsomorphismMapping {
    /// `mapping[p] = t` means pattern node `p` maps to target node `t`
    pub mapping: Vec<usize>,
}

/// VF2-inspired subgraph isomorphism engine (simplified backtracking).
///
/// Checks whether `pattern` is isomorphic to a subgraph of `target_labels /
/// target_edges`.
pub struct SubgraphIsomorphism<'a> {
    pattern: &'a PatternGraph,
    target_labels: &'a [String],
    target_edges: &'a HashSet<(usize, usize)>,
}

impl<'a> SubgraphIsomorphism<'a> {
    /// Create a new searcher
    pub fn new(
        pattern: &'a PatternGraph,
        target_labels: &'a [String],
        target_edges: &'a HashSet<(usize, usize)>,
    ) -> Self {
        Self {
            pattern,
            target_labels,
            target_edges,
        }
    }

    /// Find all subgraph isomorphisms.  Returns a list of mappings.
    pub fn find_all(&self) -> Vec<IsomorphismMapping> {
        let n_p = self.pattern.num_nodes();
        let n_t = self.target_labels.len();
        if n_p == 0 || n_p > n_t {
            return Vec::new();
        }

        let mut results = Vec::new();
        // `partial[p]` = index of the target node assigned to pattern node `p`
        let mut partial: Vec<Option<usize>> = vec![None; n_p];
        // Reverse lookup: which target nodes are already used
        let mut used = vec![false; n_t];

        self.backtrack(0, &mut partial, &mut used, &mut results);
        results
    }

    /// Recursive backtracking procedure (VF2-style)
    fn backtrack(
        &self,
        depth: usize,
        partial: &mut Vec<Option<usize>>,
        used: &mut Vec<bool>,
        results: &mut Vec<IsomorphismMapping>,
    ) {
        let n_p = self.pattern.num_nodes();
        if depth == n_p {
            // Full mapping found – record it
            let mapping: Vec<usize> = partial.iter().filter_map(|x| *x).collect();
            results.push(IsomorphismMapping { mapping });
            return;
        }

        // Try to assign each unoccupied target node to pattern node `depth`
        for t in 0..self.target_labels.len() {
            if used[t] {
                continue;
            }
            // Compatibility check 1: node labels must match
            if !self.labels_compatible(depth, t) {
                continue;
            }
            // Compatibility check 2: structural consistency with already-mapped nodes
            if !self.structurally_consistent(depth, t, partial) {
                continue;
            }

            // Extend mapping
            partial[depth] = Some(t);
            used[t] = true;

            self.backtrack(depth + 1, partial, used, results);

            // Undo
            partial[depth] = None;
            used[t] = false;
        }
    }

    /// Check whether the label of pattern node `p` is compatible with target node `t`.
    /// An empty pattern label acts as a wildcard.
    fn labels_compatible(&self, p: usize, t: usize) -> bool {
        let p_label = match self.pattern.label(p) {
            Some(l) => l,
            None => return false,
        };
        if p_label.is_empty() {
            return true; // wildcard
        }
        match self.target_labels.get(t) {
            Some(t_label) => p_label == t_label.as_str(),
            None => false,
        }
    }

    /// Check structural consistency: for every already-mapped pattern node `q < depth`,
    /// ensure that edges between `q` and `depth` are preserved in the target graph.
    fn structurally_consistent(&self, depth: usize, t: usize, partial: &[Option<usize>]) -> bool {
        for q in 0..depth {
            let mapped_q = match partial[q] {
                Some(m) => m,
                None => continue,
            };
            // If pattern has edge q→depth, target must have mapped_q→t
            if self.pattern.has_edge(q, depth) && !self.target_edges.contains(&(mapped_q, t)) {
                return false;
            }
            // If pattern has edge depth→q, target must have t→mapped_q
            if self.pattern.has_edge(depth, q) && !self.target_edges.contains(&(t, mapped_q)) {
                return false;
            }
        }
        true
    }
}

impl<T: Clone + PartialEq> DirectedGraph<T> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: T) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    pub fn add_edge(&mut self, from_idx: usize, to_idx: usize) {
        self.edges
            .entry(from_idx)
            .or_insert_with(Vec::new)
            .push(to_idx);
    }

    pub fn nodes(&self) -> &[T] {
        &self.nodes
    }

    pub fn has_edge(&self, from: &T, to: &T) -> bool {
        if let Some(from_idx) = self.nodes.iter().position(|n| n == from) {
            if let Some(to_idx) = self.nodes.iter().position(|n| n == to) {
                if let Some(neighbors) = self.edges.get(&from_idx) {
                    return neighbors.contains(&to_idx);
                }
            }
        }
        false
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }
}

/// Gate dependency graph node
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GateNode {
    /// Gate index in original circuit
    pub gate_index: usize,
    /// Gate type/name
    pub gate_type: String,
    /// Qubits this gate acts on
    pub qubits: Vec<usize>,
    /// Gate depth in the circuit
    pub depth: usize,
}

/// Circuit topology representation for hardware mapping
#[derive(Debug, Clone)]
pub struct CircuitTopology {
    /// Qubit connectivity graph
    pub qubit_graph: DirectedGraph<usize>,
    /// Gate dependency graph
    pub gate_graph: DirectedGraph<GateNode>,
    /// Critical path length
    pub critical_path_length: usize,
    /// Circuit depth
    pub circuit_depth: usize,
}

/// Hardware topology constraints
#[derive(Debug, Clone)]
pub struct HardwareTopology {
    /// Physical qubit connectivity (adjacency list)
    pub qubit_connectivity: HashMap<usize, Vec<usize>>,
    /// Number of physical qubits
    pub num_physical_qubits: usize,
    /// Gate error rates per qubit pair
    pub error_rates: HashMap<(usize, usize), f64>,
}

impl Default for HardwareTopology {
    fn default() -> Self {
        Self {
            qubit_connectivity: HashMap::new(),
            num_physical_qubits: 0,
            error_rates: HashMap::new(),
        }
    }
}

/// Configuration for graph-based transpilation
#[derive(Debug, Clone)]
pub struct SciRS2TranspilerConfig {
    /// Enable gate commutation optimization
    pub enable_commutation: bool,
    /// Enable critical path optimization
    pub enable_critical_path_opt: bool,
    /// Enable qubit routing optimization
    pub enable_routing_opt: bool,
    /// Maximum optimization passes
    pub max_optimization_passes: usize,
    /// Target hardware topology
    pub hardware_topology: Option<HardwareTopology>,
}

impl Default for SciRS2TranspilerConfig {
    fn default() -> Self {
        Self {
            enable_commutation: true,
            enable_critical_path_opt: true,
            enable_routing_opt: true,
            max_optimization_passes: 3,
            hardware_topology: None,
        }
    }
}

/// Enhanced transpiler using SciRS2 graph algorithms
pub struct SciRS2GraphTranspiler {
    config: SciRS2TranspilerConfig,
}

impl SciRS2GraphTranspiler {
    /// Create a new SciRS2 graph transpiler
    pub fn new(config: SciRS2TranspilerConfig) -> Self {
        Self { config }
    }

    /// Build gate dependency graph from circuit
    pub fn build_dependency_graph<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<DirectedGraph<GateNode>> {
        let mut graph = DirectedGraph::new();
        let mut qubit_last_gate: HashMap<usize, usize> = HashMap::new();

        // Create nodes for each gate
        for (idx, gate) in circuit.gates().iter().enumerate() {
            let node = GateNode {
                gate_index: idx,
                gate_type: gate.name().to_string(),
                qubits: gate.qubits().iter().map(|q| q.id() as usize).collect(),
                depth: 0, // Will be computed later
            };
            let node_idx = graph.add_node(node);

            // Add edges based on qubit dependencies
            for qubit in gate.qubits() {
                let q_id = qubit.id() as usize;

                // If there's a previous gate on this qubit, add dependency edge
                if let Some(&prev_idx) = qubit_last_gate.get(&q_id) {
                    graph.add_edge(prev_idx, node_idx);
                }

                // Update last gate for this qubit
                qubit_last_gate.insert(q_id, node_idx);
            }
        }

        Ok(graph)
    }

    /// Analyze circuit topology using graph algorithms
    pub fn analyze_topology<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<CircuitTopology> {
        // Build gate dependency graph
        let gate_graph = self.build_dependency_graph(circuit)?;

        // Build qubit connectivity graph
        let mut qubit_graph = DirectedGraph::new();
        let mut qubit_node_indices: HashMap<usize, usize> = HashMap::new();

        for gate in circuit.gates() {
            let qubits: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();

            // Add qubit nodes if not already present
            for &q in &qubits {
                qubit_node_indices
                    .entry(q)
                    .or_insert_with(|| qubit_graph.add_node(q));
            }

            // For two-qubit gates, add connectivity
            if qubits.len() == 2 {
                let (q0, q1) = (qubits[0], qubits[1]);
                if q0 != q1 {
                    if let (Some(&idx0), Some(&idx1)) =
                        (qubit_node_indices.get(&q0), qubit_node_indices.get(&q1))
                    {
                        qubit_graph.add_edge(idx0, idx1);
                        qubit_graph.add_edge(idx1, idx0); // Bidirectional
                    }
                }
            }
        }

        // Compute circuit depth using topological sort
        let circuit_depth = self.compute_circuit_depth(&gate_graph)?;

        // Compute critical path
        let critical_path_length = self.compute_critical_path(&gate_graph)?;

        Ok(CircuitTopology {
            qubit_graph,
            gate_graph,
            critical_path_length,
            circuit_depth,
        })
    }

    /// Compute circuit depth using simple dependency analysis
    fn compute_circuit_depth(&self, gate_graph: &DirectedGraph<GateNode>) -> DeviceResult<usize> {
        // Simplified depth computation without topological sort
        let mut depths: HashMap<usize, usize> = HashMap::new();
        let mut max_depth = 0;

        // Process all gates
        for node in gate_graph.nodes() {
            // Compute depth as 1 + max depth of predecessors
            let mut gate_depth = 0;

            // Find predecessors (gates that must execute before this one)
            for potential_pred in gate_graph.nodes() {
                if gate_graph.has_edge(potential_pred, node) {
                    if let Some(&pred_depth) = depths.get(&potential_pred.gate_index) {
                        gate_depth = gate_depth.max(pred_depth + 1);
                    }
                }
            }

            depths.insert(node.gate_index, gate_depth);
            max_depth = max_depth.max(gate_depth);
        }

        Ok(max_depth + 1)
    }

    /// Compute critical path length (longest dependency chain)
    fn compute_critical_path(&self, gate_graph: &DirectedGraph<GateNode>) -> DeviceResult<usize> {
        // Critical path = longest path in DAG
        // Simple dynamic programming approach

        let mut longest_paths: HashMap<usize, usize> = HashMap::new();
        let mut max_path_length = 0;

        for node in gate_graph.nodes() {
            let mut path_length = 0;

            // Find the longest path to this gate
            for potential_pred in gate_graph.nodes() {
                if gate_graph.has_edge(potential_pred, node) {
                    if let Some(&pred_path) = longest_paths.get(&potential_pred.gate_index) {
                        path_length = path_length.max(pred_path + 1);
                    }
                }
            }

            longest_paths.insert(node.gate_index, path_length);
            max_path_length = max_path_length.max(path_length);
        }

        Ok(max_path_length)
    }

    /// Optimize qubit routing using Dijkstra shortest paths on the hardware graph.
    ///
    /// For each logical qubit, we compute all-pairs shortest distances on the
    /// hardware coupling map and assign logical qubits to physical qubits in
    /// order of descending two-qubit gate frequency, choosing the physical qubit
    /// with the smallest average distance to already-assigned qubits.
    pub fn optimize_qubit_routing<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        hardware_topology: &HardwareTopology,
    ) -> DeviceResult<HashMap<usize, usize>> {
        // Build hardware graph for Dijkstra
        let n_phys = hardware_topology.num_physical_qubits;
        let mut hw_graph = UndirectedGraph::new(n_phys);
        for (&phys_q, neighbors) in &hardware_topology.qubit_connectivity {
            for &neighbor in neighbors {
                if phys_q < neighbor {
                    // Use error rate as edge weight, falling back to 1.0
                    let weight = hardware_topology
                        .error_rates
                        .get(&(phys_q, neighbor))
                        .copied()
                        .unwrap_or(1.0);
                    hw_graph.add_edge(phys_q, neighbor, weight);
                }
            }
        }

        // Precompute all-pairs shortest distances via Dijkstra for each physical qubit
        let all_dists: Vec<HashMap<usize, f64>> = (0..n_phys)
            .map(|src| hw_graph.dijkstra_distances(src))
            .collect();

        // Count how often each logical qubit pair appears in two-qubit gates
        let mut interaction_freq: HashMap<(usize, usize), usize> = HashMap::new();
        for gate in circuit.gates() {
            let qubits: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();
            if qubits.len() == 2 {
                let (a, b) = (qubits[0].min(qubits[1]), qubits[0].max(qubits[1]));
                *interaction_freq.entry((a, b)).or_insert(0) += 1;
            }
        }

        // Build per-logical-qubit interaction count
        let mut qubit_freq = vec![0usize; N];
        for (&(a, b), &freq) in &interaction_freq {
            if a < N {
                qubit_freq[a] += freq;
            }
            if b < N {
                qubit_freq[b] += freq;
            }
        }

        // Sort logical qubits by descending interaction frequency
        let mut sorted_logical: Vec<usize> = (0..N).collect();
        sorted_logical.sort_by(|&x, &y| qubit_freq[y].cmp(&qubit_freq[x]));

        // Greedy assignment: pick physical qubit minimizing average distance to
        // already-assigned neighbours
        let mut mapping: HashMap<usize, usize> = HashMap::new();
        let mut assigned_phys: HashSet<usize> = HashSet::new();

        for &logical in &sorted_logical {
            // Collect logical neighbours that are already assigned
            let assigned_neighbors: Vec<usize> = mapping.keys().copied().collect();

            let best_phys = (0..n_phys)
                .filter(|p| !assigned_phys.contains(p))
                .min_by(|&p1, &p2| {
                    let score = |p: usize| -> f64 {
                        if assigned_neighbors.is_empty() {
                            return p as f64; // tie-break by index
                        }
                        assigned_neighbors
                            .iter()
                            .map(|ln| {
                                let phys_n = *mapping.get(ln).unwrap_or(&0);
                                all_dists[p].get(&phys_n).copied().unwrap_or(f64::MAX)
                            })
                            .sum::<f64>()
                            / assigned_neighbors.len() as f64
                    };
                    score(p1)
                        .partial_cmp(&score(p2))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|| logical % n_phys.max(1));

            mapping.insert(logical, best_phys);
            assigned_phys.insert(best_phys);
        }

        Ok(mapping)
    }

    /// Search for occurrences of a circuit pattern in a larger circuit using
    /// the VF2 subgraph-isomorphism algorithm.
    ///
    /// The pattern is represented as a `PatternGraph` whose node labels are
    /// gate type strings (empty = wildcard).  Returns all qubit mappings found.
    pub fn find_circuit_pattern_matches<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        pattern: &PatternGraph,
    ) -> DeviceResult<Vec<IsomorphismMapping>> {
        // Build target node labels and edge set from the dependency graph
        let dep_graph = self.build_dependency_graph(circuit)?;

        let target_labels: Vec<String> = dep_graph
            .nodes()
            .iter()
            .map(|n| n.gate_type.clone())
            .collect();

        let mut target_edges: HashSet<(usize, usize)> = HashSet::new();
        for (from_idx, to_list) in &dep_graph.edges {
            for &to_idx in to_list {
                target_edges.insert((*from_idx, to_idx));
            }
        }

        let iso = SubgraphIsomorphism::new(pattern, &target_labels, &target_edges);
        Ok(iso.find_all())
    }

    /// Identify commuting gates using dependency analysis
    pub fn find_commuting_gates<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<Vec<(usize, usize)>> {
        let mut commuting_pairs = Vec::new();
        let gates = circuit.gates();

        for i in 0..gates.len() {
            for j in (i + 1)..gates.len() {
                // Check if gates commute (act on disjoint qubits)
                let qubits_i: HashSet<u32> = gates[i].qubits().iter().map(|q| q.id()).collect();
                let qubits_j: HashSet<u32> = gates[j].qubits().iter().map(|q| q.id()).collect();

                if qubits_i.is_disjoint(&qubits_j) {
                    commuting_pairs.push((i, j));
                }
            }
        }

        Ok(commuting_pairs)
    }

    /// Optimize circuit using graph-based analysis
    pub fn optimize_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<Circuit<N>> {
        // Analyze circuit topology
        let _topology = self.analyze_topology(circuit)?;

        // TODO: Implement optimization transformations
        // - Use graph analysis for gate commutation reordering
        // - Implement parallel gate scheduling
        // - Add SWAP gate insertion for routing

        // For now, return the original circuit
        Ok(circuit.clone())
    }

    /// Generate optimization report with graph analysis
    pub fn generate_optimization_report<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> DeviceResult<String> {
        let topology = self.analyze_topology(circuit)?;

        let mut report = String::from("=== SciRS2 Graph Transpiler Analysis ===\n\n");
        report.push_str(&format!("Circuit Depth: {}\n", topology.circuit_depth));
        report.push_str(&format!(
            "Critical Path Length: {}\n",
            topology.critical_path_length
        ));
        report.push_str(&format!("Number of Gates: {}\n", circuit.gates().len()));
        report.push_str(&format!("Number of Qubits: {}\n", N));

        // Qubit connectivity statistics
        let num_qubit_edges = topology.qubit_graph.num_edges();
        report.push_str(&format!("Qubit Connections: {}\n", num_qubit_edges));

        // Gate dependency statistics
        let num_dependencies = topology.gate_graph.num_edges();
        report.push_str(&format!("Gate Dependencies: {}\n", num_dependencies));

        // Commuting gate analysis
        if self.config.enable_commutation {
            let commuting = self.find_commuting_gates(circuit)?;
            report.push_str(&format!("Commuting Gate Pairs: {}\n", commuting.len()));
        }

        Ok(report)
    }
}

#[cfg(test)]
#[allow(clippy::pedantic, clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use quantrs2_circuit::prelude::*;

    #[test]
    fn test_transpiler_creation() {
        let config = SciRS2TranspilerConfig::default();
        let transpiler = SciRS2GraphTranspiler::new(config);
        assert!(transpiler.config.enable_commutation);
    }

    #[test]
    fn test_dependency_graph_building() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);
        let _ = circuit.h(1);

        let config = SciRS2TranspilerConfig::default();
        let transpiler = SciRS2GraphTranspiler::new(config);

        let graph = transpiler
            .build_dependency_graph(&circuit)
            .expect("Failed to build dependency graph");

        assert_eq!(graph.num_nodes(), 3); // H, CNOT, H
    }

    #[test]
    fn test_topology_analysis() {
        let mut circuit = Circuit::<3>::new();
        let _ = circuit.h(0);
        let _ = circuit.h(1);
        let _ = circuit.cnot(0, 1);
        let _ = circuit.cnot(1, 2);

        let config = SciRS2TranspilerConfig::default();
        let transpiler = SciRS2GraphTranspiler::new(config);

        let topology = transpiler
            .analyze_topology(&circuit)
            .expect("Failed to analyze topology");

        assert!(topology.circuit_depth > 0);
        assert!(topology.critical_path_length > 0);
    }

    #[test]
    fn test_commuting_gates_detection() {
        let mut circuit = Circuit::<4>::new();
        let _ = circuit.h(0);
        let _ = circuit.h(1); // Commutes with H(0)
        let _ = circuit.x(2); // Commutes with both
        let _ = circuit.cnot(0, 1); // Does not commute with H(0) or H(1)

        let config = SciRS2TranspilerConfig::default();
        let transpiler = SciRS2GraphTranspiler::new(config);

        let commuting = transpiler
            .find_commuting_gates(&circuit)
            .expect("Failed to find commuting gates");

        assert!(!commuting.is_empty());
    }

    #[test]
    fn test_optimization_report() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);
        let _ = circuit.measure_all();

        let config = SciRS2TranspilerConfig::default();
        let transpiler = SciRS2GraphTranspiler::new(config);

        let report = transpiler
            .generate_optimization_report(&circuit)
            .expect("Failed to generate report");

        assert!(report.contains("Circuit Depth"));
        assert!(report.contains("Critical Path"));
    }

    #[test]
    fn test_hardware_topology_creation() {
        let mut topology = HardwareTopology {
            num_physical_qubits: 5,
            ..Default::default()
        };

        // Linear connectivity: 0-1-2-3-4
        topology.qubit_connectivity.insert(0, vec![1]);
        topology.qubit_connectivity.insert(1, vec![0, 2]);
        topology.qubit_connectivity.insert(2, vec![1, 3]);
        topology.qubit_connectivity.insert(3, vec![2, 4]);
        topology.qubit_connectivity.insert(4, vec![3]);

        assert_eq!(topology.num_physical_qubits, 5);
        assert_eq!(topology.qubit_connectivity.len(), 5);
    }

    #[test]
    fn test_qubit_routing_optimization() {
        let mut circuit = Circuit::<3>::new();
        let _ = circuit.cnot(0, 1);
        let _ = circuit.cnot(1, 2);

        let mut hardware = HardwareTopology::default();
        hardware.num_physical_qubits = 5;
        hardware.qubit_connectivity.insert(0, vec![1]);
        hardware.qubit_connectivity.insert(1, vec![0, 2]);
        hardware.qubit_connectivity.insert(2, vec![1]);

        let config = SciRS2TranspilerConfig {
            enable_routing_opt: true,
            ..Default::default()
        };
        let transpiler = SciRS2GraphTranspiler::new(config);

        let mapping = transpiler
            .optimize_qubit_routing(&circuit, &hardware)
            .expect("Failed to optimize routing");

        assert_eq!(mapping.len(), 3);
    }

    // ── New algorithm tests ────────────────────────────────────────────────

    #[test]
    fn test_undirected_graph_bfs() {
        // Linear graph: 0-1-2-3
        let mut g = UndirectedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);

        let order = g.bfs(0);
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_undirected_graph_dfs() {
        // Linear graph: 0-1-2-3
        let mut g = UndirectedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);

        let order = g.dfs(0);
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_dijkstra_linear_graph() {
        // 0 -1.0- 1 -2.0- 2 -1.0- 3
        let mut g = UndirectedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 1.0);

        let dists = g.dijkstra_distances(0);
        assert!((dists[&0] - 0.0).abs() < 1e-6);
        assert!((dists[&1] - 1.0).abs() < 1e-6);
        assert!((dists[&2] - 3.0).abs() < 1e-6);
        assert!((dists[&3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_dijkstra_path() {
        let mut g = UndirectedGraph::new(5);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 3, 5.0);
        g.add_edge(3, 2, 1.0); // 0->3->2 costs 6, but 0->1->2 costs 2

        let result = g.dijkstra_path(0, 2);
        assert!(result.is_some());
        let (dist, path) = result.expect("path should exist");
        assert!(
            (dist - 2.0).abs() < 1e-6,
            "expected distance 2.0, got {}",
            dist
        );
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_pattern_graph_construction() {
        let mut pattern = PatternGraph::new();
        let h = pattern.add_node("H");
        let cx = pattern.add_node("CNOT");
        pattern.add_edge(h, cx);

        assert_eq!(pattern.num_nodes(), 2);
        assert!(pattern.has_edge(0, 1));
        assert!(!pattern.has_edge(1, 0));
    }

    #[test]
    fn test_subgraph_isomorphism_simple() {
        // Pattern: H -> CNOT
        let mut pattern = PatternGraph::new();
        let ph = pattern.add_node("H");
        let pcx = pattern.add_node("CNOT");
        pattern.add_edge(ph, pcx);

        // Target: H(q0) -> CNOT(q0,q1) -> H(q1)
        let target_labels = vec!["H".to_string(), "CNOT".to_string(), "H".to_string()];
        let mut target_edges: HashSet<(usize, usize)> = HashSet::new();
        target_edges.insert((0, 1)); // H -> CNOT
        target_edges.insert((1, 2)); // CNOT -> H

        let iso = SubgraphIsomorphism::new(&pattern, &target_labels, &target_edges);
        let mappings = iso.find_all();
        // Pattern node 0 (H) -> target 0, pattern node 1 (CNOT) -> target 1
        assert!(!mappings.is_empty(), "Should find at least one isomorphism");
    }

    #[test]
    fn test_find_circuit_pattern_matches() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);
        let _ = circuit.h(1);

        // Build pattern: any gate -> CNOT
        let mut pattern = PatternGraph::new();
        let _p0 = pattern.add_node(""); // wildcard
        let p1 = pattern.add_node("CNOT");
        pattern.add_edge(0, 1);

        let config = SciRS2TranspilerConfig::default();
        let transpiler = SciRS2GraphTranspiler::new(config);

        let matches = transpiler
            .find_circuit_pattern_matches(&circuit, &pattern)
            .expect("Pattern match should succeed");

        assert!(
            !matches.is_empty(),
            "Should find at least one pattern occurrence"
        );
    }
}
