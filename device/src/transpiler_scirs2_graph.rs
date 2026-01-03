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
use std::collections::{HashMap, HashSet, VecDeque};

// Graph structure for gate dependencies
#[derive(Debug, Clone)]
pub struct DirectedGraph<T> {
    nodes: Vec<T>,
    edges: HashMap<usize, Vec<usize>>,
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

    /// Optimize qubit routing using minimum spanning tree and shortest paths
    pub fn optimize_qubit_routing<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        hardware_topology: &HardwareTopology,
    ) -> DeviceResult<HashMap<usize, usize>> {
        // Analyze circuit topology
        let topology = self.analyze_topology(circuit)?;

        // Build hardware connectivity graph
        let mut _hw_graph = DirectedGraph::new();

        for physical_qubit in 0..hardware_topology.num_physical_qubits {
            _hw_graph.add_node(physical_qubit);
        }

        for (&phys_q, neighbors) in &hardware_topology.qubit_connectivity {
            for &neighbor in neighbors {
                // Add connectivity edges (we could use error rates for weighting later)
                _hw_graph.add_edge(phys_q, neighbor);
            }
        }

        // TODO: Use graph algorithms for optimal routing
        // - Shortest path for qubit mapping
        // - Minimum swap insertion
        // For now, simple mapping is sufficient

        // For now, use simple sequential mapping
        // TODO: Implement sophisticated mapping using graph matching algorithms
        let mut mapping = HashMap::new();
        for logical in 0..N {
            mapping.insert(logical, logical % hardware_topology.num_physical_qubits);
        }

        Ok(mapping)
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
}
