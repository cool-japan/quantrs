//! Directed Acyclic Graph (DAG) representation for quantum circuits.
//!
//! This module provides a DAG representation of quantum circuits that enables
//! advanced optimization techniques such as gate reordering, parallelization,
//! and dependency analysis.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use quantrs2_core::{gate::GateOp, qubit::QubitId};

/// A node in the circuit DAG
#[derive(Debug, Clone)]
pub struct DagNode {
    /// Unique identifier for this node
    pub id: usize,
    /// The quantum gate operation
    pub gate: Box<dyn GateOp>,
    /// Indices of predecessor nodes
    pub predecessors: Vec<usize>,
    /// Indices of successor nodes
    pub successors: Vec<usize>,
    /// Depth in the DAG (for scheduling)
    pub depth: usize,
}

/// Edge type in the DAG
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Data dependency on same qubit
    QubitDependency(u32),
    /// Classical control dependency
    ClassicalDependency,
    /// Barrier dependency
    BarrierDependency,
}

/// An edge in the circuit DAG
#[derive(Debug, Clone)]
pub struct DagEdge {
    /// Source node index
    pub source: usize,
    /// Target node index
    pub target: usize,
    /// Type of dependency
    pub edge_type: EdgeType,
}

/// DAG representation of a quantum circuit
pub struct CircuitDag {
    /// All nodes in the DAG
    nodes: Vec<DagNode>,
    /// All edges in the DAG
    edges: Vec<DagEdge>,
    /// Map from qubit ID to the last node that operated on it
    qubit_last_use: HashMap<u32, usize>,
    /// Input nodes (no predecessors)
    input_nodes: Vec<usize>,
    /// Output nodes (no successors)
    output_nodes: Vec<usize>,
}

impl CircuitDag {
    /// Create a new empty DAG
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            qubit_last_use: HashMap::new(),
            input_nodes: Vec::new(),
            output_nodes: Vec::new(),
        }
    }

    /// Add a gate to the DAG
    pub fn add_gate(&mut self, gate: Box<dyn GateOp>) -> usize {
        let node_id = self.nodes.len();
        let qubits = gate.qubits();

        // Find predecessors based on qubit dependencies
        let mut predecessors = Vec::new();
        for qubit in &qubits {
            if let Some(&last_node) = self.qubit_last_use.get(&qubit.id()) {
                predecessors.push(last_node);

                // Add edge
                self.edges.push(DagEdge {
                    source: last_node,
                    target: node_id,
                    edge_type: EdgeType::QubitDependency(qubit.id()),
                });

                // Update successor of predecessor
                self.nodes[last_node].successors.push(node_id);
            }
        }

        // Calculate depth
        let depth = if predecessors.is_empty() {
            0
        } else {
            predecessors
                .iter()
                .map(|&pred| self.nodes[pred].depth)
                .max()
                .unwrap_or(0)
                + 1
        };

        // Create new node
        let node = DagNode {
            id: node_id,
            gate,
            predecessors: predecessors.clone(),
            successors: Vec::new(),
            depth,
        };

        // Update qubit last use
        for qubit in &qubits {
            self.qubit_last_use.insert(qubit.id(), node_id);
        }

        // Update input/output nodes
        if predecessors.is_empty() {
            self.input_nodes.push(node_id);
        }

        // Remove predecessors from output nodes
        for &pred in &predecessors {
            self.output_nodes.retain(|&x| x != pred);
        }
        self.output_nodes.push(node_id);

        self.nodes.push(node);
        node_id
    }

    /// Get all nodes in the DAG
    #[must_use]
    pub fn nodes(&self) -> &[DagNode] {
        &self.nodes
    }

    /// Get all edges in the DAG
    #[must_use]
    pub fn edges(&self) -> &[DagEdge] {
        &self.edges
    }

    /// Get input nodes (no predecessors)
    #[must_use]
    pub fn input_nodes(&self) -> &[usize] {
        &self.input_nodes
    }

    /// Get output nodes (no successors)
    #[must_use]
    pub fn output_nodes(&self) -> &[usize] {
        &self.output_nodes
    }

    /// Get the maximum depth of the DAG
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Perform topological sort on the DAG
    pub fn topological_sort(&self) -> Result<Vec<usize>, String> {
        let mut in_degree = vec![0; self.nodes.len()];
        let mut sorted = Vec::new();
        let mut queue = VecDeque::new();

        // Calculate in-degrees
        for node in &self.nodes {
            in_degree[node.id] = node.predecessors.len();
        }

        // Initialize queue with nodes having no predecessors
        for (id, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(id);
            }
        }

        // Process nodes
        while let Some(node_id) = queue.pop_front() {
            sorted.push(node_id);

            // Reduce in-degree of successors
            for &succ in &self.nodes[node_id].successors {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        // Check for cycles
        if sorted.len() != self.nodes.len() {
            return Err("Circuit DAG contains a cycle".to_string());
        }

        Ok(sorted)
    }

    /// Get nodes at a specific depth
    #[must_use]
    pub fn nodes_at_depth(&self, depth: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.depth == depth)
            .map(|n| n.id)
            .collect()
    }

    /// Find the critical path (longest path) through the DAG
    #[must_use]
    pub fn critical_path(&self) -> Vec<usize> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // Dynamic programming to find longest path
        let mut longest_path_to = vec![0; self.nodes.len()];
        let mut parent = vec![None; self.nodes.len()];

        // Process nodes in topological order
        if let Ok(topo_order) = self.topological_sort() {
            for &node_id in &topo_order {
                for &succ in &self.nodes[node_id].successors {
                    let new_length = longest_path_to[node_id] + 1;
                    if new_length > longest_path_to[succ] {
                        longest_path_to[succ] = new_length;
                        parent[succ] = Some(node_id);
                    }
                }
            }
        }

        // Find the end of the longest path
        let mut end_node = 0;
        let mut max_length = 0;
        for (id, &length) in longest_path_to.iter().enumerate() {
            if length > max_length {
                max_length = length;
                end_node = id;
            }
        }

        // Reconstruct the path
        let mut path = Vec::new();
        let mut current = Some(end_node);
        while let Some(node) = current {
            path.push(node);
            current = parent[node];
        }
        path.reverse();

        path
    }

    /// Get all paths between two nodes
    #[must_use]
    pub fn paths_between(&self, start: usize, end: usize) -> Vec<Vec<usize>> {
        let mut paths = Vec::new();
        let mut current_path = vec![start];
        let mut visited = HashSet::new();

        self.find_paths_dfs(start, end, &mut current_path, &mut visited, &mut paths);

        paths
    }

    fn find_paths_dfs(
        &self,
        current: usize,
        end: usize,
        current_path: &mut Vec<usize>,
        visited: &mut HashSet<usize>,
        paths: &mut Vec<Vec<usize>>,
    ) {
        if current == end {
            paths.push(current_path.clone());
            return;
        }

        visited.insert(current);

        for &successor in &self.nodes[current].successors {
            if !visited.contains(&successor) {
                current_path.push(successor);
                self.find_paths_dfs(successor, end, current_path, visited, paths);
                current_path.pop();
            }
        }

        visited.remove(&current);
    }

    /// Check if two nodes are independent (can be executed in parallel)
    #[must_use]
    pub fn are_independent(&self, node1: usize, node2: usize) -> bool {
        // Two nodes are independent if there's no path between them
        self.paths_between(node1, node2).is_empty() && self.paths_between(node2, node1).is_empty()
    }

    /// Get all nodes that can be executed in parallel with a given node
    #[must_use]
    pub fn parallel_nodes(&self, node_id: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.id != node_id && self.are_independent(node_id, n.id))
            .map(|n| n.id)
            .collect()
    }

    /// Convert the DAG to a DOT format string for visualization
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph CircuitDAG {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n");

        // Add nodes
        for node in &self.nodes {
            dot.push_str(&format!(
                "  {} [label=\"{}: {}\"];\n",
                node.id,
                node.id,
                node.gate.name()
            ));
        }

        // Add edges
        for edge in &self.edges {
            let label = match edge.edge_type {
                EdgeType::QubitDependency(q) => format!("q{q}"),
                EdgeType::ClassicalDependency => "classical".to_string(),
                EdgeType::BarrierDependency => "barrier".to_string(),
            };
            dot.push_str(&format!(
                "  {} -> {} [label=\"{}\"];\n",
                edge.source, edge.target, label
            ));
        }

        dot.push_str("}\n");
        dot
    }
}

impl Default for CircuitDag {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CircuitDag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CircuitDag")
            .field("nodes", &self.nodes.len())
            .field("edges", &self.edges.len())
            .field("max_depth", &self.max_depth())
            .finish()
    }
}

/// Convert a Circuit into a DAG representation
#[must_use]
pub fn circuit_to_dag<const N: usize>(circuit: &crate::builder::Circuit<N>) -> CircuitDag {
    let mut dag = CircuitDag::new();

    for gate in circuit.gates() {
        // Convert Arc to Box for DAG compatibility
        let boxed_gate = gate.clone_gate();
        dag.add_gate(boxed_gate);
    }

    dag
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX};
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_dag_creation() {
        let mut dag = CircuitDag::new();

        // Add H gate on qubit 0
        let h_gate = Box::new(Hadamard { target: QubitId(0) });
        let h_id = dag.add_gate(h_gate);

        // Add X gate on qubit 1
        let x_gate = Box::new(PauliX { target: QubitId(1) });
        let x_id = dag.add_gate(x_gate);

        // Add CNOT gate on qubits 0,1
        let cnot_gate = Box::new(CNOT {
            control: QubitId(0),
            target: QubitId(1),
        });
        let cnot_id = dag.add_gate(cnot_gate);

        // Check structure
        assert_eq!(dag.nodes().len(), 3);
        assert_eq!(dag.edges().len(), 2);
        assert_eq!(dag.input_nodes(), &[h_id, x_id]);
        assert_eq!(dag.output_nodes(), &[cnot_id]);
    }

    #[test]
    fn test_topological_sort() {
        let mut dag = CircuitDag::new();

        // Create a simple circuit: H(0) -> CNOT(0,1) <- X(1)
        let h_gate = Box::new(Hadamard { target: QubitId(0) });
        let h_id = dag.add_gate(h_gate);

        let x_gate = Box::new(PauliX { target: QubitId(1) });
        let x_id = dag.add_gate(x_gate);

        let cnot_gate = Box::new(CNOT {
            control: QubitId(0),
            target: QubitId(1),
        });
        let cnot_id = dag.add_gate(cnot_gate);

        let sorted = dag.topological_sort().unwrap();

        // H and X can be in any order, but CNOT must be last
        assert_eq!(sorted.len(), 3);
        assert!(sorted.contains(&h_id));
        assert!(sorted.contains(&x_id));
        assert_eq!(sorted[2], cnot_id);
    }

    #[test]
    fn test_parallel_nodes() {
        let mut dag = CircuitDag::new();

        // Add gates on different qubits (can be parallel)
        let h0 = dag.add_gate(Box::new(Hadamard { target: QubitId(0) }));
        let h1 = dag.add_gate(Box::new(Hadamard { target: QubitId(1) }));
        let h2 = dag.add_gate(Box::new(Hadamard { target: QubitId(2) }));

        // Check that all H gates can be executed in parallel
        assert!(dag.are_independent(h0, h1));
        assert!(dag.are_independent(h0, h2));
        assert!(dag.are_independent(h1, h2));

        let parallel_to_h0 = dag.parallel_nodes(h0);
        assert!(parallel_to_h0.contains(&h1));
        assert!(parallel_to_h0.contains(&h2));
    }

    #[test]
    fn test_critical_path() {
        let mut dag = CircuitDag::new();

        // Create a circuit with a clear critical path
        // H(0) -> CNOT(0,1) -> X(0)
        //      -> X(1) -----/
        let h0 = dag.add_gate(Box::new(Hadamard { target: QubitId(0) }));
        let x1 = dag.add_gate(Box::new(PauliX { target: QubitId(1) }));
        let cnot = dag.add_gate(Box::new(CNOT {
            control: QubitId(0),
            target: QubitId(1),
        }));
        let x0 = dag.add_gate(Box::new(PauliX { target: QubitId(0) }));

        let path = dag.critical_path();

        // Critical path should be H(0) -> CNOT -> X(0)
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], h0);
        assert_eq!(path[1], cnot);
        assert_eq!(path[2], x0);
    }
}
