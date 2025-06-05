//! Graph-based circuit optimizer using SciRS2 algorithms
//!
//! This module implements advanced circuit optimization using graph representations
//! and algorithms from SciRS2 for optimal gate scheduling and optimization.

use crate::builder::Circuit;
use num_complex::Complex64;
use quantrs2_core::error::QuantRS2Error;
use quantrs2_core::qubit::QubitId;
use std::collections::{HashMap, HashSet, VecDeque};

/// Helper function to multiply two 2x2 matrices
fn matrix_multiply_2x2(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    vec![
        vec![
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        vec![
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Represents a gate in the circuit graph
#[derive(Debug, Clone, PartialEq)]
pub struct GraphGate {
    pub id: usize,
    pub gate_type: String,
    pub qubits: Vec<QubitId>,
    pub params: Vec<f64>,
    pub matrix: Option<Vec<Vec<Complex64>>>,
}

/// Edge types in the circuit graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// Data dependency (same qubit)
    DataDependency,
    /// Commutation constraint
    NonCommuting,
    /// Can be parallelized
    Parallelizable,
}

/// Circuit DAG (Directed Acyclic Graph) representation
pub struct CircuitDAG {
    nodes: Vec<GraphGate>,
    edges: HashMap<(usize, usize), EdgeType>,
    qubit_chains: HashMap<u32, Vec<usize>>, // qubit_id -> [gate_ids in order]
}

impl CircuitDAG {
    /// Create a new empty DAG
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: HashMap::new(),
            qubit_chains: HashMap::new(),
        }
    }

    /// Add a gate to the DAG
    pub fn add_gate(&mut self, gate: GraphGate) -> usize {
        let gate_id = self.nodes.len();

        // Update qubit chains
        for qubit in &gate.qubits {
            self.qubit_chains
                .entry(qubit.id())
                .or_default()
                .push(gate_id);
        }

        // Add edges for data dependencies
        for qubit in &gate.qubits {
            if let Some(chain) = self.qubit_chains.get(&qubit.id()) {
                if chain.len() > 1 {
                    let prev_gate = chain[chain.len() - 2];
                    self.edges
                        .insert((prev_gate, gate_id), EdgeType::DataDependency);
                }
            }
        }

        self.nodes.push(gate);
        gate_id
    }

    /// Check if two gates commute
    fn gates_commute(&self, g1: &GraphGate, g2: &GraphGate) -> bool {
        // Gates on different qubits always commute
        let qubits1: HashSet<_> = g1.qubits.iter().map(|q| q.id()).collect();
        let qubits2: HashSet<_> = g2.qubits.iter().map(|q| q.id()).collect();

        if qubits1.is_disjoint(&qubits2) {
            return true;
        }

        // Special cases for common gates
        match (g1.gate_type.as_str(), g2.gate_type.as_str()) {
            // Z gates always commute with each other
            ("z", "z") | ("rz", "rz") | ("z", "rz") | ("rz", "z") => true,
            // CNOT gates commute if they share only control or only target
            ("cnot", "cnot") => {
                if g1.qubits.len() == 2 && g2.qubits.len() == 2 {
                    let same_control = g1.qubits[0] == g2.qubits[0];
                    let same_target = g1.qubits[1] == g2.qubits[1];
                    same_control && same_target // Only if identical
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Compute commutation graph
    pub fn compute_commutation_edges(&mut self) {
        let n = self.nodes.len();

        for i in 0..n {
            for j in i + 1..n {
                let g1 = &self.nodes[i];
                let g2 = &self.nodes[j];

                // Skip if already connected
                if self.edges.contains_key(&(i, j)) || self.edges.contains_key(&(j, i)) {
                    continue;
                }

                // Check commutation
                if !self.gates_commute(g1, g2) {
                    // Check if they could be reordered (no data dependency path)
                    if !self.has_path(i, j) && !self.has_path(j, i) {
                        self.edges.insert((i, j), EdgeType::NonCommuting);
                    }
                } else if g1.qubits.iter().any(|q| g2.qubits.contains(q)) {
                    // Gates commute but share qubits
                    self.edges.insert((i, j), EdgeType::Parallelizable);
                }
            }
        }
    }

    /// Check if there's a path from src to dst
    fn has_path(&self, src: usize, dst: usize) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();

        queue.push_back(src);
        visited[src] = true;

        while let Some(node) = queue.pop_front() {
            if node == dst {
                return true;
            }

            for ((u, v), edge_type) in &self.edges {
                if *u == node && !visited[*v] && *edge_type == EdgeType::DataDependency {
                    visited[*v] = true;
                    queue.push_back(*v);
                }
            }
        }

        false
    }

    /// Topological sort with optimization
    pub fn optimized_topological_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_degree = vec![0; n];
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();

        // Build adjacency list and compute in-degrees
        for ((u, v), edge_type) in &self.edges {
            if *edge_type == EdgeType::DataDependency {
                adj_list.entry(*u).or_default().push(*v);
                in_degree[*v] += 1;
            }
        }

        // Priority queue for selecting next gate (minimize circuit depth)
        let mut ready: Vec<usize> = Vec::new();
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                ready.push(i);
            }
        }

        let mut result = Vec::new();
        let mut layer_qubits: HashSet<u32> = HashSet::new();

        while !ready.is_empty() {
            // Sort ready gates by number of qubits (prefer single-qubit gates)
            ready.sort_by_key(|&i| self.nodes[i].qubits.len());

            // Try to pack gates that don't conflict
            let mut next_layer = Vec::new();
            let mut used = vec![false; ready.len()];

            for (idx, &gate_id) in ready.iter().enumerate() {
                if used[idx] {
                    continue;
                }

                let gate = &self.nodes[gate_id];
                let gate_qubits: HashSet<_> = gate.qubits.iter().map(|q| q.id()).collect();

                // Check if this gate conflicts with current layer
                if gate_qubits.is_disjoint(&layer_qubits) {
                    next_layer.push(gate_id);
                    layer_qubits.extend(&gate_qubits);
                    used[idx] = true;
                }
            }

            // If no gates selected, pick the first one
            if next_layer.is_empty() && !ready.is_empty() {
                next_layer.push(ready[0]);
                used[0] = true;
            }

            // Remove selected gates from ready
            ready.retain(|&g| !next_layer.contains(&g));

            // Add to result and update graph
            for &gate_id in &next_layer {
                result.push(gate_id);

                if let Some(neighbors) = adj_list.get(&gate_id) {
                    for &neighbor in neighbors {
                        in_degree[neighbor] -= 1;
                        if in_degree[neighbor] == 0 {
                            ready.push(neighbor);
                        }
                    }
                }
            }

            layer_qubits.clear();
        }

        result
    }
}

impl Default for CircuitDAG {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph-based circuit optimizer
pub struct GraphOptimizer {
    merge_threshold: f64,
    #[allow(dead_code)]
    max_lookahead: usize,
}

impl GraphOptimizer {
    /// Create a new graph optimizer
    pub fn new() -> Self {
        Self {
            merge_threshold: 1e-6,
            max_lookahead: 10,
        }
    }

    /// Convert circuit to DAG representation
    pub fn circuit_to_dag<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
    ) -> Result<CircuitDAG, QuantRS2Error> {
        let dag = CircuitDAG::new();

        // TODO: Implement actual circuit introspection once available
        // For now, this is a placeholder

        Ok(dag)
    }

    /// Optimize gate sequence using commutation rules
    pub fn optimize_gate_sequence(&self, gates: Vec<GraphGate>) -> Vec<GraphGate> {
        let mut dag = CircuitDAG::new();

        // Add gates to DAG
        for gate in gates {
            dag.add_gate(gate);
        }

        // Compute commutation relationships
        dag.compute_commutation_edges();

        // Get optimized ordering
        let order = dag.optimized_topological_sort();

        // Return reordered gates
        order.iter().map(|&i| dag.nodes[i].clone()).collect()
    }

    /// Merge consecutive single-qubit gates
    pub fn merge_single_qubit_gates(&self, g1: &GraphGate, g2: &GraphGate) -> Option<GraphGate> {
        // Check if both are single-qubit gates on the same qubit
        if g1.qubits.len() != 1 || g2.qubits.len() != 1 || g1.qubits[0] != g2.qubits[0] {
            return None;
        }

        // Get matrices
        let m1 = g1.matrix.as_ref()?;
        let m2 = g2.matrix.as_ref()?;

        // Multiply matrices (2x2)
        let combined = matrix_multiply_2x2(m2, m1);

        // Check if it's close to a known gate
        if let Some((gate_type, params)) = self.identify_gate(&combined) {
            Some(GraphGate {
                id: g1.id, // Use first gate's ID
                gate_type,
                qubits: g1.qubits.clone(),
                params,
                matrix: Some(combined),
            })
        } else {
            // Return as generic unitary
            Some(GraphGate {
                id: g1.id,
                gate_type: "u".to_string(),
                qubits: g1.qubits.clone(),
                params: vec![],
                matrix: Some(combined),
            })
        }
    }

    /// Identify a gate from its matrix
    fn identify_gate(&self, matrix: &[Vec<Complex64>]) -> Option<(String, Vec<f64>)> {
        let tolerance = self.merge_threshold;

        // Check for Pauli gates
        if self.is_pauli_x(matrix, tolerance) {
            return Some(("x".to_string(), vec![]));
        }
        if self.is_pauli_y(matrix, tolerance) {
            return Some(("y".to_string(), vec![]));
        }
        if self.is_pauli_z(matrix, tolerance) {
            return Some(("z".to_string(), vec![]));
        }

        // Check for Hadamard
        if self.is_hadamard(matrix, tolerance) {
            return Some(("h".to_string(), vec![]));
        }

        // Check for rotation gates
        if let Some(angle) = self.is_rz(matrix, tolerance) {
            return Some(("rz".to_string(), vec![angle]));
        }

        None
    }

    fn is_pauli_x(&self, matrix: &[Vec<Complex64>], tol: f64) -> bool {
        matrix.len() == 2
            && matrix[0].len() == 2
            && (matrix[0][0].norm() < tol)
            && (matrix[0][1] - Complex64::new(1.0, 0.0)).norm() < tol
            && (matrix[1][0] - Complex64::new(1.0, 0.0)).norm() < tol
            && (matrix[1][1].norm() < tol)
    }

    fn is_pauli_y(&self, matrix: &[Vec<Complex64>], tol: f64) -> bool {
        matrix.len() == 2
            && matrix[0].len() == 2
            && (matrix[0][0].norm() < tol)
            && (matrix[0][1] - Complex64::new(0.0, -1.0)).norm() < tol
            && (matrix[1][0] - Complex64::new(0.0, 1.0)).norm() < tol
            && (matrix[1][1].norm() < tol)
    }

    fn is_pauli_z(&self, matrix: &[Vec<Complex64>], tol: f64) -> bool {
        matrix.len() == 2
            && matrix[0].len() == 2
            && (matrix[0][0] - Complex64::new(1.0, 0.0)).norm() < tol
            && (matrix[0][1].norm() < tol)
            && (matrix[1][0].norm() < tol)
            && (matrix[1][1] - Complex64::new(-1.0, 0.0)).norm() < tol
    }

    fn is_hadamard(&self, matrix: &[Vec<Complex64>], tol: f64) -> bool {
        let h_val = 1.0 / 2.0_f64.sqrt();
        matrix.len() == 2
            && matrix[0].len() == 2
            && (matrix[0][0] - Complex64::new(h_val, 0.0)).norm() < tol
            && (matrix[0][1] - Complex64::new(h_val, 0.0)).norm() < tol
            && (matrix[1][0] - Complex64::new(h_val, 0.0)).norm() < tol
            && (matrix[1][1] - Complex64::new(-h_val, 0.0)).norm() < tol
    }

    fn is_rz(&self, matrix: &[Vec<Complex64>], tol: f64) -> Option<f64> {
        if matrix.len() != 2
            || matrix[0].len() != 2
            || matrix[0][1].norm() > tol
            || matrix[1][0].norm() > tol
        {
            return None;
        }

        let phase1 = matrix[0][0].arg();
        let phase2 = matrix[1][1].arg();

        if (matrix[0][0].norm() - 1.0).abs() < tol && (matrix[1][1].norm() - 1.0).abs() < tol {
            let angle = phase2 - phase1;
            Some(angle)
        } else {
            None
        }
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub original_gate_count: usize,
    pub optimized_gate_count: usize,
    pub original_depth: usize,
    pub optimized_depth: usize,
    pub gates_removed: usize,
    pub gates_merged: usize,
}

impl OptimizationStats {
    pub fn improvement_percentage(&self) -> f64 {
        if self.original_gate_count == 0 {
            0.0
        } else {
            100.0 * (self.original_gate_count - self.optimized_gate_count) as f64
                / self.original_gate_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_construction() {
        let mut dag = CircuitDAG::new();

        let g1 = GraphGate {
            id: 0,
            gate_type: "h".to_string(),
            qubits: vec![QubitId::new(0)],
            params: vec![],
            matrix: None,
        };

        let g2 = GraphGate {
            id: 1,
            gate_type: "cnot".to_string(),
            qubits: vec![QubitId::new(0), QubitId::new(1)],
            params: vec![],
            matrix: None,
        };

        dag.add_gate(g1);
        dag.add_gate(g2);

        assert_eq!(dag.nodes.len(), 2);
        assert!(dag.edges.contains_key(&(0, 1)));
    }

    #[test]
    fn test_commutation_detection() {
        let _optimizer = GraphOptimizer::new();

        let g1 = GraphGate {
            id: 0,
            gate_type: "z".to_string(),
            qubits: vec![QubitId::new(0)],
            params: vec![],
            matrix: None,
        };

        let g2 = GraphGate {
            id: 1,
            gate_type: "z".to_string(),
            qubits: vec![QubitId::new(0)],
            params: vec![],
            matrix: None,
        };

        let dag = CircuitDAG::new();
        assert!(dag.gates_commute(&g1, &g2));
    }

    #[test]
    fn test_gate_identification() {
        let optimizer = GraphOptimizer::new();

        // Pauli X matrix
        let x_matrix = vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];

        if let Some((gate_type, _)) = optimizer.identify_gate(&x_matrix) {
            assert_eq!(gate_type, "x");
        } else {
            panic!("Failed to identify Pauli X gate");
        }
    }
}
