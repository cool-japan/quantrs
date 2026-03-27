//! # AutoParallelEngine - connections Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::builder::{Circuit, Simulator};

use super::types::{DependencyGraph, MLFeatures};

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Extract ML features from circuit and dependency graph
    pub(super) fn extract_ml_features<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> MLFeatures {
        let gates = circuit.gates();
        let num_gates = gates.len();
        let num_qubits = N;
        let circuit_depth = Self::calculate_circuit_depth(graph);
        let avg_connectivity = Self::calculate_average_connectivity(graph);
        let parallelism_factor = Self::calculate_parallelism_factor(graph);
        let gate_distribution = Self::calculate_gate_distribution(gates);
        let entanglement_score = Self::estimate_entanglement_complexity(circuit);
        MLFeatures {
            num_gates,
            num_qubits,
            circuit_depth,
            avg_connectivity,
            parallelism_factor,
            gate_distribution,
            entanglement_score,
            dependency_density: graph.edges.len() as f64 / num_gates as f64,
        }
    }
    /// Calculate circuit depth (critical path)
    pub(super) fn calculate_circuit_depth(graph: &DependencyGraph) -> usize {
        let mut depths = vec![0; graph.nodes.len()];
        for (idx, node) in graph.nodes.iter().enumerate() {
            let mut max_parent_depth = 0;
            if let Some(parents) = graph.reverse_edges.get(&idx) {
                for &parent in parents {
                    max_parent_depth = max_parent_depth.max(depths[parent]);
                }
            }
            depths[idx] = max_parent_depth + 1;
        }
        *depths.iter().max().unwrap_or(&0)
    }
    /// Calculate average gate connectivity
    pub(super) fn calculate_average_connectivity(graph: &DependencyGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 0.0;
        }
        let total_connections: usize = graph.nodes.iter().map(|n| n.qubits.len()).sum();
        total_connections as f64 / graph.nodes.len() as f64
    }
    /// Calculate parallelism factor
    pub(super) fn calculate_parallelism_factor(graph: &DependencyGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 0.0;
        }
        let independent_gates = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                graph
                    .reverse_edges
                    .get(idx)
                    .is_none_or(std::vec::Vec::is_empty)
            })
            .count();
        independent_gates as f64 / graph.nodes.len() as f64
    }
    /// Estimate entanglement complexity
    pub(super) fn estimate_entanglement_complexity<const N: usize>(circuit: &Circuit<N>) -> f64 {
        let gates = circuit.gates();
        let two_qubit_gates = gates.iter().filter(|g| g.qubits().len() >= 2).count();
        if gates.is_empty() {
            0.0
        } else {
            two_qubit_gates as f64 / gates.len() as f64
        }
    }
}
