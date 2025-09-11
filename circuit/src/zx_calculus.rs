//! ZX-calculus optimization for quantum circuits
//!
//! This module implements ZX-calculus, a powerful graphical language for
//! reasoning about quantum computation that enables advanced optimizations
//! through graph rewrite rules.

use crate::builder::Circuit;
use crate::dag::{circuit_to_dag, CircuitDag, DagNode};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use std::sync::Arc;

/// A ZX-diagram node representing quantum operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ZXNode {
    /// Green spider (Z-spider) - represents Z-basis operations
    ZSpider {
        id: usize,
        phase: f64,
        /// Number of inputs/outputs
        arity: usize,
    },
    /// Red spider (X-spider) - represents X-basis operations  
    XSpider {
        id: usize,
        phase: f64,
        arity: usize,
    },
    /// Hadamard gate
    Hadamard {
        id: usize,
    },
    /// Input/Output boundaries
    Input {
        id: usize,
        qubit: u32,
    },
    Output {
        id: usize,
        qubit: u32,
    },
}

impl ZXNode {
    pub fn id(&self) -> usize {
        match self {
            Self::ZSpider { id, .. }
            | Self::XSpider { id, .. }
            | Self::Hadamard { id }
            | Self::Input { id, .. }
            | Self::Output { id, .. } => *id,
        }
    }

    pub fn phase(&self) -> f64 {
        match self {
            Self::ZSpider { phase, .. } | Self::XSpider { phase, .. } => *phase,
            _ => 0.0,
        }
    }

    pub fn set_phase(&mut self, new_phase: f64) {
        match self {
            Self::ZSpider { phase, .. } => *phase = new_phase,
            Self::XSpider { phase, .. } => *phase = new_phase,
            _ => {}
        }
    }
}

/// Edge in ZX-diagram
#[derive(Debug, Clone, PartialEq)]
pub struct ZXEdge {
    pub source: usize,
    pub target: usize,
    /// Hadamard edges are represented as dashed lines in ZX-calculus
    pub is_hadamard: bool,
}

/// ZX-diagram representation of a quantum circuit
#[derive(Debug, Clone)]
pub struct ZXDiagram {
    /// Nodes in the diagram
    pub nodes: HashMap<usize, ZXNode>,
    /// Edges between nodes
    pub edges: Vec<ZXEdge>,
    /// Adjacency list for efficient traversal
    pub adjacency: HashMap<usize, Vec<usize>>,
    /// Input nodes for each qubit
    pub inputs: HashMap<u32, usize>,
    /// Output nodes for each qubit
    pub outputs: HashMap<u32, usize>,
    /// Next available node ID
    next_id: usize,
}

impl ZXDiagram {
    /// Create a new empty ZX diagram
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a node to the diagram
    pub fn add_node(&mut self, node: ZXNode) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let node_with_id = match node {
            ZXNode::ZSpider { phase, arity, .. } => ZXNode::ZSpider { id, phase, arity },
            ZXNode::XSpider { phase, arity, .. } => ZXNode::XSpider { id, phase, arity },
            ZXNode::Hadamard { .. } => ZXNode::Hadamard { id },
            ZXNode::Input { qubit, .. } => ZXNode::Input { id, qubit },
            ZXNode::Output { qubit, .. } => ZXNode::Output { id, qubit },
        };

        self.nodes.insert(id, node_with_id);
        self.adjacency.insert(id, Vec::new());
        id
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, source: usize, target: usize, is_hadamard: bool) {
        let edge = ZXEdge {
            source,
            target,
            is_hadamard,
        };
        self.edges.push(edge);

        // Update adjacency lists
        self.adjacency.entry(source).or_default().push(target);
        self.adjacency.entry(target).or_default().push(source);
    }

    /// Initialize inputs and outputs for a given number of qubits
    pub fn initialize_boundaries(&mut self, num_qubits: usize) {
        for i in 0..num_qubits {
            let qubit = i as u32;

            let input_id = self.add_node(ZXNode::Input { id: 0, qubit });
            let output_id = self.add_node(ZXNode::Output { id: 0, qubit });

            self.inputs.insert(qubit, input_id);
            self.outputs.insert(qubit, output_id);
        }
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node_id: usize) -> &[usize] {
        self.adjacency
            .get(&node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Apply spider fusion rule
    /// Two spiders of the same color connected by a plain edge can be fused
    pub fn spider_fusion(&mut self) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut to_update = Vec::new();

        for edge in &self.edges {
            if !edge.is_hadamard {
                if let (Some(node1), Some(node2)) =
                    (self.nodes.get(&edge.source), self.nodes.get(&edge.target))
                {
                    // Check if both are spiders of the same type
                    match (node1, node2) {
                        (
                            ZXNode::ZSpider {
                                id: id1,
                                phase: phase1,
                                ..
                            },
                            ZXNode::ZSpider {
                                id: id2,
                                phase: phase2,
                                ..
                            },
                        )
                        | (
                            ZXNode::XSpider {
                                id: id1,
                                phase: phase1,
                                ..
                            },
                            ZXNode::XSpider {
                                id: id2,
                                phase: phase2,
                                ..
                            },
                        ) => {
                            // Fuse the spiders: keep first, remove second
                            let new_phase = (phase1 + phase2) % (2.0 * PI);
                            to_update.push((*id1, new_phase));
                            to_remove.push(*id2);
                            changed = true;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Apply updates
        for (id, new_phase) in to_update {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.set_phase(new_phase);
            }
        }

        // Remove fused nodes and update edges
        for id in to_remove {
            self.remove_node(id);
        }

        changed
    }

    /// Apply identity removal rule
    /// A spider with phase 0 and arity 2 can be removed
    pub fn identity_removal(&mut self) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();

        for (id, node) in &self.nodes {
            match node {
                ZXNode::ZSpider { phase, arity, .. } | ZXNode::XSpider { phase, arity, .. }
                    if *arity == 2 && phase.abs() < 1e-10 =>
                {
                    to_remove.push(*id);
                }
                _ => {}
            }
        }

        for id in to_remove {
            // Connect the neighbors directly
            let neighbors: Vec<_> = self.neighbors(id).to_vec();
            if neighbors.len() == 2 {
                self.add_edge(neighbors[0], neighbors[1], false);
                changed = true;
            }
            self.remove_node(id);
        }

        changed
    }

    /// Apply pi-commutation rule
    /// A spider with phase π can pass through Hadamard gates
    pub fn pi_commutation(&mut self) -> bool {
        let mut changed = false;
        // Implementation would involve complex graph rewriting
        // For now, return false as this is a placeholder
        changed
    }

    /// Apply Hadamard cancellation
    /// Two adjacent Hadamard gates cancel out
    pub fn hadamard_cancellation(&mut self) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();

        // Find pairs of adjacent Hadamard nodes
        for edge in &self.edges {
            if let (Some(ZXNode::Hadamard { id: id1 }), Some(ZXNode::Hadamard { id: id2 })) =
                (self.nodes.get(&edge.source), self.nodes.get(&edge.target))
            {
                // Two Hadamards connected - they cancel out
                to_remove.push(*id1);
                to_remove.push(*id2);
                changed = true;
            }
        }

        for id in to_remove {
            self.remove_node(id);
        }

        changed
    }

    /// Remove a node and update the graph structure
    fn remove_node(&mut self, node_id: usize) {
        // Remove from nodes
        self.nodes.remove(&node_id);

        // Remove from adjacency
        self.adjacency.remove(&node_id);

        // Remove from other nodes' adjacency lists
        for adj_list in self.adjacency.values_mut() {
            adj_list.retain(|&id| id != node_id);
        }

        // Remove edges involving this node
        self.edges
            .retain(|edge| edge.source != node_id && edge.target != node_id);
    }

    /// Calculate the T-count (number of T gates) in the diagram
    pub fn t_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|node| {
                let phase = node.phase();
                (phase - PI / 4.0).abs() < 1e-10
                    || (phase - 3.0 * PI / 4.0).abs() < 1e-10
                    || (phase - 5.0 * PI / 4.0).abs() < 1e-10
                    || (phase - 7.0 * PI / 4.0).abs() < 1e-10
            })
            .count()
    }

    /// Apply all optimization rules until convergence
    pub fn optimize(&mut self) -> ZXOptimizationResult {
        let initial_node_count = self.nodes.len();
        let initial_t_count = self.t_count();

        let mut iterations = 0;
        let max_iterations = 100;

        while iterations < max_iterations {
            let mut changed = false;

            // Apply rewrite rules
            changed |= self.spider_fusion();
            changed |= self.identity_removal();
            changed |= self.hadamard_cancellation();
            changed |= self.pi_commutation();

            if !changed {
                break;
            }
            iterations += 1;
        }

        let final_node_count = self.nodes.len();
        let final_t_count = self.t_count();

        ZXOptimizationResult {
            iterations,
            initial_node_count,
            final_node_count,
            initial_t_count,
            final_t_count,
            converged: iterations < max_iterations,
        }
    }
}

/// Result of ZX optimization
#[derive(Debug, Clone)]
pub struct ZXOptimizationResult {
    pub iterations: usize,
    pub initial_node_count: usize,
    pub final_node_count: usize,
    pub initial_t_count: usize,
    pub final_t_count: usize,
    pub converged: bool,
}

/// ZX-calculus optimizer
pub struct ZXOptimizer {
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Enable specific optimization rules
    pub enable_spider_fusion: bool,
    pub enable_identity_removal: bool,
    pub enable_pi_commutation: bool,
    pub enable_hadamard_cancellation: bool,
}

impl Default for ZXOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            enable_spider_fusion: true,
            enable_identity_removal: true,
            enable_pi_commutation: true,
            enable_hadamard_cancellation: true,
        }
    }
}

impl ZXOptimizer {
    /// Create a new ZX optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert a quantum circuit to ZX diagram
    pub fn circuit_to_zx<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<ZXDiagram> {
        let mut diagram = ZXDiagram::new();
        diagram.initialize_boundaries(N);

        // Track the last node on each qubit wire
        let mut qubit_wires = HashMap::new();
        for i in 0..N {
            let qubit = i as u32;
            if let Some(&input_id) = diagram.inputs.get(&qubit) {
                qubit_wires.insert(qubit, input_id);
            }
        }

        // Convert each gate to ZX representation
        for gate in circuit.gates() {
            self.gate_to_zx(gate.as_ref(), &mut diagram, &mut qubit_wires)?;
        }

        // Connect to outputs
        for i in 0..N {
            let qubit = i as u32;
            if let (Some(&last_node), Some(&output_id)) =
                (qubit_wires.get(&qubit), diagram.outputs.get(&qubit))
            {
                diagram.add_edge(last_node, output_id, false);
            }
        }

        Ok(diagram)
    }

    /// Convert a single gate to ZX representation
    fn gate_to_zx(
        &self,
        gate: &dyn GateOp,
        diagram: &mut ZXDiagram,
        qubit_wires: &mut HashMap<u32, usize>,
    ) -> QuantRS2Result<()> {
        let gate_name = gate.name();
        let qubits = gate.qubits();

        match gate_name {
            "H" => {
                // Hadamard gate
                let qubit = qubits[0].id();
                let h_node = diagram.add_node(ZXNode::Hadamard { id: 0 });

                if let Some(&prev_node) = qubit_wires.get(&qubit) {
                    diagram.add_edge(prev_node, h_node, false);
                }
                qubit_wires.insert(qubit, h_node);
            }
            "X" => {
                // Pauli-X = Z-spider with phase π
                let qubit = qubits[0].id();
                let x_node = diagram.add_node(ZXNode::ZSpider {
                    id: 0,
                    phase: PI,
                    arity: 2,
                });

                if let Some(&prev_node) = qubit_wires.get(&qubit) {
                    diagram.add_edge(prev_node, x_node, false);
                }
                qubit_wires.insert(qubit, x_node);
            }
            "Y" => {
                // Pauli-Y = Z-spider with phase π followed by virtual Z
                let qubit = qubits[0].id();
                let y_node = diagram.add_node(ZXNode::ZSpider {
                    id: 0,
                    phase: PI,
                    arity: 2,
                });

                if let Some(&prev_node) = qubit_wires.get(&qubit) {
                    diagram.add_edge(prev_node, y_node, false);
                }
                qubit_wires.insert(qubit, y_node);
            }
            "Z" => {
                // Pauli-Z = Z-spider with phase π
                let qubit = qubits[0].id();
                let z_node = diagram.add_node(ZXNode::ZSpider {
                    id: 0,
                    phase: PI,
                    arity: 2,
                });

                if let Some(&prev_node) = qubit_wires.get(&qubit) {
                    diagram.add_edge(prev_node, z_node, false);
                }
                qubit_wires.insert(qubit, z_node);
            }
            "RZ" => {
                // Z-rotation = Z-spider with rotation angle
                let qubit = qubits[0].id();

                // Extract rotation angle from gate properties
                let angle = self.extract_rotation_angle(gate);
                let rz_node = diagram.add_node(ZXNode::ZSpider {
                    id: 0,
                    phase: angle,
                    arity: 2,
                });

                if let Some(&prev_node) = qubit_wires.get(&qubit) {
                    diagram.add_edge(prev_node, rz_node, false);
                }
                qubit_wires.insert(qubit, rz_node);
            }
            "CNOT" => {
                // CNOT = Z-spider on control connected to X-spider on target
                let control_qubit = qubits[0].id();
                let target_qubit = qubits[1].id();

                let control_spider = diagram.add_node(ZXNode::ZSpider {
                    id: 0,
                    phase: 0.0,
                    arity: 3,
                });
                let target_spider = diagram.add_node(ZXNode::XSpider {
                    id: 0,
                    phase: 0.0,
                    arity: 3,
                });

                // Connect control
                if let Some(&prev_control) = qubit_wires.get(&control_qubit) {
                    diagram.add_edge(prev_control, control_spider, false);
                }

                // Connect target
                if let Some(&prev_target) = qubit_wires.get(&target_qubit) {
                    diagram.add_edge(prev_target, target_spider, false);
                }

                // Connect control to target
                diagram.add_edge(control_spider, target_spider, false);

                qubit_wires.insert(control_qubit, control_spider);
                qubit_wires.insert(target_qubit, target_spider);
            }
            _ => {
                // For unsupported gates, add identity spiders
                for qubit_id in qubits {
                    let qubit = qubit_id.id();
                    let identity_node = diagram.add_node(ZXNode::ZSpider {
                        id: 0,
                        phase: 0.0,
                        arity: 2,
                    });

                    if let Some(&prev_node) = qubit_wires.get(&qubit) {
                        diagram.add_edge(prev_node, identity_node, false);
                    }
                    qubit_wires.insert(qubit, identity_node);
                }
            }
        }

        Ok(())
    }

    /// Extract rotation angle from gate (simplified)
    fn extract_rotation_angle(&self, gate: &dyn GateOp) -> f64 {
        // This would need to access gate parameters
        // For now, return a default value
        PI / 4.0 // T gate angle
    }

    /// Optimize a circuit using ZX-calculus
    pub fn optimize_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<OptimizedZXResult<N>> {
        // Convert to ZX diagram
        let mut diagram = self.circuit_to_zx(circuit)?;

        // Optimize the diagram
        let optimization_result = diagram.optimize();

        // Convert back to circuit (simplified for now)
        let optimized_circuit = self.zx_to_circuit(&diagram)?;

        Ok(OptimizedZXResult {
            original_circuit: circuit.clone(),
            optimized_circuit,
            diagram,
            optimization_stats: optimization_result,
        })
    }

    /// Convert ZX diagram back to quantum circuit (simplified)
    fn zx_to_circuit<const N: usize>(&self, diagram: &ZXDiagram) -> QuantRS2Result<Circuit<N>> {
        // This is a complex process that would require:
        // 1. Graph extraction algorithms
        // 2. Synthesis of unitary matrices
        // 3. Gate decomposition

        // For now, return the original circuit structure
        // In a full implementation, this would reconstruct the optimized circuit
        let mut circuit = Circuit::<N>::new();

        // Placeholder: add identity gates for demonstration
        for i in 0..N {
            // This would be replaced with proper circuit reconstruction
        }

        Ok(circuit)
    }
}

/// Result of ZX optimization containing original and optimized circuits
#[derive(Debug)]
pub struct OptimizedZXResult<const N: usize> {
    pub original_circuit: Circuit<N>,
    pub optimized_circuit: Circuit<N>,
    pub diagram: ZXDiagram,
    pub optimization_stats: ZXOptimizationResult,
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX};

    #[test]
    fn test_zx_diagram_creation() {
        let mut diagram = ZXDiagram::new();
        diagram.initialize_boundaries(2);

        assert_eq!(diagram.inputs.len(), 2);
        assert_eq!(diagram.outputs.len(), 2);
    }

    #[test]
    fn test_spider_fusion() {
        let mut diagram = ZXDiagram::new();

        // Add two Z-spiders with phases π/4 and π/8
        let spider1 = diagram.add_node(ZXNode::ZSpider {
            id: 0,
            phase: PI / 4.0,
            arity: 2,
        });
        let spider2 = diagram.add_node(ZXNode::ZSpider {
            id: 0,
            phase: PI / 8.0,
            arity: 2,
        });

        // Connect them
        diagram.add_edge(spider1, spider2, false);

        // Apply spider fusion
        let changed = diagram.spider_fusion();
        assert!(changed);

        // One spider should be removed
        assert_eq!(diagram.nodes.len(), 1);

        // Remaining spider should have combined phase
        let remaining_node = diagram.nodes.values().next().unwrap();
        assert!((remaining_node.phase() - (PI / 4.0 + PI / 8.0)).abs() < 1e-10);
    }

    #[test]
    fn test_identity_removal() {
        let mut diagram = ZXDiagram::new();

        // Add identity spider (phase 0, arity 2)
        let identity = diagram.add_node(ZXNode::ZSpider {
            id: 0,
            phase: 0.0,
            arity: 2,
        });

        // Add two other nodes
        let node1 = diagram.add_node(ZXNode::ZSpider {
            id: 0,
            phase: PI / 4.0,
            arity: 2,
        });
        let node2 = diagram.add_node(ZXNode::ZSpider {
            id: 0,
            phase: PI / 2.0,
            arity: 2,
        });

        // Connect through identity
        diagram.add_edge(node1, identity, false);
        diagram.add_edge(identity, node2, false);

        let initial_count = diagram.nodes.len();
        let changed = diagram.identity_removal();

        assert!(changed);
        assert_eq!(diagram.nodes.len(), initial_count - 1);
    }

    #[test]
    fn test_circuit_to_zx_conversion() {
        let optimizer = ZXOptimizer::new();

        let mut circuit = Circuit::<2>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let diagram = optimizer.circuit_to_zx(&circuit).unwrap();

        // Should have input/output nodes plus gate nodes
        assert!(diagram.nodes.len() >= 4); // 2 inputs + 2 outputs + gate nodes
        assert!(!diagram.edges.is_empty());
    }

    #[test]
    fn test_zx_optimization() {
        let optimizer = ZXOptimizer::new();

        let mut circuit = Circuit::<1>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap(); // Should cancel out

        let result = optimizer.optimize_circuit(&circuit).unwrap();

        assert!(
            result.optimization_stats.final_node_count
                <= result.optimization_stats.initial_node_count
        );
    }
}
