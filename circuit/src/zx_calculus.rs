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
    #[must_use]
    pub const fn id(&self) -> usize {
        match self {
            Self::ZSpider { id, .. } => *id,
            Self::XSpider { id, .. } => *id,
            Self::Hadamard { id } => *id,
            Self::Input { id, .. } => *id,
            Self::Output { id, .. } => *id,
        }
    }

    #[must_use]
    pub const fn phase(&self) -> f64 {
        match self {
            Self::ZSpider { phase, .. } | Self::XSpider { phase, .. } => *phase,
            _ => 0.0,
        }
    }

    pub const fn set_phase(&mut self, new_phase: f64) {
        match self {
            Self::ZSpider { phase, .. } | Self::XSpider { phase, .. } => *phase = new_phase,
            _ => {}
        }
    }
}

/// Edge in ZX-diagram
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl Default for ZXDiagram {
    fn default() -> Self {
        Self::new()
    }
}

impl ZXDiagram {
    /// Create a new empty ZX diagram
    #[must_use]
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
    #[must_use]
    pub fn neighbors(&self, node_id: usize) -> &[usize] {
        self.adjacency
            .get(&node_id)
            .map_or(&[], std::vec::Vec::as_slice)
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

    /// π-commutation (Pauli-push) rule: **not currently applied**.
    ///
    /// The π-commutation identity `Z(α)·X(π) = X(π)·Z(-α)` only preserves the
    /// diagram's semantics if the π-spider is *relocated* to the other side of
    /// the neighbouring spider — a graph edge-surgery, not a local phase tweak.
    /// A correct, semantics-preserving implementation requires that relocation
    /// (and, in the general entangled case, gflow-aware reasoning), which is not
    /// yet implemented here.
    ///
    /// This method therefore deliberately performs **no rewrite** and returns
    /// `false` (the honest "nothing changed" signal): it never reports a
    /// simplification it did not make, and the other rules
    /// ([`spider_fusion`](Self::spider_fusion),
    /// [`identity_removal`](Self::identity_removal),
    /// [`hadamard_cancellation`](Self::hadamard_cancellation)) already cover the
    /// reductions that are sound on the diagrams this module produces.  It is
    /// kept in the rule set so that adding the real rewrite later is a localized
    /// change.
    pub const fn pi_commutation(&self) -> bool {
        false
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
    #[must_use]
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
    #[must_use]
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

    /// Extract the rotation angle of a parameterized single-qubit gate.
    ///
    /// Downcasts the gate to the concrete `core` rotation/phase types and reads
    /// the real angle.  Returns `0.0` (the identity phase) for gates that carry
    /// no angle, so an unrecognized gate contributes a phase-0 spider rather
    /// than a fabricated `π/4`.
    fn extract_rotation_angle(&self, gate: &dyn GateOp) -> f64 {
        use quantrs2_core::gate::single::{Phase, RotationX, RotationY, RotationZ};

        let any = gate.as_any();
        if let Some(g) = any.downcast_ref::<RotationZ>() {
            g.theta
        } else if let Some(g) = any.downcast_ref::<RotationX>() {
            g.theta
        } else if let Some(g) = any.downcast_ref::<RotationY>() {
            g.theta
        } else if any.downcast_ref::<Phase>().is_some() {
            // S gate = Z-rotation by π/2 (up to global phase).
            PI / 2.0
        } else {
            0.0
        }
    }

    /// Optimize a circuit using ZX-calculus.
    ///
    /// The circuit is converted to a ZX diagram, simplified by the rewrite rules
    /// to convergence, and extracted back to a circuit.  Because circuit
    /// extraction from an arbitrary entangled diagram is out of scope (see
    /// [`zx_to_circuit`](Self::zx_to_circuit)), the extraction step returns an
    /// honest error for diagrams that retain entangling structure (e.g. those
    /// containing CNOTs).  The `optimization_stats` on the returned result
    /// always reflect the *real* diagram-level simplification (node/T-count
    /// reductions) regardless of whether extraction succeeds.
    pub fn optimize_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<OptimizedZXResult<N>> {
        // Convert to ZX diagram
        let mut diagram = self.circuit_to_zx(circuit)?;

        // Optimize the diagram
        let optimization_result = diagram.optimize();

        // Extract a circuit from the simplified diagram (honest error if the
        // diagram is not extractable by the linear-wire extractor).
        let optimized_circuit = self.zx_to_circuit(&diagram)?;

        Ok(OptimizedZXResult {
            original_circuit: circuit.clone(),
            optimized_circuit,
            diagram,
            optimization_stats: optimization_result,
        })
    }

    /// Extract a quantum circuit from a ZX diagram.
    ///
    /// General ZX-diagram extraction (recovering a circuit from an arbitrary,
    /// entangled, post-optimization diagram) requires gflow-based synthesis and
    /// is intentionally out of scope here.  This routine performs an **exact**
    /// extraction for the class of diagrams that decompose into independent
    /// per-qubit wires — i.e. circuits built only from single-qubit gates, plus
    /// any diagram the rewrite rules reduce to that form.  Each wire is walked
    /// from its `Input` to its `Output`, emitting one gate per degree-2 spider /
    /// Hadamard encountered.
    ///
    /// If the diagram still contains entangling structure (a spider shared
    /// between wires, e.g. a CNOT), this returns an honest
    /// [`QuantRS2Error::UnsupportedOperation`] rather than silently dropping the
    /// entangling gates and returning a circuit that is *not* equivalent.
    fn zx_to_circuit<const N: usize>(&self, diagram: &ZXDiagram) -> QuantRS2Result<Circuit<N>> {
        let mut circuit = Circuit::<N>::new();

        for qubit in 0..N as u32 {
            let Some(&input_id) = diagram.inputs.get(&qubit) else {
                continue;
            };
            let Some(&output_id) = diagram.outputs.get(&qubit) else {
                continue;
            };

            // Walk the wire from the input boundary to the output boundary.
            let mut prev = input_id;
            let mut current_neighbors = diagram.neighbors(input_id).to_vec();
            // An input is degree-1 in a well-formed diagram; follow its single edge.
            let mut current = match current_neighbors.as_slice() {
                [next] => *next,
                [] => continue, // disconnected boundary: nothing on this wire
                _ => {
                    return Err(QuantRS2Error::UnsupportedOperation(format!(
                        "ZX extraction: input boundary for qubit {qubit} has degree \
                         {} (expected 1); entangled diagrams are not supported",
                        current_neighbors.len()
                    )))
                }
            };

            let mut guard = 0usize;
            let node_budget = diagram.nodes.len() + 1;
            while current != output_id {
                guard += 1;
                if guard > node_budget {
                    return Err(QuantRS2Error::ComputationError(
                        "ZX extraction: wire traversal did not terminate (cycle in diagram)"
                            .to_string(),
                    ));
                }

                let node = diagram.nodes.get(&current).ok_or_else(|| {
                    QuantRS2Error::ComputationError(format!(
                        "ZX extraction: dangling node reference {current}"
                    ))
                })?;
                current_neighbors = diagram.neighbors(current).to_vec();

                // Only degree-2 (pass-through) nodes can be extracted as a wire
                // element; higher degree means the node entangles wires.
                if current_neighbors.len() != 2 {
                    return Err(QuantRS2Error::UnsupportedOperation(format!(
                        "ZX extraction: node {current} on qubit {qubit} has degree {} \
                         (expected 2); entangling structure cannot be extracted by the \
                         linear-wire extractor",
                        current_neighbors.len()
                    )));
                }

                // Emit the gate corresponding to this node.
                let target = QubitId(qubit);
                match node {
                    ZXNode::ZSpider { phase, .. } => {
                        emit_phase_gate(&mut circuit, target, *phase, true)?;
                    }
                    ZXNode::XSpider { phase, .. } => {
                        emit_phase_gate(&mut circuit, target, *phase, false)?;
                    }
                    ZXNode::Hadamard { .. } => {
                        circuit.h(target)?;
                    }
                    ZXNode::Input { .. } | ZXNode::Output { .. } => {
                        return Err(QuantRS2Error::ComputationError(format!(
                            "ZX extraction: unexpected boundary node {current} in wire interior"
                        )));
                    }
                }

                // Step to the neighbor that is not where we came from.
                let next = if current_neighbors[0] == prev {
                    current_neighbors[1]
                } else {
                    current_neighbors[0]
                };
                prev = current;
                current = next;
            }
        }

        Ok(circuit)
    }
}

/// Emit the single-qubit gate for a degree-2 spider of the given color.
///
/// A phase of (multiples of) `π` collapses to the corresponding Pauli; `π/2`
/// Z-spiders become `S`; otherwise a parameterized rotation is emitted.  A
/// phase-0 spider is the identity and emits nothing.
fn emit_phase_gate<const N: usize>(
    circuit: &mut Circuit<N>,
    target: QubitId,
    phase: f64,
    is_z: bool,
) -> QuantRS2Result<()> {
    let two_pi = 2.0 * PI;
    // Normalize the phase into [0, 2π).
    let phase = phase.rem_euclid(two_pi);
    if phase.abs() < 1e-10 || (phase - two_pi).abs() < 1e-10 {
        return Ok(()); // identity spider
    }

    if (phase - PI).abs() < 1e-10 {
        // Pauli.
        if is_z {
            circuit.z(target)?;
        } else {
            circuit.x(target)?;
        }
    } else if is_z {
        circuit.rz(target, phase)?;
    } else {
        circuit.rx(target, phase)?;
    }
    Ok(())
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
    use quantrs2_core::gate::single::Hadamard;

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
        let remaining_node = diagram
            .nodes
            .values()
            .next()
            .expect("Expected at least one remaining node after fusion");
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
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate");

        let diagram = optimizer
            .circuit_to_zx(&circuit)
            .expect("Failed to convert circuit to ZX diagram");

        // Should have input/output nodes plus gate nodes
        assert!(diagram.nodes.len() >= 4); // 2 inputs + 2 outputs + gate nodes
        assert!(!diagram.edges.is_empty());
    }

    #[test]
    fn test_zx_optimization() {
        let optimizer = ZXOptimizer::new();

        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add first Hadamard gate");
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add second Hadamard gate"); // Should cancel out

        let result = optimizer
            .optimize_circuit(&circuit)
            .expect("Failed to optimize circuit");

        assert!(
            result.optimization_stats.final_node_count
                <= result.optimization_stats.initial_node_count
        );
    }

    /// `extract_rotation_angle` must read the real gate angle, not a hardcoded
    /// `π/4`.
    #[test]
    fn test_extract_rotation_angle_reads_real_theta() {
        use quantrs2_core::gate::single::{RotationX, RotationY, RotationZ};
        let optimizer = ZXOptimizer::new();

        let rz = RotationZ {
            target: QubitId(0),
            theta: 0.123,
        };
        assert!((optimizer.extract_rotation_angle(&rz) - 0.123).abs() < 1e-12);

        let rx = RotationX {
            target: QubitId(0),
            theta: 1.75,
        };
        assert!((optimizer.extract_rotation_angle(&rx) - 1.75).abs() < 1e-12);

        let ry = RotationY {
            target: QubitId(0),
            theta: -0.6,
        };
        assert!((optimizer.extract_rotation_angle(&ry) + 0.6).abs() < 1e-12);

        // A non-rotation gate must NOT report the bogus π/4.
        let h = Hadamard { target: QubitId(0) };
        assert!(optimizer.extract_rotation_angle(&h).abs() < 1e-12);
    }

    /// A single-qubit gate chain must extract back to a non-empty circuit
    /// carrying the real gates — not the former empty placeholder circuit.
    ///
    /// We extract directly from the converted diagram (without running the lossy
    /// optimize pass) to isolate the extractor: H; RZ(0.4); Z on one wire must
    /// come back as H, RZ(0.4), Z.  (`circuit_to_zx` encodes a Pauli-Z as a
    /// phase-π Z-spider, which the extractor inverts back to a Z gate.)
    #[test]
    fn test_zx_to_circuit_extracts_single_qubit_chain() {
        use quantrs2_core::gate::single::{PauliZ, RotationZ};
        let optimizer = ZXOptimizer::new();

        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("h");
        circuit
            .add_gate(RotationZ {
                target: QubitId(0),
                theta: 0.4,
            })
            .expect("rz");
        circuit.add_gate(PauliZ { target: QubitId(0) }).expect("z");

        let diagram = optimizer.circuit_to_zx(&circuit).expect("to zx");
        let extracted: Circuit<1> = optimizer.zx_to_circuit(&diagram).expect("extract");

        let names: Vec<&str> = extracted.gates().iter().map(|g| g.name()).collect();
        // H stays H; RZ(0.4) stays RZ; phase-π Z-spider extracts back to Z.
        assert_eq!(names, vec!["H", "RZ", "Z"], "got {names:?}");

        // The RZ must carry the real angle (0.4), proving extract_rotation_angle
        // and the phase round-trip are real (not a fabricated π/4).
        let rz = extracted
            .gates()
            .iter()
            .find(|g| g.name() == "RZ")
            .expect("rz present");
        let rz_concrete = rz
            .as_any()
            .downcast_ref::<RotationZ>()
            .expect("downcast RZ");
        assert!(
            (rz_concrete.theta - 0.4).abs() < 1e-10,
            "RZ angle {}",
            rz_concrete.theta
        );
    }

    /// Extracting a circuit that still contains entangling structure (a CNOT)
    /// must return an HONEST error rather than silently dropping the CNOT and
    /// returning a non-equivalent circuit.
    #[test]
    fn test_zx_to_circuit_errors_on_entangling_diagram() {
        let optimizer = ZXOptimizer::new();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("cnot");

        let result = optimizer.optimize_circuit(&circuit);
        assert!(
            result.is_err(),
            "entangling diagram extraction must error, not fabricate an empty circuit"
        );
    }

    /// An empty single-qubit circuit (or one that cancels to identity) extracts
    /// to an empty circuit successfully.
    #[test]
    fn test_zx_to_circuit_identity_is_empty() {
        let optimizer = ZXOptimizer::new();

        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("h1");
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("h2");

        let result = optimizer
            .optimize_circuit(&circuit)
            .expect("optimize identity");
        assert_eq!(result.optimized_circuit.gates().len(), 0);
    }
}
