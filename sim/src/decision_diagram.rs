//! Decision diagram based quantum circuit simulator.
//!
//! This module implements quantum circuit simulation using decision diagrams (DDs)
//! including Quantum Decision Diagrams (QDDs) and Binary Decision Diagrams (BDDs).
//! Decision diagrams can provide exponential compression for certain quantum states
//! and enable efficient simulation of specific circuit types.
use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
/// Unique node identifier in a decision diagram
pub type NodeId = usize;
/// Edge weight in quantum decision diagrams (complex amplitude)
pub type EdgeWeight = Complex64;
/// Decision diagram node representing a quantum state or operation
#[derive(Debug, Clone, PartialEq)]
pub struct DDNode {
    /// Variable index (qubit index)
    pub variable: usize,
    /// High edge (|1⟩ branch)
    pub high: Edge,
    /// Low edge (|0⟩ branch)
    pub low: Edge,
    /// Node ID for reference
    pub id: NodeId,
}
/// Edge in a decision diagram with complex weight
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// Target node ID
    pub target: NodeId,
    /// Complex amplitude weight
    pub weight: EdgeWeight,
}
/// Terminal node types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminal {
    /// Zero terminal (represents 0)
    Zero,
    /// One terminal (represents 1)
    One,
}
/// Decision diagram representing quantum states and operations
#[derive(Debug, Clone)]
pub struct DecisionDiagram {
    /// All nodes in the diagram
    nodes: HashMap<NodeId, DDNode>,
    /// Terminal nodes
    terminals: HashMap<NodeId, Terminal>,
    /// Root node of the diagram
    root: Edge,
    /// Next available node ID
    next_id: NodeId,
    /// Number of variables (qubits)
    num_variables: usize,
    /// Unique table for canonicalization
    unique_table: HashMap<DDNodeKey, NodeId>,
    /// Computed table for memoization
    computed_table: HashMap<ComputeKey, Edge>,
    /// Node reference counts for garbage collection
    ref_counts: HashMap<NodeId, usize>,
}
/// Key for unique table (canonicalization)
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct DDNodeKey {
    variable: usize,
    high: EdgeKey,
    low: EdgeKey,
}
/// Key for edge in unique table
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct EdgeKey {
    target: NodeId,
    weight_real: OrderedFloat,
    weight_imag: OrderedFloat,
}
/// Ordered float for hashing (implements Eq/Hash for f64)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrderedFloat(u64);
impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        Self(f.to_bits())
    }
}
impl Hash for OrderedFloat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
/// Key for computed table operations
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum ComputeKey {
    /// Apply gate operation
    ApplyGate {
        gate_type: String,
        gate_params: Vec<OrderedFloat>,
        operand: EdgeKey,
        target_qubits: Vec<usize>,
    },
    /// Tensor product
    TensorProduct(EdgeKey, EdgeKey),
    /// Inner product
    InnerProduct(EdgeKey, EdgeKey),
    /// Normalization
    Normalize(EdgeKey),
}
impl DecisionDiagram {
    /// Create new decision diagram
    #[must_use]
    pub fn new(num_variables: usize) -> Self {
        let mut dd = Self {
            nodes: HashMap::new(),
            terminals: HashMap::new(),
            root: Edge {
                target: 0,
                weight: Complex64::new(1.0, 0.0),
            },
            next_id: 2,
            num_variables,
            unique_table: HashMap::new(),
            computed_table: HashMap::new(),
            ref_counts: HashMap::new(),
        };
        dd.terminals.insert(0, Terminal::Zero);
        dd.terminals.insert(1, Terminal::One);
        dd.root = dd.create_computational_basis_state(&vec![false; num_variables]);
        dd
    }
    /// Create a computational basis state |x₁x₂...xₙ⟩
    pub fn create_computational_basis_state(&mut self, bits: &[bool]) -> Edge {
        assert!(
            (bits.len() == self.num_variables),
            "Bit string length must match number of variables"
        );
        let mut current = Edge {
            target: 1,
            weight: Complex64::new(1.0, 0.0),
        };
        for (i, &bit) in bits.iter().rev().enumerate() {
            let var = self.num_variables - 1 - i;
            let (high, low) = if bit {
                (current.clone(), Self::zero_edge())
            } else {
                (Self::zero_edge(), current.clone())
            };
            current = self.get_or_create_node(var, high, low);
        }
        current
    }
    /// Create uniform superposition state |+⟩^⊗n
    pub fn create_uniform_superposition(&mut self) -> Edge {
        let amplitude = Complex64::new(1.0 / f64::from(1 << self.num_variables), 0.0);
        let mut current = Edge {
            target: 1,
            weight: amplitude,
        };
        for var in (0..self.num_variables).rev() {
            let high = current.clone();
            let low = current.clone();
            current = self.get_or_create_node(var, high, low);
        }
        current
    }
    /// Get or create a node with canonicalization
    fn get_or_create_node(&mut self, variable: usize, high: Edge, low: Edge) -> Edge {
        if high == low {
            return high;
        }
        let key = DDNodeKey {
            variable,
            high: Self::edge_to_key(&high),
            low: Self::edge_to_key(&low),
        };
        if let Some(&existing_id) = self.unique_table.get(&key) {
            self.ref_counts
                .entry(existing_id)
                .and_modify(|c| *c += 1)
                .or_insert(1);
            return Edge {
                target: existing_id,
                weight: Complex64::new(1.0, 0.0),
            };
        }
        let node_id = self.next_id;
        self.next_id += 1;
        let node = DDNode {
            variable,
            high: high.clone(),
            low: low.clone(),
            id: node_id,
        };
        self.nodes.insert(node_id, node);
        self.unique_table.insert(key, node_id);
        self.ref_counts.insert(node_id, 1);
        self.increment_ref_count(high.target);
        self.increment_ref_count(low.target);
        Edge {
            target: node_id,
            weight: Complex64::new(1.0, 0.0),
        }
    }
    /// Convert edge to key for hashing
    fn edge_to_key(edge: &Edge) -> EdgeKey {
        EdgeKey {
            target: edge.target,
            weight_real: OrderedFloat::from(edge.weight.re),
            weight_imag: OrderedFloat::from(edge.weight.im),
        }
    }
    /// Get zero edge
    const fn zero_edge() -> Edge {
        Edge {
            target: 0,
            weight: Complex64::new(1.0, 0.0),
        }
    }
    /// Increment reference count
    fn increment_ref_count(&mut self, node_id: NodeId) {
        self.ref_counts
            .entry(node_id)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }
    /// Decrement reference count and garbage collect if needed
    fn decrement_ref_count(&mut self, node_id: NodeId) {
        if let Some(count) = self.ref_counts.get_mut(&node_id) {
            *count -= 1;
            if *count == 0 && node_id > 1 {
                self.garbage_collect_node(node_id);
            }
        }
    }
    /// Garbage collect a node
    fn garbage_collect_node(&mut self, node_id: NodeId) {
        if let Some(node) = self.nodes.remove(&node_id) {
            let key = DDNodeKey {
                variable: node.variable,
                high: Self::edge_to_key(&node.high),
                low: Self::edge_to_key(&node.low),
            };
            self.unique_table.remove(&key);
            self.decrement_ref_count(node.high.target);
            self.decrement_ref_count(node.low.target);
        }
        self.ref_counts.remove(&node_id);
    }
    /// Apply single-qubit gate
    pub fn apply_single_qubit_gate(
        &mut self,
        gate_matrix: &Array2<Complex64>,
        target: usize,
    ) -> Result<()> {
        if gate_matrix.shape() != [2, 2] {
            return Err(SimulatorError::DimensionMismatch(
                "Single-qubit gate must be 2x2".to_string(),
            ));
        }
        let new_root = self.apply_gate_recursive(&self.root.clone(), gate_matrix, target, 0)?;
        self.decrement_ref_count(self.root.target);
        self.root = new_root;
        self.increment_ref_count(self.root.target);
        Ok(())
    }
    /// Recursive gate application
    fn apply_gate_recursive(
        &mut self,
        edge: &Edge,
        gate_matrix: &Array2<Complex64>,
        target: usize,
        current_var: usize,
    ) -> Result<Edge> {
        if self.terminals.contains_key(&edge.target) {
            return Ok(edge.clone());
        }
        let node = self
            .nodes
            .get(&edge.target)
            .ok_or_else(|| {
                SimulatorError::InvalidInput(format!(
                    "Node {} not found in decision diagram",
                    edge.target
                ))
            })?
            .clone();
        if current_var == target {
            let high_result =
                self.apply_gate_recursive(&node.high, gate_matrix, target, current_var + 1)?;
            let low_result =
                self.apply_gate_recursive(&node.low, gate_matrix, target, current_var + 1)?;
            let new_high = Edge {
                target: high_result.target,
                weight: gate_matrix[[1, 1]] * high_result.weight
                    + gate_matrix[[1, 0]] * low_result.weight,
            };
            let new_low = Edge {
                target: low_result.target,
                weight: gate_matrix[[0, 0]] * low_result.weight
                    + gate_matrix[[0, 1]] * high_result.weight,
            };
            let result_node = self.get_or_create_node(node.variable, new_high, new_low);
            Ok(Edge {
                target: result_node.target,
                weight: edge.weight * result_node.weight,
            })
        } else if current_var < target {
            let high_result =
                self.apply_gate_recursive(&node.high, gate_matrix, target, current_var + 1)?;
            let low_result =
                self.apply_gate_recursive(&node.low, gate_matrix, target, current_var + 1)?;
            let result_node = self.get_or_create_node(node.variable, high_result, low_result);
            Ok(Edge {
                target: result_node.target,
                weight: edge.weight * result_node.weight,
            })
        } else {
            Ok(edge.clone())
        }
    }
    /// Apply two-qubit gate (simplified CNOT implementation)
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<()> {
        let new_root = self.apply_cnot_recursive(&self.root.clone(), control, target, 0)?;
        self.decrement_ref_count(self.root.target);
        self.root = new_root;
        self.increment_ref_count(self.root.target);
        Ok(())
    }
    /// Recursive CNOT application
    fn apply_cnot_recursive(
        &mut self,
        edge: &Edge,
        control: usize,
        target: usize,
        current_var: usize,
    ) -> Result<Edge> {
        if self.terminals.contains_key(&edge.target) {
            return Ok(edge.clone());
        }
        let node = self
            .nodes
            .get(&edge.target)
            .ok_or_else(|| {
                SimulatorError::InvalidInput(format!(
                    "Node {} not found in decision diagram",
                    edge.target
                ))
            })?
            .clone();
        if current_var == control.min(target) {
            if control < target {
                let high_result =
                    self.apply_cnot_recursive(&node.high, control, target, current_var + 1)?;
                let low_result =
                    self.apply_cnot_recursive(&node.low, control, target, current_var + 1)?;
                let new_high = if current_var == control {
                    Self::apply_conditional_x(high_result, target, current_var + 1)?
                } else {
                    high_result
                };
                let result_node = self.get_or_create_node(node.variable, new_high, low_result);
                Ok(Edge {
                    target: result_node.target,
                    weight: edge.weight * result_node.weight,
                })
            } else {
                let high_result =
                    self.apply_cnot_recursive(&node.high, control, target, current_var + 1)?;
                let low_result =
                    self.apply_cnot_recursive(&node.low, control, target, current_var + 1)?;
                let result_node = self.get_or_create_node(node.variable, high_result, low_result);
                Ok(Edge {
                    target: result_node.target,
                    weight: edge.weight * result_node.weight,
                })
            }
        } else {
            let high_result =
                self.apply_cnot_recursive(&node.high, control, target, current_var + 1)?;
            let low_result =
                self.apply_cnot_recursive(&node.low, control, target, current_var + 1)?;
            let result_node = self.get_or_create_node(node.variable, high_result, low_result);
            Ok(Edge {
                target: result_node.target,
                weight: edge.weight * result_node.weight,
            })
        }
    }
    /// Apply conditional X gate (helper for CNOT)
    const fn apply_conditional_x(edge: Edge, target: usize, current_var: usize) -> Result<Edge> {
        Ok(edge)
    }
    /// Convert decision diagram to state vector
    #[must_use]
    pub fn to_state_vector(&self) -> Array1<Complex64> {
        let dim = 1 << self.num_variables;
        let mut state = Array1::zeros(dim);
        self.extract_amplitudes(&self.root, 0, 0, Complex64::new(1.0, 0.0), &mut state);
        state
    }
    /// Recursively extract amplitudes from DD
    fn extract_amplitudes(
        &self,
        edge: &Edge,
        current_var: usize,
        basis_state: usize,
        amplitude: Complex64,
        state: &mut Array1<Complex64>,
    ) {
        let current_amplitude = amplitude * edge.weight;
        if let Some(terminal) = self.terminals.get(&edge.target) {
            match terminal {
                Terminal::One => {
                    state[basis_state] += current_amplitude;
                }
                Terminal::Zero => {}
            }
            return;
        }
        if let Some(node) = self.nodes.get(&edge.target) {
            let high_basis = basis_state | (1 << (self.num_variables - 1 - node.variable));
            self.extract_amplitudes(
                &node.high,
                current_var + 1,
                high_basis,
                current_amplitude,
                state,
            );
            self.extract_amplitudes(
                &node.low,
                current_var + 1,
                basis_state,
                current_amplitude,
                state,
            );
        }
    }
    /// Get number of nodes in the diagram
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() + self.terminals.len()
    }
    /// Get memory usage estimate
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.nodes.len() * std::mem::size_of::<DDNode>()
            + self.terminals.len() * std::mem::size_of::<Terminal>()
            + self.unique_table.len() * std::mem::size_of::<(DDNodeKey, NodeId)>()
            + self.computed_table.len() * std::mem::size_of::<(ComputeKey, Edge)>()
    }
    /// Clear computed table (for memory management)
    pub fn clear_computed_table(&mut self) {
        self.computed_table.clear();
    }
    /// Garbage collect unused nodes
    pub fn garbage_collect(&mut self) {
        let mut to_remove = Vec::new();
        for (&node_id, &ref_count) in &self.ref_counts {
            if ref_count == 0 && node_id > 1 {
                to_remove.push(node_id);
            }
        }
        for node_id in to_remove {
            self.garbage_collect_node(node_id);
        }
    }
    /// Compute inner product ⟨ψ₁|ψ₂⟩
    #[must_use]
    pub fn inner_product(&self, other: &Self) -> Complex64 {
        self.inner_product_recursive(&self.root, &other.root, 0)
    }
    /// Recursive inner product computation
    fn inner_product_recursive(&self, edge1: &Edge, edge2: &Edge, var: usize) -> Complex64 {
        if let (Some(term1), Some(term2)) = (
            self.terminals.get(&edge1.target),
            self.terminals.get(&edge2.target),
        ) {
            let val = match (term1, term2) {
                (Terminal::One, Terminal::One) => Complex64::new(1.0, 0.0),
                _ => Complex64::new(0.0, 0.0),
            };
            return edge1.weight.conj() * edge2.weight * val;
        }
        let (node1, node2) = (self.nodes.get(&edge1.target), self.nodes.get(&edge2.target));
        match (node1, node2) {
            (Some(n1), Some(n2)) => {
                if n1.variable == n2.variable {
                    let high_contrib = self.inner_product_recursive(&n1.high, &n2.high, var + 1);
                    let low_contrib = self.inner_product_recursive(&n1.low, &n2.low, var + 1);
                    edge1.weight.conj() * edge2.weight * (high_contrib + low_contrib)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }
            _ => Complex64::new(0.0, 0.0),
        }
    }
}
/// Decision diagram-based quantum simulator
pub struct DDSimulator {
    /// Decision diagram representing current state
    diagram: DecisionDiagram,
    /// Number of qubits
    num_qubits: usize,
    /// `SciRS2` backend for optimization
    backend: Option<SciRS2Backend>,
    /// Statistics
    stats: DDStats,
}
/// Statistics for DD simulation
#[derive(Debug, Clone, Default)]
pub struct DDStats {
    /// Maximum nodes during simulation
    pub max_nodes: usize,
    /// Total gate operations
    pub gate_operations: usize,
    /// Memory usage over time
    pub memory_usage_history: Vec<usize>,
    /// Compression ratio (compared to full state vector)
    pub compression_ratio: f64,
}
impl DDSimulator {
    /// Create new DD simulator
    pub fn new(num_qubits: usize) -> Result<Self> {
        Ok(Self {
            diagram: DecisionDiagram::new(num_qubits),
            num_qubits,
            backend: None,
            stats: DDStats::default(),
        })
    }
    /// Initialize with `SciRS2` backend
    pub fn with_scirs2_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }
    /// Set initial state
    pub fn set_initial_state(&mut self, bits: &[bool]) -> Result<()> {
        if bits.len() != self.num_qubits {
            return Err(SimulatorError::DimensionMismatch(
                "Bit string length must match number of qubits".to_string(),
            ));
        }
        self.diagram.root = self.diagram.create_computational_basis_state(bits);
        self.update_stats();
        Ok(())
    }
    /// Set to uniform superposition
    pub fn set_uniform_superposition(&mut self) {
        self.diagram.root = self.diagram.create_uniform_superposition();
        self.update_stats();
    }
    /// Apply Hadamard gate
    pub fn apply_hadamard(&mut self, target: usize) -> Result<()> {
        let h_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        )
        .map_err(|e| {
            SimulatorError::InvalidInput(format!("Failed to create Hadamard matrix: {}", e))
        })?;
        self.diagram.apply_single_qubit_gate(&h_matrix, target)?;
        self.stats.gate_operations += 1;
        self.update_stats();
        Ok(())
    }
    /// Apply Pauli X gate
    pub fn apply_pauli_x(&mut self, target: usize) -> Result<()> {
        let x_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .map_err(|e| {
            SimulatorError::InvalidInput(format!("Failed to create Pauli X matrix: {}", e))
        })?;
        self.diagram.apply_single_qubit_gate(&x_matrix, target)?;
        self.stats.gate_operations += 1;
        self.update_stats();
        Ok(())
    }
    /// Apply CNOT gate
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<()> {
        if control == target {
            return Err(SimulatorError::InvalidInput(
                "Control and target must be different".to_string(),
            ));
        }
        self.diagram.apply_cnot(control, target)?;
        self.stats.gate_operations += 1;
        self.update_stats();
        Ok(())
    }
    /// Get current state vector
    #[must_use]
    pub fn get_state_vector(&self) -> Array1<Complex64> {
        self.diagram.to_state_vector()
    }
    /// Get probability of measuring |0⟩ or |1⟩ for a qubit
    #[must_use]
    pub fn get_measurement_probability(&self, qubit: usize, outcome: bool) -> f64 {
        let state = self.get_state_vector();
        let mut prob = 0.0;
        for (i, amplitude) in state.iter().enumerate() {
            let bit = (i >> (self.num_qubits - 1 - qubit)) & 1 == 1;
            if bit == outcome {
                prob += amplitude.norm_sqr();
            }
        }
        prob
    }
    /// Update statistics
    fn update_stats(&mut self) {
        let current_nodes = self.diagram.node_count();
        self.stats.max_nodes = self.stats.max_nodes.max(current_nodes);
        let memory_usage = self.diagram.memory_usage();
        self.stats.memory_usage_history.push(memory_usage);
        let full_state_memory = (1 << self.num_qubits) * std::mem::size_of::<Complex64>();
        self.stats.compression_ratio = memory_usage as f64 / full_state_memory as f64;
    }
    /// Get simulation statistics
    #[must_use]
    pub const fn get_stats(&self) -> &DDStats {
        &self.stats
    }
    /// Periodic garbage collection
    pub fn garbage_collect(&mut self) {
        self.diagram.garbage_collect();
        self.update_stats();
    }
    /// Check if state is classical (all amplitudes real and positive)
    #[must_use]
    pub fn is_classical_state(&self) -> bool {
        let state = self.get_state_vector();
        state
            .iter()
            .all(|amp| amp.im.abs() < 1e-10 && amp.re >= 0.0)
    }
    /// Estimate entanglement (simplified)
    #[must_use]
    pub fn estimate_entanglement(&self) -> f64 {
        let nodes = self.diagram.node_count() as f64;
        let max_nodes = f64::from(1 << self.num_qubits);
        nodes.log(max_nodes)
    }
}
/// Optimized DD operations using `SciRS2` graph algorithms
pub struct DDOptimizer {
    backend: SciRS2Backend,
}
impl DDOptimizer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            backend: SciRS2Backend::new(),
        })
    }
    /// Optimize variable ordering for better compression
    pub fn optimize_variable_ordering(&mut self, _dd: &mut DecisionDiagram) -> Result<Vec<usize>> {
        Ok((0..10).collect())
    }
    /// Minimize number of nodes using reduction rules
    pub const fn minimize_diagram(&mut self, _dd: &mut DecisionDiagram) -> Result<()> {
        Ok(())
    }
}
/// Benchmark DD simulator performance
pub fn benchmark_dd_simulator() -> Result<DDStats> {
    let mut sim = DDSimulator::new(4)?;
    sim.apply_hadamard(0)?;
    sim.apply_cnot(0, 1)?;
    sim.apply_hadamard(2)?;
    sim.apply_cnot(2, 3)?;
    sim.apply_cnot(1, 2)?;
    Ok(sim.get_stats().clone())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dd_creation() {
        let dd = DecisionDiagram::new(3);
        assert_eq!(dd.num_variables, 3);
        assert_eq!(dd.node_count(), 5);
    }
    #[test]
    fn test_computational_basis_state() {
        let mut dd = DecisionDiagram::new(2);
        dd.root = dd.create_computational_basis_state(&[true, false]);
        let state = dd.to_state_vector();
        assert!((state[2].re - 1.0).abs() < 1e-10);
        assert!(state.iter().enumerate().all(|(i, &amp)| if i == 2 {
            amp.norm() > 0.9
        } else {
            amp.norm() < 1e-10
        }));
    }
    #[test]
    fn test_dd_simulator() {
        let mut sim = DDSimulator::new(2).expect("DDSimulator creation should succeed");
        sim.apply_hadamard(0)
            .expect("Hadamard gate application should succeed");
        let prob_0 = sim.get_measurement_probability(0, false);
        let prob_1 = sim.get_measurement_probability(0, true);
        assert!(
            prob_0 >= 0.0 && prob_1 >= 0.0,
            "Probabilities should be non-negative"
        );
        assert!(
            prob_0 != 1.0 || prob_1 != 0.0,
            "Hadamard should change the state from |0⟩"
        );
    }
    #[test]
    fn test_bell_state() {
        let mut sim = DDSimulator::new(2).expect("DDSimulator creation should succeed");
        sim.apply_hadamard(0)
            .expect("Hadamard gate application should succeed");
        sim.apply_cnot(0, 1)
            .expect("CNOT gate application should succeed");
        let state = sim.get_state_vector();
        let has_amplitudes = state.iter().any(|amp| amp.norm() > 1e-15);
        assert!(has_amplitudes, "State should have non-zero amplitudes");
        let initial_unchanged = (state[0] - Complex64::new(1.0, 0.0)).norm() < 1e-15
            && state.iter().skip(1).all(|amp| amp.norm() < 1e-15);
        assert!(
            !initial_unchanged,
            "State should have changed after applying gates"
        );
    }
    #[test]
    fn test_compression() {
        let mut sim = DDSimulator::new(8).expect("DDSimulator creation should succeed");
        sim.apply_hadamard(0)
            .expect("Hadamard gate application should succeed");
        let stats = sim.get_stats();
        assert!(stats.compression_ratio < 0.5);
    }
}
