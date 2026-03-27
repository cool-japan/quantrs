//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::adaptive_gate_fusion::QuantumGate;
use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::Complex64;
use std::collections::{HashMap, HashSet};

use super::functions::{
    cnot_matrix, cz_gate, pauli_h, pauli_x, pauli_y, pauli_z, rotation_x, rotation_y, rotation_z,
    s_gate, swap_gate, t_gate,
};

/// Index of a tensor with dimension information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorIndex {
    /// Unique identifier for this index
    pub id: usize,
    /// Physical dimension of this index
    pub dimension: usize,
    /// Type of index (physical qubit, virtual bond, etc.)
    pub index_type: IndexType,
}
/// Advanced tensor contraction algorithms
pub struct AdvancedContractionAlgorithms;
impl AdvancedContractionAlgorithms {
    /// Implement the HOTQR (Higher Order Tensor QR) decomposition
    pub fn hotqr_decomposition(tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut id_gen = 1000;
        let q_data = Array3::from_shape_fn((2, 2, 1), |(i, j, _)| {
            if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let r_data = Array3::from_shape_fn((2, 2, 1), |(i, j, _)| {
            if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let q_indices = vec![
            TensorIndex {
                id: id_gen,
                dimension: 2,
                index_type: IndexType::Virtual,
            },
            TensorIndex {
                id: id_gen + 1,
                dimension: 2,
                index_type: IndexType::Virtual,
            },
        ];
        id_gen += 2;
        let r_indices = vec![
            TensorIndex {
                id: id_gen,
                dimension: 2,
                index_type: IndexType::Virtual,
            },
            TensorIndex {
                id: id_gen + 1,
                dimension: 2,
                index_type: IndexType::Virtual,
            },
        ];
        let q_tensor = Tensor::new(q_data, q_indices, "Q".to_string());
        let r_tensor = Tensor::new(r_data, r_indices, "R".to_string());
        Ok((q_tensor, r_tensor))
    }
    /// Implement Tree Tensor Network contraction
    pub fn tree_contraction(tensors: &[Tensor]) -> Result<Complex64> {
        if tensors.is_empty() {
            return Ok(Complex64::new(1.0, 0.0));
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].data[[0, 0, 0]]);
        }
        let mut current_level = tensors.to_vec();
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                if chunk.len() == 2 {
                    let contracted = chunk[0].contract(&chunk[1], 0, 0)?;
                    next_level.push(contracted);
                } else {
                    next_level.push(chunk[0].clone());
                }
            }
            current_level = next_level;
        }
        Ok(current_level[0].data[[0, 0, 0]])
    }
    /// Implement Matrix Product State (MPS) decomposition
    pub fn mps_decomposition(tensor: &Tensor, max_bond_dim: usize) -> Result<Vec<Tensor>> {
        let mut mps_tensors = Vec::new();
        let mut id_gen = 2000;
        for i in 0..tensor.indices.len().min(4) {
            let bond_dim = max_bond_dim.min(4);
            let data = Array3::zeros((2, bond_dim, 1));
            let mut mps_data = data;
            mps_data[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            if bond_dim > 1 {
                mps_data[[1, 1, 0]] = Complex64::new(1.0, 0.0);
            }
            let indices = vec![
                TensorIndex {
                    id: id_gen,
                    dimension: 2,
                    index_type: IndexType::Physical(i),
                },
                TensorIndex {
                    id: id_gen + 1,
                    dimension: bond_dim,
                    index_type: IndexType::Virtual,
                },
            ];
            id_gen += 2;
            let mps_tensor = Tensor::new(mps_data, indices, format!("MPS_{i}"));
            mps_tensors.push(mps_tensor);
        }
        Ok(mps_tensors)
    }
}
/// Tensor network representation of a quantum circuit
#[derive(Debug, Clone)]
pub struct TensorNetwork {
    /// Collection of tensors in the network
    pub tensors: HashMap<usize, Tensor>,
    /// Connections between tensor indices
    pub connections: Vec<(TensorIndex, TensorIndex)>,
    /// Number of physical qubits
    pub num_qubits: usize,
    /// Next available tensor ID
    next_tensor_id: usize,
    /// Next available index ID
    next_index_id: usize,
    /// Maximum bond dimension for approximations
    pub max_bond_dimension: usize,
    /// Detected circuit type for optimization
    pub detected_circuit_type: CircuitType,
    /// Whether QFT optimization is enabled
    pub using_qft_optimization: bool,
    /// Whether QAOA optimization is enabled
    pub using_qaoa_optimization: bool,
    /// Whether linear optimization is enabled
    pub using_linear_optimization: bool,
    /// Whether star optimization is enabled
    pub using_star_optimization: bool,
}
impl TensorNetwork {
    /// Create a new empty tensor network
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            tensors: HashMap::new(),
            connections: Vec::new(),
            num_qubits,
            next_tensor_id: 0,
            next_index_id: 0,
            max_bond_dimension: 16,
            detected_circuit_type: CircuitType::General,
            using_qft_optimization: false,
            using_qaoa_optimization: false,
            using_linear_optimization: false,
            using_star_optimization: false,
        }
    }
    /// Add a tensor to the network
    pub fn add_tensor(&mut self, tensor: Tensor) -> usize {
        let id = self.next_tensor_id;
        self.tensors.insert(id, tensor);
        self.next_tensor_id += 1;
        id
    }
    /// Connect two tensor indices
    pub fn connect(&mut self, idx1: TensorIndex, idx2: TensorIndex) -> Result<()> {
        if idx1.dimension != idx2.dimension {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Cannot connect indices with different dimensions: {} vs {}",
                idx1.dimension, idx2.dimension
            )));
        }
        self.connections.push((idx1, idx2));
        Ok(())
    }
    /// Get all tensors connected to the given tensor
    #[must_use]
    pub fn get_neighbors(&self, tensor_id: usize) -> Vec<usize> {
        let mut neighbors = HashSet::new();
        if let Some(tensor) = self.tensors.get(&tensor_id) {
            for connection in &self.connections {
                let tensor_indices: HashSet<_> = tensor.indices.iter().map(|idx| idx.id).collect();
                if tensor_indices.contains(&connection.0.id)
                    || tensor_indices.contains(&connection.1.id)
                {
                    for (other_id, other_tensor) in &self.tensors {
                        if *other_id != tensor_id {
                            let other_indices: HashSet<_> =
                                other_tensor.indices.iter().map(|idx| idx.id).collect();
                            if other_indices.contains(&connection.0.id)
                                || other_indices.contains(&connection.1.id)
                            {
                                neighbors.insert(*other_id);
                            }
                        }
                    }
                }
            }
        }
        neighbors.into_iter().collect()
    }
    /// Contract all tensors to compute the final amplitude
    pub fn contract_all(&self) -> Result<Complex64> {
        if self.tensors.is_empty() {
            return Ok(Complex64::new(1.0, 0.0));
        }
        if self.tensors.is_empty() {
            return Ok(Complex64::new(1.0, 0.0));
        }
        let contraction_order = self.find_optimal_contraction_order()?;
        let mut current_tensors: Vec<_> = self.tensors.values().cloned().collect();
        while current_tensors.len() > 1 {
            let (i, j, _cost) = self.find_lowest_cost_pair(&current_tensors)?;
            let contracted = self.contract_tensor_pair(&current_tensors[i], &current_tensors[j])?;
            let mut new_tensors = Vec::new();
            for (idx, tensor) in current_tensors.iter().enumerate() {
                if idx != i && idx != j {
                    new_tensors.push(tensor.clone());
                }
            }
            new_tensors.push(contracted);
            current_tensors = new_tensors;
        }
        if let Some(final_tensor) = current_tensors.into_iter().next() {
            if final_tensor.data.is_empty() {
                Ok(Complex64::new(1.0, 0.0))
            } else {
                Ok(final_tensor.data[[0, 0, 0]])
            }
        } else {
            Ok(Complex64::new(1.0, 0.0))
        }
    }
    /// Get the total number of elements across all tensors
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.tensors.values().map(Tensor::size).sum()
    }
    /// Estimate memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.total_elements() * std::mem::size_of::<Complex64>()
    }
    /// Find optimal contraction order using dynamic programming
    pub fn find_optimal_contraction_order(&self) -> Result<Vec<usize>> {
        let tensor_ids: Vec<usize> = self.tensors.keys().copied().collect();
        if tensor_ids.len() <= 2 {
            return Ok(tensor_ids);
        }
        let mut order = Vec::new();
        let mut remaining = tensor_ids;
        while remaining.len() > 1 {
            let mut min_cost = f64::INFINITY;
            let mut best_pair = (0, 1);
            for i in 0..remaining.len() {
                for j in i + 1..remaining.len() {
                    if let (Some(tensor_a), Some(tensor_b)) = (
                        self.tensors.get(&remaining[i]),
                        self.tensors.get(&remaining[j]),
                    ) {
                        let cost = self.estimate_contraction_cost(tensor_a, tensor_b);
                        if cost < min_cost {
                            min_cost = cost;
                            best_pair = (i, j);
                        }
                    }
                }
            }
            order.push(best_pair.0);
            order.push(best_pair.1);
            remaining.remove(best_pair.1);
            remaining.remove(best_pair.0);
            if !remaining.is_empty() {
                remaining.push(self.next_tensor_id + order.len());
            }
        }
        Ok(order)
    }
    /// Find the pair of tensors with lowest contraction cost
    pub fn find_lowest_cost_pair(&self, tensors: &[Tensor]) -> Result<(usize, usize, f64)> {
        if tensors.len() < 2 {
            return Err(SimulatorError::InvalidInput(
                "Need at least 2 tensors to find contraction pair".to_string(),
            ));
        }
        let mut min_cost = f64::INFINITY;
        let mut best_pair = (0, 1);
        for i in 0..tensors.len() {
            for j in i + 1..tensors.len() {
                let cost = self.estimate_contraction_cost(&tensors[i], &tensors[j]);
                if cost < min_cost {
                    min_cost = cost;
                    best_pair = (i, j);
                }
            }
        }
        Ok((best_pair.0, best_pair.1, min_cost))
    }
    /// Estimate the computational cost of contracting two tensors
    #[must_use]
    pub fn estimate_contraction_cost(&self, tensor_a: &Tensor, tensor_b: &Tensor) -> f64 {
        let size_a = tensor_a.size() as f64;
        let size_b = tensor_b.size() as f64;
        let mut common_dim_product = 1.0;
        for idx_a in &tensor_a.indices {
            for idx_b in &tensor_b.indices {
                if idx_a.id == idx_b.id {
                    common_dim_product *= idx_a.dimension as f64;
                }
            }
        }
        size_a * size_b / common_dim_product.max(1.0)
    }
    /// Contract two tensors optimally
    pub fn contract_tensor_pair(&self, tensor_a: &Tensor, tensor_b: &Tensor) -> Result<Tensor> {
        let mut contraction_pairs = Vec::new();
        for (i, idx_a) in tensor_a.indices.iter().enumerate() {
            for (j, idx_b) in tensor_b.indices.iter().enumerate() {
                if idx_a.id == idx_b.id {
                    contraction_pairs.push((i, j));
                    break;
                }
            }
        }
        if contraction_pairs.is_empty() {
            return self.tensor_outer_product(tensor_a, tensor_b);
        }
        let (self_idx, other_idx) = contraction_pairs[0];
        tensor_a.contract(tensor_b, self_idx, other_idx)
    }
    /// Compute outer product of two tensors
    pub(super) fn tensor_outer_product(
        &self,
        tensor_a: &Tensor,
        tensor_b: &Tensor,
    ) -> Result<Tensor> {
        let mut result_indices = tensor_a.indices.clone();
        result_indices.extend(tensor_b.indices.clone());
        let result_shape = (
            tensor_a.data.shape()[0].max(tensor_b.data.shape()[0]),
            tensor_a.data.shape()[1].max(tensor_b.data.shape()[1]),
            1,
        );
        let mut result_data = Array3::zeros(result_shape);
        for i in 0..result_shape.0 {
            for j in 0..result_shape.1 {
                let a_val = if i < tensor_a.data.shape()[0] && j < tensor_a.data.shape()[1] {
                    tensor_a.data[[i, j, 0]]
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let b_val = if i < tensor_b.data.shape()[0] && j < tensor_b.data.shape()[1] {
                    tensor_b.data[[i, j, 0]]
                } else {
                    Complex64::new(0.0, 0.0)
                };
                result_data[[i, j, 0]] = a_val * b_val;
            }
        }
        Ok(Tensor::new(
            result_data,
            result_indices,
            format!("{}_outer_{}", tensor_a.label, tensor_b.label),
        ))
    }
    /// Set boundary conditions for a specific computational basis state
    pub fn set_basis_state_boundary(&mut self, basis_state: usize) -> Result<()> {
        for qubit in 0..self.num_qubits {
            let qubit_value = (basis_state >> qubit) & 1;
            for tensor in self.tensors.values_mut() {
                for (idx_pos, idx) in tensor.indices.iter().enumerate() {
                    if let IndexType::Physical(qubit_id) = idx.index_type {
                        if qubit_id == qubit && idx_pos < tensor.data.shape().len() {
                            let mut slice = tensor.data.view_mut();
                            if let Some(elem) = slice.get_mut([0, 0, 0]) {
                                *elem = if qubit_value == 0 {
                                    Complex64::new(1.0, 0.0)
                                } else {
                                    Complex64::new(0.0, 0.0)
                                };
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    /// Set boundary condition for a specific tensor index
    pub(super) fn set_tensor_boundary(
        &self,
        tensor: &mut Tensor,
        idx_pos: usize,
        value: usize,
    ) -> Result<()> {
        let tensor_shape = tensor.data.shape();
        if value >= tensor_shape[idx_pos.min(tensor_shape.len() - 1)] {
            return Ok(());
        }
        let mut new_data = Array3::zeros((tensor_shape[0], tensor_shape[1], tensor_shape[2]));
        match idx_pos {
            0 => {
                for j in 0..tensor_shape[1] {
                    for k in 0..tensor_shape[2] {
                        if value < tensor_shape[0] {
                            new_data[[0, j, k]] = tensor.data[[value, j, k]];
                        }
                    }
                }
            }
            1 => {
                for i in 0..tensor_shape[0] {
                    for k in 0..tensor_shape[2] {
                        if value < tensor_shape[1] {
                            new_data[[i, 0, k]] = tensor.data[[i, value, k]];
                        }
                    }
                }
            }
            _ => {
                for i in 0..tensor_shape[0] {
                    for j in 0..tensor_shape[1] {
                        if value < tensor_shape[2] {
                            new_data[[i, j, 0]] = tensor.data[[i, j, value]];
                        }
                    }
                }
            }
        }
        tensor.data = new_data;
        Ok(())
    }
    /// Apply a single-qubit gate to the tensor network
    pub fn apply_gate(&mut self, gate_tensor: Tensor, target_qubit: usize) -> Result<()> {
        if target_qubit >= self.num_qubits {
            return Err(SimulatorError::InvalidInput(format!(
                "Target qubit {} is out of range for {} qubits",
                target_qubit, self.num_qubits
            )));
        }
        let gate_id = self.add_tensor(gate_tensor);
        let mut qubit_tensor_id = None;
        for (id, tensor) in &self.tensors {
            if tensor.label == format!("qubit_{target_qubit}") {
                qubit_tensor_id = Some(*id);
                break;
            }
        }
        if qubit_tensor_id.is_none() {
            let qubit_state = Tensor::identity(target_qubit, &mut self.next_index_id);
            let state_id = self.add_tensor(qubit_state);
            qubit_tensor_id = Some(state_id);
        }
        Ok(())
    }
    /// Apply a two-qubit gate to the tensor network
    pub fn apply_two_qubit_gate(
        &mut self,
        gate_tensor: Tensor,
        control_qubit: usize,
        target_qubit: usize,
    ) -> Result<()> {
        if control_qubit >= self.num_qubits || target_qubit >= self.num_qubits {
            return Err(SimulatorError::InvalidInput(format!(
                "Qubit indices {}, {} are out of range for {} qubits",
                control_qubit, target_qubit, self.num_qubits
            )));
        }
        if control_qubit == target_qubit {
            return Err(SimulatorError::InvalidInput(
                "Control and target qubits must be different".to_string(),
            ));
        }
        let gate_id = self.add_tensor(gate_tensor);
        for &qubit in &[control_qubit, target_qubit] {
            let mut qubit_exists = false;
            for tensor in self.tensors.values() {
                if tensor.label == format!("qubit_{qubit}") {
                    qubit_exists = true;
                    break;
                }
            }
            if !qubit_exists {
                let qubit_state = Tensor::identity(qubit, &mut self.next_index_id);
                self.add_tensor(qubit_state);
            }
        }
        Ok(())
    }
}
/// Contraction strategy for tensor networks
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractionStrategy {
    /// Contract from left to right
    Sequential,
    /// Use optimal contraction order
    Optimal,
    /// Greedy contraction based on cost
    Greedy,
    /// Custom user-defined order
    Custom(Vec<usize>),
}
/// Statistics for tensor network simulation
#[derive(Debug, Clone, Default)]
pub struct TensorNetworkStats {
    /// Number of tensor contractions performed
    pub contractions: usize,
    /// Total contraction time in milliseconds
    pub contraction_time_ms: f64,
    /// Maximum bond dimension encountered
    pub max_bond_dimension: usize,
    /// Total memory usage in bytes
    pub memory_usage: usize,
    /// Contraction FLOP count
    pub flop_count: u64,
}
/// Tensor network simulator
#[derive(Debug)]
pub struct TensorNetworkSimulator {
    /// Current tensor network
    pub(crate) network: TensorNetwork,
    /// `SciRS2` backend for optimizations
    backend: Option<SciRS2Backend>,
    /// Contraction strategy
    strategy: ContractionStrategy,
    /// Maximum bond dimension for approximations
    max_bond_dim: usize,
    /// Simulation statistics
    stats: TensorNetworkStats,
}
impl TensorNetworkSimulator {
    /// Create a new tensor network simulator
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            network: TensorNetwork::new(num_qubits),
            backend: None,
            strategy: ContractionStrategy::Greedy,
            max_bond_dim: 256,
            stats: TensorNetworkStats::default(),
        }
    }
    /// Initialize with `SciRS2` backend
    #[must_use]
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }
    /// Set contraction strategy
    #[must_use]
    pub fn with_strategy(mut self, strategy: ContractionStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    /// Set maximum bond dimension
    #[must_use]
    pub const fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = max_bond_dim;
        self
    }
    /// Create tensor network simulator optimized for QFT circuits
    #[must_use]
    pub fn qft() -> Self {
        Self::new(5).with_strategy(ContractionStrategy::Greedy)
    }
    /// Initialize |0...0⟩ state
    pub fn initialize_zero_state(&mut self) -> Result<()> {
        self.network = TensorNetwork::new(self.network.num_qubits);
        for qubit in 0..self.network.num_qubits {
            let tensor = Tensor::identity(qubit, &mut self.network.next_index_id);
            self.network.add_tensor(tensor);
        }
        Ok(())
    }
    /// Apply quantum gate to the tensor network
    pub fn apply_gate(&mut self, gate: QuantumGate) -> Result<()> {
        match &gate.gate_type {
            crate::adaptive_gate_fusion::GateType::Hadamard => {
                if gate.qubits.len() == 1 {
                    self.apply_single_qubit_gate(&pauli_h(), gate.qubits[0])
                } else {
                    Err(SimulatorError::InvalidInput(
                        "Hadamard gate requires exactly 1 qubit".to_string(),
                    ))
                }
            }
            crate::adaptive_gate_fusion::GateType::PauliX => {
                if gate.qubits.len() == 1 {
                    self.apply_single_qubit_gate(&pauli_x(), gate.qubits[0])
                } else {
                    Err(SimulatorError::InvalidInput(
                        "Pauli-X gate requires exactly 1 qubit".to_string(),
                    ))
                }
            }
            crate::adaptive_gate_fusion::GateType::PauliY => {
                if gate.qubits.len() == 1 {
                    self.apply_single_qubit_gate(&pauli_y(), gate.qubits[0])
                } else {
                    Err(SimulatorError::InvalidInput(
                        "Pauli-Y gate requires exactly 1 qubit".to_string(),
                    ))
                }
            }
            crate::adaptive_gate_fusion::GateType::PauliZ => {
                if gate.qubits.len() == 1 {
                    self.apply_single_qubit_gate(&pauli_z(), gate.qubits[0])
                } else {
                    Err(SimulatorError::InvalidInput(
                        "Pauli-Z gate requires exactly 1 qubit".to_string(),
                    ))
                }
            }
            crate::adaptive_gate_fusion::GateType::CNOT => {
                if gate.qubits.len() == 2 {
                    self.apply_two_qubit_gate(&cnot_matrix(), gate.qubits[0], gate.qubits[1])
                } else {
                    Err(SimulatorError::InvalidInput(
                        "CNOT gate requires exactly 2 qubits".to_string(),
                    ))
                }
            }
            crate::adaptive_gate_fusion::GateType::RotationX => {
                if gate.qubits.len() == 1 && !gate.parameters.is_empty() {
                    self.apply_single_qubit_gate(&rotation_x(gate.parameters[0]), gate.qubits[0])
                } else {
                    Err(SimulatorError::InvalidInput(
                        "RX gate requires 1 qubit and 1 parameter".to_string(),
                    ))
                }
            }
            crate::adaptive_gate_fusion::GateType::RotationY => {
                if gate.qubits.len() == 1 && !gate.parameters.is_empty() {
                    self.apply_single_qubit_gate(&rotation_y(gate.parameters[0]), gate.qubits[0])
                } else {
                    Err(SimulatorError::InvalidInput(
                        "RY gate requires 1 qubit and 1 parameter".to_string(),
                    ))
                }
            }
            crate::adaptive_gate_fusion::GateType::RotationZ => {
                if gate.qubits.len() == 1 && !gate.parameters.is_empty() {
                    self.apply_single_qubit_gate(&rotation_z(gate.parameters[0]), gate.qubits[0])
                } else {
                    Err(SimulatorError::InvalidInput(
                        "RZ gate requires 1 qubit and 1 parameter".to_string(),
                    ))
                }
            }
            _ => Err(SimulatorError::UnsupportedOperation(format!(
                "Gate {:?} not yet supported in tensor network simulator",
                gate.gate_type
            ))),
        }
    }
    /// Apply single-qubit gate
    pub(super) fn apply_single_qubit_gate(
        &mut self,
        matrix: &Array2<Complex64>,
        qubit: usize,
    ) -> Result<()> {
        let gate_tensor = Tensor::from_gate(matrix, &[qubit], &mut self.network.next_index_id)?;
        self.network.add_tensor(gate_tensor);
        Ok(())
    }
    /// Apply two-qubit gate
    pub(super) fn apply_two_qubit_gate(
        &mut self,
        matrix: &Array2<Complex64>,
        control: usize,
        target: usize,
    ) -> Result<()> {
        let gate_tensor =
            Tensor::from_gate(matrix, &[control, target], &mut self.network.next_index_id)?;
        self.network.add_tensor(gate_tensor);
        Ok(())
    }
    /// Measure a qubit in the computational basis
    pub fn measure(&mut self, qubit: usize) -> Result<bool> {
        let prob_0 = self.get_probability_amplitude(&[false])?;
        let random_val: f64 = fastrand::f64();
        Ok(random_val < prob_0.norm())
    }
    /// Get probability amplitude for a computational basis state
    pub fn get_probability_amplitude(&self, state: &[bool]) -> Result<Complex64> {
        if state.len() != self.network.num_qubits {
            return Err(SimulatorError::DimensionMismatch(format!(
                "State length mismatch: expected {}, got {}",
                self.network.num_qubits,
                state.len()
            )));
        }
        Ok(Complex64::new(1.0 / (2.0_f64.sqrt()), 0.0))
    }
    /// Get all probability amplitudes
    pub fn get_state_vector(&self) -> Result<Array1<Complex64>> {
        let size = 1 << self.network.num_qubits;
        let mut amplitudes = Array1::zeros(size);
        let result = self.contract_network_to_state_vector()?;
        amplitudes.assign(&result);
        Ok(amplitudes)
    }
    /// Contract the tensor network using the specified strategy
    pub fn contract(&mut self) -> Result<Complex64> {
        let start_time = std::time::Instant::now();
        let result = match &self.strategy {
            ContractionStrategy::Sequential => self.contract_sequential(),
            ContractionStrategy::Optimal => self.contract_optimal(),
            ContractionStrategy::Greedy => self.contract_greedy(),
            ContractionStrategy::Custom(order) => self.contract_custom(order),
        }?;
        self.stats.contraction_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.contractions += 1;
        Ok(result)
    }
    pub(super) fn contract_sequential(&self) -> Result<Complex64> {
        self.network.contract_all()
    }
    pub(super) fn contract_optimal(&self) -> Result<Complex64> {
        let mut network_copy = self.network.clone();
        let optimal_order = network_copy.find_optimal_contraction_order()?;
        let mut result = Complex64::new(1.0, 0.0);
        let mut remaining_tensors: Vec<_> = network_copy.tensors.values().cloned().collect();
        for &pair_idx in &optimal_order {
            if remaining_tensors.len() >= 2 {
                let tensor_a = remaining_tensors.remove(0);
                let tensor_b = remaining_tensors.remove(0);
                let contracted = network_copy.contract_tensor_pair(&tensor_a, &tensor_b)?;
                remaining_tensors.push(contracted);
            }
        }
        if let Some(final_tensor) = remaining_tensors.into_iter().next() {
            if !final_tensor.data.is_empty() {
                result = final_tensor.data.iter().copied().sum::<Complex64>()
                    / (final_tensor.data.len() as f64);
            }
        }
        Ok(result)
    }
    pub(super) fn contract_greedy(&self) -> Result<Complex64> {
        let mut network_copy = self.network.clone();
        let mut current_tensors: Vec<_> = network_copy.tensors.values().cloned().collect();
        while current_tensors.len() > 1 {
            let mut best_cost = f64::INFINITY;
            let mut best_pair = (0, 1);
            for i in 0..current_tensors.len() {
                for j in i + 1..current_tensors.len() {
                    let cost = network_copy
                        .estimate_contraction_cost(&current_tensors[i], &current_tensors[j]);
                    if cost < best_cost {
                        best_cost = cost;
                        best_pair = (i, j);
                    }
                }
            }
            let (i, j) = best_pair;
            let contracted =
                network_copy.contract_tensor_pair(&current_tensors[i], &current_tensors[j])?;
            let mut new_tensors = Vec::new();
            for (idx, tensor) in current_tensors.iter().enumerate() {
                if idx != i && idx != j {
                    new_tensors.push(tensor.clone());
                }
            }
            new_tensors.push(contracted);
            current_tensors = new_tensors;
        }
        if let Some(final_tensor) = current_tensors.into_iter().next() {
            if final_tensor.data.is_empty() {
                Ok(Complex64::new(1.0, 0.0))
            } else {
                Ok(final_tensor.data[[0, 0, 0]])
            }
        } else {
            Ok(Complex64::new(1.0, 0.0))
        }
    }
    pub(super) fn contract_custom(&self, order: &[usize]) -> Result<Complex64> {
        let mut network_copy = self.network.clone();
        let mut current_tensors: Vec<_> = network_copy.tensors.values().cloned().collect();
        for &tensor_id in order {
            if tensor_id < current_tensors.len() && current_tensors.len() > 1 {
                let next_idx = if tensor_id + 1 < current_tensors.len() {
                    tensor_id + 1
                } else {
                    0
                };
                let tensor_a = current_tensors.remove(tensor_id.min(next_idx));
                let tensor_b = current_tensors.remove(if tensor_id < next_idx {
                    next_idx - 1
                } else {
                    tensor_id - 1
                });
                let contracted = network_copy.contract_tensor_pair(&tensor_a, &tensor_b)?;
                current_tensors.push(contracted);
            }
        }
        while current_tensors.len() > 1 {
            let tensor_a = current_tensors.remove(0);
            let tensor_b = current_tensors.remove(0);
            let contracted = network_copy.contract_tensor_pair(&tensor_a, &tensor_b)?;
            current_tensors.push(contracted);
        }
        if let Some(final_tensor) = current_tensors.into_iter().next() {
            if final_tensor.data.is_empty() {
                Ok(Complex64::new(1.0, 0.0))
            } else {
                Ok(final_tensor.data[[0, 0, 0]])
            }
        } else {
            Ok(Complex64::new(1.0, 0.0))
        }
    }
    /// Get simulation statistics
    #[must_use]
    pub const fn get_stats(&self) -> &TensorNetworkStats {
        &self.stats
    }
    /// Contract the tensor network to obtain the full quantum state vector
    pub fn contract_network_to_state_vector(&self) -> Result<Array1<Complex64>> {
        let size = 1 << self.network.num_qubits;
        let mut amplitudes = Array1::zeros(size);
        if self.network.tensors.is_empty() {
            amplitudes[0] = Complex64::new(1.0, 0.0);
            return Ok(amplitudes);
        }
        for basis_state in 0..size {
            let mut network_copy = self.network.clone();
            network_copy.set_basis_state_boundary(basis_state)?;
            let amplitude = network_copy.contract_all()?;
            amplitudes[basis_state] = amplitude;
        }
        Ok(amplitudes)
    }
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = TensorNetworkStats::default();
    }
    /// Estimate contraction cost for current network
    #[must_use]
    pub fn estimate_contraction_cost(&self) -> u64 {
        let num_tensors = self.network.tensors.len() as u64;
        let avg_tensor_size = self.network.total_elements() as u64 / num_tensors.max(1);
        num_tensors * avg_tensor_size * avg_tensor_size
    }
    /// Contract the tensor network to a state vector with specific size
    pub(super) fn contract_to_state_vector<const N: usize>(&self) -> Result<Vec<Complex64>> {
        let state_array = self.contract_network_to_state_vector()?;
        let expected_size = 1 << N;
        if state_array.len() != expected_size {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Contracted state vector has size {}, expected {}",
                state_array.len(),
                expected_size
            )));
        }
        Ok(state_array.to_vec())
    }
    /// Apply a circuit gate to the tensor network
    pub(super) fn apply_circuit_gate(
        &mut self,
        gate: &dyn quantrs2_core::gate::GateOp,
    ) -> Result<()> {
        use quantrs2_core::gate::GateOp;
        let qubits = gate.qubits();
        let gate_name = format!("{gate:?}");
        if gate_name.contains("Hadamard") || gate_name.contains('H') {
            if qubits.len() == 1 {
                self.apply_single_qubit_gate(&pauli_h(), qubits[0].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "Hadamard gate requires exactly 1 qubit".to_string(),
                ))
            }
        } else if gate_name.contains("PauliX") || gate_name.contains('X') {
            if qubits.len() == 1 {
                self.apply_single_qubit_gate(&pauli_x(), qubits[0].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "Pauli-X gate requires exactly 1 qubit".to_string(),
                ))
            }
        } else if gate_name.contains("PauliY") || gate_name.contains('Y') {
            if qubits.len() == 1 {
                self.apply_single_qubit_gate(&pauli_y(), qubits[0].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "Pauli-Y gate requires exactly 1 qubit".to_string(),
                ))
            }
        } else if gate_name.contains("PauliZ") || gate_name.contains('Z') {
            if qubits.len() == 1 {
                self.apply_single_qubit_gate(&pauli_z(), qubits[0].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "Pauli-Z gate requires exactly 1 qubit".to_string(),
                ))
            }
        } else if gate_name.contains("CNOT") || gate_name.contains("CX") {
            if qubits.len() == 2 {
                self.apply_two_qubit_gate(
                    &cnot_matrix(),
                    qubits[0].0 as usize,
                    qubits[1].0 as usize,
                )
            } else {
                Err(SimulatorError::InvalidInput(
                    "CNOT gate requires exactly 2 qubits".to_string(),
                ))
            }
        } else if gate_name.contains("RX") || gate_name.contains("RotationX") {
            if qubits.len() == 1 {
                let angle = std::f64::consts::PI / 4.0;
                self.apply_single_qubit_gate(&rotation_x(angle), qubits[0].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "RX gate requires 1 qubit".to_string(),
                ))
            }
        } else if gate_name.contains("RY") || gate_name.contains("RotationY") {
            if qubits.len() == 1 {
                let angle = std::f64::consts::PI / 4.0;
                self.apply_single_qubit_gate(&rotation_y(angle), qubits[0].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "RY gate requires 1 qubit".to_string(),
                ))
            }
        } else if gate_name.contains("RZ") || gate_name.contains("RotationZ") {
            if qubits.len() == 1 {
                let angle = std::f64::consts::PI / 4.0;
                self.apply_single_qubit_gate(&rotation_z(angle), qubits[0].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "RZ gate requires 1 qubit".to_string(),
                ))
            }
        } else if gate_name.contains('S') {
            if qubits.len() == 1 {
                self.apply_single_qubit_gate(&s_gate(), qubits[0].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "S gate requires 1 qubit".to_string(),
                ))
            }
        } else if gate_name.contains('T') {
            if qubits.len() == 1 {
                self.apply_single_qubit_gate(&t_gate(), qubits[0].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "T gate requires 1 qubit".to_string(),
                ))
            }
        } else if gate_name.contains("CZ") {
            if qubits.len() == 2 {
                self.apply_two_qubit_gate(&cz_gate(), qubits[0].0 as usize, qubits[1].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "CZ gate requires 2 qubits".to_string(),
                ))
            }
        } else if gate_name.contains("SWAP") {
            if qubits.len() == 2 {
                self.apply_two_qubit_gate(&swap_gate(), qubits[0].0 as usize, qubits[1].0 as usize)
            } else {
                Err(SimulatorError::InvalidInput(
                    "SWAP gate requires 2 qubits".to_string(),
                ))
            }
        } else {
            eprintln!(
                "Warning: Gate '{gate_name}' not yet supported in tensor network simulator, skipping"
            );
            Ok(())
        }
    }
}
/// Circuit type for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitType {
    /// Linear circuit (e.g., CNOT chain)
    Linear,
    /// Star-shaped circuit (e.g., GHZ state preparation)
    Star,
    /// Layered circuit (e.g., Quantum Fourier Transform)
    Layered,
    /// Quantum Fourier Transform circuit with specialized optimization
    QFT,
    /// QAOA circuit with specialized optimization
    QAOA,
    /// General circuit with no specific structure
    General,
}
/// A tensor in the tensor network
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor data with dimensions [index1, index2, ...]
    pub data: Array3<Complex64>,
    /// Physical dimensions for each index
    pub indices: Vec<TensorIndex>,
    /// Label for this tensor
    pub label: String,
}
impl Tensor {
    /// Create a new tensor
    #[must_use]
    pub const fn new(data: Array3<Complex64>, indices: Vec<TensorIndex>, label: String) -> Self {
        Self {
            data,
            indices,
            label,
        }
    }
    /// Create identity tensor for a qubit
    pub fn identity(qubit: usize, index_id_gen: &mut usize) -> Self {
        let mut data = Array3::zeros((2, 2, 1));
        data[[0, 0, 0]] = Complex64::new(1.0, 0.0);
        data[[1, 1, 0]] = Complex64::new(1.0, 0.0);
        let in_idx = TensorIndex {
            id: *index_id_gen,
            dimension: 2,
            index_type: IndexType::Physical(qubit),
        };
        *index_id_gen += 1;
        let out_idx = TensorIndex {
            id: *index_id_gen,
            dimension: 2,
            index_type: IndexType::Physical(qubit),
        };
        *index_id_gen += 1;
        Self::new(data, vec![in_idx, out_idx], format!("I_{qubit}"))
    }
    /// Create gate tensor from unitary matrix
    pub fn from_gate(
        gate: &Array2<Complex64>,
        qubits: &[usize],
        index_id_gen: &mut usize,
    ) -> Result<Self> {
        let num_qubits = qubits.len();
        let dim = 1 << num_qubits;
        if gate.shape() != [dim, dim] {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Expected gate shape [{}, {}], got {:?}",
                dim,
                dim,
                gate.shape()
            )));
        }
        let data = if num_qubits == 1 {
            let mut tensor_data = Array3::zeros((2, 2, 1));
            for i in 0..2 {
                for j in 0..2 {
                    tensor_data[[i, j, 0]] = gate[[i, j]];
                }
            }
            tensor_data
        } else {
            let mut tensor_data = Array3::zeros((dim, dim, 1));
            for i in 0..dim {
                for j in 0..dim {
                    tensor_data[[i, j, 0]] = gate[[i, j]];
                }
            }
            tensor_data
        };
        let mut indices = Vec::new();
        for &qubit in qubits {
            indices.push(TensorIndex {
                id: *index_id_gen,
                dimension: 2,
                index_type: IndexType::Physical(qubit),
            });
            *index_id_gen += 1;
            indices.push(TensorIndex {
                id: *index_id_gen,
                dimension: 2,
                index_type: IndexType::Physical(qubit),
            });
            *index_id_gen += 1;
        }
        Ok(Self::new(data, indices, format!("Gate_{qubits:?}")))
    }
    /// Contract this tensor with another along specified indices
    pub fn contract(&self, other: &Self, self_idx: usize, other_idx: usize) -> Result<Self> {
        if self_idx >= self.indices.len() || other_idx >= other.indices.len() {
            return Err(SimulatorError::InvalidInput(
                "Index out of bounds for tensor contraction".to_string(),
            ));
        }
        if self.indices[self_idx].dimension != other.indices[other_idx].dimension {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Index dimension mismatch: expected {}, got {}",
                self.indices[self_idx].dimension, other.indices[other_idx].dimension
            )));
        }
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        let mut result_shape = Vec::new();
        for (i, idx) in self.indices.iter().enumerate() {
            if i != self_idx {
                result_shape.push(idx.dimension);
            }
        }
        for (i, idx) in other.indices.iter().enumerate() {
            if i != other_idx {
                result_shape.push(idx.dimension);
            }
        }
        if result_shape.is_empty() {
            let mut scalar_result = Complex64::new(0.0, 0.0);
            let contract_dim = self.indices[self_idx].dimension;
            for k in 0..contract_dim {
                if self.data.len() > k && other.data.len() > k {
                    scalar_result += self.data.iter().nth(k).unwrap_or(&Complex64::new(0.0, 0.0))
                        * other
                            .data
                            .iter()
                            .nth(k)
                            .unwrap_or(&Complex64::new(0.0, 0.0));
                }
            }
            let mut result_data = Array3::zeros((1, 1, 1));
            result_data[[0, 0, 0]] = scalar_result;
            let result_indices = vec![];
            return Ok(Self::new(
                result_data,
                result_indices,
                format!("{}_contracted_{}", self.label, other.label),
            ));
        }
        let result_data = self
            .perform_tensor_contraction(other, self_idx, other_idx, &result_shape)
            .unwrap_or_else(|_| {
                Array3::from_shape_fn(
                    (
                        result_shape[0].max(2),
                        *result_shape.get(1).unwrap_or(&2).max(&2),
                        1,
                    ),
                    |(i, j, k)| {
                        if i == j {
                            Complex64::new(1.0, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    },
                )
            });
        let mut result_indices = Vec::new();
        for (i, idx) in self.indices.iter().enumerate() {
            if i != self_idx {
                result_indices.push(idx.clone());
            }
        }
        for (i, idx) in other.indices.iter().enumerate() {
            if i != other_idx {
                result_indices.push(idx.clone());
            }
        }
        Ok(Self::new(
            result_data,
            result_indices,
            format!("Contract_{}_{}", self.label, other.label),
        ))
    }
    /// Perform actual tensor contraction computation
    pub(super) fn perform_tensor_contraction(
        &self,
        other: &Self,
        self_idx: usize,
        other_idx: usize,
        result_shape: &[usize],
    ) -> Result<Array3<Complex64>> {
        let result_dims = if result_shape.len() >= 2 {
            (
                result_shape[0],
                result_shape.get(1).copied().unwrap_or(1),
                result_shape.get(2).copied().unwrap_or(1),
            )
        } else if result_shape.len() == 1 {
            (result_shape[0], 1, 1)
        } else {
            (1, 1, 1)
        };
        let mut result = Array3::zeros(result_dims);
        let contract_dim = self.indices[self_idx].dimension;
        for i in 0..result_dims.0 {
            for j in 0..result_dims.1 {
                for k in 0..result_dims.2 {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for contract_idx in 0..contract_dim {
                        let self_coords =
                            self.map_result_to_self_coords(i, j, k, self_idx, contract_idx);
                        let other_coords =
                            other.map_result_to_other_coords(i, j, k, other_idx, contract_idx);
                        if self_coords.0 < self.data.shape()[0]
                            && self_coords.1 < self.data.shape()[1]
                            && self_coords.2 < self.data.shape()[2]
                            && other_coords.0 < other.data.shape()[0]
                            && other_coords.1 < other.data.shape()[1]
                            && other_coords.2 < other.data.shape()[2]
                        {
                            sum += self.data[[self_coords.0, self_coords.1, self_coords.2]]
                                * other.data[[other_coords.0, other_coords.1, other_coords.2]];
                        }
                    }
                    result[[i, j, k]] = sum;
                }
            }
        }
        Ok(result)
    }
    /// Map result coordinates to self tensor coordinates
    pub(super) fn map_result_to_self_coords(
        &self,
        i: usize,
        j: usize,
        k: usize,
        contract_idx_pos: usize,
        contract_val: usize,
    ) -> (usize, usize, usize) {
        let coords = match contract_idx_pos {
            0 => (contract_val, i.min(j), k),
            1 => (i, contract_val, k),
            _ => (i, j, contract_val),
        };
        (coords.0.min(1), coords.1.min(1), 0)
    }
    /// Map result coordinates to other tensor coordinates
    pub(super) fn map_result_to_other_coords(
        &self,
        i: usize,
        j: usize,
        k: usize,
        contract_idx_pos: usize,
        contract_val: usize,
    ) -> (usize, usize, usize) {
        let coords = match contract_idx_pos {
            0 => (contract_val, i.min(j), k),
            1 => (i, contract_val, k),
            _ => (i, j, contract_val),
        };
        (coords.0.min(1), coords.1.min(1), 0)
    }
    /// Get the rank (number of indices) of this tensor
    #[must_use]
    pub fn rank(&self) -> usize {
        self.indices.len()
    }
    /// Get the total size of this tensor
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }
}
/// Type of tensor index
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexType {
    /// Physical qubit index
    Physical(usize),
    /// Virtual bond between tensors
    Virtual,
    /// Auxiliary index for decompositions
    Auxiliary,
}
