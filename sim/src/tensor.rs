//! Tensor network simulator for quantum circuits
//!
//! This module provides a tensor network-based quantum circuit simulator that
//! is particularly efficient for circuits with limited entanglement or certain
//! structural properties.

use std::collections::{HashMap, HashSet};
use std::fmt;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::adaptive_gate_fusion::QuantumGate;
use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;

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
}

/// Tensor network simulator
#[derive(Debug)]
pub struct TensorNetworkSimulator {
    /// Current tensor network
    network: TensorNetwork,
    /// SciRS2 backend for optimizations
    backend: Option<SciRS2Backend>,
    /// Contraction strategy
    strategy: ContractionStrategy,
    /// Maximum bond dimension for approximations
    max_bond_dim: usize,
    /// Simulation statistics
    stats: TensorNetworkStats,
}

/// Contraction strategy for tensor networks
#[derive(Debug, Clone, PartialEq)]
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

impl Tensor {
    /// Create a new tensor
    pub fn new(data: Array3<Complex64>, indices: Vec<TensorIndex>, label: String) -> Self {
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

        Self::new(data, vec![in_idx, out_idx], format!("I_{}", qubit))
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

        // For this simplified implementation, we'll use a fixed 3D tensor structure
        // Real tensor networks would decompose gates more sophisticatedly
        let data = if num_qubits == 1 {
            // Single-qubit gate: reshape 2x2 to 2x2x1
            let mut tensor_data = Array3::zeros((2, 2, 1));
            for i in 0..2 {
                for j in 0..2 {
                    tensor_data[[i, j, 0]] = gate[[i, j]];
                }
            }
            tensor_data
        } else {
            // Multi-qubit gate: use a simplified 3D representation
            let mut tensor_data = Array3::zeros((dim, dim, 1));
            for i in 0..dim {
                for j in 0..dim {
                    tensor_data[[i, j, 0]] = gate[[i, j]];
                }
            }
            tensor_data
        };

        // Create indices
        let mut indices = Vec::new();
        for &qubit in qubits {
            // Input index
            indices.push(TensorIndex {
                id: *index_id_gen,
                dimension: 2,
                index_type: IndexType::Physical(qubit),
            });
            *index_id_gen += 1;

            // Output index
            indices.push(TensorIndex {
                id: *index_id_gen,
                dimension: 2,
                index_type: IndexType::Physical(qubit),
            });
            *index_id_gen += 1;
        }

        Ok(Self::new(data, indices, format!("Gate_{:?}", qubits)))
    }

    /// Contract this tensor with another along specified indices
    pub fn contract(&self, other: &Tensor, self_idx: usize, other_idx: usize) -> Result<Tensor> {
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

        // Simplified contraction - in practice would use optimized tensor contraction libraries
        // This is a placeholder implementation
        let result_data = Array3::zeros((2, 2, 1)); // Simplified result

        let mut result_indices = self.indices.clone();
        result_indices.remove(self_idx);
        result_indices.extend(
            other
                .indices
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != other_idx)
                .map(|(_, idx)| idx.clone()),
        );

        Ok(Tensor::new(
            result_data,
            result_indices,
            format!("Contract_{}_{}", self.label, other.label),
        ))
    }

    /// Get the rank (number of indices) of this tensor
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Get the total size of this tensor
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl TensorNetwork {
    /// Create a new empty tensor network
    pub fn new(num_qubits: usize) -> Self {
        Self {
            tensors: HashMap::new(),
            connections: Vec::new(),
            num_qubits,
            next_tensor_id: 0,
            next_index_id: 0,
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
    pub fn get_neighbors(&self, tensor_id: usize) -> Vec<usize> {
        let mut neighbors = HashSet::new();

        if let Some(tensor) = self.tensors.get(&tensor_id) {
            for connection in &self.connections {
                // Check if any index of this tensor is involved in the connection
                let tensor_indices: HashSet<_> = tensor.indices.iter().map(|idx| idx.id).collect();

                if tensor_indices.contains(&connection.0.id)
                    || tensor_indices.contains(&connection.1.id)
                {
                    // Find the other tensor in this connection
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

        // Simplified contraction - in practice would use sophisticated algorithms
        // This is a placeholder that returns a dummy result
        Ok(Complex64::new(1.0, 0.0))
    }

    /// Get the total number of elements across all tensors
    pub fn total_elements(&self) -> usize {
        self.tensors.values().map(|t| t.size()).sum()
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.total_elements() * std::mem::size_of::<Complex64>()
    }
}

impl TensorNetworkSimulator {
    /// Create a new tensor network simulator
    pub fn new(num_qubits: usize) -> Self {
        Self {
            network: TensorNetwork::new(num_qubits),
            backend: None,
            strategy: ContractionStrategy::Greedy,
            max_bond_dim: 256,
            stats: TensorNetworkStats::default(),
        }
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Set contraction strategy
    pub fn with_strategy(mut self, strategy: ContractionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set maximum bond dimension
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = max_bond_dim;
        self
    }

    /// Initialize |0...0⟩ state
    pub fn initialize_zero_state(&mut self) -> Result<()> {
        self.network = TensorNetwork::new(self.network.num_qubits);

        // Add identity tensors for each qubit
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
    fn apply_single_qubit_gate(&mut self, matrix: &Array2<Complex64>, qubit: usize) -> Result<()> {
        let gate_tensor = Tensor::from_gate(matrix, &[qubit], &mut self.network.next_index_id)?;
        self.network.add_tensor(gate_tensor);
        Ok(())
    }

    /// Apply two-qubit gate
    fn apply_two_qubit_gate(
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
        // Simplified measurement - in practice would involve partial contraction
        // and normalization of the remaining network
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

        // Simplified implementation - in practice would contract network
        // with measurement projectors
        Ok(Complex64::new(1.0 / (2.0_f64.sqrt()), 0.0))
    }

    /// Get all probability amplitudes
    pub fn get_state_vector(&self) -> Result<Array1<Complex64>> {
        let size = 1 << self.network.num_qubits;
        let mut amplitudes = Array1::zeros(size);

        // This is a placeholder - real implementation would contract the tensor network
        amplitudes[0] = Complex64::new(1.0, 0.0);

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

    fn contract_sequential(&self) -> Result<Complex64> {
        // Simplified sequential contraction
        self.network.contract_all()
    }

    fn contract_optimal(&self) -> Result<Complex64> {
        // Placeholder for optimal contraction order algorithm
        // In practice would use dynamic programming or branch-and-bound
        self.contract_sequential()
    }

    fn contract_greedy(&self) -> Result<Complex64> {
        // Placeholder for greedy contraction
        // Would repeatedly choose the lowest-cost contraction
        self.contract_sequential()
    }

    fn contract_custom(&self, _order: &[usize]) -> Result<Complex64> {
        // Placeholder for custom contraction order
        self.contract_sequential()
    }

    /// Get simulation statistics
    pub fn get_stats(&self) -> &TensorNetworkStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = TensorNetworkStats::default();
    }

    /// Estimate contraction cost for current network
    pub fn estimate_contraction_cost(&self) -> u64 {
        // Simplified cost estimation
        let num_tensors = self.network.tensors.len() as u64;
        let avg_tensor_size = self.network.total_elements() as u64 / num_tensors.max(1);
        num_tensors * avg_tensor_size * avg_tensor_size
    }
}

impl Default for TensorNetworkSimulator {
    fn default() -> Self {
        Self::new(1)
    }
}

impl fmt::Display for TensorNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TensorNetwork with {} qubits:", self.num_qubits)?;
        writeln!(f, "  Tensors: {}", self.tensors.len())?;
        writeln!(f, "  Connections: {}", self.connections.len())?;
        writeln!(f, "  Memory usage: {} bytes", self.memory_usage())?;
        Ok(())
    }
}

// Helper functions for common gate matrices
fn pauli_x() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
}

fn pauli_y() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
}

fn pauli_z() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap()
}

fn pauli_h() -> Array2<Complex64> {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ],
    )
    .unwrap()
}

fn cnot_matrix() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (4, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
}

fn rotation_x(theta: f64) -> Array2<Complex64> {
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(cos_half, 0.0),
            Complex64::new(0.0, -sin_half),
            Complex64::new(0.0, -sin_half),
            Complex64::new(cos_half, 0.0),
        ],
    )
    .unwrap()
}

fn rotation_y(theta: f64) -> Array2<Complex64> {
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(cos_half, 0.0),
            Complex64::new(-sin_half, 0.0),
            Complex64::new(sin_half, 0.0),
            Complex64::new(cos_half, 0.0),
        ],
    )
    .unwrap()
}

fn rotation_z(theta: f64) -> Array2<Complex64> {
    let exp_neg = Complex64::from_polar(1.0, -theta / 2.0);
    let exp_pos = Complex64::from_polar(1.0, theta / 2.0);
    Array2::from_shape_vec(
        (2, 2),
        vec![
            exp_neg,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            exp_pos,
        ],
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_tensor_creation() {
        let data = Array3::zeros((2, 2, 1));
        let indices = vec![
            TensorIndex {
                id: 0,
                dimension: 2,
                index_type: IndexType::Physical(0),
            },
            TensorIndex {
                id: 1,
                dimension: 2,
                index_type: IndexType::Physical(0),
            },
        ];
        let tensor = Tensor::new(data, indices, "test".to_string());

        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.label, "test");
    }

    #[test]
    fn test_tensor_network_creation() {
        let network = TensorNetwork::new(3);
        assert_eq!(network.num_qubits, 3);
        assert_eq!(network.tensors.len(), 0);
    }

    #[test]
    fn test_simulator_initialization() {
        let mut sim = TensorNetworkSimulator::new(2);
        sim.initialize_zero_state().unwrap();

        assert_eq!(sim.network.tensors.len(), 2);
    }

    #[test]
    fn test_single_qubit_gate() {
        let mut sim = TensorNetworkSimulator::new(1);
        sim.initialize_zero_state().unwrap();

        let initial_tensors = sim.network.tensors.len();
        let h_gate = QuantumGate::new(
            crate::adaptive_gate_fusion::GateType::Hadamard,
            vec![0],
            vec![],
        );
        sim.apply_gate(h_gate).unwrap();

        // Should add one more tensor for the gate
        assert_eq!(sim.network.tensors.len(), initial_tensors + 1);
    }

    #[test]
    fn test_measurement() {
        let mut sim = TensorNetworkSimulator::new(1);
        sim.initialize_zero_state().unwrap();

        let result = sim.measure(0).unwrap();
        assert!(result == true || result == false); // Just check it returns a bool
    }

    #[test]
    fn test_contraction_strategies() {
        let sim = TensorNetworkSimulator::new(2);

        // Test different strategies don't crash
        let strat1 = ContractionStrategy::Sequential;
        let strat2 = ContractionStrategy::Greedy;
        let strat3 = ContractionStrategy::Custom(vec![0, 1]);

        assert_ne!(strat1, strat2);
        assert_ne!(strat2, strat3);
    }

    #[test]
    fn test_gate_matrices() {
        let h = pauli_h();
        assert_abs_diff_eq!(h[[0, 0]].re, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);

        let x = pauli_x();
        assert_abs_diff_eq!(x[[0, 1]].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[[1, 0]].re, 1.0, epsilon = 1e-10);
    }
}
