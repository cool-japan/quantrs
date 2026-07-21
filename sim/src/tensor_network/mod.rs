//! Tensor Network simulator implementation
//!
//! This module provides a tensor network-based quantum circuit simulator.
//! Tensor networks can be more efficient than state vector simulation for
//! circuits with specific structures or limited entanglement.

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi, single, GateOp},
    qubit::QubitId,
    register::Register,
};

use scirs2_core::ndarray::{Array, ArrayD, Dimension, IxDyn};
use scirs2_core::ndarray_ext::manipulation;
use scirs2_core::parallel_ops::*;
use scirs2_core::Complex64;
use std::collections::HashMap;

pub mod contraction;
pub mod opt_contraction;
pub mod tensor;

use contraction::ContractableNetwork;
use opt_contraction::{ContractionOptMethod, OptimizedTensorNetwork, PathOptimizer};
use tensor::{Tensor, TensorIndex};

/// A simulator for quantum circuits using tensor network methods
#[derive(Debug, Clone)]
pub struct TensorNetworkSimulator {
    /// Maximum bond dimension for tensor network decompositions
    max_bond_dimension: usize,

    /// Optimization level (0-3)
    optimization_level: u8,

    /// Contraction strategy to use
    contraction_strategy: ContractionStrategy,

    /// Optimizer for tensor network contraction
    path_optimizer: PathOptimizer,
}

/// Enum representing different types of quantum circuits
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

/// Enum representing different contraction strategies for tensor networks
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractionStrategy {
    /// Greedy contraction strategy (good general purpose algorithm)
    Greedy,

    /// Strategy optimized for linear circuits (e.g., CNOT chains)
    Linear,

    /// Strategy optimized for star-shaped circuits (e.g., GHZ state preparation)
    Star,

    /// Strategy optimized for Quantum Fourier Transform circuits
    QFT,

    /// Strategy optimized for QAOA circuits
    QAOA,

    /// Custom identifier for extensions
    Custom,
}

impl TensorNetworkSimulator {
    /// Create a new tensor network simulator with default settings
    pub fn new() -> Self {
        Self {
            max_bond_dimension: 16,
            optimization_level: 1,
            contraction_strategy: ContractionStrategy::Greedy,
            path_optimizer: PathOptimizer::default(),
        }
    }

    /// Create a new tensor network simulator optimized for QFT circuits
    pub fn qft() -> Self {
        Self {
            max_bond_dimension: 16,
            optimization_level: 2,
            contraction_strategy: ContractionStrategy::QFT,
            path_optimizer: PathOptimizer::default()
                .with_method(ContractionOptMethod::Hybrid)
                .with_max_bond_dimension(32),
        }
    }

    /// Create a new tensor network simulator optimized for QAOA circuits
    pub fn qaoa() -> Self {
        Self {
            max_bond_dimension: 16,
            optimization_level: 2,
            contraction_strategy: ContractionStrategy::QAOA,
            path_optimizer: PathOptimizer::default()
                .with_method(ContractionOptMethod::Hybrid)
                .with_max_bond_dimension(32),
        }
    }

    /// Create a new tensor network simulator with specified bond dimension
    #[must_use]
    pub const fn with_bond_dimension(mut self, max_bond_dimension: usize) -> Self {
        self.max_bond_dimension = max_bond_dimension;
        self.path_optimizer = self
            .path_optimizer
            .with_max_bond_dimension(max_bond_dimension);
        self
    }

    /// Set the optimization level
    ///
    /// 0 = No optimization
    /// 1 = Basic optimizations
    /// 2 = Advanced optimizations
    /// 3 = Aggressive optimizations (may impact accuracy)
    #[must_use]
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        self.optimization_level = level.min(3);

        // Set the contraction method based on optimization level
        self.path_optimizer = match level {
            0 => self
                .path_optimizer
                .with_method(ContractionOptMethod::Greedy),
            1 => self
                .path_optimizer
                .with_method(ContractionOptMethod::Greedy),
            2 => self
                .path_optimizer
                .with_method(ContractionOptMethod::DynamicProgramming),
            3 => self
                .path_optimizer
                .with_method(ContractionOptMethod::Hybrid),
            _ => self
                .path_optimizer
                .with_method(ContractionOptMethod::Greedy),
        };

        self
    }

    /// Set the contraction strategy
    #[must_use]
    pub fn with_contraction_strategy(mut self, strategy: ContractionStrategy) -> Self {
        self.contraction_strategy = strategy.clone();

        // Set the appropriate optimization method based on strategy
        self.path_optimizer = match &strategy {
            ContractionStrategy::QFT => self
                .path_optimizer
                .with_method(ContractionOptMethod::DynamicProgramming)
                .with_max_bond_dimension(32),
            ContractionStrategy::QAOA => self
                .path_optimizer
                .with_method(ContractionOptMethod::DynamicProgramming)
                .with_max_bond_dimension(32),
            ContractionStrategy::Linear => self
                .path_optimizer
                .with_method(ContractionOptMethod::Greedy),
            ContractionStrategy::Star => self
                .path_optimizer
                .with_method(ContractionOptMethod::Greedy),
            _ => self.path_optimizer,
        };

        self
    }

    /// Analyze the structure of a quantum circuit to determine the best simulation approach
    fn analyze_circuit_structure<const N: usize>(&self, circuit: &Circuit<N>) -> CircuitType {
        // Count different types of gates
        let mut single_qubit_gates = 0;
        let mut cnot_gates = 0;
        let mut other_two_qubit_gates = 0;
        let mut multi_qubit_gates = 0;
        let mut hadamard_gates = 0;
        let mut rotation_gates = 0;
        let mut phase_gates = 0;
        let mut x_rotation_gates = 0;
        let mut controlled_phase_gates = 0;
        let mut swap_gates = 0;

        // Track connections between qubits
        let mut qubit_connections =
            std::collections::HashMap::<usize, std::collections::HashSet<usize>>::new();

        // Analyze each gate
        for gate in circuit.gates() {
            let qubits = gate.qubits();
            let gate_name = gate.name();

            if qubits.len() == 1 {
                // Single-qubit gate
                single_qubit_gates += 1;

                match gate_name {
                    "H" => hadamard_gates += 1,
                    "RX" => {
                        rotation_gates += 1;
                        x_rotation_gates += 1;
                    }
                    "RY" | "RZ" => rotation_gates += 1,
                    "S" | "T" | "S†" | "T†" => phase_gates += 1,
                    _ => {}
                }
            } else if qubits.len() == 2 {
                // Two-qubit gate
                if gate_name == "CNOT" {
                    cnot_gates += 1;
                } else if gate_name == "SWAP" {
                    swap_gates += 1;
                } else if gate_name == "CZ" || gate_name == "CS" || gate_name == "CRZ" {
                    controlled_phase_gates += 1;
                    other_two_qubit_gates += 1;
                } else {
                    other_two_qubit_gates += 1;
                }

                // Record connection between qubits
                let q1 = qubits[0].id() as usize;
                let q2 = qubits[1].id() as usize;

                qubit_connections.entry(q1).or_default().insert(q2);
                qubit_connections.entry(q2).or_default().insert(q1);
            } else {
                // Multi-qubit gate
                multi_qubit_gates += 1;
            }
        }

        // Check for QFT circuit pattern
        if self.is_qft_pattern(hadamard_gates, controlled_phase_gates, swap_gates, N) {
            return CircuitType::QFT;
        }

        // Check for QAOA circuit pattern
        if self.is_qaoa_pattern(x_rotation_gates, cnot_gates) {
            return CircuitType::QAOA;
        }

        // Check for linear structure (chain of CNOTs)
        if is_linear_structure(&qubit_connections, N)
            && other_two_qubit_gates == 0
            && multi_qubit_gates == 0
        {
            return CircuitType::Linear;
        }

        // Check for star structure (like GHZ state preparation)
        if is_star_structure(&qubit_connections, N) {
            return CircuitType::Star;
        }

        // Check for layered structure (like QFT)
        if is_layered_structure(circuit) {
            return CircuitType::Layered;
        }

        // Default to general circuit
        CircuitType::General
    }

    /// Check if the circuit matches a QFT pattern
    const fn is_qft_pattern(
        &self,
        hadamard_count: usize,
        controlled_phase_count: usize,
        swap_count: usize,
        num_qubits: usize,
    ) -> bool {
        // QFT on n qubits typically has:
        // - n Hadamard gates
        // - n*(n-1)/2 controlled-phase gates
        // - n/2 SWAP gates (for qubit reversal)

        let expected_controlled_phase = (num_qubits * (num_qubits - 1)) / 2;
        let expected_swap = num_qubits / 2;

        // Allow some flexibility in the counts
        hadamard_count >= num_qubits
            && controlled_phase_count >= expected_controlled_phase / 2
            && (swap_count == 0 || swap_count >= expected_swap / 2)
    }

    /// Check if the circuit matches a QAOA pattern
    const fn is_qaoa_pattern(&self, x_rotation_count: usize, cnot_count: usize) -> bool {
        // QAOA typically has:
        // - X rotations for the mixer Hamiltonian
        // - CNOT gates + Z rotations for the problem Hamiltonian

        // Simple heuristic: if we have both X rotations and CNOT gates, it might be QAOA
        x_rotation_count > 0 && cnot_count > 0
    }
}

/// Check if the circuit has a linear structure (chain of qubits)
fn is_linear_structure(
    qubit_connections: &std::collections::HashMap<usize, std::collections::HashSet<usize>>,
    num_qubits: usize,
) -> bool {
    // Check that each qubit is connected to at most 2 others
    for i in 0..num_qubits {
        if let Some(connections) = qubit_connections.get(&i) {
            if connections.len() > 2 {
                return false;
            }
        }
    }

    // Count endpoints (qubits with only one connection)
    let num_endpoints = (0..num_qubits)
        .filter(|&i| {
            qubit_connections
                .get(&i)
                .is_some_and(|conns| conns.len() == 1)
        })
        .count();

    // A chain has exactly 2 endpoints
    num_endpoints == 2
}

/// Check if the circuit has a star structure (central qubit connected to many others)
fn is_star_structure(
    qubit_connections: &std::collections::HashMap<usize, std::collections::HashSet<usize>>,
    num_qubits: usize,
) -> bool {
    // Count degree of each qubit
    let mut high_degree_qubits = 0;
    let mut leaf_qubits = 0;

    for i in 0..num_qubits {
        if let Some(connections) = qubit_connections.get(&i) {
            if connections.len() > 2 {
                high_degree_qubits += 1;
            } else if connections.len() == 1 {
                leaf_qubits += 1;
            }
        }
    }

    // A star has one high-degree qubit and many leaf qubits
    high_degree_qubits == 1 && leaf_qubits >= 3
}

/// Check if the circuit has a layered structure (like QFT)
fn is_layered_structure<const N: usize>(circuit: &Circuit<N>) -> bool {
    // This is a simplified check - a proper analysis would be more complex
    // Here we just count rotations and controlled gates, which are common in QFT

    let mut rotation_gates = 0;
    let mut controlled_gates = 0;

    for gate in circuit.gates() {
        match gate.name() {
            "RZ" | "RY" | "RX" => rotation_gates += 1,
            "CNOT" | "CZ" | "CY" | "CH" | "CS" | "CRX" | "CRY" | "CRZ" => controlled_gates += 1,
            _ => {}
        }
    }

    // QFT-like circuits have many rotation gates and controlled operations
    rotation_gates >= N / 2 && controlled_gates >= N / 2
}

impl TensorNetworkSimulator {
    /// Apply a single-qubit gate to a tensor network
    fn apply_single_qubit_gate<const N: usize>(
        &self,
        network: &mut TensorNetwork,
        gate_matrix: &[Complex64],
        target: QubitId,
    ) -> QuantRS2Result<()> {
        let target_idx = target.id() as usize;
        if target_idx >= N {
            return Err(QuantRS2Error::InvalidQubitId(target.id()));
        }

        // Create a gate tensor from the matrix
        let gate_tensor = Tensor::from_matrix(gate_matrix, 2);

        // Insert or contract the gate tensor with the qubit tensor
        network.apply_gate(gate_tensor, target_idx)?;

        Ok(())
    }

    /// Apply a two-qubit gate to a tensor network
    fn apply_two_qubit_gate<const N: usize>(
        &self,
        network: &mut TensorNetwork,
        gate_matrix: &[Complex64],
        control: QubitId,
        target: QubitId,
    ) -> QuantRS2Result<()> {
        let control_idx = control.id() as usize;
        let target_idx = target.id() as usize;

        if control_idx >= N || target_idx >= N {
            return Err(QuantRS2Error::InvalidQubitId(if control_idx >= N {
                control.id()
            } else {
                target.id()
            }));
        }

        if control_idx == target_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "Control and target qubits must be different".into(),
            ));
        }

        // Create a gate tensor from the matrix
        let gate_tensor = Tensor::from_matrix(gate_matrix, 4);

        // Insert or contract the gate tensor with the qubit tensors
        network.apply_two_qubit_gate(gate_tensor, control_idx, target_idx)?;

        Ok(())
    }
}

impl Default for TensorNetworkSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Simulator<N> for TensorNetworkSimulator {
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>> {
        // Initialize a tensor network representing |0...0⟩
        let mut network = TensorNetwork::new(N);

        // Set the maximum bond dimension based on the optimization level
        network.max_bond_dimension = match self.optimization_level {
            0 => 4,  // Minimal optimization
            1 => 16, // Default
            2 => 32, // Advanced
            3 => 64, // Aggressive
            _ => 16, // Default for unknown levels
        };

        // Analyze the circuit structure to choose the best simulation approach
        let circuit_type = self.analyze_circuit_structure(circuit);

        // Choose an appropriate contraction strategy based on the circuit type
        // if one hasn't been explicitly set
        let effective_strategy = match &self.contraction_strategy {
            ContractionStrategy::Greedy => {
                // Auto-select strategy based on circuit type
                match circuit_type {
                    CircuitType::QFT => ContractionStrategy::QFT,
                    CircuitType::QAOA => ContractionStrategy::QAOA,
                    CircuitType::Linear => ContractionStrategy::Linear,
                    CircuitType::Star => ContractionStrategy::Star,
                    _ => ContractionStrategy::Greedy,
                }
            }
            // If a specific strategy was chosen, use that
            _ => self.contraction_strategy.clone(),
        };

        // Store the detected circuit type for later use when creating the state vector
        network.detected_circuit_type = circuit_type;

        // Apply the chosen contraction strategy
        match effective_strategy {
            ContractionStrategy::QFT => {
                // Set parameters optimized for QFT circuits
                network.max_bond_dimension = network.max_bond_dimension.max(32);
                // Use specialized contraction order for QFT
                // In a real implementation, this would set a custom contraction path
                // For now, we just set the flag
                network.using_qft_optimization = true;
            }
            ContractionStrategy::QAOA => {
                // Set parameters optimized for QAOA circuits
                network.max_bond_dimension = network.max_bond_dimension.max(32);
                // Use specialized contraction order for QAOA
                // In a real implementation, this would set a custom contraction path
                // For now, we just set the flag
                network.using_qaoa_optimization = true;
            }
            ContractionStrategy::Linear => {
                // Optimizations for linear circuits
                network.max_bond_dimension = network.max_bond_dimension.max(16);
                network.using_linear_optimization = true;
            }
            ContractionStrategy::Star => {
                // Optimizations for star-shaped circuits
                network.max_bond_dimension = network.max_bond_dimension.max(16);
                network.using_star_optimization = true;
            }
            _ => {
                // Default settings for greedy or custom strategies
                // No special optimization flag set
            }
        }

        // Apply each gate in the circuit
        for gate in circuit.gates() {
            match gate.name() {
                // Single-qubit gates
                "H" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::Hadamard>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "X" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PauliX>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "Y" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PauliY>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "Z" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PauliZ>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                // Rotation gates
                "RX" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::RotationX>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "RY" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::RotationY>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "RZ" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::RotationZ>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                // Phase gates
                "S" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::Phase>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "T" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::T>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "S†" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PhaseDagger>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "T†" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::TDagger>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "√X" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::SqrtX>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }
                "√X†" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::SqrtXDagger>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut network, &matrix, g.target)?;
                    }
                }

                // Two-qubit gates
                "CNOT" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CNOT>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut network, &matrix, g.control, g.target)?;
                    }
                }
                "CZ" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CZ>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut network, &matrix, g.control, g.target)?;
                    }
                }
                "SWAP" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::SWAP>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut network, &matrix, g.qubit1, g.qubit2)?;
                    }
                }
                "CY" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CY>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut network, &matrix, g.control, g.target)?;
                    }
                }
                "CH" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CH>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut network, &matrix, g.control, g.target)?;
                    }
                }
                "CS" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CS>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut network, &matrix, g.control, g.target)?;
                    }
                }
                "CRX" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CRX>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut network, &matrix, g.control, g.target)?;
                    }
                }
                "CRY" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CRY>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut network, &matrix, g.control, g.target)?;
                    }
                }
                "CRZ" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CRZ>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut network, &matrix, g.control, g.target)?;
                    }
                }

                // Three-qubit gates
                "Toffoli" | "Fredkin" => {
                    return Err(QuantRS2Error::UnsupportedOperation(format!(
                        "Gate {} not yet implemented for tensor network simulator",
                        gate.name()
                    )));
                }

                _ => {
                    return Err(QuantRS2Error::UnsupportedOperation(format!(
                        "Gate {} not supported",
                        gate.name()
                    )));
                }
            }
        }

        // Contract the entire network to obtain the final state vector
        let amplitudes = network.contract_to_statevector()?;

        // Create register from final state
        Register::<N>::with_amplitudes(amplitudes)
    }
}

/// A tensor network representation of a quantum state
#[derive(Debug, Clone)]
pub struct TensorNetwork {
    /// Number of qubits in the network
    num_qubits: usize,

    /// Tensors in the network, indexed by their ID
    tensors: HashMap<usize, Tensor>,

    /// Connections between tensors
    connections: Vec<(TensorIndex, TensorIndex)>,

    /// Ordered record of gate applications used to compute the final state.
    ///
    /// Each entry is `(gate_tensor, qubits)` where `gate_tensor` has rank `2k`
    /// with axis layout `[out_0,…,out_{k-1}, in_0,…,in_{k-1}]` (its row-major
    /// flattening equals the standard `2ᵏ × 2ᵏ` gate matrix), and `qubits[i]` is
    /// the register qubit that gate leg `i` acts on. For a two-qubit gate the
    /// order is `[control, target]`, matching the little-endian basis convention
    /// (`basis = control·2 + target`) used throughout the crate.
    applied_gates: Vec<(Tensor, Vec<usize>)>,

    /// Greedy contraction order cached by [`ContractableNetwork::optimize_contraction_order`].
    cached_contraction_path: Option<contraction::ContractionPath>,

    /// Next available tensor ID
    next_id: usize,

    /// Maximum bond dimension for tensor decompositions
    max_bond_dimension: usize,

    /// Detected circuit type
    detected_circuit_type: CircuitType,

    /// Flag indicating QFT optimization is being used
    using_qft_optimization: bool,

    /// Flag indicating QAOA optimization is being used
    using_qaoa_optimization: bool,

    /// Flag indicating linear circuit optimization is being used
    using_linear_optimization: bool,

    /// Flag indicating star circuit optimization is being used
    using_star_optimization: bool,
}

impl TensorNetwork {
    /// Create a new tensor network representing the |0...0⟩ state
    pub fn new(num_qubits: usize) -> Self {
        let mut network = Self {
            num_qubits,
            tensors: HashMap::new(),
            connections: Vec::new(),
            applied_gates: Vec::new(),
            cached_contraction_path: None,
            next_id: 0,
            max_bond_dimension: 16,
            detected_circuit_type: CircuitType::General,
            using_qft_optimization: false,
            using_qaoa_optimization: false,
            using_linear_optimization: false,
            using_star_optimization: false,
        };

        // Initialize each qubit to |0⟩
        for i in 0..num_qubits {
            let qubit_tensor = Tensor::qubit_zero();
            network.add_tensor(qubit_tensor, i);
        }

        network
    }

    /// Add a tensor to the network
    fn add_tensor(&mut self, tensor: Tensor, qubit_index: usize) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        self.tensors.insert(id, tensor);

        id
    }

    /// Record a single-qubit gate application.
    ///
    /// The gate tensor (rank 2, layout `[out, in]`) is appended to the ordered
    /// gate list. It is later contracted into the state tensor by
    /// [`Self::contract_to_statevector`]; nothing is fabricated here.
    pub fn apply_gate(&mut self, gate_tensor: Tensor, qubit_index: usize) -> QuantRS2Result<()> {
        if gate_tensor.rank != 2 {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "Single-qubit gate tensor must have rank 2, got {}",
                gate_tensor.rank
            )));
        }
        self.applied_gates.push((gate_tensor, vec![qubit_index]));
        Ok(())
    }

    /// Record a two-qubit gate application.
    ///
    /// The gate tensor (rank 4, layout `[out_c, out_t, in_c, in_t]`) and the
    /// `[control, target]` qubit pair are appended to the ordered gate list for
    /// later contraction into the state tensor.
    pub fn apply_two_qubit_gate(
        &mut self,
        gate_tensor: Tensor,
        control_index: usize,
        target_index: usize,
    ) -> QuantRS2Result<()> {
        if gate_tensor.rank != 4 {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "Two-qubit gate tensor must have rank 4, got {}",
                gate_tensor.rank
            )));
        }
        self.applied_gates
            .push((gate_tensor, vec![control_index, target_index]));
        Ok(())
    }

    /// Contract the entire network to produce the final state vector.
    ///
    /// This performs a genuine tensor contraction: the state is held as a rank-`N`
    /// tensor (`N` = number of qubits, every index of dimension 2) initialized to
    /// `|0…0⟩`, and each recorded gate tensor is contracted into it in
    /// application order (a sequential contraction path). The resulting amplitudes
    /// are read out in the crate-standard little-endian basis (qubit `q` occupies
    /// bit `q` of the flat index), so the output matches the state-vector
    /// simulator exactly. No circuit-type heuristics or canned states are used —
    /// the actual gate matrices and qubit indices fully determine the result.
    pub fn contract_to_statevector(&self) -> QuantRS2Result<Vec<Complex64>> {
        let n = self.num_qubits;
        if n == 0 {
            return Ok(vec![Complex64::new(1.0, 0.0)]);
        }

        // State tensor: axis `j` corresponds to qubit `n-1-j`, so the natural
        // row-major (C-order) traversal reproduces the little-endian statevector.
        let shape = vec![2usize; n];
        let mut state_data = ArrayD::<Complex64>::zeros(IxDyn(shape.as_slice()));
        state_data[IxDyn(&vec![0usize; n])] = Complex64::new(1.0, 0.0);
        let mut state = Tensor::new(state_data);

        for (gate, qubits) in &self.applied_gates {
            // Map each gate leg to the state axis of the qubit it acts on.
            let state_axes: Vec<usize> = qubits.iter().map(|&q| n - 1 - q).collect();
            state = Self::apply_gate_to_state(&state, gate, &state_axes)?;
        }

        // Read amplitudes in logical C-order = little-endian statevector order.
        Ok(state.data.iter().copied().collect())
    }

    /// Contract a `k`-qubit gate tensor into the rank-`N` state tensor, leaving
    /// the state's axes in canonical qubit order.
    ///
    /// `gate` has rank `2k` with layout `[out_0,…,out_{k-1}, in_0,…,in_{k-1}]`;
    /// `state_axes[i]` is the state axis contracted with gate input leg `i` (and
    /// which receives output leg `i`). This is the Einstein summation
    ///   `state'[…] = Σ_{a} gate[o…,a…] · state[…a…]`
    /// carried out by explicit index iteration — correct for any `k` and any
    /// qubit placement (including non-adjacent control/target).
    fn apply_gate_to_state(
        state: &Tensor,
        gate: &Tensor,
        state_axes: &[usize],
    ) -> QuantRS2Result<Tensor> {
        let n = state.rank;
        let k = state_axes.len();

        if gate.rank != 2 * k {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "Gate rank {} does not match 2×(qubits acted on) = {}",
                gate.rank,
                2 * k
            )));
        }
        for &axis in state_axes {
            if axis >= n {
                return Err(QuantRS2Error::CircuitValidationFailed(format!(
                    "Gate acts on state axis {axis} but state has rank {n}"
                )));
            }
        }

        let mut result = ArrayD::<Complex64>::zeros(IxDyn(state.data.shape()));
        for (out_idx, slot) in result.indexed_iter_mut() {
            let out = out_idx.slice();
            let mut acc = Complex64::new(0.0, 0.0);

            // Sum over the 2^k input configurations of the gate's input legs.
            for in_config in 0..(1usize << k) {
                let mut gate_idx = vec![0usize; 2 * k];
                let mut src = out.to_vec();
                for (i, &axis) in state_axes.iter().enumerate() {
                    // Output leg value is fixed by the current output multi-index.
                    gate_idx[i] = out[axis];
                    // Input leg value ranges over in_config (leg 0 = most significant).
                    let bit = (in_config >> (k - 1 - i)) & 1;
                    gate_idx[k + i] = bit;
                    src[axis] = bit;
                }
                acc += gate.data[IxDyn(&gate_idx)] * state.data[IxDyn(&src)];
            }
            *slot = acc;
        }

        Ok(Tensor::new(result))
    }
}

impl ContractableNetwork for TensorNetwork {
    /// Genuinely contract the two referenced tensors, replacing them with their
    /// contraction over any shared bonds (or their outer product if they share
    /// none) and returning the ID of the merged tensor.
    fn contract_tensors(&mut self, tensor_id1: usize, tensor_id2: usize) -> QuantRS2Result<usize> {
        let t1 = self.tensors.get(&tensor_id1).cloned().ok_or_else(|| {
            QuantRS2Error::CircuitValidationFailed(format!("Tensor {tensor_id1} not found"))
        })?;
        let t2 = self.tensors.get(&tensor_id2).cloned().ok_or_else(|| {
            QuantRS2Error::CircuitValidationFailed(format!("Tensor {tensor_id2} not found"))
        })?;

        // Determine shared axes between the two tensors from the connection list.
        let shared = contraction::shared_axis_pairs(&self.connections, tensor_id1, tensor_id2);
        let merged = contraction::contract_pair_multi(&t1, &t2, &shared)?;

        let new_id = self.next_id;
        self.next_id += 1;
        self.tensors.remove(&tensor_id1);
        self.tensors.remove(&tensor_id2);
        self.tensors.insert(new_id, merged);

        // Drop connections that referenced either contracted tensor; the merged
        // tensor's remaining open legs are tracked by later contraction steps.
        self.connections.retain(|(a, b)| {
            ![tensor_id1, tensor_id2].contains(&a.tensor_id)
                && ![tensor_id1, tensor_id2].contains(&b.tensor_id)
        });

        Ok(new_id)
    }

    /// Compute and cache a greedy contraction order over the current network.
    ///
    /// This runs the real greedy path optimizer
    /// ([`contraction::calculate_greedy_contraction_path`]) and stores the result
    /// for reuse; it is no longer a no-op placeholder.
    fn optimize_contraction_order(&mut self) -> QuantRS2Result<()> {
        let path =
            contraction::calculate_greedy_contraction_path(&self.tensors, &self.connections)?;
        self.cached_contraction_path = Some(path);
        Ok(())
    }
}
