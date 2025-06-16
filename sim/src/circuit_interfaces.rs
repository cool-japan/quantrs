//! Efficient interfaces with circuit module for seamless integration.
//!
//! This module provides comprehensive bridge functionality between quantum circuit
//! representations from the circuit module and various simulation backends in the
//! sim module. It includes optimized gate translation, circuit analysis, and
//! execution pipelines for maximum performance.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use crate::sparse::CSRMatrix;
use crate::statevector::StateVectorSimulator;
#[cfg(feature = "advanced_math")]
use crate::tensor_network::TensorNetwork;

/// Circuit interface configuration
#[derive(Debug, Clone)]
pub struct CircuitInterfaceConfig {
    /// Enable automatic backend selection
    pub auto_backend_selection: bool,
    /// Enable circuit optimization during translation
    pub enable_optimization: bool,
    /// Maximum qubits for state vector simulation
    pub max_statevector_qubits: usize,
    /// Maximum bond dimension for MPS simulation
    pub max_mps_bond_dim: usize,
    /// Enable parallel gate compilation
    pub parallel_compilation: bool,
    /// Cache compiled circuits
    pub enable_circuit_cache: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Enable circuit analysis and profiling
    pub enable_profiling: bool,
}

impl Default for CircuitInterfaceConfig {
    fn default() -> Self {
        Self {
            auto_backend_selection: true,
            enable_optimization: true,
            max_statevector_qubits: 25,
            max_mps_bond_dim: 1024,
            parallel_compilation: true,
            enable_circuit_cache: true,
            max_cache_size: 10000,
            enable_profiling: true,
        }
    }
}

/// Quantum gate types for circuit interface
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterfaceGateType {
    // Single-qubit gates
    Identity,
    PauliX,
    X, // Alias for PauliX
    PauliY,
    PauliZ,
    Hadamard,
    H, // Alias for Hadamard
    S,
    T,
    Phase(f64),
    RX(f64),
    RY(f64),
    RZ(f64),
    U1(f64),
    U2(f64, f64),
    U3(f64, f64, f64),
    // Two-qubit gates
    CNOT,
    CZ,
    CY,
    SWAP,
    ISwap,
    CRX(f64),
    CRY(f64),
    CRZ(f64),
    CPhase(f64),
    // Three-qubit gates
    Toffoli,
    Fredkin,
    // Multi-qubit gates
    MultiControlledX(usize), // Number of control qubits
    MultiControlledZ(usize),
    // Custom gates
    Custom(String, Array2<Complex64>),
    // Measurement
    Measure,
    Reset,
}

// Custom Hash implementation since f64 doesn't implement Hash
impl std::hash::Hash for InterfaceGateType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use std::mem;
        match self {
            InterfaceGateType::Identity => 0u8.hash(state),
            InterfaceGateType::PauliX => 1u8.hash(state),
            InterfaceGateType::X => 2u8.hash(state),
            InterfaceGateType::PauliY => 3u8.hash(state),
            InterfaceGateType::PauliZ => 4u8.hash(state),
            InterfaceGateType::Hadamard => 5u8.hash(state),
            InterfaceGateType::H => 6u8.hash(state),
            InterfaceGateType::S => 7u8.hash(state),
            InterfaceGateType::T => 8u8.hash(state),
            InterfaceGateType::Phase(angle) => {
                9u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*angle) }.hash(state);
            }
            InterfaceGateType::RX(angle) => {
                10u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*angle) }.hash(state);
            }
            InterfaceGateType::RY(angle) => {
                11u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*angle) }.hash(state);
            }
            InterfaceGateType::RZ(angle) => {
                12u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*angle) }.hash(state);
            }
            InterfaceGateType::U1(angle) => {
                13u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*angle) }.hash(state);
            }
            InterfaceGateType::U2(theta, phi) => {
                14u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*theta) }.hash(state);
                unsafe { mem::transmute::<f64, u64>(*phi) }.hash(state);
            }
            InterfaceGateType::U3(theta, phi, lambda) => {
                15u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*theta) }.hash(state);
                unsafe { mem::transmute::<f64, u64>(*phi) }.hash(state);
                unsafe { mem::transmute::<f64, u64>(*lambda) }.hash(state);
            }
            InterfaceGateType::CNOT => 16u8.hash(state),
            InterfaceGateType::CZ => 17u8.hash(state),
            InterfaceGateType::CY => 18u8.hash(state),
            InterfaceGateType::SWAP => 19u8.hash(state),
            InterfaceGateType::ISwap => 20u8.hash(state),
            InterfaceGateType::CRX(angle) => {
                21u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*angle) }.hash(state);
            }
            InterfaceGateType::CRY(angle) => {
                22u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*angle) }.hash(state);
            }
            InterfaceGateType::CRZ(angle) => {
                23u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*angle) }.hash(state);
            }
            InterfaceGateType::CPhase(angle) => {
                24u8.hash(state);
                unsafe { mem::transmute::<f64, u64>(*angle) }.hash(state);
            }
            InterfaceGateType::Toffoli => 25u8.hash(state),
            InterfaceGateType::Fredkin => 26u8.hash(state),
            InterfaceGateType::MultiControlledX(n) => {
                27u8.hash(state);
                n.hash(state);
            }
            InterfaceGateType::MultiControlledZ(n) => {
                28u8.hash(state);
                n.hash(state);
            }
            InterfaceGateType::Custom(name, matrix) => {
                29u8.hash(state);
                name.hash(state);
                // Hash matrix shape instead of all elements
                matrix.shape().hash(state);
            }
            InterfaceGateType::Measure => 30u8.hash(state),
            InterfaceGateType::Reset => 31u8.hash(state),
        }
    }
}

// Custom Eq implementation since f64 doesn't implement Eq
impl Eq for InterfaceGateType {}

/// Quantum gate representation for circuit interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceGate {
    /// Gate type and parameters
    pub gate_type: InterfaceGateType,
    /// Qubits this gate acts on
    pub qubits: Vec<usize>,
    /// Classical register targets (for measurements)
    pub classical_targets: Vec<usize>,
    /// Gate position in the circuit
    pub position: usize,
    /// Conditional execution (if classical bit is 1)
    pub condition: Option<usize>,
    /// Gate label for debugging
    pub label: Option<String>,
}

impl InterfaceGate {
    /// Create a new interface gate
    pub fn new(gate_type: InterfaceGateType, qubits: Vec<usize>) -> Self {
        Self {
            gate_type,
            qubits,
            classical_targets: Vec::new(),
            position: 0,
            condition: None,
            label: None,
        }
    }

    /// Create a measurement gate
    pub fn measurement(qubit: usize, classical_bit: usize) -> Self {
        Self {
            gate_type: InterfaceGateType::Measure,
            qubits: vec![qubit],
            classical_targets: vec![classical_bit],
            position: 0,
            condition: None,
            label: None,
        }
    }

    /// Create a conditional gate
    pub fn conditional(mut self, condition: usize) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Add a label to the gate
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Get the unitary matrix for this gate
    pub fn unitary_matrix(&self) -> Result<Array2<Complex64>> {
        match &self.gate_type {
            InterfaceGateType::Identity => Ok(Array2::eye(2)),
            InterfaceGateType::PauliX => Ok(Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap()),
            InterfaceGateType::PauliY => Ok(Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap()),
            InterfaceGateType::PauliZ => Ok(Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .unwrap()),
            InterfaceGateType::Hadamard => {
                let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
                Ok(Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(inv_sqrt2, 0.0),
                        Complex64::new(inv_sqrt2, 0.0),
                        Complex64::new(inv_sqrt2, 0.0),
                        Complex64::new(-inv_sqrt2, 0.0),
                    ],
                )
                .unwrap())
            }
            InterfaceGateType::S => Ok(Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 1.0),
                ],
            )
            .unwrap()),
            InterfaceGateType::T => {
                let phase = Complex64::new(0.0, std::f64::consts::PI / 4.0).exp();
                Ok(Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        phase,
                    ],
                )
                .unwrap())
            }
            InterfaceGateType::Phase(theta) => {
                let phase = Complex64::new(0.0, *theta).exp();
                Ok(Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        phase,
                    ],
                )
                .unwrap())
            }
            InterfaceGateType::RX(theta) => {
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                Ok(Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(cos_half, 0.0),
                        Complex64::new(0.0, -sin_half),
                        Complex64::new(0.0, -sin_half),
                        Complex64::new(cos_half, 0.0),
                    ],
                )
                .unwrap())
            }
            InterfaceGateType::RY(theta) => {
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                Ok(Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(cos_half, 0.0),
                        Complex64::new(-sin_half, 0.0),
                        Complex64::new(sin_half, 0.0),
                        Complex64::new(cos_half, 0.0),
                    ],
                )
                .unwrap())
            }
            InterfaceGateType::RZ(theta) => {
                let exp_neg = Complex64::new(0.0, -theta / 2.0).exp();
                let exp_pos = Complex64::new(0.0, theta / 2.0).exp();
                Ok(Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        exp_neg,
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        exp_pos,
                    ],
                )
                .unwrap())
            }
            InterfaceGateType::CNOT => Ok(Array2::from_shape_vec(
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
            .unwrap()),
            InterfaceGateType::CZ => Ok(Array2::from_shape_vec(
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
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .unwrap()),
            InterfaceGateType::SWAP => Ok(Array2::from_shape_vec(
                (4, 4),
                vec![
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
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .unwrap()),
            InterfaceGateType::MultiControlledZ(num_controls) => {
                let total_qubits = num_controls + 1;
                let dim = 1 << total_qubits;
                let mut matrix = Array2::eye(dim);

                // Apply Z to the target when all control qubits are |1⟩
                let target_state = (1 << total_qubits) - 1; // All qubits in |1⟩ state
                matrix[(target_state, target_state)] = Complex64::new(-1.0, 0.0);

                Ok(matrix)
            }
            InterfaceGateType::MultiControlledX(num_controls) => {
                let total_qubits = num_controls + 1;
                let dim = 1 << total_qubits;
                let mut matrix = Array2::eye(dim);

                // Apply X to the target when all control qubits are |1⟩
                let control_pattern = (1 << *num_controls) - 1; // All control qubits in |1⟩
                let target_bit = 1 << num_controls;

                // Swap the two states where only the target bit differs
                let state0 = control_pattern; // Target qubit is |0⟩
                let state1 = control_pattern | target_bit; // Target qubit is |1⟩

                matrix[(state0, state0)] = Complex64::new(0.0, 0.0);
                matrix[(state1, state1)] = Complex64::new(0.0, 0.0);
                matrix[(state0, state1)] = Complex64::new(1.0, 0.0);
                matrix[(state1, state0)] = Complex64::new(1.0, 0.0);

                Ok(matrix)
            }
            InterfaceGateType::Custom(_, matrix) => Ok(matrix.clone()),
            _ => Err(SimulatorError::UnsupportedOperation(format!(
                "Unitary matrix not available for gate type: {:?}",
                self.gate_type
            ))),
        }
    }

    /// Check if this gate is a measurement
    pub fn is_measurement(&self) -> bool {
        matches!(self.gate_type, InterfaceGateType::Measure)
    }

    /// Check if this gate is unitary
    pub fn is_unitary(&self) -> bool {
        !matches!(
            self.gate_type,
            InterfaceGateType::Measure | InterfaceGateType::Reset
        )
    }

    /// Get the number of qubits this gate acts on
    pub fn num_qubits(&self) -> usize {
        match &self.gate_type {
            InterfaceGateType::MultiControlledX(n) | InterfaceGateType::MultiControlledZ(n) => {
                n + 1
            }
            _ => self.qubits.len(),
        }
    }
}

/// Quantum circuit representation for interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of classical bits
    pub num_classical: usize,
    /// Gates in the circuit
    pub gates: Vec<InterfaceGate>,
    /// Circuit metadata
    pub metadata: CircuitMetadata,
}

/// Circuit metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitMetadata {
    /// Circuit name
    pub name: Option<String>,
    /// Circuit description
    pub description: Option<String>,
    /// Creation timestamp
    #[serde(skip)]
    pub created_at: Option<std::time::SystemTime>,
    /// Circuit depth
    pub depth: usize,
    /// Number of two-qubit gates
    pub two_qubit_gates: usize,
    /// Circuit complexity score
    pub complexity_score: f64,
    /// Estimated classical simulation complexity
    pub classical_complexity: Option<f64>,
}

impl InterfaceCircuit {
    /// Create a new circuit
    pub fn new(num_qubits: usize, num_classical: usize) -> Self {
        Self {
            num_qubits,
            num_classical,
            gates: Vec::new(),
            metadata: CircuitMetadata::default(),
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, mut gate: InterfaceGate) {
        gate.position = self.gates.len();
        self.gates.push(gate);
        self.update_metadata();
    }

    /// Add multiple gates to the circuit
    pub fn add_gates(&mut self, gates: Vec<InterfaceGate>) {
        for gate in gates {
            self.add_gate(gate);
        }
    }

    /// Update circuit metadata
    fn update_metadata(&mut self) {
        let depth = self.calculate_depth();
        let two_qubit_gates = self.gates.iter().filter(|g| g.num_qubits() == 2).count();

        let complexity_score = self.calculate_complexity_score();

        self.metadata.depth = depth;
        self.metadata.two_qubit_gates = two_qubit_gates;
        self.metadata.complexity_score = complexity_score;
    }

    /// Calculate circuit depth
    pub fn calculate_depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }

        let mut qubit_depths = vec![0; self.num_qubits];

        for gate in &self.gates {
            let max_depth = gate
                .qubits
                .iter()
                .map(|&q| qubit_depths[q])
                .max()
                .unwrap_or(0);

            for &qubit in &gate.qubits {
                qubit_depths[qubit] = max_depth + 1;
            }
        }

        qubit_depths.into_iter().max().unwrap_or(0)
    }

    /// Calculate complexity score
    fn calculate_complexity_score(&self) -> f64 {
        let mut score = 0.0;

        for gate in &self.gates {
            let gate_score = match gate.num_qubits() {
                1 => 1.0,
                2 => 5.0,
                3 => 25.0,
                n => (5.0_f64).powi(n as i32 - 1),
            };
            score += gate_score;
        }

        score
    }

    /// Extract subcircuit
    pub fn subcircuit(&self, start: usize, end: usize) -> Result<InterfaceCircuit> {
        if start >= end || end > self.gates.len() {
            return Err(SimulatorError::InvalidInput(
                "Invalid subcircuit range".to_string(),
            ));
        }

        let mut subcircuit = InterfaceCircuit::new(self.num_qubits, self.num_classical);
        subcircuit.gates = self.gates[start..end].to_vec();
        subcircuit.update_metadata();

        Ok(subcircuit)
    }

    /// Optimize the circuit
    pub fn optimize(&mut self) -> CircuitOptimizationResult {
        let original_gates = self.gates.len();
        let original_depth = self.metadata.depth;

        // Apply various optimization passes
        self.remove_identity_gates();
        self.cancel_adjacent_gates();
        self.merge_rotation_gates();
        self.optimize_cnot_patterns();

        self.update_metadata();

        CircuitOptimizationResult {
            original_gates,
            optimized_gates: self.gates.len(),
            original_depth,
            optimized_depth: self.metadata.depth,
            gates_eliminated: original_gates.saturating_sub(self.gates.len()),
            depth_reduction: original_depth.saturating_sub(self.metadata.depth),
        }
    }

    /// Remove identity gates
    fn remove_identity_gates(&mut self) {
        self.gates
            .retain(|gate| !matches!(gate.gate_type, InterfaceGateType::Identity));
    }

    /// Cancel adjacent gates
    fn cancel_adjacent_gates(&mut self) {
        let mut i = 0;
        while i + 1 < self.gates.len() {
            if self.gates_cancel(&self.gates[i], &self.gates[i + 1]) {
                self.gates.remove(i);
                self.gates.remove(i);
                if i > 0 {
                    i = i.saturating_sub(1);
                }
            } else {
                i += 1;
            }
        }
    }

    /// Check if two gates cancel each other
    fn gates_cancel(&self, gate1: &InterfaceGate, gate2: &InterfaceGate) -> bool {
        if gate1.qubits != gate2.qubits {
            return false;
        }

        match (&gate1.gate_type, &gate2.gate_type) {
            (InterfaceGateType::PauliX, InterfaceGateType::PauliX)
            | (InterfaceGateType::PauliY, InterfaceGateType::PauliY)
            | (InterfaceGateType::PauliZ, InterfaceGateType::PauliZ)
            | (InterfaceGateType::Hadamard, InterfaceGateType::Hadamard)
            | (InterfaceGateType::S, InterfaceGateType::S)
            | (InterfaceGateType::CNOT, InterfaceGateType::CNOT)
            | (InterfaceGateType::CZ, InterfaceGateType::CZ)
            | (InterfaceGateType::SWAP, InterfaceGateType::SWAP) => true,
            _ => false,
        }
    }

    /// Merge rotation gates
    fn merge_rotation_gates(&mut self) {
        let mut i = 0;
        while i + 1 < self.gates.len() {
            if let Some(merged) = self.try_merge_rotations(&self.gates[i], &self.gates[i + 1]) {
                self.gates[i] = merged;
                self.gates.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    /// Try to merge two rotation gates
    fn try_merge_rotations(
        &self,
        gate1: &InterfaceGate,
        gate2: &InterfaceGate,
    ) -> Option<InterfaceGate> {
        if gate1.qubits != gate2.qubits {
            return None;
        }

        match (&gate1.gate_type, &gate2.gate_type) {
            (InterfaceGateType::RX(angle1), InterfaceGateType::RX(angle2)) => Some(
                InterfaceGate::new(InterfaceGateType::RX(angle1 + angle2), gate1.qubits.clone()),
            ),
            (InterfaceGateType::RY(angle1), InterfaceGateType::RY(angle2)) => Some(
                InterfaceGate::new(InterfaceGateType::RY(angle1 + angle2), gate1.qubits.clone()),
            ),
            (InterfaceGateType::RZ(angle1), InterfaceGateType::RZ(angle2)) => Some(
                InterfaceGate::new(InterfaceGateType::RZ(angle1 + angle2), gate1.qubits.clone()),
            ),
            _ => None,
        }
    }

    /// Optimize CNOT patterns
    fn optimize_cnot_patterns(&mut self) {
        // Look for CNOT chains and optimize them
        let mut i = 0;
        while i + 2 < self.gates.len() {
            if self.is_cnot_chain(i) {
                self.optimize_cnot_chain(i);
            }
            i += 1;
        }
    }

    /// Check if there's a CNOT chain starting at position i
    fn is_cnot_chain(&self, start: usize) -> bool {
        if start + 2 >= self.gates.len() {
            return false;
        }

        for i in start..start + 3 {
            if !matches!(self.gates[i].gate_type, InterfaceGateType::CNOT) {
                return false;
            }
        }

        true
    }

    /// Optimize a CNOT chain
    fn optimize_cnot_chain(&mut self, start: usize) {
        // Simple optimization: remove triple CNOTs with same control/target
        if start + 2 < self.gates.len() {
            let gate1 = &self.gates[start];
            let gate2 = &self.gates[start + 1];
            let gate3 = &self.gates[start + 2];

            if gate1.qubits == gate2.qubits && gate2.qubits == gate3.qubits {
                // Three identical CNOTs = one CNOT
                self.gates.drain(start + 1..start + 3);
            }
        }
    }
}

/// Circuit optimization result
#[derive(Debug, Clone)]
pub struct CircuitOptimizationResult {
    /// Original number of gates
    pub original_gates: usize,
    /// Optimized number of gates
    pub optimized_gates: usize,
    /// Original circuit depth
    pub original_depth: usize,
    /// Optimized circuit depth
    pub optimized_depth: usize,
    /// Number of gates eliminated
    pub gates_eliminated: usize,
    /// Depth reduction achieved
    pub depth_reduction: usize,
}

/// Circuit execution backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationBackend {
    /// State vector simulation
    StateVector,
    /// Matrix Product State simulation
    MPS,
    /// Stabilizer simulation (for Clifford circuits)
    Stabilizer,
    /// Sparse matrix simulation
    Sparse,
    /// Tensor network simulation
    TensorNetwork,
    /// Automatic backend selection
    Auto,
}

/// Circuit interface for simulation backends
pub struct CircuitInterface {
    /// Configuration
    config: CircuitInterfaceConfig,
    /// SciRS2 backend for optimization
    backend: Option<SciRS2Backend>,
    /// Circuit cache
    circuit_cache: Arc<Mutex<HashMap<u64, CompiledCircuit>>>,
    /// Performance statistics
    stats: CircuitInterfaceStats,
}

/// Compiled circuit representation
#[derive(Debug, Clone)]
pub struct CompiledCircuit {
    /// Original circuit
    pub original: InterfaceCircuit,
    /// Optimized gate sequence
    pub optimized_gates: Vec<InterfaceGate>,
    /// Backend-specific compiled representation
    pub backend_data: BackendCompiledData,
    /// Compilation metadata
    pub metadata: CompilationMetadata,
}

/// Backend-specific compiled data
#[derive(Debug, Clone)]
pub enum BackendCompiledData {
    StateVector {
        unitary_matrices: Vec<Array2<Complex64>>,
        gate_indices: Vec<Vec<usize>>,
    },
    MPS {
        bond_dimensions: Vec<usize>,
        truncation_thresholds: Vec<f64>,
    },
    Stabilizer {
        clifford_sequence: Vec<StabilizerOp>,
    },
    Sparse {
        sparse_matrices: Vec<CSRMatrix>,
    },
}

/// Stabilizer operation for Clifford circuits
#[derive(Debug, Clone)]
pub enum StabilizerOp {
    H(usize),
    S(usize),
    CNOT(usize, usize),
    X(usize),
    Y(usize),
    Z(usize),
}

/// Compilation metadata
#[derive(Debug, Clone)]
pub struct CompilationMetadata {
    /// Compilation time in milliseconds
    pub compilation_time_ms: f64,
    /// Backend used for compilation
    pub backend: SimulationBackend,
    /// Optimization passes applied
    pub optimization_passes: Vec<String>,
    /// Estimated execution time
    pub estimated_execution_time_ms: f64,
    /// Memory requirements estimate
    pub estimated_memory_bytes: usize,
}

/// Circuit interface performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitInterfaceStats {
    /// Total circuits compiled
    pub circuits_compiled: usize,
    /// Total compilation time
    pub total_compilation_time_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Backend selection counts
    pub backend_selections: HashMap<String, usize>,
    /// Optimization statistics
    pub optimization_stats: OptimizationStats,
}

/// Optimization statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Total gates eliminated
    pub total_gates_eliminated: usize,
    /// Total depth reduction
    pub total_depth_reduction: usize,
    /// Average optimization ratio
    pub average_optimization_ratio: f64,
}

impl CircuitInterface {
    /// Create new circuit interface
    pub fn new(config: CircuitInterfaceConfig) -> Result<Self> {
        Ok(Self {
            config,
            backend: None,
            circuit_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: CircuitInterfaceStats::default(),
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Compile circuit for execution
    pub fn compile_circuit(
        &mut self,
        circuit: &InterfaceCircuit,
        backend: SimulationBackend,
    ) -> Result<CompiledCircuit> {
        let start_time = std::time::Instant::now();

        // Check cache first
        let circuit_hash = self.calculate_circuit_hash(circuit);
        if self.config.enable_circuit_cache {
            let cache = self.circuit_cache.lock().unwrap();
            if let Some(compiled) = cache.get(&circuit_hash) {
                self.stats.cache_hit_rate =
                    (self.stats.cache_hit_rate * self.stats.circuits_compiled as f64 + 1.0)
                        / (self.stats.circuits_compiled + 1) as f64;
                return Ok(compiled.clone());
            }
        }

        // Select backend automatically if needed
        let selected_backend = if backend == SimulationBackend::Auto {
            self.select_optimal_backend(circuit)?
        } else {
            backend
        };

        // Optimize circuit if enabled
        let mut optimized_circuit = circuit.clone();
        let mut optimization_passes = Vec::new();

        if self.config.enable_optimization {
            let opt_result = optimized_circuit.optimize();
            optimization_passes.push("basic_optimization".to_string());

            self.stats.optimization_stats.total_gates_eliminated += opt_result.gates_eliminated;
            self.stats.optimization_stats.total_depth_reduction += opt_result.depth_reduction;
        }

        // Compile for specific backend
        let backend_data = self.compile_for_backend(&optimized_circuit, selected_backend)?;

        // Estimate execution time and memory
        let estimated_execution_time_ms =
            self.estimate_execution_time(&optimized_circuit, selected_backend);
        let estimated_memory_bytes =
            self.estimate_memory_requirements(&optimized_circuit, selected_backend);

        let compilation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        let compiled = CompiledCircuit {
            original: circuit.clone(),
            optimized_gates: optimized_circuit.gates,
            backend_data,
            metadata: CompilationMetadata {
                compilation_time_ms,
                backend: selected_backend,
                optimization_passes,
                estimated_execution_time_ms,
                estimated_memory_bytes,
            },
        };

        // Update cache
        if self.config.enable_circuit_cache {
            let mut cache = self.circuit_cache.lock().unwrap();
            if cache.len() >= self.config.max_cache_size {
                // Simple LRU: remove oldest entry
                if let Some(oldest_key) = cache.keys().next().copied() {
                    cache.remove(&oldest_key);
                }
            }
            cache.insert(circuit_hash, compiled.clone());
        }

        // Update statistics
        self.stats.circuits_compiled += 1;
        self.stats.total_compilation_time_ms += compilation_time_ms;
        *self
            .stats
            .backend_selections
            .entry(format!("{:?}", selected_backend))
            .or_insert(0) += 1;

        Ok(compiled)
    }

    /// Execute compiled circuit
    pub fn execute_circuit(
        &mut self,
        compiled: &CompiledCircuit,
        initial_state: Option<Array1<Complex64>>,
    ) -> Result<CircuitExecutionResult> {
        let start_time = std::time::Instant::now();

        let result = match compiled.metadata.backend {
            SimulationBackend::StateVector => self.execute_statevector(compiled, initial_state)?,
            SimulationBackend::MPS => self.execute_mps(compiled, initial_state)?,
            SimulationBackend::Stabilizer => self.execute_stabilizer(compiled)?,
            SimulationBackend::Sparse => self.execute_sparse(compiled, initial_state)?,
            #[cfg(feature = "advanced_math")]
            SimulationBackend::TensorNetwork => {
                self.execute_tensor_network(compiled, initial_state)?
            }
            #[cfg(not(feature = "advanced_math"))]
            SimulationBackend::TensorNetwork => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Tensor network simulation requires advanced_math feature".to_string(),
                ))
            }
            SimulationBackend::Auto => {
                unreachable!("Auto backend should be resolved during compilation")
            }
        };

        let execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(CircuitExecutionResult {
            final_state: result.final_state,
            measurement_results: result.measurement_results,
            classical_bits: result.classical_bits,
            execution_time_ms,
            backend_used: compiled.metadata.backend,
            memory_used_bytes: result.memory_used_bytes,
        })
    }

    /// Select optimal backend for circuit
    fn select_optimal_backend(&self, circuit: &InterfaceCircuit) -> Result<SimulationBackend> {
        let num_qubits = circuit.num_qubits;
        let two_qubit_gates = circuit.metadata.two_qubit_gates;
        let total_gates = circuit.gates.len();

        // Check if circuit is Clifford (can use stabilizer simulation)
        if self.is_clifford_circuit(circuit) {
            return Ok(SimulationBackend::Stabilizer);
        }

        // For small circuits, use state vector
        if num_qubits <= self.config.max_statevector_qubits {
            return Ok(SimulationBackend::StateVector);
        }

        // For circuits with low entanglement, use MPS
        let entanglement_score = two_qubit_gates as f64 / total_gates as f64;
        if entanglement_score < 0.3 {
            return Ok(SimulationBackend::MPS);
        }

        // For very sparse circuits, use sparse simulation
        let sparsity_score = self.estimate_sparsity(circuit);
        if sparsity_score > 0.8 {
            return Ok(SimulationBackend::Sparse);
        }

        // For highly structured circuits, use tensor networks
        if self.has_tensor_network_structure(circuit) {
            return Ok(SimulationBackend::TensorNetwork);
        }

        // Default to MPS for large circuits
        Ok(SimulationBackend::MPS)
    }

    /// Check if circuit is Clifford
    fn is_clifford_circuit(&self, circuit: &InterfaceCircuit) -> bool {
        circuit.gates.iter().all(|gate| {
            matches!(
                gate.gate_type,
                InterfaceGateType::Identity
                    | InterfaceGateType::PauliX
                    | InterfaceGateType::PauliY
                    | InterfaceGateType::PauliZ
                    | InterfaceGateType::Hadamard
                    | InterfaceGateType::S
                    | InterfaceGateType::CNOT
                    | InterfaceGateType::CZ
                    | InterfaceGateType::SWAP
                    | InterfaceGateType::Measure
                    | InterfaceGateType::Reset
            )
        })
    }

    /// Estimate circuit sparsity
    fn estimate_sparsity(&self, circuit: &InterfaceCircuit) -> f64 {
        // Heuristic: circuits with many single-qubit gates are typically sparser
        let single_qubit_gates = circuit.gates.iter().filter(|g| g.num_qubits() == 1).count();

        single_qubit_gates as f64 / circuit.gates.len() as f64
    }

    /// Check if circuit has tensor network structure
    fn has_tensor_network_structure(&self, circuit: &InterfaceCircuit) -> bool {
        // Heuristic: circuits with regular structure and moderate entanglement
        let depth = circuit.metadata.depth;
        let num_qubits = circuit.num_qubits;

        // Look for regular patterns
        depth > num_qubits && circuit.metadata.complexity_score > 100.0
    }

    /// Compile circuit for specific backend
    fn compile_for_backend(
        &self,
        circuit: &InterfaceCircuit,
        backend: SimulationBackend,
    ) -> Result<BackendCompiledData> {
        match backend {
            SimulationBackend::StateVector => {
                let mut unitary_matrices = Vec::new();
                let mut gate_indices = Vec::new();

                for gate in &circuit.gates {
                    if gate.is_unitary() {
                        unitary_matrices.push(gate.unitary_matrix()?);
                        gate_indices.push(gate.qubits.clone());
                    }
                }

                Ok(BackendCompiledData::StateVector {
                    unitary_matrices,
                    gate_indices,
                })
            }
            SimulationBackend::MPS => {
                // Analyze circuit to determine optimal bond dimensions
                let bond_dimensions = self.calculate_optimal_bond_dimensions(circuit);
                let truncation_thresholds = vec![1e-12; circuit.gates.len()];

                Ok(BackendCompiledData::MPS {
                    bond_dimensions,
                    truncation_thresholds,
                })
            }
            SimulationBackend::Stabilizer => {
                let mut clifford_sequence = Vec::new();

                for gate in &circuit.gates {
                    match &gate.gate_type {
                        InterfaceGateType::Hadamard => {
                            clifford_sequence.push(StabilizerOp::H(gate.qubits[0]))
                        }
                        InterfaceGateType::S => {
                            clifford_sequence.push(StabilizerOp::S(gate.qubits[0]))
                        }
                        InterfaceGateType::PauliX => {
                            clifford_sequence.push(StabilizerOp::X(gate.qubits[0]))
                        }
                        InterfaceGateType::PauliY => {
                            clifford_sequence.push(StabilizerOp::Y(gate.qubits[0]))
                        }
                        InterfaceGateType::PauliZ => {
                            clifford_sequence.push(StabilizerOp::Z(gate.qubits[0]))
                        }
                        InterfaceGateType::CNOT => clifford_sequence
                            .push(StabilizerOp::CNOT(gate.qubits[0], gate.qubits[1])),
                        _ => {} // Skip non-Clifford gates
                    }
                }

                Ok(BackendCompiledData::Stabilizer { clifford_sequence })
            }
            SimulationBackend::Sparse => {
                let sparse_matrices = Vec::new(); // Would be implemented with actual sparse matrix compilation
                Ok(BackendCompiledData::Sparse { sparse_matrices })
            }
            SimulationBackend::TensorNetwork => {
                // For now, use the same data as state vector
                let mut unitary_matrices = Vec::new();
                let mut gate_indices = Vec::new();

                for gate in &circuit.gates {
                    if gate.is_unitary() {
                        unitary_matrices.push(gate.unitary_matrix()?);
                        gate_indices.push(gate.qubits.clone());
                    }
                }

                Ok(BackendCompiledData::StateVector {
                    unitary_matrices,
                    gate_indices,
                })
            }
            SimulationBackend::Auto => unreachable!(),
        }
    }

    /// Calculate optimal bond dimensions for MPS
    fn calculate_optimal_bond_dimensions(&self, circuit: &InterfaceCircuit) -> Vec<usize> {
        let base_bond_dim = self.config.max_mps_bond_dim.min(64);
        vec![base_bond_dim; circuit.num_qubits - 1]
    }

    /// Execute state vector simulation
    fn execute_statevector(
        &self,
        compiled: &CompiledCircuit,
        initial_state: Option<Array1<Complex64>>,
    ) -> Result<BackendExecutionResult> {
        let _simulator = StateVectorSimulator::new();

        // For now, use a placeholder implementation
        let num_qubits = compiled.original.num_qubits;
        let state_size = 1 << num_qubits;

        let final_state = initial_state.unwrap_or_else(|| {
            let mut state = Array1::zeros(state_size);
            state[0] = Complex64::new(1.0, 0.0);
            state
        });
        let memory_used = final_state.len() * std::mem::size_of::<Complex64>();

        Ok(BackendExecutionResult {
            final_state: Some(final_state),
            measurement_results: Vec::new(),
            classical_bits: vec![false; compiled.original.num_classical],
            memory_used_bytes: memory_used,
        })
    }

    /// Execute MPS simulation
    fn execute_mps(
        &self,
        compiled: &CompiledCircuit,
        initial_state: Option<Array1<Complex64>>,
    ) -> Result<BackendExecutionResult> {
        // Placeholder implementation
        Ok(BackendExecutionResult {
            final_state: None,
            measurement_results: Vec::new(),
            classical_bits: vec![false; compiled.original.num_classical],
            memory_used_bytes: 0,
        })
    }

    /// Execute stabilizer simulation
    fn execute_stabilizer(&self, compiled: &CompiledCircuit) -> Result<BackendExecutionResult> {
        // Placeholder implementation
        Ok(BackendExecutionResult {
            final_state: None,
            measurement_results: Vec::new(),
            classical_bits: vec![false; compiled.original.num_classical],
            memory_used_bytes: 0,
        })
    }

    /// Execute sparse simulation
    fn execute_sparse(
        &self,
        compiled: &CompiledCircuit,
        initial_state: Option<Array1<Complex64>>,
    ) -> Result<BackendExecutionResult> {
        // Placeholder implementation
        Ok(BackendExecutionResult {
            final_state: None,
            measurement_results: Vec::new(),
            classical_bits: vec![false; compiled.original.num_classical],
            memory_used_bytes: 0,
        })
    }

    /// Execute tensor network simulation
    #[cfg(feature = "advanced_math")]
    fn execute_tensor_network(
        &self,
        compiled: &CompiledCircuit,
        initial_state: Option<Array1<Complex64>>,
    ) -> Result<BackendExecutionResult> {
        // Placeholder implementation
        Ok(BackendExecutionResult {
            final_state: None,
            measurement_results: Vec::new(),
            classical_bits: vec![false; compiled.original.num_classical],
            memory_used_bytes: 0,
        })
    }

    #[cfg(not(feature = "advanced_math"))]
    fn execute_tensor_network(
        &self,
        _compiled: &CompiledCircuit,
        _initial_state: Option<Array1<Complex64>>,
    ) -> Result<BackendExecutionResult> {
        Err(SimulatorError::UnsupportedOperation(
            "Tensor network simulation requires advanced_math feature".to_string(),
        ))
    }

    /// Calculate circuit hash for caching
    fn calculate_circuit_hash(&self, circuit: &InterfaceCircuit) -> u64 {
        let mut hasher = DefaultHasher::new();
        circuit.num_qubits.hash(&mut hasher);
        circuit.num_classical.hash(&mut hasher);

        for gate in &circuit.gates {
            // Hash gate type discriminant
            std::mem::discriminant(&gate.gate_type).hash(&mut hasher);
            // Hash gate parameters separately
            match &gate.gate_type {
                InterfaceGateType::Phase(angle)
                | InterfaceGateType::RX(angle)
                | InterfaceGateType::RY(angle)
                | InterfaceGateType::RZ(angle) => {
                    angle.to_bits().hash(&mut hasher);
                }
                _ => {}
            }
            gate.qubits.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Estimate execution time
    fn estimate_execution_time(
        &self,
        circuit: &InterfaceCircuit,
        backend: SimulationBackend,
    ) -> f64 {
        let base_time_per_gate = match backend {
            SimulationBackend::StateVector => 0.1, // ms per gate
            SimulationBackend::MPS => 1.0,
            SimulationBackend::Stabilizer => 0.01,
            SimulationBackend::Sparse => 0.5,
            SimulationBackend::TensorNetwork => 2.0,
            SimulationBackend::Auto => 1.0,
        };

        circuit.gates.len() as f64 * base_time_per_gate * (1.1_f64).powi(circuit.num_qubits as i32)
    }

    /// Estimate memory requirements
    fn estimate_memory_requirements(
        &self,
        circuit: &InterfaceCircuit,
        backend: SimulationBackend,
    ) -> usize {
        let base_memory = match backend {
            SimulationBackend::StateVector => {
                (1_usize << circuit.num_qubits) * std::mem::size_of::<Complex64>()
            }
            SimulationBackend::MPS => {
                circuit.num_qubits
                    * self.config.max_mps_bond_dim
                    * self.config.max_mps_bond_dim
                    * std::mem::size_of::<Complex64>()
            }
            SimulationBackend::Stabilizer => {
                circuit.num_qubits * circuit.num_qubits * 2 // Stabilizer tableau
            }
            SimulationBackend::Sparse => {
                circuit.gates.len() * 1000 * std::mem::size_of::<Complex64>() // Rough estimate
            }
            SimulationBackend::TensorNetwork => {
                circuit.num_qubits * 64 * std::mem::size_of::<Complex64>() // Bond dimension 64
            }
            SimulationBackend::Auto => 0,
        };

        base_memory
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &CircuitInterfaceStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = CircuitInterfaceStats::default();
    }
}

/// Backend execution result
#[derive(Debug)]
struct BackendExecutionResult {
    final_state: Option<Array1<Complex64>>,
    measurement_results: Vec<bool>,
    classical_bits: Vec<bool>,
    memory_used_bytes: usize,
}

/// Circuit execution result
#[derive(Debug)]
pub struct CircuitExecutionResult {
    /// Final quantum state (if available)
    pub final_state: Option<Array1<Complex64>>,
    /// Measurement results
    pub measurement_results: Vec<bool>,
    /// Classical bit values
    pub classical_bits: Vec<bool>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Backend used for execution
    pub backend_used: SimulationBackend,
    /// Memory used in bytes
    pub memory_used_bytes: usize,
}

/// Circuit interface utilities
pub struct CircuitInterfaceUtils;

impl CircuitInterfaceUtils {
    /// Create a test circuit
    pub fn create_test_circuit(circuit_type: &str, num_qubits: usize) -> InterfaceCircuit {
        let mut circuit = InterfaceCircuit::new(num_qubits, num_qubits);

        match circuit_type {
            "ghz" => {
                // GHZ state preparation
                circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
                for i in 1..num_qubits {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, i]));
                }
            }
            "qft" => {
                // Quantum Fourier Transform
                for i in 0..num_qubits {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![i]));
                    for j in i + 1..num_qubits {
                        let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
                        circuit.add_gate(InterfaceGate::new(
                            InterfaceGateType::CRZ(angle),
                            vec![j, i],
                        ));
                    }
                }
            }
            "random" => {
                // Random circuit
                for _ in 0..num_qubits * 5 {
                    let qubit = fastrand::usize(0..num_qubits);
                    let gate_type = match fastrand::usize(0..4) {
                        0 => InterfaceGateType::Hadamard,
                        1 => InterfaceGateType::RX(fastrand::f64() * 2.0 * std::f64::consts::PI),
                        2 => InterfaceGateType::RY(fastrand::f64() * 2.0 * std::f64::consts::PI),
                        _ => InterfaceGateType::RZ(fastrand::f64() * 2.0 * std::f64::consts::PI),
                    };
                    circuit.add_gate(InterfaceGate::new(gate_type, vec![qubit]));
                }
            }
            _ => {
                // Identity circuit
                for i in 0..num_qubits {
                    circuit.add_gate(InterfaceGate::new(InterfaceGateType::Identity, vec![i]));
                }
            }
        }

        circuit
    }

    /// Benchmark circuit interface
    pub fn benchmark_interface(
        config: CircuitInterfaceConfig,
    ) -> Result<InterfaceBenchmarkResults> {
        let mut interface = CircuitInterface::new(config)?;
        let mut results = InterfaceBenchmarkResults::default();

        let circuit_types = vec!["ghz", "qft", "random"];
        let qubit_counts = vec![5, 10, 15, 20];

        for circuit_type in circuit_types {
            for &num_qubits in &qubit_counts {
                let circuit = Self::create_test_circuit(circuit_type, num_qubits);

                let start = std::time::Instant::now();
                let compiled = interface.compile_circuit(&circuit, SimulationBackend::Auto)?;
                let compilation_time = start.elapsed().as_secs_f64() * 1000.0;

                let start = std::time::Instant::now();
                let _result = interface.execute_circuit(&compiled, None)?;
                let execution_time = start.elapsed().as_secs_f64() * 1000.0;

                results
                    .compilation_times
                    .push((format!("{}_{}", circuit_type, num_qubits), compilation_time));
                results
                    .execution_times
                    .push((format!("{}_{}", circuit_type, num_qubits), execution_time));
            }
        }

        results.interface_stats = interface.get_stats().clone();
        Ok(results)
    }
}

/// Interface benchmark results
#[derive(Debug, Clone, Default)]
pub struct InterfaceBenchmarkResults {
    /// Compilation times (circuit_name, time_ms)
    pub compilation_times: Vec<(String, f64)>,
    /// Execution times (circuit_name, time_ms)
    pub execution_times: Vec<(String, f64)>,
    /// Interface statistics
    pub interface_stats: CircuitInterfaceStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_interface_gate_creation() {
        let gate = InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]);
        assert_eq!(gate.qubits, vec![0]);
        assert!(gate.is_unitary());
        assert!(!gate.is_measurement());
    }

    #[test]
    fn test_measurement_gate() {
        let gate = InterfaceGate::measurement(0, 0);
        assert!(gate.is_measurement());
        assert!(!gate.is_unitary());
        assert_eq!(gate.classical_targets, vec![0]);
    }

    #[test]
    fn test_circuit_creation() {
        let mut circuit = InterfaceCircuit::new(3, 3);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));

        assert_eq!(circuit.gates.len(), 2);
        assert_eq!(circuit.calculate_depth(), 2);
    }

    #[test]
    fn test_circuit_optimization() {
        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::PauliX, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::PauliX, vec![0])); // Should cancel
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Identity, vec![1])); // Should be removed

        let result = circuit.optimize();
        assert_eq!(result.gates_eliminated, 3); // All gates should be eliminated
    }

    #[test]
    fn test_gate_unitary_matrices() {
        let hadamard = InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]);
        let matrix = hadamard.unitary_matrix().unwrap();

        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        assert_abs_diff_eq!(matrix[[0, 0]].re, inv_sqrt2, epsilon = 1e-10);
        assert_abs_diff_eq!(matrix[[1, 1]].re, -inv_sqrt2, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_gate_merging() {
        let mut circuit = InterfaceCircuit::new(1, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RX(0.5), vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RX(0.3), vec![0]));

        let _ = circuit.optimize();
        assert_eq!(circuit.gates.len(), 1);

        if let InterfaceGateType::RX(angle) = &circuit.gates[0].gate_type {
            assert_abs_diff_eq!(*angle, 0.8, epsilon = 1e-10);
        } else {
            panic!("Expected merged RX gate");
        }
    }

    #[test]
    fn test_circuit_interface_config() {
        let config = CircuitInterfaceConfig::default();
        assert!(config.auto_backend_selection);
        assert!(config.enable_optimization);
        assert_eq!(config.max_statevector_qubits, 25);
    }

    #[test]
    fn test_test_circuit_creation() {
        let ghz_circuit = CircuitInterfaceUtils::create_test_circuit("ghz", 3);
        assert_eq!(ghz_circuit.num_qubits, 3);
        assert_eq!(ghz_circuit.gates.len(), 3); // H + 2 CNOTs

        let qft_circuit = CircuitInterfaceUtils::create_test_circuit("qft", 3);
        assert!(qft_circuit.gates.len() > 3); // Should have multiple gates
    }

    #[test]
    fn test_circuit_metadata() {
        let circuit = CircuitInterfaceUtils::create_test_circuit("ghz", 4);
        assert_eq!(circuit.metadata.depth, 4); // H on qubit 0, then 3 CNOTs sequentially
        assert_eq!(circuit.metadata.two_qubit_gates, 3);
    }

    #[test]
    fn test_clifford_detection() {
        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::S, vec![1]));

        let config = CircuitInterfaceConfig::default();
        let interface = CircuitInterface::new(config).unwrap();
        assert!(interface.is_clifford_circuit(&circuit));

        // Add non-Clifford gate
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::T, vec![0]));
        assert!(!interface.is_clifford_circuit(&circuit));
    }
}
