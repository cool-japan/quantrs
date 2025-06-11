//! Logical Encoding Configuration Types

use std::collections::HashMap;

/// Logical encoding system
#[derive(Debug, Clone)]
pub struct LogicalEncoding {
    /// Stabilizer generators
    pub stabilizers: Vec<PauliOperator>,
    /// Logical operators
    pub logical_operators: Vec<LogicalOperatorSet>,
    /// Code space
    pub code_space: CodeSpace,
    /// Encoding circuits
    pub encoding_circuits: Vec<QuantumCircuit>,
    /// Decoding data
    pub decoding_data: DecodingData,
}

/// Pauli operator representation
#[derive(Debug, Clone)]
pub struct PauliOperator {
    /// Pauli string (I, X, Y, Z for each qubit)
    pub pauli_string: Vec<PauliType>,
    /// Phase factor
    pub phase: f64,
    /// Coefficient
    pub coefficient: f64,
    /// Support (qubits on which operator acts non-trivially)
    pub support: Vec<usize>,
}

/// Pauli operator types
#[derive(Debug, Clone, PartialEq)]
pub enum PauliType {
    /// Identity
    I,
    /// Pauli X
    X,
    /// Pauli Y
    Y,
    /// Pauli Z
    Z,
}

/// Logical operator set
#[derive(Debug, Clone)]
pub struct LogicalOperatorSet {
    /// Logical qubit index
    pub logical_qubit: usize,
    /// Logical X operator
    pub logical_x: PauliOperator,
    /// Logical Z operator
    pub logical_z: PauliOperator,
    /// Logical Y operator (derived)
    pub logical_y: PauliOperator,
}

/// Code space definition
#[derive(Debug, Clone)]
pub struct CodeSpace {
    /// Basis states of the code space
    pub basis_states: Vec<LogicalBasisState>,
    /// Projector onto code space
    pub code_projector: Vec<Vec<f64>>,
    /// Dimension of code space
    pub dimension: usize,
    /// Distance of the code
    pub distance: usize,
}

/// Logical basis state
#[derive(Debug, Clone)]
pub struct LogicalBasisState {
    /// Logical state label
    pub label: String,
    /// Physical state representation
    pub physical_state: Vec<f64>,
    /// Stabilizer eigenvalues
    pub stabilizer_eigenvalues: Vec<i8>,
}

/// Quantum circuit representation
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    /// Circuit gates
    pub gates: Vec<QuantumGate>,
    /// Circuit depth
    pub depth: usize,
    /// Qubit count
    pub num_qubits: usize,
    /// Classical registers for measurements
    pub classical_registers: Vec<ClassicalRegister>,
}

/// Quantum gate
#[derive(Debug, Clone)]
pub struct QuantumGate {
    /// Gate type
    pub gate_type: GateType,
    /// Target qubits
    pub target_qubits: Vec<usize>,
    /// Control qubits
    pub control_qubits: Vec<usize>,
    /// Gate parameters
    pub parameters: Vec<f64>,
    /// Gate time
    pub gate_time: f64,
}

/// Quantum gate types
#[derive(Debug, Clone, PartialEq)]
pub enum GateType {
    /// Pauli X
    X,
    /// Pauli Y
    Y,
    /// Pauli Z
    Z,
    /// Hadamard
    H,
    /// Phase gate
    S,
    /// T gate
    T,
    /// CNOT
    CNOT,
    /// Controlled-Z
    CZ,
    /// Rotation gates
    RX(f64),
    RY(f64),
    RZ(f64),
    /// Measurement
    Measurement,
}

/// Classical register
#[derive(Debug, Clone)]
pub struct ClassicalRegister {
    /// Register name
    pub name: String,
    /// Number of bits
    pub num_bits: usize,
}

/// Decoding data
#[derive(Debug, Clone)]
pub struct DecodingData {
    /// Syndrome lookup table
    pub syndrome_table: HashMap<Vec<i8>, ErrorPattern>,
    /// Decoding algorithm
    pub decoding_algorithm: DecodingAlgorithm,
    /// Decoding performance
    pub decoding_performance: DecodingPerformance,
}

/// Error pattern
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    /// Error locations
    pub error_locations: Vec<usize>,
    /// Error types
    pub error_types: Vec<PauliType>,
    /// Correction operations
    pub correction_operations: Vec<QuantumGate>,
}

/// Decoding algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum DecodingAlgorithm {
    LookupTable,
    MinimumWeight,
    BeliefPropagation,
    NeuralNetwork,
    MaximumLikelihood,
}

/// Decoding performance metrics
#[derive(Debug, Clone)]
pub struct DecodingPerformance {
    /// Logical error rate
    pub logical_error_rate: f64,
    /// Decoding time
    pub decoding_time: std::time::Duration,
    /// Success probability
    pub success_probability: f64,
    /// Threshold estimate
    pub threshold_estimate: f64,
}