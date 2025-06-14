//! Resource Constraints Configuration Types

use std::time::Duration;

/// Resource constraints for QEC
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum physical qubits
    pub max_physical_qubits: usize,
    /// Maximum circuit depth
    pub max_circuit_depth: usize,
    /// Maximum correction time
    pub max_correction_time: Duration,
    /// Memory requirements
    pub memory_constraints: MemoryConstraints,
    /// Connectivity constraints
    pub connectivity_constraints: ConnectivityConstraints,
}

/// Memory constraints
#[derive(Debug, Clone)]
pub struct MemoryConstraints {
    /// Classical memory for syndrome storage
    pub syndrome_memory: usize,
    /// Quantum memory for code states
    pub quantum_memory: usize,
    /// Lookup table memory for decoding
    pub lookup_table_memory: usize,
}

/// Connectivity constraints
#[derive(Debug, Clone)]
pub struct ConnectivityConstraints {
    /// Qubit connectivity graph
    pub connectivity_graph: Vec<Vec<bool>>,
    /// Maximum interaction range
    pub max_interaction_range: usize,
    /// Routing overhead
    pub routing_overhead: f64,
}
