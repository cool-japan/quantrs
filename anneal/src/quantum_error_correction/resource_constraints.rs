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

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_physical_qubits: 1000,
            max_circuit_depth: 10_000,
            max_correction_time: Duration::from_secs(60),
            memory_constraints: MemoryConstraints::default(),
            connectivity_constraints: ConnectivityConstraints::default(),
        }
    }
}

impl Default for MemoryConstraints {
    fn default() -> Self {
        Self {
            syndrome_memory: 1024 * 1024, // 1 MB
            quantum_memory: 1024,
            lookup_table_memory: 1024 * 1024, // 1 MB
        }
    }
}

impl Default for ConnectivityConstraints {
    fn default() -> Self {
        Self {
            connectivity_graph: vec![],
            max_interaction_range: 10,
            routing_overhead: 1.2,
        }
    }
}
