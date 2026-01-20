//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Simplified variational circuit for internal use
/// This is used by lstm, attention, and gnn modules for building quantum circuits
#[derive(Debug, Clone)]
pub struct VariationalCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Gates and parameters
    pub gates: Vec<(String, Vec<usize>, Vec<String>)>,
}
impl VariationalCircuit {
    /// Create a new variational circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
        }
    }
    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate_name: &str, qubits: Vec<usize>, params: Vec<String>) {
        self.gates.push((gate_name.to_string(), qubits, params));
    }
    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.gates.iter().map(|(_, _, params)| params.len()).sum()
    }
}
