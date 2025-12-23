//! Error Correction Code Types

/// Error correction code types
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCorrectionCode {
    /// Surface code
    SurfaceCode,
    /// Color code
    ColorCode,
    /// Repetition code
    RepetitionCode,
    /// Steane code
    SteaneCode,
    /// Shor code
    ShorCode,
    /// CSS codes
    CSSCode,
    /// Stabilizer codes
    StabilizerCode,
    /// Topological codes
    TopologicalCode,
    /// LDPC codes
    LDPCCode,
}

/// Code parameters
#[derive(Debug, Clone)]
pub struct CodeParameters {
    /// Distance of the code
    pub distance: usize,
    /// Number of logical qubits
    pub num_logical_qubits: usize,
    /// Number of physical qubits
    pub num_physical_qubits: usize,
    /// Number of ancilla qubits
    pub num_ancilla_qubits: usize,
    /// Code rate
    pub code_rate: f64,
    /// Threshold probability
    pub threshold_probability: f64,
}

impl Default for CodeParameters {
    fn default() -> Self {
        Self {
            distance: 3,
            num_logical_qubits: 1,
            num_physical_qubits: 9,
            num_ancilla_qubits: 4,
            code_rate: 0.11,
            threshold_probability: 0.01,
        }
    }
}
