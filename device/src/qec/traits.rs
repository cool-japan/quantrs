//! Quantum Error Correction Trait Definitions
//!
//! This module defines the core traits for quantum error correction:
//! - `SyndromeDetector`: For detecting error syndromes from measurements
//! - `ErrorCorrector`: For applying error corrections based on syndromes
//! - `QuantumErrorCode`: For defining quantum error correction codes

use std::collections::HashMap;

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;

use crate::DeviceError;

use super::{CorrectionOperation, LogicalOperator, StabilizerGroup, SyndromePattern};

/// Result type for QEC operations
pub type QECResult<T> = Result<T, DeviceError>;

/// Trait for syndrome detection in quantum error correction
pub trait SyndromeDetector {
    /// Detect error syndromes from measurement results
    fn detect_syndromes(
        &self,
        measurements: &HashMap<String, Vec<i32>>,
        stabilizers: &[StabilizerGroup],
    ) -> QECResult<Vec<SyndromePattern>>;

    /// Validate a detected syndrome against historical patterns
    fn validate_syndrome(
        &self,
        syndrome: &SyndromePattern,
        history: &[SyndromePattern],
    ) -> QECResult<bool>;
}

/// Trait for error correction in quantum systems
pub trait ErrorCorrector {
    /// Apply error corrections based on detected syndromes
    fn correct_errors(
        &self,
        syndromes: &[SyndromePattern],
        code: &dyn QuantumErrorCode,
    ) -> QECResult<Vec<CorrectionOperation>>;

    /// Estimate the fidelity of a proposed correction operation
    fn estimate_correction_fidelity(
        &self,
        correction: &CorrectionOperation,
        current_state: Option<&Array1<Complex64>>,
    ) -> QECResult<f64>;
}

/// Trait defining a quantum error correction code
pub trait QuantumErrorCode {
    /// Get the stabilizer generators for this code
    fn get_stabilizers(&self) -> Vec<StabilizerGroup>;

    /// Get the logical operators for this code
    fn get_logical_operators(&self) -> Vec<LogicalOperator>;

    /// Get the code distance (minimum weight of logical operators)
    fn distance(&self) -> usize;

    /// Get the number of physical data qubits
    fn num_data_qubits(&self) -> usize;

    /// Get the number of ancilla qubits for syndrome measurement
    fn num_ancilla_qubits(&self) -> usize;

    /// Get the number of logical qubits encoded
    fn logical_qubit_count(&self) -> usize;

    /// Encode a logical state into the code space
    fn encode_logical_state(
        &self,
        logical_state: &Array1<Complex64>,
    ) -> QECResult<Array1<Complex64>>;
}
