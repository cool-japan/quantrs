//! Quantum Error Correction Configuration Types

use crate::ising::IsingError;
use thiserror::Error;

/// Errors that can occur in quantum error correction
#[derive(Error, Debug)]
pub enum QuantumErrorCorrectionError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Error correction code error
    #[error("Error correction code error: {0}")]
    CodeError(String),

    /// Syndrome detection error
    #[error("Syndrome detection error: {0}")]
    SyndromeError(String),

    /// Logical operation error
    #[error("Logical operation error: {0}")]
    LogicalOperationError(String),

    /// Decoding error
    #[error("Decoding error: {0}")]
    DecodingError(String),

    /// Threshold error
    #[error("Threshold error: {0}")]
    ThresholdError(String),

    /// Resource estimation error
    #[error("Resource estimation error: {0}")]
    ResourceEstimationError(String),
}

/// Result type for quantum error correction operations
pub type QECResult<T> = Result<T, QuantumErrorCorrectionError>;

/// Configuration for quantum error correction
#[derive(Debug, Clone)]
pub struct QECConfig {
    /// Error correction code type
    pub code_type: ErrorCorrectionCode,
    /// Code parameters
    pub code_parameters: CodeParameters,
    /// Error threshold
    pub error_threshold: f64,
    /// Correction frequency
    pub correction_frequency: f64,
    /// Logical operations
    pub logical_operations: Vec<LogicalOperation>,
    /// Fault tolerance level
    pub fault_tolerance_level: usize,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Annealing integration
    pub annealing_integration: AnnealingIntegration,
}

// Forward declarations for types that will be defined in other modules
use super::{
    annealing_integration::AnnealingIntegration, codes::CodeParameters, codes::ErrorCorrectionCode,
    logical_operations::LogicalOperation, resource_constraints::ResourceConstraints,
};
