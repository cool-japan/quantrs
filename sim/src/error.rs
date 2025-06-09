//! Error types for the quantum simulator module.

use thiserror::Error;
use quantrs2_core::error::QuantRS2Error;

/// Error types for quantum simulation
#[derive(Debug, Clone, Error)]
pub enum SimulatorError {
    /// Invalid number of qubits
    #[error("Invalid number of qubits: {0}")]
    InvalidQubits(usize),

    /// Invalid gate specification
    #[error("Invalid gate: {0}")]
    InvalidGate(String),

    /// Computation error during simulation
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Gate target out of bounds
    #[error("Gate target out of bounds: {0}")]
    IndexOutOfBounds(usize),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// GPU not available
    #[error("GPU not available")]
    GPUNotAvailable,

    /// Shader compilation failed
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationError(String),

    /// GPU execution error
    #[error("GPU execution error: {0}")]
    GPUExecutionError(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Numerical error
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Core error
    #[error("Core error: {0}")]
    CoreError(#[from] QuantRS2Error),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinalgError(String),
}

/// Result type for simulator operations
pub type Result<T> = std::result::Result<T, SimulatorError>;