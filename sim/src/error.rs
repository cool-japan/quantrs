//! Error types for the quantum simulator module.

use quantrs2_core::error::QuantRS2Error;
use thiserror::Error;

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

    /// General GPU error
    #[error("GPU error: {0}")]
    GpuError(String),

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

    /// Initialization failed
    #[error("Initialization failed: {0}")]
    InitializationFailed(String),

    /// Memory allocation error
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Initialization error
    #[error("Initialization error: {0}")]
    InitializationError(String),

    /// Operation not supported
    #[error("Operation not supported: {0}")]
    OperationNotSupported(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Memory allocation failed
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),

    /// Resource exhausted
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Result type for simulator operations
pub type Result<T> = std::result::Result<T, SimulatorError>;

impl From<ndarray::ShapeError> for SimulatorError {
    fn from(err: ndarray::ShapeError) -> Self {
        SimulatorError::DimensionMismatch(err.to_string())
    }
}
