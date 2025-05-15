use thiserror::Error;

/// Common error types for quantum operations
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum QuantRS2Error {
    /// Error when a qubit is not in a valid range
    #[error("Invalid qubit ID {0}, must be within the valid range for this operation")]
    InvalidQubitId(u32),

    /// Error when an operation is not supported
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Error when a gate application fails
    #[error("Failed to apply gate: {0}")]
    GateApplicationFailed(String),

    /// Error when circuit validation fails
    #[error("Circuit validation failed: {0}")]
    CircuitValidationFailed(String),

    /// Error when backend execution fails
    #[error("Backend execution failed: {0}")]
    BackendExecutionFailed(String),

    /// Error when unsupported qubit count is requested
    #[error("Unsupported qubit count {0}: {1}")]
    UnsupportedQubits(usize, String),

    /// Error when invalid input is provided
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Result type for quantum operations
pub type QuantRS2Result<T> = Result<T, QuantRS2Error>;
