use thiserror::Error;

/// Common error types for quantum operations
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum QuantrsError {
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
}

/// Result type for quantum operations
pub type QuantrsResult<T> = Result<T, QuantrsError>;
