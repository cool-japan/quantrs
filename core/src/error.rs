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

    /// Error during computation
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    /// Routing error
    #[error("Routing error: {0}")]
    RoutingError(String),

    /// Matrix construction error
    #[error("Matrix construction error: {0}")]
    MatrixConstruction(String),

    /// Matrix inversion error
    #[error("Matrix inversion error: {0}")]
    MatrixInversion(String),

    /// Optimization failed error
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    /// Tensor network error
    #[error("Tensor network error: {0}")]
    TensorNetwork(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Invalid gate operation
    #[error("Invalid gate operation: {0}")]
    InvalidGateOp(String),

    /// UltraThink mode error variants
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Quantum decoherence: {0}")]
    QuantumDecoherence(String),

    #[error("No storage available: {0}")]
    NoStorageAvailable(String),

    #[error("Calibration not found: {0}")]
    CalibrationNotFound(String),

    #[error("Access denied: {0}")]
    AccessDenied(String),

    #[error("Storage capacity exceeded: {0}")]
    StorageCapacityExceeded(String),

    #[error("Hardware target not found: {0}")]
    HardwareTargetNotFound(String),

    #[error("Gate fusion error: {0}")]
    GateFusionError(String),

    #[error("Unsupported gate: {0}")]
    UnsupportedGate(String),

    #[error("Compilation timeout: {0}")]
    CompilationTimeout(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Node unavailable: {0}")]
    NodeUnavailable(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("No hardware available: {0}")]
    NoHardwareAvailable(String),

    #[error("State not found: {0}")]
    StateNotFound(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("QKD failure: {0}")]
    QKDFailure(String),
}

/// Result type for quantum operations
pub type QuantRS2Result<T> = Result<T, QuantRS2Error>;

impl From<ndarray::ShapeError> for QuantRS2Error {
    fn from(err: ndarray::ShapeError) -> Self {
        QuantRS2Error::InvalidInput(format!("Shape error: {}", err))
    }
}

#[cfg(feature = "mps")]
impl From<ndarray_linalg::error::LinalgError> for QuantRS2Error {
    fn from(err: ndarray_linalg::error::LinalgError) -> Self {
        QuantRS2Error::LinalgError(format!("Linear algebra error: {}", err))
    }
}

impl From<std::io::Error> for QuantRS2Error {
    fn from(err: std::io::Error) -> Self {
        QuantRS2Error::RuntimeError(format!("I/O error: {}", err))
    }
}

impl From<Box<bincode::ErrorKind>> for QuantRS2Error {
    fn from(err: Box<bincode::ErrorKind>) -> Self {
        QuantRS2Error::RuntimeError(format!("Serialization error: {:?}", err))
    }
}

impl From<serde_json::Error> for QuantRS2Error {
    fn from(err: serde_json::Error) -> Self {
        QuantRS2Error::RuntimeError(format!("JSON error: {}", err))
    }
}
