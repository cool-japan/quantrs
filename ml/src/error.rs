use quantrs2_core::error::QuantRS2Error;
use std::io;
use thiserror::Error;

/// Type alias for Result with MLError as error type
pub type Result<T> = std::result::Result<T, MLError>;

/// Error type for Machine Learning operations
#[derive(Error, Debug)]
pub enum MLError {
    /// Error during training or inference
    #[error("Machine learning error: {0}")]
    MLOperationError(String),

    /// Error during model creation
    #[error("Model creation error: {0}")]
    ModelCreationError(String),

    /// Error in optimization process
    #[error("Optimization error: {0}")]
    OptimizationError(String),

    /// Error in data handling
    #[error("Data error: {0}")]
    DataError(String),

    /// Error in quantum circuit execution
    #[error("Circuit execution error: {0}")]
    CircuitExecutionError(String),

    /// Error during feature extraction
    #[error("Feature extraction error: {0}")]
    FeatureExtractionError(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Not supported
    #[error("Not supported: {0}")]
    NotSupported(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Model not trained
    #[error("Model not trained: {0}")]
    ModelNotTrained(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IOError(#[from] io::Error),

    /// Quantum error
    #[error("Quantum error: {0}")]
    QuantumError(#[from] QuantRS2Error),

    /// Shape error from ndarray
    #[error("Shape error: {0}")]
    ShapeError(#[from] scirs2_core::ndarray::ShapeError),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Numerical error during computation
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Backend error
    #[error("Backend error: {0}")]
    BackendError(String),
}

impl From<String> for MLError {
    fn from(s: String) -> Self {
        MLError::ComputationError(s)
    }
}
