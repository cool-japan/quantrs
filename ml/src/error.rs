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

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IOError(#[from] io::Error),

    /// Quantum error
    #[error("Quantum error: {0}")]
    QuantumError(#[from] QuantRS2Error),
}
