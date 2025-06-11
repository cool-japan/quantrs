//! Error types for RL embedding optimization

use crate::ising::IsingError;
use thiserror::Error;

/// Errors that can occur in RL embedding optimization
#[derive(Error, Debug)]
pub enum RLEmbeddingError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Neural network error
    #[error("Neural network error: {0}")]
    NeuralNetworkError(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Embedding error
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Hardware error
    #[error("Hardware error: {0}")]
    HardwareError(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    OptimizationError(String),
}

/// Result type for RL embedding operations
pub type RLEmbeddingResult<T> = Result<T, RLEmbeddingError>;
