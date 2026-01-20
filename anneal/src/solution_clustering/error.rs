//! Error handling for solution clustering operations

use thiserror::Error;

/// Errors that can occur in solution clustering and analysis
#[derive(Error, Debug)]
pub enum ClusteringError {
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Clustering algorithm error
    #[error("Clustering algorithm error: {0}")]
    AlgorithmError(String),

    /// Data processing error
    #[error("Data processing error: {0}")]
    DataError(String),

    /// Statistical analysis error
    #[error("Statistical analysis error: {0}")]
    StatisticalError(String),

    /// Visualization error
    #[error("Visualization error: {0}")]
    VisualizationError(String),

    /// Convergence error
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Insufficient data
    #[error("Insufficient data: need at least {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },
}

/// Result type for clustering operations
pub type ClusteringResult<T> = Result<T, ClusteringError>;
