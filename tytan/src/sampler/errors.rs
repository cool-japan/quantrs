//! Error types for samplers

use quantrs2_anneal::{AnnealingError, IsingError};
use thiserror::Error;

/// Errors that can occur during sampling
#[derive(Error, Debug)]
pub enum SamplerError {
    /// Error when the input parameters are invalid
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Error in the underlying annealing simulator
    #[error("Annealing error: {0}")]
    AnnealingError(#[from] AnnealingError),

    /// Error in the Ising model
    #[error("Ising model error: {0}")]
    IsingError(#[from] IsingError),

    /// Error in GPU operations
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Error when D-Wave API is unavailable
    #[error("D-Wave API unavailable: {0}")]
    DWaveUnavailable(String),

    /// Error during API communication
    #[error("API communication error: {0}")]
    ApiError(String),

    /// Error in D-Wave operations
    #[cfg(feature = "dwave")]
    #[error("D-Wave error: {0}")]
    DWaveError(#[from] quantrs2_anneal::dwave::DWaveError),

    /// Feature not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

impl From<String> for SamplerError {
    fn from(s: String) -> Self {
        SamplerError::InvalidParameter(s)
    }
}

/// Result type for sampling operations
pub type SamplerResult<T> = Result<T, SamplerError>;
