//! Error types for advanced quantum algorithms

use crate::ising::IsingError;
use thiserror::Error;

/// Errors that can occur in advanced quantum algorithms
#[derive(Error, Debug)]
pub enum AdvancedQuantumError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Quantum circuit error
    #[error("Quantum circuit error: {0}")]
    QuantumCircuitError(String),

    /// Algorithm convergence error
    #[error("Algorithm convergence error: {0}")]
    ConvergenceError(String),

    /// Parameter optimization error
    #[error("Parameter optimization error: {0}")]
    ParameterError(String),

    /// Zeno effect error
    #[error("Zeno effect error: {0}")]
    ZenoError(String),

    /// Adiabatic error
    #[error("Adiabatic error: {0}")]
    AdiabaticError(String),

    /// Counterdiabatic error
    #[error("Counterdiabatic error: {0}")]
    CounterdiabaticError(String),

    /// Noise model error
    #[error("Noise model error: {0}")]
    NoiseModelError(String),

    /// No algorithm available
    #[error("No algorithm available")]
    NoAlgorithmAvailable,

    /// Ensemble failed
    #[error("Ensemble optimization failed")]
    EnsembleFailed,

    /// Algorithm not found
    #[error("Algorithm not found: {0}")]
    AlgorithmNotFound(String),
}

/// Result type for advanced quantum algorithms
pub type AdvancedQuantumResult<T> = Result<T, AdvancedQuantumError>;
