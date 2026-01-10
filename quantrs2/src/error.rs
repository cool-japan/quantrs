//! # Unified Error Handling for QuantRS2
//!
//! This module provides a unified error handling interface that aggregates error types
//! from all QuantRS2 subcrates and provides convenient conversion traits.
//!
//! ## Basic Usage
//!
//! ```rust,ignore
//! use quantrs2::error::{QuantRS2Error, QuantRS2Result};
//!
//! fn quantum_operation() -> QuantRS2Result<()> {
//!     // Your quantum code here
//!     Ok(())
//! }
//! ```
//!
//! ## Error Categories
//!
//! Errors are categorized by their source and severity:
//! - **Core Errors**: From quantrs2-core (qubit errors, gate errors, etc.)
//! - **Circuit Errors**: From quantrs2-circuit (validation, compilation errors)
//! - **Simulation Errors**: From quantrs2-sim (backend, execution errors)
//! - **Hardware Errors**: From quantrs2-device (connectivity, API errors)
//! - **Algorithm Errors**: From quantrs2-ml (optimization, convergence errors)
//! - **Annealing Errors**: From quantrs2-anneal (QUBO, solver errors)
//!
//! ## Error Conversion
//!
//! All subcrate errors can be automatically converted to the unified `QuantRS2Error` type:
//!
//! ```rust,ignore
//! use quantrs2::prelude::circuits::*;
//! use quantrs2::error::QuantRS2Result;
//!
//! fn build_circuit() -> QuantRS2Result<Circuit<4>> {
//!     let mut circuit = Circuit::<4>::new();
//!     circuit.h(QubitId::new(0))?;  // Errors convert automatically
//!     Ok(circuit)
//! }
//! ```

#![allow(clippy::doc_markdown)] // QuantRS2 is a proper name
#![allow(clippy::must_use_candidate)] // Error handling functions naturally return values
#![allow(clippy::too_many_lines)] // with_context needs comprehensive match arms

// Re-export core error types - always available
pub use crate::core::error::{QuantRS2Error, QuantRS2Result};

/// Error category enumeration for better error handling and diagnostics.
///
/// This allows categorizing errors by their origin, which is useful for:
/// - Error logging and monitoring
/// - Error recovery strategies
/// - User-facing error messages
/// - Debugging and troubleshooting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Errors from core quantum operations (gates, qubits, registers)
    Core,
    /// Errors from circuit construction and manipulation
    Circuit,
    /// Errors from quantum simulation
    Simulation,
    /// Errors from quantum hardware integration
    Hardware,
    /// Errors from quantum algorithms and ML
    Algorithm,
    /// Errors from quantum annealing
    Annealing,
    /// Errors from symbolic computation
    Symbolic,
    /// Runtime errors (I/O, serialization, etc.)
    Runtime,
    /// Unknown or uncategorized errors
    Unknown,
}

impl ErrorCategory {
    /// Returns a human-readable name for the error category.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Core => "Core",
            Self::Circuit => "Circuit",
            Self::Simulation => "Simulation",
            Self::Hardware => "Hardware",
            Self::Algorithm => "Algorithm",
            Self::Annealing => "Annealing",
            Self::Symbolic => "Symbolic",
            Self::Runtime => "Runtime",
            Self::Unknown => "Unknown",
        }
    }

    /// Returns a detailed description of what this category represents.
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Core => "Errors from core quantum operations (gates, qubits, registers)",
            Self::Circuit => "Errors from circuit construction and manipulation",
            Self::Simulation => "Errors from quantum simulation backends",
            Self::Hardware => "Errors from quantum hardware integration",
            Self::Algorithm => "Errors from quantum algorithms and machine learning",
            Self::Annealing => "Errors from quantum annealing operations",
            Self::Symbolic => "Errors from symbolic computation",
            Self::Runtime => "Runtime errors (I/O, serialization, network, etc.)",
            Self::Unknown => "Uncategorized or unknown errors",
        }
    }
}

/// Extension trait for `QuantRS2Error` providing additional error handling utilities.
pub trait QuantRS2ErrorExt {
    /// Categorize the error based on its variant.
    fn category(&self) -> ErrorCategory;

    /// Check if the error is recoverable (e.g., can be retried).
    fn is_recoverable(&self) -> bool;

    /// Get a user-friendly error message.
    fn user_message(&self) -> String;

    /// Check if the error is related to invalid input.
    fn is_invalid_input(&self) -> bool;

    /// Check if the error is related to resource limitations.
    fn is_resource_error(&self) -> bool;
}

impl QuantRS2ErrorExt for QuantRS2Error {
    fn category(&self) -> ErrorCategory {
        match self {
            // Core errors - combined for simplicity
            Self::InvalidQubitId(_)
            | Self::InvalidGateOp(_)
            | Self::LinalgError(_)
            | Self::MatrixConstruction(_)
            | Self::MatrixInversion(_)
            | Self::InvalidInput(_)
            | Self::InvalidParameter(_)
            | Self::InvalidOperation(_)
            | Self::UnsupportedOperation(_)
            | Self::UnsupportedGate(_)
            | Self::DivisionByZero => ErrorCategory::Core,

            // Circuit errors
            Self::CircuitValidationFailed(_)
            | Self::RoutingError(_)
            | Self::GateApplicationFailed(_)
            | Self::GateFusionError(_)
            | Self::CompilationTimeout(_) => ErrorCategory::Circuit,

            // Simulation errors
            Self::BackendExecutionFailed(_)
            | Self::UnsupportedQubits(_, _)
            | Self::TensorNetwork(_)
            | Self::ComputationError(_)
            | Self::QuantumDecoherence(_)
            | Self::StateNotFound(_) => ErrorCategory::Simulation,

            // Hardware errors
            Self::NetworkError(_)
            | Self::NoHardwareAvailable(_)
            | Self::CalibrationNotFound(_)
            | Self::HardwareTargetNotFound(_)
            | Self::NodeNotFound(_)
            | Self::NodeUnavailable(_)
            | Self::QKDFailure(_) => ErrorCategory::Hardware,

            // Algorithm errors
            Self::OptimizationFailed(_) => ErrorCategory::Algorithm,

            // Runtime errors
            Self::RuntimeError(_)
            | Self::ExecutionError(_)
            | Self::NoStorageAvailable(_)
            | Self::StorageCapacityExceeded(_)
            | Self::AccessDenied(_)
            | Self::LockPoisoned(_)
            | Self::IndexOutOfBounds { .. } => ErrorCategory::Runtime,
        }
    }

    fn is_recoverable(&self) -> bool {
        match self {
            // Transient errors that might succeed on retry
            Self::NetworkError(_)
            | Self::NodeUnavailable(_)
            | Self::NoHardwareAvailable(_)
            | Self::BackendExecutionFailed(_) => true,

            // All other errors are not recoverable
            _ => false,
        }
    }

    fn user_message(&self) -> String {
        match self {
            Self::InvalidQubitId(id) => format!("Qubit {id} is not valid for this operation"),
            Self::UnsupportedOperation(op) => {
                format!("Operation '{op}' is not supported")
            }
            Self::UnsupportedQubits(count, msg) => {
                format!("Cannot handle {count} qubits: {msg}")
            }
            Self::NetworkError(msg) => {
                format!("Network connection failed: {msg}. Please check your internet connection and try again.")
            }
            Self::NoHardwareAvailable(msg) => {
                format!(
                    "No quantum hardware is currently available: {msg}. Please try again later."
                )
            }
            Self::OptimizationFailed(msg) => {
                format!("Optimization did not converge: {msg}. Try adjusting parameters.")
            }
            Self::BackendExecutionFailed(msg) => {
                format!("Simulation failed: {msg}. This might be a temporary issue.")
            }
            // For other errors, use the Display implementation
            _ => format!("{self}"),
        }
    }

    fn is_invalid_input(&self) -> bool {
        matches!(
            self,
            Self::InvalidInput(_)
                | Self::InvalidParameter(_)
                | Self::InvalidQubitId(_)
                | Self::InvalidOperation(_)
                | Self::InvalidGateOp(_)
        )
    }

    fn is_resource_error(&self) -> bool {
        matches!(
            self,
            Self::NoStorageAvailable(_)
                | Self::StorageCapacityExceeded(_)
                | Self::NoHardwareAvailable(_)
                | Self::UnsupportedQubits(_, _)
        )
    }
}

/// Helper function to create a more informative error message with context.
///
/// # Example
///
/// ```rust,ignore
/// use quantrs2::error::{with_context, QuantRS2Error};
///
/// fn operation() -> Result<(), QuantRS2Error> {
///     Err(with_context(
///         QuantRS2Error::InvalidQubitId(5),
///         "while building Bell state circuit"
///     ))
/// }
/// ```
#[must_use]
pub fn with_context(error: QuantRS2Error, context: &str) -> QuantRS2Error {
    match error {
        QuantRS2Error::InvalidQubitId(id) => {
            QuantRS2Error::InvalidQubitId(id) // Can't add context to this variant
        }
        QuantRS2Error::DivisionByZero => QuantRS2Error::DivisionByZero,
        // For all string-based variants, prepend the context
        QuantRS2Error::UnsupportedOperation(msg) => {
            QuantRS2Error::UnsupportedOperation(format!("{context}: {msg}"))
        }
        QuantRS2Error::GateApplicationFailed(msg) => {
            QuantRS2Error::GateApplicationFailed(format!("{context}: {msg}"))
        }
        QuantRS2Error::CircuitValidationFailed(msg) => {
            QuantRS2Error::CircuitValidationFailed(format!("{context}: {msg}"))
        }
        QuantRS2Error::BackendExecutionFailed(msg) => {
            QuantRS2Error::BackendExecutionFailed(format!("{context}: {msg}"))
        }
        QuantRS2Error::UnsupportedQubits(count, msg) => {
            QuantRS2Error::UnsupportedQubits(count, format!("{context}: {msg}"))
        }
        QuantRS2Error::InvalidInput(msg) => {
            QuantRS2Error::InvalidInput(format!("{context}: {msg}"))
        }
        QuantRS2Error::ComputationError(msg) => {
            QuantRS2Error::ComputationError(format!("{context}: {msg}"))
        }
        QuantRS2Error::LinalgError(msg) => QuantRS2Error::LinalgError(format!("{context}: {msg}")),
        QuantRS2Error::RoutingError(msg) => {
            QuantRS2Error::RoutingError(format!("{context}: {msg}"))
        }
        QuantRS2Error::MatrixConstruction(msg) => {
            QuantRS2Error::MatrixConstruction(format!("{context}: {msg}"))
        }
        QuantRS2Error::MatrixInversion(msg) => {
            QuantRS2Error::MatrixInversion(format!("{context}: {msg}"))
        }
        QuantRS2Error::OptimizationFailed(msg) => {
            QuantRS2Error::OptimizationFailed(format!("{context}: {msg}"))
        }
        QuantRS2Error::TensorNetwork(msg) => {
            QuantRS2Error::TensorNetwork(format!("{context}: {msg}"))
        }
        QuantRS2Error::RuntimeError(msg) => {
            QuantRS2Error::RuntimeError(format!("{context}: {msg}"))
        }
        QuantRS2Error::ExecutionError(msg) => {
            QuantRS2Error::ExecutionError(format!("{context}: {msg}"))
        }
        QuantRS2Error::InvalidGateOp(msg) => {
            QuantRS2Error::InvalidGateOp(format!("{context}: {msg}"))
        }
        QuantRS2Error::InvalidParameter(msg) => {
            QuantRS2Error::InvalidParameter(format!("{context}: {msg}"))
        }
        QuantRS2Error::QuantumDecoherence(msg) => {
            QuantRS2Error::QuantumDecoherence(format!("{context}: {msg}"))
        }
        QuantRS2Error::NoStorageAvailable(msg) => {
            QuantRS2Error::NoStorageAvailable(format!("{context}: {msg}"))
        }
        QuantRS2Error::CalibrationNotFound(msg) => {
            QuantRS2Error::CalibrationNotFound(format!("{context}: {msg}"))
        }
        QuantRS2Error::AccessDenied(msg) => {
            QuantRS2Error::AccessDenied(format!("{context}: {msg}"))
        }
        QuantRS2Error::StorageCapacityExceeded(msg) => {
            QuantRS2Error::StorageCapacityExceeded(format!("{context}: {msg}"))
        }
        QuantRS2Error::HardwareTargetNotFound(msg) => {
            QuantRS2Error::HardwareTargetNotFound(format!("{context}: {msg}"))
        }
        QuantRS2Error::GateFusionError(msg) => {
            QuantRS2Error::GateFusionError(format!("{context}: {msg}"))
        }
        QuantRS2Error::UnsupportedGate(msg) => {
            QuantRS2Error::UnsupportedGate(format!("{context}: {msg}"))
        }
        QuantRS2Error::CompilationTimeout(msg) => {
            QuantRS2Error::CompilationTimeout(format!("{context}: {msg}"))
        }
        QuantRS2Error::NodeNotFound(msg) => {
            QuantRS2Error::NodeNotFound(format!("{context}: {msg}"))
        }
        QuantRS2Error::NodeUnavailable(msg) => {
            QuantRS2Error::NodeUnavailable(format!("{context}: {msg}"))
        }
        QuantRS2Error::NetworkError(msg) => {
            QuantRS2Error::NetworkError(format!("{context}: {msg}"))
        }
        QuantRS2Error::NoHardwareAvailable(msg) => {
            QuantRS2Error::NoHardwareAvailable(format!("{context}: {msg}"))
        }
        QuantRS2Error::StateNotFound(msg) => {
            QuantRS2Error::StateNotFound(format!("{context}: {msg}"))
        }
        QuantRS2Error::InvalidOperation(msg) => {
            QuantRS2Error::InvalidOperation(format!("{context}: {msg}"))
        }
        QuantRS2Error::QKDFailure(msg) => QuantRS2Error::QKDFailure(format!("{context}: {msg}")),
        QuantRS2Error::LockPoisoned(msg) => {
            QuantRS2Error::LockPoisoned(format!("{context}: {msg}"))
        }
        QuantRS2Error::IndexOutOfBounds { index, len } => {
            QuantRS2Error::IndexOutOfBounds { index, len } // Can't add context to this variant
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_category() {
        let error = QuantRS2Error::InvalidQubitId(5);
        assert_eq!(error.category(), ErrorCategory::Core);

        let error = QuantRS2Error::CircuitValidationFailed("test".into());
        assert_eq!(error.category(), ErrorCategory::Circuit);

        let error = QuantRS2Error::NetworkError("connection failed".into());
        assert_eq!(error.category(), ErrorCategory::Hardware);
    }

    #[test]
    fn test_is_recoverable() {
        let error = QuantRS2Error::NetworkError("timeout".into());
        assert!(error.is_recoverable());

        let error = QuantRS2Error::InvalidQubitId(5);
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_is_invalid_input() {
        let error = QuantRS2Error::InvalidInput("bad value".into());
        assert!(error.is_invalid_input());

        let error = QuantRS2Error::NetworkError("timeout".into());
        assert!(!error.is_invalid_input());
    }

    #[test]
    fn test_with_context() {
        let error = QuantRS2Error::InvalidInput("bad parameter".into());
        let contextualized = with_context(error, "in circuit builder");

        match contextualized {
            QuantRS2Error::InvalidInput(msg) => {
                assert!(msg.contains("in circuit builder"));
                assert!(msg.contains("bad parameter"));
            }
            _ => panic!("Expected InvalidInput variant"),
        }
    }

    #[test]
    fn test_user_message() {
        let error = QuantRS2Error::InvalidQubitId(42);
        let msg = error.user_message();
        assert!(msg.contains("42"));
        assert!(msg.contains("not valid"));
    }

    #[test]
    fn test_category_name_and_description() {
        assert_eq!(ErrorCategory::Core.name(), "Core");
        assert!(ErrorCategory::Circuit.description().contains("circuit"));
        assert!(ErrorCategory::Hardware.description().contains("hardware"));
    }
}
