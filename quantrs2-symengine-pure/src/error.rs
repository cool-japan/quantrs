//! Error types for the symbolic mathematics library.
//!
//! This module provides error types that follow the "No unwrap policy"
//! ensuring all fallible operations return proper Result types.

use std::fmt;
use thiserror::Error;

/// Result type alias for symbolic operations
pub type SymEngineResult<T> = Result<T, SymEngineError>;

/// Errors that can occur during symbolic computation
#[derive(Error, Debug, Clone)]
pub enum SymEngineError {
    /// Failed to parse an expression from a string
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Invalid operation on expressions
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Division by zero in symbolic computation
    #[error("Division by zero")]
    DivisionByZero,

    /// Undefined result (e.g., 0/0, inf - inf)
    #[error("Undefined result: {0}")]
    Undefined(String),

    /// Type mismatch in expression
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    /// Invalid symbol name
    #[error("Invalid symbol name: {0}")]
    InvalidSymbol(String),

    /// Matrix dimension mismatch
    #[error("Matrix dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Evaluation error (e.g., undefined variable)
    #[error("Evaluation error: {0}")]
    EvaluationError(String),

    /// Differentiation error
    #[error("Differentiation error: {0}")]
    DifferentiationError(String),

    /// Simplification failed
    #[error("Simplification error: {0}")]
    SimplificationError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Internal error (should not happen in normal operation)
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Feature not yet implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Quantum-specific error
    #[error("Quantum error: {0}")]
    QuantumError(String),
}

impl SymEngineError {
    /// Create a parse error with the given message
    #[must_use]
    pub fn parse<S: Into<String>>(msg: S) -> Self {
        Self::ParseError(msg.into())
    }

    /// Create an invalid operation error
    #[must_use]
    pub fn invalid_op<S: Into<String>>(msg: S) -> Self {
        Self::InvalidOperation(msg.into())
    }

    /// Create an evaluation error
    #[must_use]
    pub fn eval<S: Into<String>>(msg: S) -> Self {
        Self::EvaluationError(msg.into())
    }

    /// Create a differentiation error
    #[must_use]
    pub fn diff<S: Into<String>>(msg: S) -> Self {
        Self::DifferentiationError(msg.into())
    }

    /// Create a not implemented error
    #[must_use]
    pub fn not_impl<S: Into<String>>(msg: S) -> Self {
        Self::NotImplemented(msg.into())
    }

    /// Create a quantum error
    #[must_use]
    pub fn quantum<S: Into<String>>(msg: S) -> Self {
        Self::QuantumError(msg.into())
    }

    /// Create a type mismatch error
    #[must_use]
    pub fn type_mismatch<S1: Into<String>, S2: Into<String>>(expected: S1, actual: S2) -> Self {
        Self::TypeMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a dimension mismatch error
    #[must_use]
    pub fn dimension<S: Into<String>>(msg: S) -> Self {
        Self::DimensionMismatch(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = SymEngineError::parse("invalid syntax");
        assert!(err.to_string().contains("Parse error"));

        let err = SymEngineError::DivisionByZero;
        assert!(err.to_string().contains("Division by zero"));

        let err = SymEngineError::type_mismatch("Symbol", "Number");
        assert!(err.to_string().contains("Symbol"));
        assert!(err.to_string().contains("Number"));
    }
}
