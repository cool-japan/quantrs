//! Error handling for SymEngine operations.

use std::fmt;
use thiserror::Error;

/// Result type for SymEngine operations
pub type SymEngineResult<T> = Result<T, SymEngineError>;

/// Errors that can occur during SymEngine operations
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum SymEngineError {
    /// No error occurred
    #[error("No exception")]
    NoException,

    /// Runtime error occurred in SymEngine
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },

    /// Division by zero error
    #[error("Division by zero")]
    DivisionByZero,

    /// Operation not implemented
    #[error("Operation not implemented")]
    NotImplemented,

    /// Domain error (e.g., invalid input for function)
    #[error("Domain error")]
    DomainError,

    /// Parse error when parsing string expressions
    #[error("Parse error")]
    ParseError,

    /// Memory allocation error
    #[error("Memory allocation failed")]
    MemoryError,

    /// Invalid operation or arguments
    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    /// Unknown error code from SymEngine
    #[error("Unknown error code: {code}")]
    Unknown { code: i32 },
}

impl From<symengine_sys::SymEngineError> for SymEngineError {
    fn from(sys_error: symengine_sys::SymEngineError) -> Self {
        match sys_error {
            symengine_sys::SymEngineError::NoException => SymEngineError::NoException,
            symengine_sys::SymEngineError::RuntimeError(msg) => SymEngineError::RuntimeError { message: msg },
            symengine_sys::SymEngineError::DivisionByZero => SymEngineError::DivisionByZero,
            symengine_sys::SymEngineError::NotImplemented => SymEngineError::NotImplemented,
            symengine_sys::SymEngineError::DomainError => SymEngineError::DomainError,
            symengine_sys::SymEngineError::ParseError => SymEngineError::ParseError,
            symengine_sys::SymEngineError::Unknown(code) => SymEngineError::Unknown { code },
        }
    }
}

impl SymEngineError {
    /// Create a new runtime error with a custom message
    pub fn runtime_error(message: impl Into<String>) -> Self {
        SymEngineError::RuntimeError {
            message: message.into(),
        }
    }

    /// Create a new invalid operation error with a custom message
    pub fn invalid_operation(message: impl Into<String>) -> Self {
        SymEngineError::InvalidOperation {
            message: message.into(),
        }
    }

    /// Check if this error indicates a critical failure
    pub fn is_critical(&self) -> bool {
        matches!(self, SymEngineError::MemoryError | SymEngineError::Unknown { .. })
    }

    /// Get the error code for this error (compatible with SymEngine C API)
    pub fn error_code(&self) -> i32 {
        match self {
            SymEngineError::NoException => 0,
            SymEngineError::RuntimeError { .. } => 1,
            SymEngineError::DivisionByZero => 2,
            SymEngineError::NotImplemented => 3,
            SymEngineError::DomainError => 4,
            SymEngineError::ParseError => 5,
            SymEngineError::MemoryError => 6,
            SymEngineError::InvalidOperation { .. } => 7,
            SymEngineError::Unknown { code } => *code,
        }
    }
}

/// Helper function to check SymEngine result codes
pub fn check_result(code: i32) -> SymEngineResult<()> {
    symengine_sys::check_result(code).map_err(SymEngineError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let runtime_err = SymEngineError::runtime_error("test message");
        assert!(matches!(runtime_err, SymEngineError::RuntimeError { .. }));

        let invalid_op_err = SymEngineError::invalid_operation("invalid op");
        assert!(matches!(invalid_op_err, SymEngineError::InvalidOperation { .. }));
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(SymEngineError::NoException.error_code(), 0);
        assert_eq!(SymEngineError::DivisionByZero.error_code(), 2);
        assert_eq!(SymEngineError::ParseError.error_code(), 5);
    }

    #[test]
    fn test_critical_errors() {
        assert!(SymEngineError::MemoryError.is_critical());
        assert!(SymEngineError::Unknown { code: 999 }.is_critical());
        assert!(!SymEngineError::ParseError.is_critical());
    }

    #[test]
    fn test_error_display() {
        let err = SymEngineError::runtime_error("test");
        assert!(err.to_string().contains("test"));

        let div_zero = SymEngineError::DivisionByZero;
        assert!(div_zero.to_string().contains("Division by zero"));
    }
}