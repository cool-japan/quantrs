//! Error handling for `SymEngine` operations.

use thiserror::Error;

/// Result type for `SymEngine` operations
pub type SymEngineResult<T> = Result<T, SymEngineError>;

/// Errors that can occur during `SymEngine` operations
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum SymEngineError {
    /// No error occurred
    #[error("No exception")]
    NoException,

    /// Runtime error occurred in `SymEngine`
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

    /// Unknown error code from `SymEngine`
    #[error("Unknown error code: {code}")]
    Unknown { code: i32 },
}

impl From<quantrs2_symengine_sys::SymEngineError> for SymEngineError {
    fn from(sys_error: quantrs2_symengine_sys::SymEngineError) -> Self {
        match sys_error {
            quantrs2_symengine_sys::SymEngineError::NoException => Self::NoException,
            quantrs2_symengine_sys::SymEngineError::RuntimeError(msg) => {
                Self::RuntimeError { message: msg }
            }
            quantrs2_symengine_sys::SymEngineError::DivisionByZero => Self::DivisionByZero,
            quantrs2_symengine_sys::SymEngineError::NotImplemented => Self::NotImplemented,
            quantrs2_symengine_sys::SymEngineError::DomainError => Self::DomainError,
            quantrs2_symengine_sys::SymEngineError::ParseError => Self::ParseError,
            quantrs2_symengine_sys::SymEngineError::Unknown(code) => Self::Unknown { code },
        }
    }
}

impl SymEngineError {
    /// Create a new runtime error with a custom message
    pub fn runtime_error(message: impl Into<String>) -> Self {
        Self::RuntimeError {
            message: message.into(),
        }
    }

    /// Create a new invalid operation error with a custom message
    pub fn invalid_operation(message: impl Into<String>) -> Self {
        Self::InvalidOperation {
            message: message.into(),
        }
    }

    /// Check if this error indicates a critical failure
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        matches!(self, Self::MemoryError | Self::Unknown { .. })
    }

    /// Get the error code for this error (compatible with `SymEngine` C API)
    #[must_use]
    pub const fn error_code(&self) -> i32 {
        match self {
            Self::NoException => 0,
            Self::RuntimeError { .. } => 1,
            Self::DivisionByZero => 2,
            Self::NotImplemented => 3,
            Self::DomainError => 4,
            Self::ParseError => 5,
            Self::MemoryError => 6,
            Self::InvalidOperation { .. } => 7,
            Self::Unknown { code } => *code,
        }
    }
}

/// Helper function to check `SymEngine` result codes
pub fn check_result(code: i32) -> SymEngineResult<()> {
    quantrs2_symengine_sys::check_result(code).map_err(SymEngineError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let runtime_err = SymEngineError::runtime_error("test message");
        assert!(matches!(runtime_err, SymEngineError::RuntimeError { .. }));

        let invalid_op_err = SymEngineError::invalid_operation("invalid op");
        assert!(matches!(
            invalid_op_err,
            SymEngineError::InvalidOperation { .. }
        ));
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
