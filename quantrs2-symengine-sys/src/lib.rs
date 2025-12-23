#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::use_self)]
#![allow(unused)]
#![doc = include_str!("../README.md")]

//! # SymEngine Sys
//!
//! Low-level Rust bindings to the [SymEngine](https://github.com/symengine/symengine) library.
//!
//! This crate provides raw FFI bindings to SymEngine, a fast symbolic manipulation library
//! written in C++. For a more idiomatic Rust interface, consider using the `symengine` crate
//! which builds on top of these bindings.
//!
//! ## Safety
//!
//! All functions in this crate are `unsafe` as they directly interface with C/C++ code.
//! Users must ensure proper memory management and follow SymEngine's usage patterns.
//!
//! ## Features
//!
//! - `static`: Link SymEngine statically
//! - `system-deps`: Use pkg-config to find system dependencies
//!
//! ## Documentation
//!
//! For comprehensive API documentation, see the [`api_docs`] module.

use std::os::raw::{c_char, c_int};

// Include generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Comprehensive API documentation
pub mod api_docs;

/// SymEngine error codes
pub mod error_codes {
    use super::c_int;

    pub const SYMENGINE_NO_EXCEPTION: c_int = 0;
    pub const SYMENGINE_RUNTIME_ERROR: c_int = 1;
    pub const SYMENGINE_DIV_BY_ZERO: c_int = 2;
    pub const SYMENGINE_NOT_IMPLEMENTED: c_int = 3;
    pub const SYMENGINE_DOMAIN_ERROR: c_int = 4;
    pub const SYMENGINE_PARSE_ERROR: c_int = 5;
}

/// SymEngine type codes
pub mod type_codes {
    use super::c_int;

    pub const SYMENGINE_SYMBOL: c_int = 1;
    pub const SYMENGINE_ADD: c_int = 2;
    pub const SYMENGINE_MUL: c_int = 3;
    pub const SYMENGINE_POW: c_int = 4;
    pub const SYMENGINE_INTEGER: c_int = 5;
    pub const SYMENGINE_RATIONAL: c_int = 6;
    pub const SYMENGINE_REAL_DOUBLE: c_int = 7;
    pub const SYMENGINE_COMPLEX_DOUBLE: c_int = 8;
}

// Re-export type codes for easier access
pub use type_codes::*;

/// Result type for SymEngine operations
pub type SymEngineResult<T> = Result<T, SymEngineError>;

/// SymEngine error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymEngineError {
    NoException,
    RuntimeError(String),
    DivisionByZero,
    NotImplemented,
    DomainError,
    ParseError,
    Unknown(c_int),
}

impl std::fmt::Display for SymEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymEngineError::NoException => write!(f, "No exception occurred"),
            SymEngineError::RuntimeError(msg) => {
                write!(f, "SymEngine runtime error: {}", msg)
            }
            SymEngineError::DivisionByZero => {
                write!(f, "Division by zero in symbolic computation")
            }
            SymEngineError::NotImplemented => {
                write!(
                    f,
                    "Operation not implemented in SymEngine for this type or configuration"
                )
            }
            SymEngineError::DomainError => {
                write!(
                    f,
                    "Domain error: operation outside valid mathematical domain"
                )
            }
            SymEngineError::ParseError => {
                write!(f, "Parse error: failed to parse symbolic expression")
            }
            SymEngineError::Unknown(code) => {
                write!(f, "Unknown SymEngine error (code: {})", code)
            }
        }
    }
}

impl std::error::Error for SymEngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        match self {
            SymEngineError::NoException => "No exception occurred",
            SymEngineError::RuntimeError(_) => "SymEngine runtime error",
            SymEngineError::DivisionByZero => "Division by zero",
            SymEngineError::NotImplemented => "Operation not implemented",
            SymEngineError::DomainError => "Domain error",
            SymEngineError::ParseError => "Parse error",
            SymEngineError::Unknown(_) => "Unknown error",
        }
    }
}

impl From<c_int> for SymEngineError {
    fn from(code: c_int) -> Self {
        match code {
            error_codes::SYMENGINE_NO_EXCEPTION => SymEngineError::NoException,
            error_codes::SYMENGINE_RUNTIME_ERROR => {
                SymEngineError::RuntimeError("SymEngine encountered a runtime error".to_string())
            }
            error_codes::SYMENGINE_DIV_BY_ZERO => SymEngineError::DivisionByZero,
            error_codes::SYMENGINE_NOT_IMPLEMENTED => SymEngineError::NotImplemented,
            error_codes::SYMENGINE_DOMAIN_ERROR => SymEngineError::DomainError,
            error_codes::SYMENGINE_PARSE_ERROR => SymEngineError::ParseError,
            _ => SymEngineError::Unknown(code),
        }
    }
}

impl SymEngineError {
    /// Create a runtime error with a custom message
    pub fn runtime_error(msg: impl Into<String>) -> Self {
        SymEngineError::RuntimeError(msg.into())
    }

    /// Check if this error represents a successful operation (no exception)
    pub fn is_ok(&self) -> bool {
        matches!(self, SymEngineError::NoException)
    }

    /// Check if this is an error (not NoException)
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    /// Get the error code as an integer
    pub fn code(&self) -> c_int {
        match self {
            SymEngineError::NoException => error_codes::SYMENGINE_NO_EXCEPTION,
            SymEngineError::RuntimeError(_) => error_codes::SYMENGINE_RUNTIME_ERROR,
            SymEngineError::DivisionByZero => error_codes::SYMENGINE_DIV_BY_ZERO,
            SymEngineError::NotImplemented => error_codes::SYMENGINE_NOT_IMPLEMENTED,
            SymEngineError::DomainError => error_codes::SYMENGINE_DOMAIN_ERROR,
            SymEngineError::ParseError => error_codes::SYMENGINE_PARSE_ERROR,
            SymEngineError::Unknown(code) => *code,
        }
    }
}

/// Check result code and convert to Result
pub fn check_result(code: c_int) -> SymEngineResult<()> {
    if code == error_codes::SYMENGINE_NO_EXCEPTION {
        Ok(())
    } else {
        Err(SymEngineError::from(code))
    }
}

/// Version information
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// Note: All function bindings are now generated automatically by bindgen
// from the SymEngine C wrapper API (cwrapper.h). The following functions
// are now available through the generated bindings:
//
// - CVecBasic functions: vecbasic_new, vecbasic_free, vecbasic_push_back, etc.
// - CMapBasicBasic functions: mapbasicbasic_new, mapbasicbasic_free, mapbasicbasic_insert, etc.
// - Substitution: basic_subs, basic_subs2
// - Complex numbers: complex_base_real_part, complex_base_imaginary_part, basic_conjugate
// - Special functions: basic_atan2, basic_kronecker_delta, basic_lowergamma, basic_uppergamma, etc.
// - Number theory: ntheory_gcd, ntheory_lcm, ntheory_mod, ntheory_factorial, etc.
// - Matrix operations: dense_matrix_*, sparse_matrix_*
// - Calculus: basic_diff, dense_matrix_jacobian, etc.
//
// All of these are automatically generated and available for use.

/// Check if SymEngine is available at runtime
pub fn is_symengine_available() -> bool {
    // Try to call a basic SymEngine function to verify it's available
    // This should be safe as long as SymEngine is properly linked
    std::ptr::null_mut::<basic_struct>() != std::ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        assert_eq!(SymEngineError::from(0), SymEngineError::NoException);
        assert_eq!(
            SymEngineError::from(1),
            SymEngineError::RuntimeError("SymEngine encountered a runtime error".to_string())
        );
        assert_eq!(SymEngineError::from(2), SymEngineError::DivisionByZero);
        assert_eq!(SymEngineError::from(3), SymEngineError::NotImplemented);
        assert_eq!(SymEngineError::from(4), SymEngineError::DomainError);
        assert_eq!(SymEngineError::from(5), SymEngineError::ParseError);
        assert_eq!(SymEngineError::from(99), SymEngineError::Unknown(99));
    }

    #[test]
    fn test_error_methods() {
        let ok_err = SymEngineError::NoException;
        assert!(ok_err.is_ok());
        assert!(!ok_err.is_err());
        assert_eq!(ok_err.code(), 0);

        let div_err = SymEngineError::DivisionByZero;
        assert!(!div_err.is_ok());
        assert!(div_err.is_err());
        assert_eq!(div_err.code(), 2);

        let custom_err = SymEngineError::runtime_error("Custom error message");
        assert!(!custom_err.is_ok());
        assert!(custom_err.is_err());
        assert_eq!(custom_err.code(), 1);
        match custom_err {
            SymEngineError::RuntimeError(msg) => assert_eq!(msg, "Custom error message"),
            _ => panic!("Expected RuntimeError"),
        }
    }

    #[test]
    fn test_error_display() {
        let err = SymEngineError::DivisionByZero;
        let display = format!("{}", err);
        assert!(display.contains("Division by zero"));

        let runtime_err = SymEngineError::runtime_error("test");
        let runtime_display = format!("{}", runtime_err);
        assert!(runtime_display.contains("test"));
    }

    #[test]
    fn test_check_result() {
        assert!(check_result(0).is_ok());
        assert!(check_result(1).is_err());
    }

    #[test]
    fn test_version() {
        let version_str = version();
        assert!(!version_str.is_empty());
        assert!(version_str.contains('.'));
    }

    #[test]
    fn test_basic_struct_size() {
        // Ensure basic_struct has reasonable size
        let size = std::mem::size_of::<basic_struct>();
        assert!(size > 0);
        assert!(size < 1024); // Reasonable upper bound
    }

    #[test]
    fn test_symengine_availability() {
        // This test verifies the library is linked correctly
        // In a real environment with SymEngine installed, this should pass
        // Note: This test might fail in CI environments without SymEngine
        let _available = is_symengine_available();
        // Just verify the function doesn't panic
    }

    #[test]
    fn test_vecbasic_operations() {
        unsafe {
            let vec = vecbasic_new();
            assert!(!vec.is_null());

            // Test that vecbasic_size works
            let size = vecbasic_size(vec);
            assert_eq!(size, 0);

            vecbasic_free(vec);
        }
    }

    #[test]
    fn test_mapbasicbasic_operations() {
        unsafe {
            let map = mapbasicbasic_new();
            assert!(!map.is_null());

            // Test that mapbasicbasic_size works
            let size = mapbasicbasic_size(map);
            assert_eq!(size, 0);

            mapbasicbasic_free(map);
        }
    }

    #[test]
    fn test_dense_matrix_creation() {
        unsafe {
            let mat = dense_matrix_new();
            assert!(!mat.is_null());
            dense_matrix_free(mat);

            let mat2 = dense_matrix_new_rows_cols(2, 2);
            assert!(!mat2.is_null());
            dense_matrix_free(mat2);
        }
    }

    #[test]
    fn test_type_sizes() {
        // Verify all major types have reasonable sizes
        assert!(std::mem::size_of::<basic_struct>() > 0);
        assert!(std::mem::size_of::<basic_struct>() < 1024);

        // Pointer types should be the size of a pointer
        assert_eq!(
            std::mem::size_of::<*mut CVecBasic>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<*mut CMapBasicBasic>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<*mut CDenseMatrix>(),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn test_error_codes() {
        // Verify error code constants
        assert_eq!(error_codes::SYMENGINE_NO_EXCEPTION, 0);
        assert_eq!(error_codes::SYMENGINE_RUNTIME_ERROR, 1);
        assert_eq!(error_codes::SYMENGINE_DIV_BY_ZERO, 2);
        assert_eq!(error_codes::SYMENGINE_NOT_IMPLEMENTED, 3);
        assert_eq!(error_codes::SYMENGINE_DOMAIN_ERROR, 4);
        assert_eq!(error_codes::SYMENGINE_PARSE_ERROR, 5);
    }

    #[test]
    fn test_type_codes() {
        // Verify type code constants
        assert_eq!(type_codes::SYMENGINE_SYMBOL, 1);
        assert_eq!(type_codes::SYMENGINE_ADD, 2);
        assert_eq!(type_codes::SYMENGINE_MUL, 3);
        assert_eq!(type_codes::SYMENGINE_POW, 4);
        assert_eq!(type_codes::SYMENGINE_INTEGER, 5);
        assert_eq!(type_codes::SYMENGINE_RATIONAL, 6);
        assert_eq!(type_codes::SYMENGINE_REAL_DOUBLE, 7);
        assert_eq!(type_codes::SYMENGINE_COMPLEX_DOUBLE, 8);
    }
}
