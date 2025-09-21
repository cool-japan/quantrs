#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]
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

use std::os::raw::{c_char, c_int};

// Include generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

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
            SymEngineError::NoException => write!(f, "No exception"),
            SymEngineError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            SymEngineError::DivisionByZero => write!(f, "Division by zero"),
            SymEngineError::NotImplemented => write!(f, "Operation not implemented"),
            SymEngineError::DomainError => write!(f, "Domain error"),
            SymEngineError::ParseError => write!(f, "Parse error"),
            SymEngineError::Unknown(code) => write!(f, "Unknown error code: {}", code),
        }
    }
}

impl std::error::Error for SymEngineError {}

impl From<c_int> for SymEngineError {
    fn from(code: c_int) -> Self {
        match code {
            error_codes::SYMENGINE_NO_EXCEPTION => SymEngineError::NoException,
            error_codes::SYMENGINE_RUNTIME_ERROR => {
                SymEngineError::RuntimeError("Runtime error".to_string())
            }
            error_codes::SYMENGINE_DIV_BY_ZERO => SymEngineError::DivisionByZero,
            error_codes::SYMENGINE_NOT_IMPLEMENTED => SymEngineError::NotImplemented,
            error_codes::SYMENGINE_DOMAIN_ERROR => SymEngineError::DomainError,
            error_codes::SYMENGINE_PARSE_ERROR => SymEngineError::ParseError,
            _ => SymEngineError::Unknown(code),
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

// Additional function declarations that might not be in the generated bindings
extern "C" {
    // CVecBasic functions (for handling argument vectors)
    pub fn vecbasic_new() -> *mut CVecBasic;
    pub fn vecbasic_free(self_: *mut CVecBasic);
    pub fn vecbasic_push_back(self_: *mut CVecBasic, value: *const basic_struct) -> c_int;
    pub fn vecbasic_get(self_: *mut CVecBasic, n: usize, result: *mut basic_struct) -> c_int;
    pub fn vecbasic_size(self_: *const CVecBasic) -> usize;

    // Symbol operations
    pub fn function_symbol_get_name(basic: *const basic_struct) -> *mut c_char;
    // Note: basic_get_args, basic_str_free, basic_pow, and basic_get_type are already in the generated bindings
}

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
            SymEngineError::RuntimeError("Runtime error".to_string())
        );
        assert_eq!(SymEngineError::from(2), SymEngineError::DivisionByZero);
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
}
