//! `OpenQASM` 3.0 import/export functionality
//!
//! This module provides support for converting between `QuantRS2` circuits
//! and `OpenQASM` 3.0 format, enabling interoperability with other quantum
//! computing frameworks.

pub mod ast;
pub mod exporter;
pub mod parser;
pub mod validator;

pub use ast::{QasmGate, QasmProgram, QasmRegister, QasmStatement};
pub use exporter::{export_qasm3, ExportOptions, QasmExporter};
pub use parser::{parse_qasm3, ParseError, QasmParser};
pub use validator::{validate_qasm3, ValidationError};

/// QASM version supported
pub const QASM_VERSION: &str = "3.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(QASM_VERSION, "3.0");
    }
}
