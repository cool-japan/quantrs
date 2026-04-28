//! OpenQASM import/export functionality
//!
//! This module provides support for converting between `QuantRS2` circuits
//! and OpenQASM format (both 2.0 and 3.0), enabling interoperability with
//! other quantum computing frameworks.
//!
//! ## QASM 2.0
//!
//! Use [`circuit_to_qasm`] and [`qasm_to_gates`] for round-trip QASM 2.0
//! support compatible with Qiskit and other standard tools.
//!
//! ## QASM 3.0
//!
//! Use [`export_qasm3`] / [`parse_qasm3`] for QASM 3.0 features.

// ─── OpenQASM 2.0 ──────────────────────────────────────────────────────────
pub mod error;
pub mod export;
pub mod import;

pub use error::QasmError;
pub use export::{
    circuit_to_qasm, extract_params, gate_name_to_qasm2, qasm2_gate_param_count,
    qasm2_gate_qubit_count,
};
pub use import::qasm_to_gates;

// ─── OpenQASM 3.0 ──────────────────────────────────────────────────────────
pub mod ast;
pub mod exporter;
pub mod parser;
pub mod validator;

pub use ast::{QasmGate, QasmProgram, QasmRegister, QasmStatement};
pub use exporter::{export_qasm3, ExportOptions, QasmExporter};
pub use parser::{parse_qasm3, ParseError, QasmParser};
pub use validator::{validate_qasm3, ValidationError};

/// QASM 3.0 version string
pub const QASM_VERSION: &str = "3.0";
/// QASM 2.0 version string
pub const QASM2_VERSION: &str = "2.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(QASM_VERSION, "3.0");
        assert_eq!(QASM2_VERSION, "2.0");
    }
}
