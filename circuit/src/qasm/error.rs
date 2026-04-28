//! Error types for OpenQASM 2.0 import/export

use thiserror::Error;

/// Errors that can occur during QASM 2.0 operations
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum QasmError {
    /// The gate is not supported in QASM 2.0
    #[error("Unsupported gate: {0}")]
    UnsupportedGate(String),

    /// The QASM input is syntactically invalid
    #[error("Parse error at line {line}: {message}")]
    ParseError {
        /// Line number where the error occurred (1-indexed)
        line: usize,
        /// Human-readable error description
        message: String,
    },

    /// A register referenced in the QASM was not declared
    #[error("Undefined register: {0}")]
    UndefinedRegister(String),

    /// A qubit index is out of range for its register
    #[error("Qubit index {index} out of range for register '{register}' (size {size})")]
    QubitIndexOutOfRange {
        /// Register name
        register: String,
        /// Attempted index
        index: usize,
        /// Register size
        size: usize,
    },

    /// An invalid or unsupported gate parameter was encountered
    #[error("Invalid parameter in gate '{gate}': {message}")]
    InvalidParameter {
        /// Gate name
        gate: String,
        /// Error description
        message: String,
    },

    /// Wrong number of parameters for a gate
    #[error("Gate '{gate}' expects {expected} parameter(s), got {actual}")]
    WrongParameterCount {
        /// Gate name
        gate: String,
        /// Expected count
        expected: usize,
        /// Actual count
        actual: usize,
    },

    /// Wrong number of qubits for a gate
    #[error("Gate '{gate}' expects {expected} qubit(s), got {actual}")]
    WrongQubitCount {
        /// Gate name
        gate: String,
        /// Expected count
        expected: usize,
        /// Actual count
        actual: usize,
    },

    /// A string formatting error during export
    #[error("Format error: {0}")]
    FormatError(#[from] std::fmt::Error),

    /// No qubits in circuit (cannot determine register size)
    #[error("Circuit has no qubits")]
    EmptyCircuit,

    /// An expression in the QASM could not be evaluated
    #[error("Cannot evaluate expression '{0}'")]
    ExpressionError(String),
}

impl QasmError {
    /// Create a parse error with position information
    pub fn parse(line: usize, message: impl Into<String>) -> Self {
        Self::ParseError {
            line,
            message: message.into(),
        }
    }
}
