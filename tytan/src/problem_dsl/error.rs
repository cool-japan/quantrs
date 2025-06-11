//! Error types for the problem DSL.

use std::fmt;

/// Parse error
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Parse error at line {}, column {}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for ParseError {}

/// Type error
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub location: String,
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Type error at {}: {}", self.location, self.message)
    }
}

impl std::error::Error for TypeError {}

impl From<TypeError> for ParseError {
    fn from(type_error: TypeError) -> Self {
        ParseError {
            message: type_error.message,
            line: 0,
            column: 0,
        }
    }
}

/// Compilation error
#[derive(Debug, Clone)]
pub struct CompileError {
    pub message: String,
    pub context: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Compilation error in {}: {}", self.context, self.message)
    }
}

impl std::error::Error for CompileError {}
