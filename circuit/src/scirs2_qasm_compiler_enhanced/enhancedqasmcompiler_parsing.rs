//! # EnhancedQASMCompiler - parsing Methods
//!
//! This module contains method implementations for `EnhancedQASMCompiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
    register::Register,
};

use super::types::{ErrorRecovery, ParsedQASM, QASMLexer, QASMParser, QASMVersion, Token, AST};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    /// Parse QASM file
    pub fn parse_file(&self, path: &str) -> QuantRS2Result<ParsedQASM> {
        let source = std::fs::read_to_string(path)?;
        let ast = self.parse_with_recovery(&Self::lexical_analysis(&source)?)?;
        Ok(ParsedQASM {
            version: Self::detect_version(&source)?,
            ast,
            metadata: Self::extract_metadata(&source)?,
            includes: Self::extract_includes(&source)?,
        })
    }
    /// Lexical analysis
    pub(super) fn lexical_analysis(source: &str) -> QuantRS2Result<Vec<Token>> {
        QASMLexer::tokenize(source)
    }
    /// Parse with error recovery
    pub(super) fn parse_with_recovery(&self, tokens: &[Token]) -> QuantRS2Result<AST> {
        match QASMParser::parse(tokens) {
            Ok(ast) => Ok(ast),
            Err(e) if self.config.enable_error_recovery => {
                let recovered = ErrorRecovery::recover_from_parse_error(tokens, &e)?;
                Ok(recovered)
            }
            Err(e) => Err(QuantRS2Error::InvalidOperation(e.to_string())),
        }
    }
    pub(super) fn detect_version(source: &str) -> QuantRS2Result<QASMVersion> {
        if source.contains("OPENQASM 3") {
            Ok(QASMVersion::QASM3)
        } else if source.contains("OPENQASM 2") {
            Ok(QASMVersion::QASM2)
        } else {
            Ok(QASMVersion::OpenQASM)
        }
    }
}
