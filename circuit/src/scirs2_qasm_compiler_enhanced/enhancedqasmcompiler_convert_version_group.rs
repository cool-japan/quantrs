//! # EnhancedQASMCompiler - convert_version_group Methods
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

use super::types::{CodeGenerator, QASMVersion};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    /// Convert between QASM versions
    pub fn convert_version(
        &self,
        source: &str,
        target_version: QASMVersion,
    ) -> QuantRS2Result<String> {
        let ast = self.parse_with_recovery(&Self::lexical_analysis(source)?)?;
        let converted_ast = Self::convert_ast_version(ast, target_version)?;
        CodeGenerator::generate_qasm(&converted_ast, target_version)
    }
}
