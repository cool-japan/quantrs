//! # EnhancedQASMCompiler - export_cirq_group Methods
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

use super::types::{CodeGenerator, CompilationTarget, AST};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    pub(super) fn export_cirq(ast: &AST) -> QuantRS2Result<Vec<u8>> {
        let code = CodeGenerator::generate(ast, CompilationTarget::Cirq)?;
        Ok(code.python_code.into_bytes())
    }
}
