//! # EnhancedQASMCompiler - export_binary_group Methods
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

use super::types::AST;

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    pub(super) fn export_binary(ast: &AST) -> QuantRS2Result<Vec<u8>> {
        let bytes = oxicode::serde::encode_to_vec(ast, oxicode::config::standard())?;
        Ok(bytes)
    }
}
