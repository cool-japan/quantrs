//! # EnhancedQASMCompiler - detect_ast_version_group Methods
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

use super::types::{QASMVersion, AST};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    pub(super) fn detect_ast_version(_ast: &AST) -> QuantRS2Result<QASMVersion> {
        Ok(QASMVersion::QASM3)
    }
}
