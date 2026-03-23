//! # EnhancedQASMCompiler - collect_info_group Methods
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
    pub(super) fn collect_info(ast: &AST) -> QuantRS2Result<Vec<String>> {
        Ok(vec![
            format!("AST nodes: {}", ast.node_count()),
            format!("Max depth: {}", ast.max_depth()),
        ])
    }
}
