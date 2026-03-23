//! # EnhancedQASMCompiler - eliminate_dead_code_group Methods
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

use super::types::{DeadCodeAnalyzer, AST};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    /// Dead code elimination
    pub(super) fn eliminate_dead_code(ast: AST) -> QuantRS2Result<AST> {
        let dead_nodes = DeadCodeAnalyzer::find_dead_code(&ast)?;
        Ok(ast.remove_nodes(dead_nodes))
    }
}
