//! # EnhancedQASMCompiler - optimize_loops_group Methods
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

use super::types::{LoopOptimizer, AST};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    /// Loop optimization
    pub(super) fn optimize_loops(ast: AST) -> QuantRS2Result<AST> {
        LoopOptimizer::optimize(ast)
    }
}
