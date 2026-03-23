//! # EnhancedQASMCompiler - propagate_constants_group Methods
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

use super::types::{ConstantPropagator, AST};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    /// Constant propagation
    pub(super) fn propagate_constants(ast: AST) -> QuantRS2Result<AST> {
        ConstantPropagator::propagate(ast)
    }
}
