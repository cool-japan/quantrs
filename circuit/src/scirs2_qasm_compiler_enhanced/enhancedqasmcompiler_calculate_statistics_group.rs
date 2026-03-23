//! # EnhancedQASMCompiler - calculate_statistics_group Methods
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

use super::types::{CompilationStatistics, Token};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    /// Calculate compilation statistics
    pub(super) fn calculate_statistics(tokens: &[Token]) -> QuantRS2Result<CompilationStatistics> {
        Ok(CompilationStatistics {
            token_count: tokens.len(),
            line_count: Self::count_lines(tokens),
            gate_count: Self::count_gates(tokens),
            qubit_count: Self::count_qubits(tokens),
            classical_bit_count: Self::count_classical_bits(tokens),
            function_count: Self::count_functions(tokens),
            include_count: Self::count_includes(tokens),
        })
    }
    pub(super) fn count_lines(tokens: &[Token]) -> usize {
        tokens.iter().map(|t| t.line).max().unwrap_or(0)
    }
    pub(super) fn count_gates(tokens: &[Token]) -> usize {
        tokens.iter().filter(|t| t.is_gate()).count()
    }
    pub(super) fn count_qubits(_tokens: &[Token]) -> usize {
        0
    }
    pub(super) fn count_classical_bits(_tokens: &[Token]) -> usize {
        0
    }
    pub(super) fn count_functions(tokens: &[Token]) -> usize {
        tokens.iter().filter(|t| t.is_function()).count()
    }
    pub(super) fn count_includes(tokens: &[Token]) -> usize {
        tokens.iter().filter(|t| t.is_include()).count()
    }
}
