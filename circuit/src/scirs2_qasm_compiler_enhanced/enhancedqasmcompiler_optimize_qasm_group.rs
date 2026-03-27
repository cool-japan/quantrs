//! # EnhancedQASMCompiler - optimize_qasm_group Methods
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

use super::types::{
    ASTStatistics, CodeGenerator, ImprovementMetrics, MLOptimizer, OptimizedQASM, QASMOptimizer,
    AST,
};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    /// Optimize QASM code
    pub fn optimize_qasm(&self, source: &str) -> QuantRS2Result<OptimizedQASM> {
        let ast = self.parse_with_recovery(&Self::lexical_analysis(source)?)?;
        let original_stats = Self::calculate_ast_stats(&ast)?;
        let optimized_ast = self.optimize_ast(ast)?;
        let optimized_stats = Self::calculate_ast_stats(&optimized_ast)?;
        let optimized_code =
            CodeGenerator::generate_qasm(&optimized_ast, self.config.base_config.qasm_version)?;
        Ok(OptimizedQASM {
            original_code: source.to_string(),
            optimized_code,
            original_stats: original_stats.clone(),
            optimized_stats: optimized_stats.clone(),
            optimizations_applied: self.optimizer.get_applied_optimizations(),
            improvement_metrics: Self::calculate_improvements(&original_stats, &optimized_stats)?,
        })
    }
    /// Optimize AST
    pub(super) fn optimize_ast(&self, ast: AST) -> QuantRS2Result<AST> {
        let mut optimized = ast;
        optimized = QASMOptimizer::optimize(optimized)?;
        if self.ml_optimizer.is_some() {
            optimized = MLOptimizer::optimize(optimized)?;
        }
        if self.config.analysis_options.dead_code_elimination {
            optimized = Self::eliminate_dead_code(optimized)?;
        }
        if self.config.analysis_options.constant_propagation {
            optimized = Self::propagate_constants(optimized)?;
        }
        if self.config.analysis_options.loop_optimization {
            optimized = Self::optimize_loops(optimized)?;
        }
        Ok(optimized)
    }
    pub(super) fn calculate_ast_stats(ast: &AST) -> QuantRS2Result<ASTStatistics> {
        Ok(ASTStatistics {
            node_count: ast.node_count(),
            gate_count: ast.gate_count(),
            depth: AST::circuit_depth(),
            two_qubit_gates: AST::two_qubit_gate_count(),
            parameter_count: AST::parameter_count(),
        })
    }
    pub(super) fn calculate_improvements(
        original: &ASTStatistics,
        optimized: &ASTStatistics,
    ) -> QuantRS2Result<ImprovementMetrics> {
        Ok(ImprovementMetrics {
            gate_reduction: (original.gate_count - optimized.gate_count) as f64
                / original.gate_count as f64,
            depth_reduction: (original.depth - optimized.depth) as f64 / original.depth as f64,
            two_qubit_reduction: (original.two_qubit_gates - optimized.two_qubit_gates) as f64
                / original.two_qubit_gates.max(1) as f64,
        })
    }
}
