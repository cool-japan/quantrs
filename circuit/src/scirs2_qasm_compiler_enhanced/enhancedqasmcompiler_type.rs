//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::buffer_pool::BufferPool;
use std::sync::{Arc, Mutex};

use super::types::{
    CodeGenerator, CompilationCache, EnhancedQASMConfig, ErrorRecovery, MLOptimizer, QASMOptimizer,
    QASMParser, SemanticAnalyzer,
};

/// Enhanced QASM compiler
pub struct EnhancedQASMCompiler {
    pub(super) config: EnhancedQASMConfig,
    pub(super) parser: Arc<QASMParser>,
    pub(super) semantic_analyzer: Arc<SemanticAnalyzer>,
    pub(super) optimizer: Arc<QASMOptimizer>,
    pub(super) code_generator: Arc<CodeGenerator>,
    pub(super) ml_optimizer: Option<Arc<MLOptimizer>>,
    pub(super) error_recovery: Arc<ErrorRecovery>,
    pub(super) buffer_pool: BufferPool<f64>,
    pub(super) cache: Arc<Mutex<CompilationCache>>,
}
