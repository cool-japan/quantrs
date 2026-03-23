//! # EnhancedQASMConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedQASMConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AnalysisOptions, CompilationTarget, EnhancedQASMConfig, ExportFormat, OptimizationLevel,
    QASMCompilerConfig,
};

impl Default for EnhancedQASMConfig {
    fn default() -> Self {
        Self {
            base_config: QASMCompilerConfig::default(),
            enable_ml_optimization: true,
            enable_multi_version: true,
            enable_semantic_analysis: true,
            enable_realtime_validation: true,
            enable_error_recovery: true,
            enable_visual_ast: true,
            compilation_targets: vec![
                CompilationTarget::QuantRS2,
                CompilationTarget::Qiskit,
                CompilationTarget::Cirq,
            ],
            optimization_level: OptimizationLevel::Aggressive,
            analysis_options: AnalysisOptions::default(),
            export_formats: vec![
                ExportFormat::QuantRS2Native,
                ExportFormat::QASM3,
                ExportFormat::OpenQASM,
            ],
        }
    }
}
