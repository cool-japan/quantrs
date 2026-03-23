//! # EnhancedLintingConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedLintingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::parallel_ops_stubs::*;
use crate::scirs2_quantum_linter::{LintSeverity, LintingConfig, QuantumGate};

use super::types::{EnhancedLintingConfig, HardwareArchitecture, ReportFormat};

impl Default for EnhancedLintingConfig {
    fn default() -> Self {
        Self {
            base_config: LintingConfig::default(),
            enable_ml_pattern_detection: true,
            enable_algorithm_recognition: true,
            enable_complexity_analysis: true,
            enable_noise_resilience_check: true,
            enable_topological_optimization: true,
            enable_qec_pattern_detection: true,
            enable_cross_compilation_check: true,
            enable_hardware_specific_linting: true,
            target_architectures: vec![
                HardwareArchitecture::IBMQ,
                HardwareArchitecture::IonQ,
                HardwareArchitecture::Simulator,
            ],
            pattern_database_version: "1.0.0".to_string(),
            max_analysis_depth: 1000,
            enable_incremental_linting: true,
            custom_rules: Vec::new(),
            report_format: ReportFormat::Detailed,
        }
    }
}
