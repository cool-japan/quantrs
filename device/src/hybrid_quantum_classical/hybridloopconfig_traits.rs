//! # Hybridloopconfig - Trait Implementations
//!
//! This module contains trait implementations for `Hybridloopconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for HybridLoopConfig {
    fn default() -> Self {
        Self {
            strategy: HybridLoopStrategy::VariationalOptimization,
            optimization_config: HybridOptimizationConfig::default(),
            feedback_config: FeedbackControlConfig::default(),
            classical_config: ClassicalComputationConfig::default(),
            quantum_config: QuantumExecutionConfig::default(),
            convergence_config: ConvergenceConfig::default(),
            performance_config: HybridPerformanceConfig::default(),
            error_handling_config: ErrorHandlingConfig::default(),
        }
    }
}
