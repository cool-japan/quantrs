//! Enhanced Hybrid Quantum-Classical Algorithms with Advanced SciRS2 Optimization
//!
//! This module provides state-of-the-art hybrid quantum-classical algorithms with
//! ML-driven optimization, adaptive parameter learning, real-time performance
//! tuning, and comprehensive benchmarking powered by SciRS2's optimization tools.

mod config;
mod data_types;
mod executor;
mod history;
mod optimizers;
mod quantum_types;
mod results;
mod support;

// Public API exports
pub use config::*;
pub use data_types::*;
pub use executor::EnhancedHybridExecutor;
pub use history::*;
pub use quantum_types::*;
pub use results::*;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_enhanced_hybrid_executor_creation() {
        let config = EnhancedHybridConfig::default();
        let executor = EnhancedHybridExecutor::new(config);
        // Executor created successfully
    }

    #[test]
    fn test_default_configuration() {
        let config = EnhancedHybridConfig::default();
        assert_eq!(config.base_config.max_iterations, 1000);
        assert!(config.enable_ml_optimization);
        assert!(config.algorithm_variants.contains(&HybridAlgorithm::VQE));
    }

    #[test]
    fn test_optimization_history() {
        let mut history = OptimizationHistory::new();
        history.record(0, 1.0, Array1::zeros(5));
        history.record(1, 0.9, Array1::zeros(5));

        assert_eq!(history.iterations.len(), 2);
        assert!(history.get_recent_improvement(1).unwrap() > 0.0);
    }
}
