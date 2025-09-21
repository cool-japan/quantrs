//! Mixed-precision quantum simulation module.
//!
//! This module provides adaptive precision algorithms that automatically
//! select optimal numerical precision (f16, f32, f64) for different parts
//! of quantum computations, leveraging performance optimization while
//! maintaining required accuracy.

pub mod analysis;
pub mod config;
pub mod simulator;
pub mod state_vector;

// Re-export commonly used types and structs
pub use analysis::{AnalysisSummary, PerformanceMetrics, PrecisionAnalysis, PrecisionAnalyzer};
pub use config::{
    AdaptiveStrategy, MixedPrecisionConfig, MixedPrecisionContext, PrecisionLevel, QuantumPrecision,
};
pub use simulator::{MixedPrecisionSimulator, MixedPrecisionStats};
pub use state_vector::MixedPrecisionStateVector;

use crate::error::Result;

/// Initialize the mixed-precision subsystem
pub fn initialize() -> Result<()> {
    // Perform any necessary initialization
    #[cfg(feature = "advanced_math")]
    {
        // Initialize SciRS2 mixed precision context if available
        let _context = MixedPrecisionContext::new(AdaptiveStrategy::ErrorBased(1e-6));
    }

    Ok(())
}

/// Check if mixed-precision features are available
pub fn is_available() -> bool {
    cfg!(feature = "advanced_math")
}

/// Get supported precision levels
pub fn get_supported_precisions() -> Vec<QuantumPrecision> {
    vec![
        QuantumPrecision::Half,
        QuantumPrecision::Single,
        QuantumPrecision::Double,
        QuantumPrecision::Adaptive,
    ]
}

/// Create a default configuration for accuracy
pub fn default_accuracy_config() -> MixedPrecisionConfig {
    MixedPrecisionConfig::for_accuracy()
}

/// Create a default configuration for performance
pub fn default_performance_config() -> MixedPrecisionConfig {
    MixedPrecisionConfig::for_performance()
}

/// Create a balanced configuration
pub fn default_balanced_config() -> MixedPrecisionConfig {
    MixedPrecisionConfig::balanced()
}

/// Validate a mixed-precision configuration
pub fn validate_config(config: &MixedPrecisionConfig) -> Result<()> {
    config.validate()
}

/// Estimate memory usage for a given configuration and number of qubits
pub fn estimate_memory_usage(config: &MixedPrecisionConfig, num_qubits: usize) -> usize {
    config.estimate_memory_usage(num_qubits)
}

/// Calculate memory savings compared to double precision
pub fn calculate_memory_savings(config: &MixedPrecisionConfig, num_qubits: usize) -> f64 {
    simulator::utils::memory_savings(config, num_qubits)
}

/// Get performance improvement factor for a precision level
pub fn get_performance_factor(precision: QuantumPrecision) -> f64 {
    simulator::utils::performance_improvement_factor(precision)
}

/// Benchmark different precision levels
pub fn benchmark_precisions() -> Result<analysis::PrecisionAnalysis> {
    let mut analyzer = PrecisionAnalyzer::new();
    Ok(analyzer.analyze_for_tolerance(1e-6))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use num_complex::Complex64;

    #[test]
    fn test_precision_initialization() {
        let result = initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_supported_precisions() {
        let precisions = get_supported_precisions();
        assert_eq!(precisions.len(), 4);
        assert!(precisions.contains(&QuantumPrecision::Half));
        assert!(precisions.contains(&QuantumPrecision::Single));
        assert!(precisions.contains(&QuantumPrecision::Double));
        assert!(precisions.contains(&QuantumPrecision::Adaptive));
    }

    #[test]
    fn test_config_creation() {
        let accuracy_config = default_accuracy_config();
        assert_eq!(
            accuracy_config.state_vector_precision,
            QuantumPrecision::Double
        );

        let performance_config = default_performance_config();
        assert_eq!(
            performance_config.state_vector_precision,
            QuantumPrecision::Half
        );

        let balanced_config = default_balanced_config();
        assert_eq!(
            balanced_config.state_vector_precision,
            QuantumPrecision::Single
        );
    }

    #[test]
    fn test_config_validation() {
        let config = MixedPrecisionConfig::default();
        assert!(validate_config(&config).is_ok());

        let mut invalid_config = config.clone();
        invalid_config.error_tolerance = -1.0;
        assert!(validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_memory_estimation() {
        let config = MixedPrecisionConfig::default();
        let memory_4q = estimate_memory_usage(&config, 4);
        let memory_8q = estimate_memory_usage(&config, 8);

        // Memory should scale exponentially with qubits
        assert!(memory_8q > memory_4q * 10);
    }

    #[test]
    fn test_precision_properties() {
        assert_eq!(QuantumPrecision::Half.memory_factor(), 0.25);
        assert_eq!(QuantumPrecision::Single.memory_factor(), 0.5);
        assert_eq!(QuantumPrecision::Double.memory_factor(), 1.0);

        assert!(QuantumPrecision::Half.typical_error() > QuantumPrecision::Single.typical_error());
        assert!(
            QuantumPrecision::Single.typical_error() > QuantumPrecision::Double.typical_error()
        );
    }

    #[test]
    fn test_precision_transitions() {
        assert_eq!(
            QuantumPrecision::Half.higher_precision(),
            Some(QuantumPrecision::Single)
        );
        assert_eq!(
            QuantumPrecision::Single.higher_precision(),
            Some(QuantumPrecision::Double)
        );
        assert_eq!(QuantumPrecision::Double.higher_precision(), None);

        assert_eq!(
            QuantumPrecision::Double.lower_precision(),
            Some(QuantumPrecision::Single)
        );
        assert_eq!(
            QuantumPrecision::Single.lower_precision(),
            Some(QuantumPrecision::Half)
        );
        assert_eq!(QuantumPrecision::Half.lower_precision(), None);
    }

    #[test]
    fn test_state_vector_creation() {
        let state = MixedPrecisionStateVector::new(4, QuantumPrecision::Single);
        assert_eq!(state.len(), 4);
        assert_eq!(state.precision(), QuantumPrecision::Single);

        let basis_state =
            MixedPrecisionStateVector::computational_basis(2, QuantumPrecision::Double);
        assert_eq!(basis_state.len(), 4);
        assert_eq!(basis_state.precision(), QuantumPrecision::Double);
    }

    #[test]
    fn test_state_vector_operations() {
        let mut state = MixedPrecisionStateVector::new(4, QuantumPrecision::Single);

        // Test setting and getting amplitudes
        let amplitude = Complex64::new(0.5, 0.3);
        assert!(state.set_amplitude(0, amplitude).is_ok());

        // For single precision, we need to account for precision loss
        let retrieved_amplitude = state.amplitude(0).unwrap();
        assert!((retrieved_amplitude.re - amplitude.re).abs() < 1e-6);
        assert!((retrieved_amplitude.im - amplitude.im).abs() < 1e-6);

        // Test probability calculation
        let prob = state.probability(0).unwrap();
        assert!((prob - amplitude.norm_sqr()).abs() < 1e-6);
    }

    #[test]
    fn test_precision_conversion() {
        let state_single = MixedPrecisionStateVector::new(4, QuantumPrecision::Single);
        let state_double = state_single.to_precision(QuantumPrecision::Double);

        assert!(state_double.is_ok());
        let converted = state_double.unwrap();
        assert_eq!(converted.precision(), QuantumPrecision::Double);
        assert_eq!(converted.len(), 4);
    }

    #[test]
    fn test_simulator_creation() {
        let config = MixedPrecisionConfig::default();
        let simulator = MixedPrecisionSimulator::new(2, config);

        assert!(simulator.is_ok());
        let sim = simulator.unwrap();
        assert!(sim.get_state().is_some());
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::new(100.0, 1024, 10.0, 5.0);
        assert_eq!(metrics.execution_time_ms, 100.0);
        assert_eq!(metrics.memory_usage_bytes, 1024);
        assert_eq!(metrics.throughput_ops_per_sec, 10.0);
        assert_eq!(metrics.energy_efficiency, 5.0);

        let score = metrics.performance_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_precision_analysis() {
        let mut analysis = PrecisionAnalysis::new();
        analysis.add_recommendation("test_op".to_string(), QuantumPrecision::Single);
        analysis.add_error_estimate(QuantumPrecision::Single, 1e-6);

        let metrics = PerformanceMetrics::new(50.0, 512, 20.0, 10.0);
        analysis.add_performance_metrics(QuantumPrecision::Single, metrics);

        analysis.calculate_quality_score();

        assert_eq!(
            analysis.get_best_precision("test_op"),
            Some(QuantumPrecision::Single)
        );
        assert!(analysis.quality_score > 0.0);
    }

    #[test]
    fn test_analyzer() {
        let mut analyzer = PrecisionAnalyzer::new();
        let result = analyzer.analyze_for_tolerance(1e-6);
        assert!(!result.error_estimates.is_empty());
        assert!(!result.performance_metrics.is_empty());
    }
}
