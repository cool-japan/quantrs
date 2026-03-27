//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

use super::types::{
    ConstraintConfig, ErrorCorrectionProtocol, ErrorMitigationConfig, MeasurementDatabase,
    NoiseCharacterizationConfig, OptimizationAlgorithm, OptimizationConfig, PhysicalConstraint,
    QuantumStateTomography, ReconstructedState, ReconstructionMethodType, RegularizationConfig,
    StatisticalTest, TomographyConfig, TomographyError, TomographyType, UncertaintyAnalysis,
    ValidationConfig, ValidationMetric, ValidationResult,
};

/// Reconstruction method trait
pub trait ReconstructionMethod: Send + Sync {
    /// Reconstruct quantum state from measurement data
    fn reconstruct_state(
        &self,
        data: &MeasurementDatabase,
    ) -> Result<ReconstructedState, TomographyError>;
    /// Get method name
    fn get_method_name(&self) -> &str;
    /// Get method parameters
    fn get_parameters(&self) -> HashMap<String, f64>;
    /// Validate reconstruction
    fn validate_reconstruction(
        &self,
        state: &ReconstructedState,
        data: &MeasurementDatabase,
    ) -> ValidationResult;
}
/// Fidelity estimator trait
pub trait FidelityEstimator: Send + Sync {
    /// Estimate fidelity between two states
    fn estimate_fidelity(
        &self,
        state1: &ReconstructedState,
        state2: &ReconstructedState,
    ) -> Result<f64, TomographyError>;
    /// Estimate fidelity with measurement data
    fn estimate_fidelity_from_data(
        &self,
        state: &ReconstructedState,
        data: &MeasurementDatabase,
    ) -> Result<f64, TomographyError>;
    /// Get estimator name
    fn get_estimator_name(&self) -> &str;
}
/// Error propagation method trait
pub trait ErrorPropagationMethod: Send + Sync + std::fmt::Debug {
    /// Propagate measurement errors to state reconstruction
    fn propagate_errors(
        &self,
        measurement_errors: &Array1<f64>,
        reconstruction: &ReconstructedState,
    ) -> Array2<f64>;
    /// Get method name
    fn get_method_name(&self) -> &str;
}
/// Uncertainty quantification method trait
pub trait UncertaintyQuantificationMethod: Send + Sync + std::fmt::Debug {
    /// Quantify reconstruction uncertainty
    fn quantify_uncertainty(
        &self,
        data: &MeasurementDatabase,
        reconstruction: &ReconstructedState,
    ) -> UncertaintyAnalysis;
    /// Get method name
    fn get_method_name(&self) -> &str;
}
/// Create default tomography configuration
pub fn create_default_tomography_config() -> TomographyConfig {
    TomographyConfig {
        tomography_type: TomographyType::QuantumState,
        shots_per_setting: 1000,
        measurement_bases: Vec::new(),
        reconstruction_method: ReconstructionMethodType::MaximumLikelihood,
        error_mitigation: ErrorMitigationConfig {
            readout_error_correction: true,
            gate_error_mitigation: false,
            symmetry_verification: true,
            noise_characterization: NoiseCharacterizationConfig {
                coherent_errors: false,
                incoherent_errors: true,
                crosstalk: false,
                temporal_correlations: false,
                spatial_correlations: false,
            },
            error_correction_protocols: vec![ErrorCorrectionProtocol::ZeroNoiseExtrapolation],
        },
        optimization: OptimizationConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            regularization: RegularizationConfig {
                l1_strength: 0.0,
                l2_strength: 0.001,
                nuclear_norm_strength: 0.0,
                trace_norm_strength: 0.0,
                entropy_strength: 0.0,
            },
            constraints: ConstraintConfig {
                trace_preserving: true,
                completely_positive: true,
                hermitian: true,
                positive_semidefinite: true,
                physical_constraints: vec![
                    PhysicalConstraint::UnitTrace,
                    PhysicalConstraint::PositiveEigenvalues,
                ],
            },
            algorithm: OptimizationAlgorithm::GradientDescent,
        },
        validation: ValidationConfig {
            cross_validation_folds: 5,
            bootstrap_samples: 1000,
            confidence_level: 0.95,
            validation_metrics: vec![ValidationMetric::Fidelity, ValidationMetric::TraceDistance],
            statistical_tests: vec![StatisticalTest::ChiSquared, StatisticalTest::Bootstrap],
        },
    }
}
/// Create tomography system for given number of qubits
pub fn create_tomography_system(num_qubits: usize) -> QuantumStateTomography {
    let config = create_default_tomography_config();
    QuantumStateTomography::new(num_qubits, config)
}
/// Create shadow tomography configuration
pub fn create_shadow_tomography_config(num_shadows: usize) -> TomographyConfig {
    let mut config = create_default_tomography_config();
    config.tomography_type = TomographyType::ShadowTomography { num_shadows };
    config.shots_per_setting = 100;
    config
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tomography_creation() {
        let tomography = create_tomography_system(2);
        assert_eq!(tomography.num_qubits, 2);
        assert_eq!(
            tomography.config.tomography_type,
            TomographyType::QuantumState
        );
    }
    #[test]
    fn test_pauli_measurement_generation() {
        let tomography = create_tomography_system(2);
        let mut measurements = tomography
            .generate_pauli_measurements()
            .expect("Pauli measurement generation should succeed");
        assert_eq!(measurements.len(), 9);
    }
    #[test]
    fn test_shadow_measurement_generation() {
        let tomography = create_tomography_system(2);
        let mut measurements = tomography
            .generate_shadow_measurements(100)
            .expect("Shadow measurement generation should succeed");
        assert_eq!(measurements.len(), 100);
    }
    #[test]
    fn test_outcome_to_index() {
        let tomography = create_tomography_system(3);
        assert_eq!(tomography.outcome_to_index(&[0, 0, 0]), 0);
        assert_eq!(tomography.outcome_to_index(&[1, 0, 0]), 1);
        assert_eq!(tomography.outcome_to_index(&[0, 1, 0]), 2);
        assert_eq!(tomography.outcome_to_index(&[1, 1, 1]), 7);
    }
    #[test]
    fn test_purity_computation() {
        let tomography = create_tomography_system(1);
        let pure_state = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.0])
            .expect("Array creation from valid shape and data should succeed");
        let purity = tomography.compute_purity(&pure_state);
        assert!((purity - 1.0).abs() < 1e-10);
        let mixed_state = Array2::eye(2) / 2.0;
        let mixed_purity = tomography.compute_purity(&mixed_state);
        assert!((mixed_purity - 0.5).abs() < 1e-10);
    }
    #[test]
    fn test_entropy_computation() {
        let tomography = create_tomography_system(1);
        let pure_state = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.0])
            .expect("Array creation from valid shape and data should succeed");
        let entropy = tomography.compute_von_neumann_entropy(&pure_state);
        assert!(entropy.abs() < 1e-10);
    }
}
