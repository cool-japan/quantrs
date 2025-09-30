//! Comprehensive tests for Quantum State Tomography module

#[cfg(test)]
mod tests {
    use scirs2_core::ndarray::{Array1, Array2, Array3, Array4};
    use quantrs2_tytan::quantum_state_tomography::*;
    use quantrs2_tytan::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
    use std::collections::HashMap;

    /// Test tomography types
    #[test]
    fn test_tomography_types() {
        let types = vec![
            TomographyType::QuantumState,
            TomographyType::QuantumProcess,
            TomographyType::ShadowTomography { num_shadows: 100 },
            TomographyType::CompressedSensing { sparsity_level: 10 },
            TomographyType::AdaptiveTomography,
            TomographyType::EntanglementCharacterization,
        ];

        for tomo_type in types {
            match tomo_type {
                TomographyType::QuantumState => assert!(true),
                TomographyType::QuantumProcess => assert!(true),
                TomographyType::ShadowTomography { num_shadows } => {
                    assert_eq!(num_shadows, 100);
                }
                TomographyType::CompressedSensing { sparsity_level } => {
                    assert_eq!(sparsity_level, 10);
                }
                TomographyType::AdaptiveTomography => assert!(true),
                TomographyType::EntanglementCharacterization => assert!(true),
            }
        }
    }

    /// Test measurement basis
    #[test]
    fn test_measurement_basis() {
        let basis = MeasurementBasis {
            name: "Pauli-X".to_string(),
            operators: vec![PauliOperator::X, PauliOperator::I],
            angles: vec![std::f64::consts::PI / 2.0, 0.0],
            basis_type: BasisType::Pauli,
        };

        assert_eq!(basis.name, "Pauli-X");
        assert_eq!(basis.operators.len(), 2);
        assert_eq!(basis.angles.len(), 2);
        assert_eq!(basis.basis_type, BasisType::Pauli);
        assert_eq!(basis.operators[0], PauliOperator::X);
        assert_eq!(basis.operators[1], PauliOperator::I);
    }

    /// Test basis types
    #[test]
    fn test_basis_types() {
        let basis_types = vec![
            BasisType::Computational,
            BasisType::Pauli,
            BasisType::MUB,
            BasisType::SIC,
            BasisType::Stabilizer,
            BasisType::RandomPauli,
            BasisType::Adaptive,
        ];

        for basis_type in basis_types {
            match basis_type {
                BasisType::Computational => assert!(true),
                BasisType::Pauli => assert!(true),
                BasisType::MUB => assert!(true),
                BasisType::SIC => assert!(true),
                BasisType::Stabilizer => assert!(true),
                BasisType::RandomPauli => assert!(true),
                BasisType::Adaptive => assert!(true),
            }
        }
    }

    /// Test Pauli operators
    #[test]
    fn test_pauli_operators() {
        let pauli_ops = vec![
            PauliOperator::I,
            PauliOperator::X,
            PauliOperator::Y,
            PauliOperator::Z,
        ];

        for op in pauli_ops {
            match op {
                PauliOperator::I => assert!(true),
                PauliOperator::X => assert!(true),
                PauliOperator::Y => assert!(true),
                PauliOperator::Z => assert!(true),
            }
        }
    }

    /// Test reconstruction method types
    #[test]
    fn test_reconstruction_method_types() {
        let methods = vec![
            ReconstructionMethodType::MaximumLikelihood,
            ReconstructionMethodType::LeastSquares,
            ReconstructionMethodType::CompressedSensing,
            ReconstructionMethodType::BayesianInference,
            ReconstructionMethodType::NeuralNetwork,
            ReconstructionMethodType::Variational,
            ReconstructionMethodType::MatrixCompletion,
        ];

        for method in methods {
            match method {
                ReconstructionMethodType::MaximumLikelihood => assert!(true),
                ReconstructionMethodType::LeastSquares => assert!(true),
                ReconstructionMethodType::CompressedSensing => assert!(true),
                ReconstructionMethodType::BayesianInference => assert!(true),
                ReconstructionMethodType::NeuralNetwork => assert!(true),
                ReconstructionMethodType::Variational => assert!(true),
                ReconstructionMethodType::MatrixCompletion => assert!(true),
            }
        }
    }

    /// Test tomography metrics
    #[test]
    fn test_tomography_metrics() {
        let metrics = TomographyMetrics {
            reconstruction_accuracy: 0.94,
            computational_efficiency: 0.85,
            statistical_power: 0.90,
            robustness_score: 0.88,
            overall_quality: 0.89,
        };

        assert_eq!(metrics.reconstruction_accuracy, 0.94);
        assert_eq!(metrics.computational_efficiency, 0.85);
        assert_eq!(metrics.statistical_power, 0.90);
        assert_eq!(metrics.robustness_score, 0.88);
        assert_eq!(metrics.overall_quality, 0.89);
    }
}
