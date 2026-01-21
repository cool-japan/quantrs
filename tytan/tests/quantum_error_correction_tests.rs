#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Comprehensive tests for Quantum Error Correction module

#[cfg(test)]
mod tests {
    use quantrs2_tytan::quantum_error_correction::*;
    use quantrs2_tytan::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
    use scirs2_core::ndarray::{Array1, Array2, Array3, Array4};
    use std::collections::HashMap;

    /// Test basic QEC configuration
    #[test]
    fn test_qec_config() {
        let config = QECConfig {
            code_type: QuantumCodeType::SurfaceCode {
                lattice_type: LatticeType::Square,
            },
            code_distance: 5,
            correction_frequency: 1000.0,
            syndrome_method: SyndromeExtractionMethod::Standard,
            decoding_algorithm: DecodingAlgorithm::MWPM,
            error_mitigation: ErrorMitigationConfig {
                zero_noise_extrapolation: true,
                probabilistic_error_cancellation: false,
                symmetry_verification: false,
                virtual_distillation: false,
                error_amplification: ErrorAmplificationConfig {
                    amplification_factors: vec![1.0, 2.0, 3.0],
                    max_amplification: 5.0,
                    strategy: AmplificationStrategy::Linear,
                },
                clifford_data_regression: false,
            },
            adaptive_correction: AdaptiveCorrectionConfig {
                adaptive_thresholding: true,
                dynamic_distance: false,
                real_time_code_switching: false,
                performance_adaptation: PerformanceAdaptationConfig {
                    error_rate_threshold: 0.01,
                    monitoring_window: 100,
                    adaptation_sensitivity: 0.5,
                    min_adaptation_interval: 10.0,
                },
                learning_adaptation: LearningAdaptationConfig {
                    reinforcement_learning: true,
                    learning_rate: 0.01,
                    replay_buffer_size: 1000,
                    update_frequency: 10,
                },
            },
            threshold_estimation: ThresholdEstimationConfig {
                real_time_estimation: false,
                estimation_method: ThresholdEstimationMethod::MonteCarlo,
                confidence_level: 0.95,
                update_frequency: 100,
            },
        };

        assert_eq!(config.code_distance, 5);
        assert_eq!(config.correction_frequency, 1000.0);
        assert_eq!(config.syndrome_method, SyndromeExtractionMethod::Standard);
        assert_eq!(config.decoding_algorithm, DecodingAlgorithm::MWPM);
    }

    /// Test quantum code types
    #[test]
    fn test_quantum_code_types() {
        let surface_code = QuantumCodeType::SurfaceCode {
            lattice_type: LatticeType::Square,
        };
        let color_code = QuantumCodeType::ColorCode {
            color_scheme: ColorScheme::ThreeColor,
        };
        let stabilizer_code = QuantumCodeType::StabilizerCode {
            generators: vec!["XXXX".to_string(), "ZZZZ".to_string()],
        };
        let topological_code = QuantumCodeType::TopologicalCode {
            code_family: TopologicalFamily::ToricCode,
        };

        match surface_code {
            QuantumCodeType::SurfaceCode { lattice_type } => {
                assert_eq!(lattice_type, LatticeType::Square);
            }
            _ => panic!("Wrong code type"),
        }

        match color_code {
            QuantumCodeType::ColorCode { color_scheme } => {
                assert_eq!(color_scheme, ColorScheme::ThreeColor);
            }
            _ => panic!("Wrong code type"),
        }

        match stabilizer_code {
            QuantumCodeType::StabilizerCode { generators } => {
                assert_eq!(generators.len(), 2);
                assert_eq!(generators[0], "XXXX");
                assert_eq!(generators[1], "ZZZZ");
            }
            _ => panic!("Wrong code type"),
        }

        match topological_code {
            QuantumCodeType::TopologicalCode { code_family } => {
                assert_eq!(code_family, TopologicalFamily::ToricCode);
            }
            _ => panic!("Wrong code type"),
        }
    }

    /// Test lattice types
    #[test]
    fn test_lattice_types() {
        let lattice_types = [
            LatticeType::Square,
            LatticeType::Triangular,
            LatticeType::Hexagonal,
            LatticeType::Kagome,
        ];

        for lattice_type in lattice_types {
            match lattice_type {
                LatticeType::Square
                | LatticeType::Triangular
                | LatticeType::Hexagonal
                | LatticeType::Kagome => {}
            }
        }
    }

    /// Test color schemes
    #[test]
    fn test_color_schemes() {
        let schemes = [
            ColorScheme::ThreeColor,
            ColorScheme::FourColor,
            ColorScheme::HexagonalColor,
        ];

        for scheme in schemes {
            match scheme {
                ColorScheme::ThreeColor | ColorScheme::FourColor | ColorScheme::HexagonalColor => {}
            }
        }
    }

    /// Test topological families
    #[test]
    fn test_topological_families() {
        let families = [
            TopologicalFamily::ToricCode,
            TopologicalFamily::PlanarCode,
            TopologicalFamily::HyperbolicCode,
            TopologicalFamily::FractalCode,
        ];

        for family in families {
            match family {
                TopologicalFamily::ToricCode
                | TopologicalFamily::PlanarCode
                | TopologicalFamily::HyperbolicCode
                | TopologicalFamily::FractalCode => {}
            }
        }
    }

    /// Test syndrome extraction methods
    #[test]
    fn test_syndrome_extraction_methods() {
        let methods = [
            SyndromeExtractionMethod::Standard,
            SyndromeExtractionMethod::FlagBased,
            SyndromeExtractionMethod::Repeated { num_repetitions: 3 },
            SyndromeExtractionMethod::Adaptive,
            SyndromeExtractionMethod::Concurrent,
        ];

        for method in methods {
            match method {
                SyndromeExtractionMethod::Standard
                | SyndromeExtractionMethod::FlagBased
                | SyndromeExtractionMethod::Adaptive
                | SyndromeExtractionMethod::Concurrent => {}
                SyndromeExtractionMethod::Repeated { num_repetitions } => {
                    assert_eq!(num_repetitions, 3);
                }
            }
        }
    }

    /// Test decoding algorithms
    #[test]
    fn test_decoding_algorithms() {
        let algorithms = [
            DecodingAlgorithm::MWPM,
            DecodingAlgorithm::BeliefPropagation,
            DecodingAlgorithm::NeuralNetwork {
                architecture: "CNN".to_string(),
            },
            DecodingAlgorithm::UnionFind,
            DecodingAlgorithm::Trellis,
            DecodingAlgorithm::MachineLearning {
                model_type: MLModelType::RNN,
            },
        ];

        for algorithm in algorithms {
            match algorithm {
                DecodingAlgorithm::MWPM
                | DecodingAlgorithm::BeliefPropagation
                | DecodingAlgorithm::UnionFind
                | DecodingAlgorithm::Trellis => {}
                DecodingAlgorithm::NeuralNetwork { architecture } => {
                    assert_eq!(architecture, "CNN");
                }
                DecodingAlgorithm::MachineLearning { model_type } => {
                    assert_eq!(model_type, MLModelType::RNN);
                }
                _ => {}
            }
        }
    }

    /// Test ML model types
    #[test]
    fn test_ml_model_types() {
        let models = [
            MLModelType::RNN,
            MLModelType::CNN,
            MLModelType::Transformer,
            MLModelType::GNN,
            MLModelType::VAE,
            MLModelType::ConvolutionalNN,
        ];

        for model in models {
            match model {
                MLModelType::RNN
                | MLModelType::CNN
                | MLModelType::Transformer
                | MLModelType::GNN
                | MLModelType::VAE
                | MLModelType::ConvolutionalNN => {}
                _ => {}
            }
        }
    }

    /// Test error mitigation methods
    #[test]
    fn test_error_mitigation_methods() {
        let methods = [
            ErrorMitigationMethod::ZeroNoiseExtrapolation,
            ErrorMitigationMethod::ReadoutErrorCorrection,
            ErrorMitigationMethod::VirtualDistillation,
            ErrorMitigationMethod::SymmetryVerification,
            ErrorMitigationMethod::PostSelection,
            ErrorMitigationMethod::TwirlingProtocols,
        ];

        for method in methods {
            match method {
                ErrorMitigationMethod::ZeroNoiseExtrapolation
                | ErrorMitigationMethod::ReadoutErrorCorrection
                | ErrorMitigationMethod::VirtualDistillation
                | ErrorMitigationMethod::SymmetryVerification
                | ErrorMitigationMethod::PostSelection
                | ErrorMitigationMethod::TwirlingProtocols => {}
            }
        }
    }
}
