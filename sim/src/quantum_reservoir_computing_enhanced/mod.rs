//! Enhanced Quantum Reservoir Computing Framework - Ultrathink Mode Implementation
//!
//! This module provides a comprehensive implementation of quantum reservoir computing (QRC),
//! a cutting-edge computational paradigm that leverages the high-dimensional, nonlinear
//! dynamics of quantum systems for temporal information processing and machine learning.
//! This ultrathink mode implementation includes advanced learning algorithms, sophisticated
//! reservoir topologies, real-time adaptation, and comprehensive analysis tools.
//!
//! ## Core Features
//! - **Advanced Quantum Reservoirs**: Multiple sophisticated architectures including scale-free,
//!   hierarchical, modular, and adaptive topologies
//! - **Comprehensive Learning Algorithms**: Ridge regression, LASSO, Elastic Net, RLS, Kalman
//!   filtering, neural network readouts, and meta-learning approaches
//! - **Time Series Modeling**: ARIMA-like capabilities, nonlinear autoregressive models,
//!   memory kernels, and temporal correlation analysis
//! - **Real-time Adaptation**: Online learning algorithms with forgetting factors, plasticity
//!   mechanisms, and adaptive reservoir modification
//! - **Memory Analysis Tools**: Quantum memory capacity estimation, nonlinear memory measures,
//!   temporal information processing capacity, and correlation analysis
//! - **Hardware-aware Optimization**: Device-specific compilation, noise-aware training,
//!   error mitigation, and platform-specific optimizations
//! - **Comprehensive Benchmarking**: Multiple datasets, statistical significance testing,
//!   comparative analysis, and performance validation frameworks
//! - **Advanced Quantum Dynamics**: Unitary evolution, open system dynamics, NISQ simulation,
//!   adiabatic processes, and quantum error correction integration

mod analysis;
mod config;
mod reservoir;
mod state;
mod time_series;
mod types;

// Re-export all public types
pub use analysis::*;
pub use config::*;
pub use reservoir::*;
pub use state::*;
pub use time_series::*;
pub use types::*;

use crate::error::Result;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Comprehensive benchmark for enhanced quantum reservoir computing
pub fn benchmark_enhanced_quantum_reservoir_computing() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    // Test different enhanced reservoir configurations
    let configs = vec![
        QuantumReservoirConfig {
            num_qubits: 6,
            architecture: QuantumReservoirArchitecture::RandomCircuit,
            learning_config: AdvancedLearningConfig {
                algorithm: LearningAlgorithm::Ridge,
                ..Default::default()
            },
            ..Default::default()
        },
        QuantumReservoirConfig {
            num_qubits: 8,
            architecture: QuantumReservoirArchitecture::ScaleFree,
            learning_config: AdvancedLearningConfig {
                algorithm: LearningAlgorithm::LASSO,
                ..Default::default()
            },
            ..Default::default()
        },
        QuantumReservoirConfig {
            num_qubits: 6,
            architecture: QuantumReservoirArchitecture::HierarchicalModular,
            learning_config: AdvancedLearningConfig {
                algorithm: LearningAlgorithm::RecursiveLeastSquares,
                ..Default::default()
            },
            memory_config: MemoryAnalysisConfig {
                enable_capacity_estimation: true,
                enable_nonlinear: true,
                ..Default::default()
            },
            ..Default::default()
        },
        QuantumReservoirConfig {
            num_qubits: 8,
            architecture: QuantumReservoirArchitecture::Grid,
            dynamics: ReservoirDynamics::Floquet,
            input_encoding: InputEncoding::Angle,
            output_measurement: OutputMeasurement::TemporalCorrelations,
            ..Default::default()
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        let start = std::time::Instant::now();

        let mut qrc = QuantumReservoirComputerEnhanced::new(config)?;

        // Generate enhanced test data
        let training_data = ReservoirTrainingData::new(
            (0..200)
                .map(|i| {
                    Array1::from_vec(vec![
                        (f64::from(i) * 0.1).sin(),
                        (f64::from(i) * 0.1).cos(),
                        (f64::from(i) * 0.05).sin() * (f64::from(i) * 0.2).cos(),
                    ])
                })
                .collect(),
            (0..200)
                .map(|i| Array1::from_vec(vec![f64::from(i).mul_add(0.1, 1.0).sin()]))
                .collect(),
            (0..200).map(|i| f64::from(i) * 0.1).collect(),
        );

        // Train and test
        let training_result = qrc.train(&training_data)?;

        let time = start.elapsed().as_secs_f64() * 1000.0;
        results.insert(format!("enhanced_config_{i}"), time);

        // Add enhanced performance metrics
        let metrics = qrc.get_metrics();
        results.insert(
            format!("enhanced_config_{i}_accuracy"),
            metrics.prediction_accuracy,
        );
        results.insert(
            format!("enhanced_config_{i}_memory_capacity"),
            training_result.memory_capacity,
        );
        results.insert(
            format!("enhanced_config_{i}_nonlinear_capacity"),
            training_result.nonlinear_capacity,
        );
        results.insert(
            format!("enhanced_config_{i}_processing_capacity"),
            training_result.processing_capacity,
        );
        results.insert(
            format!("enhanced_config_{i}_quantum_advantage"),
            metrics.quantum_advantage,
        );
        results.insert(
            format!("enhanced_config_{i}_efficiency"),
            metrics.reservoir_efficiency,
        );

        // Memory analysis results
        let memory_analyzer = qrc.get_memory_analysis();
        if let Some(&linear_capacity) = memory_analyzer.capacity_estimates.get("linear") {
            results.insert(
                format!("enhanced_config_{i}_linear_memory"),
                linear_capacity,
            );
        }
        if let Some(&total_capacity) = memory_analyzer.capacity_estimates.get("total") {
            results.insert(format!("enhanced_config_{i}_total_memory"), total_capacity);
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_quantum_reservoir_creation() {
        let config = QuantumReservoirConfig::default();
        let qrc = QuantumReservoirComputerEnhanced::new(config);
        assert!(qrc.is_ok());
    }

    #[test]
    fn test_enhanced_reservoir_state_creation() {
        let state = QuantumReservoirState::new(3, 10);
        assert_eq!(state.state_vector.len(), 8); // 2^3
        assert_eq!(state.state_history.capacity(), 10);
        assert_eq!(state.time_index, 0);
        assert!(state.memory_metrics.total_capacity >= 0.0);
    }

    #[test]
    fn test_enhanced_input_processing() {
        let config = QuantumReservoirConfig {
            num_qubits: 3,
            evolution_steps: 2,
            ..Default::default()
        };
        let mut qrc = QuantumReservoirComputerEnhanced::new(config).expect("Failed to create QRC");

        let input = Array1::from_vec(vec![0.5, 0.3, 0.8]);
        let result = qrc.process_input(&input);
        assert!(result.is_ok());

        let features = result.expect("Failed to process input");
        assert!(!features.is_empty());
    }

    #[test]
    fn test_enhanced_architectures() {
        let architectures = vec![
            QuantumReservoirArchitecture::RandomCircuit,
            QuantumReservoirArchitecture::SpinChain,
            QuantumReservoirArchitecture::ScaleFree,
            QuantumReservoirArchitecture::HierarchicalModular,
            QuantumReservoirArchitecture::Ring,
            QuantumReservoirArchitecture::Grid,
        ];

        for arch in architectures {
            let config = QuantumReservoirConfig {
                num_qubits: 4,
                architecture: arch,
                evolution_steps: 2,
                ..Default::default()
            };

            let qrc = QuantumReservoirComputerEnhanced::new(config);
            assert!(qrc.is_ok(), "Failed for architecture: {arch:?}");
        }
    }

    #[test]
    fn test_advanced_learning_algorithms() {
        let algorithms = vec![
            LearningAlgorithm::Ridge,
            LearningAlgorithm::LASSO,
            LearningAlgorithm::ElasticNet,
            LearningAlgorithm::RecursiveLeastSquares,
        ];

        for algorithm in algorithms {
            let config = QuantumReservoirConfig {
                num_qubits: 3,
                learning_config: AdvancedLearningConfig {
                    algorithm,
                    ..Default::default()
                },
                ..Default::default()
            };

            let qrc = QuantumReservoirComputerEnhanced::new(config);
            assert!(qrc.is_ok(), "Failed for algorithm: {algorithm:?}");
        }
    }

    #[test]
    fn test_enhanced_encoding_methods() {
        let encodings = vec![
            InputEncoding::Amplitude,
            InputEncoding::Phase,
            InputEncoding::BasisState,
            InputEncoding::Angle,
        ];

        for encoding in encodings {
            let config = QuantumReservoirConfig {
                num_qubits: 3,
                input_encoding: encoding,
                ..Default::default()
            };
            let mut qrc =
                QuantumReservoirComputerEnhanced::new(config).expect("Failed to create QRC");

            let input = Array1::from_vec(vec![0.5, 0.3]);
            let result = qrc.encode_input(&input);
            assert!(result.is_ok(), "Failed for encoding: {encoding:?}");
        }
    }

    #[test]
    fn test_enhanced_measurement_strategies() {
        let measurements = vec![
            OutputMeasurement::PauliExpectation,
            OutputMeasurement::Probability,
            OutputMeasurement::Correlations,
            OutputMeasurement::Entanglement,
            OutputMeasurement::QuantumFisherInformation,
            OutputMeasurement::Variance,
            OutputMeasurement::QuantumCoherence,
            OutputMeasurement::Purity,
            OutputMeasurement::TemporalCorrelations,
        ];

        for measurement in measurements {
            let config = QuantumReservoirConfig {
                num_qubits: 3,
                output_measurement: measurement,
                ..Default::default()
            };

            let qrc = QuantumReservoirComputerEnhanced::new(config);
            assert!(qrc.is_ok(), "Failed for measurement: {measurement:?}");
        }
    }

    #[test]
    fn test_enhanced_reservoir_dynamics() {
        let dynamics = vec![
            ReservoirDynamics::Unitary,
            ReservoirDynamics::Open,
            ReservoirDynamics::NISQ,
            ReservoirDynamics::Floquet,
        ];

        for dynamic in dynamics {
            let config = QuantumReservoirConfig {
                num_qubits: 3,
                dynamics: dynamic,
                evolution_steps: 1,
                ..Default::default()
            };

            let mut qrc =
                QuantumReservoirComputerEnhanced::new(config).expect("Failed to create QRC");
            let result = qrc.evolve_reservoir();
            assert!(result.is_ok(), "Failed for dynamics: {dynamic:?}");
        }
    }

    #[test]
    fn test_memory_analysis() {
        let config = QuantumReservoirConfig {
            num_qubits: 4,
            memory_config: MemoryAnalysisConfig {
                enable_capacity_estimation: true,
                enable_nonlinear: true,
                enable_ipc: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let qrc = QuantumReservoirComputerEnhanced::new(config).expect("Failed to create QRC");
        let memory_analyzer = qrc.get_memory_analysis();

        assert!(memory_analyzer.config.enable_capacity_estimation);
        assert!(memory_analyzer.config.enable_nonlinear);
        assert!(memory_analyzer.config.enable_ipc);
    }

    #[test]
    fn test_enhanced_training_data() {
        let training_data = ReservoirTrainingData::new(
            vec![
                Array1::from_vec(vec![0.1, 0.2]),
                Array1::from_vec(vec![0.3, 0.4]),
            ],
            vec![Array1::from_vec(vec![0.5]), Array1::from_vec(vec![0.6])],
            vec![0.0, 1.0],
        )
        .with_features(vec![
            Array1::from_vec(vec![0.7, 0.8]),
            Array1::from_vec(vec![0.9, 1.0]),
        ])
        .with_labels(vec![0, 1])
        .with_weights(vec![1.0, 1.0]);

        assert_eq!(training_data.len(), 2);
        assert!(training_data.features.is_some());
        assert!(training_data.labels.is_some());
        assert!(training_data.sample_weights.is_some());

        let (train, test) = training_data.train_test_split(0.5);
        assert_eq!(train.len(), 1);
        assert_eq!(test.len(), 1);
    }

    #[test]
    fn test_time_series_predictor() {
        let config = TimeSeriesConfig::default();
        let predictor = TimeSeriesPredictor::new(&config);

        assert_eq!(predictor.arima_params.ar_coeffs.len(), config.ar_order);
        assert_eq!(predictor.arima_params.ma_coeffs.len(), config.ma_order);
        assert_eq!(predictor.nar_state.order, config.nar_order);
    }

    #[test]
    fn test_enhanced_metrics_tracking() {
        let config = QuantumReservoirConfig::default();
        let qrc = QuantumReservoirComputerEnhanced::new(config).expect("Failed to create QRC");

        let metrics = qrc.get_metrics();
        assert_eq!(metrics.training_examples, 0);
        assert_eq!(metrics.prediction_accuracy, 0.0);
        assert_eq!(metrics.memory_capacity, 0.0);
        assert_eq!(metrics.nonlinear_memory_capacity, 0.0);
        assert_eq!(metrics.quantum_advantage, 0.0);
    }

    #[test]
    fn test_enhanced_feature_sizes() {
        let measurements = vec![
            (OutputMeasurement::PauliExpectation, 24), // 8 qubits * 3 Pauli
            (OutputMeasurement::QuantumFisherInformation, 8), // 8 qubits
            (OutputMeasurement::Variance, 24),         // 8 qubits * 3 Pauli
            (OutputMeasurement::Purity, 1),            // Single value
        ];

        for (measurement, expected_size) in measurements {
            let config = QuantumReservoirConfig {
                num_qubits: 8,
                output_measurement: measurement,
                ..Default::default()
            };

            let feature_size = QuantumReservoirComputerEnhanced::calculate_feature_size(&config);
            assert_eq!(
                feature_size, expected_size,
                "Feature size mismatch for {:?}: expected {}, got {}",
                measurement, expected_size, feature_size
            );
        }
    }
}
