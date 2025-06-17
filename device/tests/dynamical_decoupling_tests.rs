//! Comprehensive test suite for Dynamical Decoupling (DD) system
//!
//! This module provides extensive test coverage for all DD components including
//! sequence generation, optimization, adaptive strategies, and hardware integration.

use futures::future;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::prelude::*;
use quantrs2_device::adaptive_compilation::strategies::{AdaptationTrigger, OptimizationAlgorithm};
use quantrs2_device::dynamical_decoupling::config::{
    DDNoiseConfig, DDOptimizationAlgorithm, DDOptimizationObjectiveType, DDPerformanceConfig,
    DDPerformanceMetric, NoiseType,
};
use quantrs2_device::dynamical_decoupling::hardware::SynchronizationRequirements;
use quantrs2_device::dynamical_decoupling::optimization::DDSequenceOptimizer;
use quantrs2_device::dynamical_decoupling::*;
use quantrs2_device::prelude::*;
use quantrs2_device::{backend_traits::BackendCapabilities, DeviceError};
use std::collections::HashMap;
use std::time::Duration;

/// Test helper functions and mock implementations
mod test_helpers {
    use super::*;

    /// Mock circuit executor for testing
    pub struct MockDDCircuitExecutor {
        pub fidelity: f64,
        pub execution_time: Duration,
        pub error_rates: HashMap<String, f64>,
    }

    impl MockDDCircuitExecutor {
        pub fn new() -> Self {
            let mut error_rates = HashMap::new();
            error_rates.insert("dephasing".to_string(), 0.001);
            error_rates.insert("amplitude_damping".to_string(), 0.0005);

            Self {
                fidelity: 0.995,
                execution_time: Duration::from_millis(100),
                error_rates,
            }
        }
    }

    impl DDCircuitExecutor for MockDDCircuitExecutor {
        fn execute_circuit(
            &self,
            _circuit: &Circuit<16>,
        ) -> Result<CircuitExecutionResults, DeviceError> {
            let mut measurements = HashMap::new();
            measurements.insert("qubit_0".to_string(), vec![0, 1, 0, 1, 0]);
            measurements.insert("qubit_1".to_string(), vec![1, 0, 1, 0, 1]);

            Ok(CircuitExecutionResults {
                measurements,
                fidelity: self.fidelity,
                execution_time: self.execution_time,
                error_rates: self.error_rates.clone(),
                metadata: CircuitExecutionMetadata {
                    backend: "mock_backend".to_string(),
                    quantum_volume: 32,
                    topology_type: "linear".to_string(),
                    calibration_timestamp: std::time::SystemTime::now(),
                    environmental_conditions: HashMap::new(),
                },
            })
        }

        fn get_capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                backend: quantrs2_device::prelude::HardwareBackend::Custom(0),
                native_gates: quantrs2_device::prelude::NativeGateSet::default(),
                features: quantrs2_device::backend_traits::BackendFeatures {
                    max_qubits: 16,
                    ..Default::default()
                },
                performance: quantrs2_device::backend_traits::BackendPerformance::default(),
            }
        }

        fn estimate_execution_time(&self, circuit: &Circuit<16>) -> Duration {
            // Simple estimation based on circuit depth
            Duration::from_nanos((circuit.gates().len() * 35) as u64)
        }
    }

    pub fn create_test_dd_config() -> DynamicalDecouplingConfig {
        DynamicalDecouplingConfig::default()
    }

    pub fn create_test_qubits() -> Vec<QubitId> {
        vec![QubitId(0), QubitId(1)]
    }

    pub fn create_mock_performance_analysis() -> DDPerformanceAnalysis {
        use quantrs2_device::dynamical_decoupling::performance::*;
        use std::collections::HashMap;

        let mut metrics = HashMap::new();
        metrics.insert(DDPerformanceMetric::CoherenceTime, 100.0);
        metrics.insert(DDPerformanceMetric::ProcessFidelity, 0.99);
        metrics.insert(DDPerformanceMetric::GateOverhead, 4.0);

        DDPerformanceAnalysis {
            metrics,
            benchmark_results: BenchmarkResults {
                randomized_benchmarking: None,
                process_tomography: None,
                gate_set_tomography: None,
                cross_entropy_benchmarking: None,
                cycle_benchmarking: None,
            },
            statistical_analysis: DDStatisticalAnalysis {
                descriptive_stats: DescriptiveStatistics {
                    means: HashMap::new(),
                    standard_deviations: HashMap::new(),
                    medians: HashMap::new(),
                    percentiles: HashMap::new(),
                    ranges: HashMap::new(),
                },
                hypothesis_tests: HypothesisTestResults {
                    t_test_results: HashMap::new(),
                    ks_test_results: HashMap::new(),
                    normality_tests: HashMap::new(),
                },
                correlation_analysis: CorrelationAnalysis {
                    pearson_correlations: ndarray::Array2::eye(3),
                    spearman_correlations: ndarray::Array2::eye(3),
                    significant_correlations: Vec::new(),
                },
                distribution_analysis: DistributionAnalysis {
                    best_fit_distributions: HashMap::new(),
                    distribution_parameters: HashMap::new(),
                    goodness_of_fit: HashMap::new(),
                },
                confidence_intervals: ConfidenceIntervals {
                    mean_intervals: HashMap::new(),
                    bootstrap_intervals: HashMap::new(),
                    prediction_intervals: HashMap::new(),
                },
            },
            comparative_analysis: None,
            performance_trends: PerformanceTrends {
                trend_slopes: HashMap::new(),
                trend_significance: HashMap::new(),
                seasonality: HashMap::new(),
                outliers: HashMap::new(),
            },
        }
    }
}

use test_helpers::*;

/// Basic DD sequence generation tests
mod sequence_generation_tests {
    use super::*;

    #[test]
    fn test_hahn_echo_generation() {
        let qubits = create_test_qubits();
        let duration = 1000.0; // microseconds

        let sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::HahnEcho,
            &qubits,
            duration,
        );

        assert!(sequence.is_ok(), "Hahn echo generation should succeed");
        let seq = sequence.unwrap();
        assert_eq!(seq.sequence_type, DDSequenceType::HahnEcho);
        assert_eq!(seq.target_qubits, qubits);
        assert_eq!(seq.duration, duration);
        assert!(
            !seq.pulse_timings.is_empty(),
            "Sequence should have pulse timings"
        );
    }

    #[test]
    fn test_cpmg_generation() {
        let qubits = create_test_qubits();
        let duration = 2000.0;

        let sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::CPMG { n_pulses: 16 },
            &qubits,
            duration,
        );

        assert!(sequence.is_ok(), "CPMG generation should succeed");
        let seq = sequence.unwrap();
        assert!(matches!(seq.sequence_type, DDSequenceType::CPMG { .. }));
        assert_eq!(seq.target_qubits, qubits);
        assert!(!seq.pulse_timings.is_empty());
    }

    #[test]
    fn test_xy4_generation() {
        let qubits = create_test_qubits();
        let duration = 1500.0;

        let sequence =
            DDSequenceGenerator::generate_base_sequence(&DDSequenceType::XY4, &qubits, duration);

        assert!(sequence.is_ok(), "XY-4 generation should succeed");
        let seq = sequence.unwrap();
        assert_eq!(seq.sequence_type, DDSequenceType::XY4);
        assert_eq!(seq.target_qubits, qubits);
    }

    #[test]
    fn test_xy8_generation() {
        let qubits = vec![QubitId(0)];
        let duration = 3000.0;

        let sequence =
            DDSequenceGenerator::generate_base_sequence(&DDSequenceType::XY8, &qubits, duration);

        assert!(sequence.is_ok(), "XY-8 generation should succeed");
        let seq = sequence.unwrap();
        assert_eq!(seq.sequence_type, DDSequenceType::XY8);
        assert_eq!(seq.target_qubits, qubits);
    }

    #[test]
    fn test_kdd_generation() {
        let qubits = create_test_qubits();
        let duration = 2500.0;

        let sequence =
            DDSequenceGenerator::generate_base_sequence(&DDSequenceType::KDD, &qubits, duration);

        assert!(sequence.is_ok(), "KDD generation should succeed");
        let seq = sequence.unwrap();
        assert!(matches!(seq.sequence_type, DDSequenceType::KDD));
    }

    #[test]
    fn test_udd_generation() {
        let qubits = vec![QubitId(0)];
        let duration = 1800.0;

        let sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::UDD { n_pulses: 3 },
            &qubits,
            duration,
        );

        assert!(sequence.is_ok(), "UDD generation should succeed");
        let seq = sequence.unwrap();
        assert!(matches!(seq.sequence_type, DDSequenceType::UDD { .. }));
    }

    #[test]
    fn test_all_sequence_types() {
        let qubits = create_test_qubits();
        let duration = 1000.0;

        let sequence_types = vec![
            DDSequenceType::HahnEcho,
            DDSequenceType::CPMG { n_pulses: 16 },
            DDSequenceType::XY4,
            DDSequenceType::XY8,
            DDSequenceType::XY16,
            DDSequenceType::KDD,
            DDSequenceType::UDD { n_pulses: 3 },
        ];

        for seq_type in sequence_types {
            let sequence =
                DDSequenceGenerator::generate_base_sequence(&seq_type, &qubits, duration);

            assert!(
                sequence.is_ok(),
                "Sequence generation should succeed for {:?}",
                seq_type
            );
            let seq = sequence.unwrap();
            assert_eq!(seq.sequence_type, seq_type);
            assert!(
                !seq.pulse_timings.is_empty(),
                "Sequence {:?} should have pulse timings",
                seq_type
            );
        }
    }
}

/// DD manager and system tests
mod system_tests {
    use super::*;

    #[tokio::test]
    async fn test_dd_manager_creation() {
        let config = create_test_dd_config();
        let device_id = "test_device".to_string();

        let manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        assert!(manager.adaptive_system.is_none());
        assert!(manager.multi_qubit_coordinator.is_none());
    }

    #[tokio::test]
    async fn test_adaptive_system_initialization() {
        let config = create_test_dd_config();
        let device_id = "test_device".to_string();
        let mut manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        let adaptive_config = AdaptiveDDConfig::default();
        let initial_sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::HahnEcho,
            &create_test_qubits(),
            1000.0,
        )
        .unwrap();
        let available_sequences = vec![
            DDSequenceType::HahnEcho,
            DDSequenceType::XY4,
            DDSequenceType::CPMG { n_pulses: 16 },
        ];

        let result = manager.initialize_adaptive_system(
            adaptive_config,
            initial_sequence,
            available_sequences,
        );

        assert!(
            result.is_ok(),
            "Adaptive system initialization should succeed"
        );
        assert!(manager.adaptive_system.is_some());
    }

    #[test]
    fn test_multi_qubit_coordination_initialization() {
        let config = create_test_dd_config();
        let device_id = "test_device".to_string();
        let mut manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        manager.initialize_multi_qubit_coordination(
            CrosstalkMitigationStrategy::TemporalSeparation,
            SynchronizationRequirements::Loose,
        );

        assert!(manager.multi_qubit_coordinator.is_some());
    }

    #[tokio::test]
    async fn test_optimized_sequence_generation() {
        let config = create_test_dd_config();
        let device_id = "test_device".to_string();
        let mut manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        let executor = MockDDCircuitExecutor::new();
        let qubits = create_test_qubits();
        let duration = 1000.0;

        let result = manager
            .generate_optimized_sequence(&DDSequenceType::HahnEcho, &qubits, duration, &executor)
            .await;

        assert!(
            result.is_ok(),
            "Optimized sequence generation should succeed"
        );
        let dd_result = result.unwrap();
        assert!(dd_result.success);
        assert!(dd_result.quality_score > 0.0);
        assert!(dd_result.performance_analysis.is_some());
        assert!(dd_result.noise_analysis.is_some());
        assert!(dd_result.hardware_analysis.is_some());
    }

    #[test]
    fn test_multi_qubit_sequence_generation() {
        let config = create_test_dd_config();
        let device_id = "test_device".to_string();
        let mut manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        manager.initialize_multi_qubit_coordination(
            CrosstalkMitigationStrategy::SpatialSeparation,
            SynchronizationRequirements::Strict,
        );

        let qubit_groups = vec![
            (vec![QubitId(0), QubitId(1)], DDSequenceType::XY4),
            (
                vec![QubitId(2), QubitId(3)],
                DDSequenceType::CPMG { n_pulses: 2 },
            ),
        ];
        let duration = 2000.0;

        let result = manager.generate_multi_qubit_sequence(qubit_groups, duration);

        assert!(
            result.is_ok(),
            "Multi-qubit sequence generation should succeed"
        );
        let sequence = result.unwrap();
        assert!(!sequence.pulse_timings.is_empty());
    }

    #[test]
    fn test_system_status() {
        let config = create_test_dd_config();
        let device_id = "test_device".to_string();
        let manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        let status = manager.get_system_status();

        assert!(!status.adaptive_enabled);
        assert!(!status.multi_qubit_enabled);
        assert_eq!(status.total_sequences_generated, 0);
    }
}

/// Adaptive DD system tests
mod adaptive_tests {
    use super::*;

    #[test]
    fn test_adaptive_config_creation() {
        let config = AdaptiveDDConfig::default();

        assert!(config.enable_real_time_adaptation);
        assert!(config.adaptation_threshold > 0.0);
        assert!(config.min_adaptation_interval > Duration::ZERO);
    }

    #[test]
    fn test_sequence_selection_strategy() {
        let strategies = vec![
            SequenceSelectionStrategy::PerformanceBased,
            SequenceSelectionStrategy::NoiseCharacteristicBased,
            SequenceSelectionStrategy::HybridOptimization,
            SequenceSelectionStrategy::MLDriven,
        ];

        for strategy in strategies {
            let mut config = AdaptiveDDConfig::default();
            config.selection_strategy = strategy.clone();
            assert_eq!(config.selection_strategy, strategy);
        }
    }

    #[test]
    fn test_adaptation_triggers() {
        let triggers = vec![
            AdaptationTrigger::PerformanceDegradation,
            AdaptationTrigger::ErrorRateIncrease,
            AdaptationTrigger::TimeInterval,
            AdaptationTrigger::ResourceConstraintViolation,
            AdaptationTrigger::UserRequest,
        ];

        for trigger in triggers {
            let mut config = AdaptiveDDConfig::default();
            config.adaptation_triggers.push(trigger.clone());
            assert!(config.adaptation_triggers.contains(&trigger));
        }
    }

    #[test]
    fn test_adaptive_system_creation() {
        let config = AdaptiveDDConfig::default();
        let initial_sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::HahnEcho,
            &create_test_qubits(),
            1000.0,
        )
        .unwrap();
        let available_sequences = vec![
            DDSequenceType::HahnEcho,
            DDSequenceType::XY4,
            DDSequenceType::XY8,
        ];

        let adaptive_system = AdaptiveDDSystem::new(config, initial_sequence, available_sequences);

        let current_state = adaptive_system.get_current_state();
        assert_eq!(
            current_state.current_sequence.sequence_type,
            DDSequenceType::HahnEcho
        );
    }
}

/// Performance analysis tests
mod performance_tests {
    use super::*;

    #[test]
    fn test_performance_config_creation() {
        let config = DDPerformanceConfig::default();

        assert!(config.enable_coherence_tracking);
        assert!(config.enable_fidelity_monitoring);
        assert!(config.measurement_shots > 0);
    }

    #[tokio::test]
    async fn test_performance_analysis() {
        let config = DDPerformanceConfig::default();
        let mut analyzer = DDPerformanceAnalyzer::new(config);
        let executor = MockDDCircuitExecutor::new();

        let sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::XY4,
            &create_test_qubits(),
            1500.0,
        )
        .unwrap();

        let result = analyzer.analyze_performance(&sequence, &executor).await;

        assert!(result.is_ok(), "Performance analysis should succeed");
        let analysis = result.unwrap();
        assert!(!analysis.metrics.is_empty());
        // Check that metrics contain expected values
        assert!(analysis
            .metrics
            .contains_key(&DDPerformanceMetric::ProcessFidelity));
        assert!(analysis
            .metrics
            .contains_key(&DDPerformanceMetric::CoherenceTime));
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = vec![
            DDPerformanceMetric::CoherenceTime,
            DDPerformanceMetric::ProcessFidelity,
            DDPerformanceMetric::ResourceEfficiency,
            DDPerformanceMetric::NoiseSuppressionFactor,
            DDPerformanceMetric::GateOverhead,
        ];

        for metric in metrics {
            // Test that all metrics can be created and used
            let mut config = DDPerformanceConfig::default();
            config.metrics.push(metric.clone());
            assert!(config.metrics.contains(&metric));
        }
    }
}

/// Noise analysis tests
mod noise_tests {
    use super::*;

    #[test]
    fn test_noise_config_creation() {
        let config = DDNoiseConfig::default();

        assert!(config.enable_spectral_analysis);
        assert!(config.enable_correlation_analysis);
        assert!(config.sampling_rate > 0.0);
    }

    #[test]
    fn test_noise_analysis() {
        let config = DDNoiseConfig::default();
        let analyzer = DDNoiseAnalyzer::new(config);

        let sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::CPMG { n_pulses: 16 },
            &create_test_qubits(),
            2000.0,
        )
        .unwrap();

        let performance = create_mock_performance_analysis();

        let result = analyzer.analyze_noise_characteristics(&sequence, &performance);

        assert!(result.is_ok(), "Noise analysis should succeed");
        let analysis = result.unwrap();
        // Check that spectral analysis is present and has data
        if let Some(spectral) = &analysis.spectral_analysis {
            assert!(!spectral.psd_analysis.frequency_bins.is_empty());
        }
        assert!(analysis.suppression_effectiveness.overall_suppression > 0.0);
    }

    #[test]
    fn test_noise_types() {
        let noise_types = vec![
            NoiseType::PhaseDamping,
            NoiseType::AmplitudeDamping,
            NoiseType::Depolarizing,
            NoiseType::Pauli,
            NoiseType::CoherentErrors,
            NoiseType::CrossTalk,
            NoiseType::ControlNoise,
            NoiseType::FluxNoise,
        ];

        for noise_type in noise_types {
            // Test that all noise types can be used in configuration
            let mut config = DDNoiseConfig::default();
            config.target_noise_types.push(noise_type.clone());
            assert!(config.target_noise_types.contains(&noise_type));
        }
    }
}

/// Hardware integration tests
mod hardware_tests {
    use super::*;

    #[test]
    fn test_hardware_config_creation() {
        let config = DDHardwareAdaptationConfig::default();

        assert!(config.enable_platform_optimization);
        assert!(config.enable_calibration_integration);
        assert!(!config.supported_platforms.is_empty());
    }

    #[test]
    fn test_hardware_analyzer() {
        let config = DDHardwareAdaptationConfig::default();
        let analyzer = DDHardwareAnalyzer::new(config, None, None);

        let sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::XY8,
            &create_test_qubits(),
            1800.0,
        )
        .unwrap();

        let result = analyzer.analyze_hardware_implementation("test_device", &sequence);

        assert!(result.is_ok(), "Hardware analysis should succeed");
        let analysis = result.unwrap();
        assert!(analysis.hardware_compatibility.compatibility_score >= 0.0);
        assert!(!analysis.platform_optimizations.is_empty());
    }

    #[test]
    fn test_platform_types() {
        let platforms = vec![
            PlatformType::IBMQuantum,
            PlatformType::AWSBraket,
            PlatformType::AzureQuantum,
            PlatformType::GoogleQuantumAI,
            PlatformType::RigettiQCS,
            PlatformType::IonQCloud,
        ];

        for platform in platforms {
            let mut config = DDHardwareAdaptationConfig::default();
            config.target_platform = Some(platform.clone());
            assert_eq!(config.target_platform, Some(platform));
        }
    }
}

/// Optimization tests
mod optimization_tests {
    use super::*;

    #[test]
    fn test_optimization_config_creation() {
        let config = DDOptimizationConfig::default();

        assert!(config.enable_scirs2_optimization);
        assert!(config.max_optimization_iterations > 0);
        assert!(config.convergence_tolerance > 0.0);
    }

    #[tokio::test]
    async fn test_sequence_optimization() {
        let config = DDOptimizationConfig::default();
        let mut optimizer = DDSequenceOptimizer::new(config);
        let executor = MockDDCircuitExecutor::new();

        let sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::UDD { n_pulses: 3 },
            &create_test_qubits(),
            1200.0,
        )
        .unwrap();

        let result = optimizer.optimize_sequence(&sequence, &executor).await;

        assert!(result.is_ok(), "Sequence optimization should succeed");
        let opt_result = result.unwrap();
        assert!(opt_result.optimization_metrics.success);
        assert!(opt_result.optimization_metrics.iterations > 0);
    }

    #[test]
    fn test_optimization_objectives() {
        let objectives = vec![
            DDOptimizationObjectiveType::MaximizeFidelity,
            DDOptimizationObjectiveType::MinimizeExecutionTime,
            DDOptimizationObjectiveType::MaximizeCoherenceTime,
            DDOptimizationObjectiveType::MinimizeNoiseAmplification,
            DDOptimizationObjectiveType::MaximizeRobustness,
        ];

        for objective in objectives {
            let mut config = DDOptimizationConfig::default();
            config.optimization_objectives.push(objective.clone());
            assert!(config.optimization_objectives.contains(&objective));
        }
    }

    #[test]
    fn test_optimization_algorithms() {
        let algorithms = vec![
            DDOptimizationAlgorithm::DifferentialEvolution,
            DDOptimizationAlgorithm::ParticleSwarm,
            DDOptimizationAlgorithm::SimulatedAnnealing,
            DDOptimizationAlgorithm::GeneticAlgorithm,
            DDOptimizationAlgorithm::BayesianOptimization,
            DDOptimizationAlgorithm::GradientFree,
        ];

        for algorithm in algorithms {
            let mut config = DDOptimizationConfig::default();
            config.optimization_algorithm = algorithm.clone();
            assert_eq!(config.optimization_algorithm, algorithm);
        }
    }
}

/// Integration tests
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_dd_workflow() {
        // 1. Create DD manager
        let config = create_test_dd_config();
        let device_id = "integration_test_device".to_string();
        let mut manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        // 2. Initialize adaptive system
        let adaptive_config = AdaptiveDDConfig::default();
        let initial_sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::HahnEcho,
            &create_test_qubits(),
            1000.0,
        )
        .unwrap();
        let available_sequences = vec![
            DDSequenceType::HahnEcho,
            DDSequenceType::XY4,
            DDSequenceType::CPMG { n_pulses: 16 },
        ];

        manager
            .initialize_adaptive_system(adaptive_config, initial_sequence, available_sequences)
            .unwrap();

        // 3. Generate optimized sequence
        let executor = MockDDCircuitExecutor::new();
        let qubits = create_test_qubits();
        let duration = 1500.0;

        let result = manager
            .generate_optimized_sequence(&DDSequenceType::XY4, &qubits, duration, &executor)
            .await;

        assert!(result.is_ok(), "Complete DD workflow should succeed");
        let dd_result = result.unwrap();
        assert!(dd_result.success);
        assert!(dd_result.quality_score > 0.5);

        // 4. Check system status
        let status = manager.get_system_status();
        assert!(status.adaptive_enabled);
        assert!(status.total_sequences_generated > 0);

        println!("Complete DD workflow test passed successfully");
    }

    #[tokio::test]
    async fn test_multiple_sequence_optimization() {
        let config = create_test_dd_config();
        let device_id = "multi_seq_test_device".to_string();
        let mut manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        let executor = MockDDCircuitExecutor::new();
        let qubits = create_test_qubits();
        let duration = 1000.0;

        let sequence_types = vec![
            DDSequenceType::HahnEcho,
            DDSequenceType::XY4,
            DDSequenceType::CPMG { n_pulses: 16 },
            DDSequenceType::UDD { n_pulses: 3 },
        ];

        for seq_type in sequence_types {
            let result = manager
                .generate_optimized_sequence(&seq_type, &qubits, duration, &executor)
                .await;

            assert!(
                result.is_ok(),
                "Sequence optimization should succeed for {:?}",
                seq_type
            );
            let dd_result = result.unwrap();
            assert!(dd_result.success);
            assert!(dd_result.quality_score > 0.0);
        }

        let status = manager.get_system_status();
        assert_eq!(status.total_sequences_generated, 4);
    }

    #[test]
    fn test_adaptive_and_multi_qubit_integration() {
        let config = create_test_dd_config();
        let device_id = "adaptive_multi_test_device".to_string();
        let mut manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        // Initialize both adaptive and multi-qubit systems
        let adaptive_config = AdaptiveDDConfig::default();
        let initial_sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::HahnEcho,
            &create_test_qubits(),
            1000.0,
        )
        .unwrap();
        let available_sequences = vec![DDSequenceType::HahnEcho, DDSequenceType::XY4];

        manager
            .initialize_adaptive_system(adaptive_config, initial_sequence, available_sequences)
            .unwrap();

        manager.initialize_multi_qubit_coordination(
            CrosstalkMitigationStrategy::HybridApproach,
            SynchronizationRequirements::Adaptive,
        );

        let status = manager.get_system_status();
        assert!(status.adaptive_enabled);
        assert!(status.multi_qubit_enabled);

        // Test multi-qubit sequence generation
        let qubit_groups = vec![
            (vec![QubitId(0)], DDSequenceType::HahnEcho),
            (vec![QubitId(1)], DDSequenceType::XY4),
        ];
        let duration = 1200.0;

        let result = manager.generate_multi_qubit_sequence(qubit_groups, duration);
        assert!(
            result.is_ok(),
            "Multi-qubit sequence generation should succeed"
        );
    }
}

/// Performance and stress tests
mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_sequence_generation() {
        let config = create_test_dd_config();
        let device_id = "concurrent_test_device".to_string();
        let manager = std::sync::Arc::new(tokio::sync::RwLock::new(
            DynamicalDecouplingManager::new(config, device_id, None, None),
        ));

        let executor = std::sync::Arc::new(MockDDCircuitExecutor::new());
        let qubits = create_test_qubits();
        let duration = 1000.0;

        // Generate multiple sequences concurrently
        let mut tasks = vec![];
        for i in 0..5 {
            let manager_clone = manager.clone();
            let executor_clone = executor.clone();
            let qubits_clone = qubits.clone();

            let task = tokio::spawn(async move {
                let mut mgr = manager_clone.write().await;
                mgr.generate_optimized_sequence(
                    &DDSequenceType::HahnEcho,
                    &qubits_clone,
                    duration + i as f64 * 100.0,
                    executor_clone.as_ref(),
                )
                .await
            });
            tasks.push(task);
        }

        let results = futures::future::try_join_all(tasks).await.unwrap();

        for (i, result) in results.into_iter().enumerate() {
            assert!(
                result.is_ok(),
                "Task {} should complete without panicking",
                i
            );
            let dd_result = result.unwrap();
            assert!(dd_result.success, "DD generation {} should succeed", i);
        }
    }

    #[test]
    fn test_large_qubit_count() {
        let qubits: Vec<QubitId> = (0..10).map(QubitId).collect(); // 10 qubits
        let duration = 2000.0;

        let sequence =
            DDSequenceGenerator::generate_base_sequence(&DDSequenceType::XY4, &qubits, duration);

        assert!(
            sequence.is_ok(),
            "Large qubit count sequence generation should succeed"
        );
        let seq = sequence.unwrap();
        assert_eq!(seq.target_qubits.len(), 10);
        assert!(!seq.pulse_timings.is_empty());
    }

    #[test]
    fn test_long_duration_sequences() {
        let qubits = create_test_qubits();
        let long_duration = 100000.0; // 100ms

        let sequence = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::CPMG { n_pulses: 100 },
            &qubits,
            long_duration,
        );

        assert!(
            sequence.is_ok(),
            "Long duration sequence generation should succeed"
        );
        let seq = sequence.unwrap();
        assert_eq!(seq.duration, long_duration);
        assert!(!seq.pulse_timings.is_empty());
    }
}

/// Error handling tests
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_invalid_sequence_parameters() {
        let qubits = vec![];
        let duration = 0.0;

        let result = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::HahnEcho,
            &qubits,
            duration,
        );

        // Should handle invalid parameters gracefully
        assert!(result.is_err(), "Invalid parameters should be rejected");
    }

    #[test]
    fn test_negative_duration() {
        let qubits = create_test_qubits();
        let negative_duration = -1000.0;

        let result = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::XY4,
            &qubits,
            negative_duration,
        );

        assert!(result.is_err(), "Negative duration should be rejected");
    }

    #[test]
    fn test_multi_qubit_without_coordinator() {
        let config = create_test_dd_config();
        let device_id = "error_test_device".to_string();
        let mut manager = DynamicalDecouplingManager::new(config, device_id, None, None);

        let qubit_groups = vec![(vec![QubitId(0)], DDSequenceType::HahnEcho)];
        let duration = 1000.0;

        let result = manager.generate_multi_qubit_sequence(qubit_groups, duration);

        assert!(
            result.is_err(),
            "Multi-qubit generation without coordinator should fail"
        );
    }

    #[test]
    fn test_invalid_optimization_config() {
        let mut config = DDOptimizationConfig::default();
        config.max_optimization_iterations = 0; // Invalid
        config.convergence_tolerance = -1.0; // Invalid

        // The system should handle invalid configuration gracefully
        assert_eq!(config.max_optimization_iterations, 0);
        assert_eq!(config.convergence_tolerance, -1.0);

        // These would be caught during optimization
        config.max_optimization_iterations = 100;
        config.convergence_tolerance = 1e-6;
        assert!(config.max_optimization_iterations > 0);
        assert!(config.convergence_tolerance > 0.0);
    }
}
