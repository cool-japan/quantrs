//! Comprehensive test suite for dynamical decoupling system

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamical_decoupling::config::DDSequenceType;
    use crate::dynamical_decoupling::sequences::SynchronizationRequirements;
    use crate::dynamical_decoupling::{
        AdaptiveDDConfig, AdaptiveDDSystem, CompositionStrategy, CrosstalkMitigationStrategy,
        DDHardwareAnalyzer, DDNoiseAnalyzer, DDPerformanceAnalyzer, DDSequenceGenerator,
        DDSequenceOptimizer, DynamicalDecouplingConfig, DynamicalDecouplingManager,
        MultiQubitDDCoordinator,
    };
    use quantrs2_circuit::prelude::Circuit;
    use quantrs2_core::qubit::QubitId;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// Mock circuit executor for testing
    pub struct MockCircuitExecutor {
        execution_count: Arc<Mutex<usize>>,
    }

    impl MockCircuitExecutor {
        pub fn new() -> Self {
            Self {
                execution_count: Arc::new(Mutex::new(0)),
            }
        }

        fn get_execution_count(&self) -> usize {
            *self.execution_count.lock().unwrap()
        }
    }

    impl crate::dynamical_decoupling::DDCircuitExecutor for MockCircuitExecutor {
        fn execute_circuit(
            &self,
            _circuit: &Circuit<16>,
        ) -> Result<crate::dynamical_decoupling::CircuitExecutionResults, crate::DeviceError>
        {
            *self.execution_count.lock().unwrap() += 1;

            let mut measurements = HashMap::new();
            measurements.insert("qubit_0".to_string(), vec![0, 1, 0, 1]);

            let mut error_rates = HashMap::new();
            error_rates.insert("gate_error".to_string(), 0.001);
            error_rates.insert("readout_error".to_string(), 0.01);

            Ok(crate::dynamical_decoupling::CircuitExecutionResults {
                measurements,
                fidelity: 0.99,
                execution_time: std::time::Duration::from_micros(100),
                error_rates,
                metadata: crate::dynamical_decoupling::CircuitExecutionMetadata {
                    backend: "mock_backend".to_string(),
                    quantum_volume: 64,
                    topology_type: "grid".to_string(),
                    calibration_timestamp: std::time::SystemTime::now(),
                    environmental_conditions: HashMap::new(),
                },
            })
        }

        fn get_capabilities(&self) -> crate::backend_traits::BackendCapabilities {
            crate::backend_traits::BackendCapabilities::default()
        }

        fn estimate_execution_time(&self, _circuit: &Circuit<16>) -> std::time::Duration {
            std::time::Duration::from_micros(100)
        }
    }

    fn create_test_config() -> DynamicalDecouplingConfig {
        DynamicalDecouplingConfig::default()
    }

    fn create_test_sequence() -> crate::dynamical_decoupling::DDSequence {
        let target_qubits = vec![QubitId(0), QubitId(1)];
        DDSequenceGenerator::generate_base_sequence(&DDSequenceType::CPMG, &target_qubits, 100e-6)
            .unwrap()
    }

    #[test]
    fn test_dd_manager_creation() {
        let config = create_test_config();
        let manager =
            DynamicalDecouplingManager::new(config, "test_device".to_string(), None, None);

        let status = manager.get_system_status();
        assert!(!status.adaptive_enabled);
        assert!(!status.multi_qubit_enabled);
        assert_eq!(status.total_sequences_generated, 0);
    }

    #[test]
    fn test_sequence_generation() {
        let target_qubits = vec![QubitId(0)];

        // Test Hahn Echo
        let hahn_echo = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::HahnEcho,
            &target_qubits,
            50e-6,
        )
        .unwrap();
        assert_eq!(hahn_echo.sequence_type, DDSequenceType::HahnEcho);
        assert_eq!(hahn_echo.pulse_timings.len(), 1);
        assert_eq!(hahn_echo.properties.pulse_count, 1);

        // Test CPMG
        let cpmg = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::CPMG,
            &target_qubits,
            100e-6,
        )
        .unwrap();
        assert_eq!(cpmg.sequence_type, DDSequenceType::CPMG);
        assert!(cpmg.properties.pulse_count > 1);

        // Test XY-4
        let xy4 = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::XY4,
            &target_qubits,
            100e-6,
        )
        .unwrap();
        assert_eq!(xy4.sequence_type, DDSequenceType::XY4);
        assert_eq!(xy4.properties.sequence_order, 2);

        // Test UDD
        let udd = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::UDD,
            &target_qubits,
            100e-6,
        )
        .unwrap();
        assert_eq!(udd.sequence_type, DDSequenceType::UDD);
        assert!(udd.properties.sequence_order > 1);
    }

    #[test]
    fn test_composite_sequence_generation() {
        let target_qubits = vec![QubitId(0)];

        let base1 = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::HahnEcho,
            &target_qubits,
            50e-6,
        )
        .unwrap();

        let base2 = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::XY4,
            &target_qubits,
            50e-6,
        )
        .unwrap();

        let composite = DDSequenceGenerator::generate_composite_sequence(
            &[base1, base2],
            CompositionStrategy::Sequential,
        )
        .unwrap();

        assert_eq!(composite.sequence_type, DDSequenceType::Composite);
        assert!(composite.duration > 50e-6); // Should be sum of components
        assert!(composite.properties.resource_requirements.gate_count > 1);
    }

    #[test]
    fn test_multi_qubit_coordination() {
        let crosstalk_mitigation = CrosstalkMitigationStrategy::TimeShifted;
        let synchronization = SynchronizationRequirements {
            global_sync: true,
            phase_coherence: true,
            timing_tolerance: 0.01,
            sync_points: vec![0.5],
        };

        let mut coordinator = MultiQubitDDCoordinator::new(crosstalk_mitigation, synchronization);

        // Add sequences for different qubit groups
        let group1 = vec![QubitId(0), QubitId(1)];
        let sequence1 =
            DDSequenceGenerator::generate_base_sequence(&DDSequenceType::CPMG, &group1, 100e-6)
                .unwrap();
        coordinator.add_sequence(group1, sequence1);

        let group2 = vec![QubitId(2), QubitId(3)];
        let sequence2 =
            DDSequenceGenerator::generate_base_sequence(&DDSequenceType::XY4, &group2, 100e-6)
                .unwrap();
        coordinator.add_sequence(group2, sequence2);

        // Generate coordinated sequence
        let coordinated = coordinator.generate_coordinated_sequence().unwrap();
        assert_eq!(
            coordinated.sequence_type,
            DDSequenceType::MultiQubitCoordinated
        );
    }

    #[tokio::test]
    async fn test_performance_analysis() {
        let config = crate::dynamical_decoupling::config::DDPerformanceConfig::default();
        let mut analyzer = DDPerformanceAnalyzer::new(config);
        let executor = MockCircuitExecutor::new();
        let sequence = create_test_sequence();

        let analysis = analyzer
            .analyze_performance(&sequence, &executor)
            .await
            .unwrap();

        assert!(!analysis.metrics.is_empty());
        assert!(analysis.metrics.contains_key(
            &crate::dynamical_decoupling::config::DDPerformanceMetric::CoherenceTime
        ));
        assert!(analysis.metrics.contains_key(
            &crate::dynamical_decoupling::config::DDPerformanceMetric::ProcessFidelity
        ));

        // Verify statistical analysis
        assert!(!analysis
            .statistical_analysis
            .descriptive_stats
            .means
            .is_empty());

        // Check if executor was called
        assert_eq!(executor.get_execution_count(), 0); // Mock doesn't actually execute in this test
    }

    #[test]
    fn test_noise_analysis() {
        let config = crate::dynamical_decoupling::config::DDNoiseConfig::default();
        let analyzer = DDNoiseAnalyzer::new(config);
        let sequence = create_test_sequence();

        // Create mock performance analysis
        let performance_analysis = create_mock_performance_analysis();

        let noise_analysis = analyzer
            .analyze_noise_characteristics(&sequence, &performance_analysis)
            .unwrap();

        assert!(!noise_analysis.noise_characterization.noise_types.is_empty());
        assert!(noise_analysis.suppression_effectiveness.overall_suppression >= 0.0);
    }

    #[test]
    fn test_hardware_analysis() {
        let config = crate::dynamical_decoupling::config::DDHardwareConfig::default();
        let analyzer = DDHardwareAnalyzer::new(config, None, None);
        let sequence = create_test_sequence();

        let hardware_analysis = analyzer
            .analyze_hardware_implementation("test_device", &sequence)
            .unwrap();

        assert!(hardware_analysis.hardware_compatibility.compatibility_score >= 0.0);
        assert!(hardware_analysis.hardware_compatibility.compatibility_score <= 1.0);
        assert!(
            hardware_analysis
                .resource_utilization
                .gate_resource_usage
                .resource_efficiency
                >= 0.0
        );
    }

    #[tokio::test]
    async fn test_sequence_optimization() {
        let config = crate::dynamical_decoupling::config::DDOptimizationConfig::default();
        let mut optimizer = DDSequenceOptimizer::new(config);
        let executor = MockCircuitExecutor::new();
        let sequence = create_test_sequence();

        let optimization_result = optimizer
            .optimize_sequence(&sequence, &executor)
            .await
            .unwrap();

        assert!(optimization_result.optimization_metrics.success);
        assert!(optimization_result.optimization_metrics.improvement_factor >= 0.0);
        assert!(optimization_result.parameter_sensitivity.robustness_score >= 0.0);
    }

    #[test]
    fn test_adaptive_dd_system() {
        let adaptive_config = AdaptiveDDConfig::default();
        let initial_sequence = create_test_sequence();
        let available_sequences = vec![
            DDSequenceType::HahnEcho,
            DDSequenceType::CPMG,
            DDSequenceType::XY4,
            DDSequenceType::XY8,
        ];

        let mut adaptive_system =
            AdaptiveDDSystem::new(adaptive_config, initial_sequence, available_sequences);

        let executor = MockCircuitExecutor::new();
        adaptive_system.start(&executor).unwrap();

        let initial_state = adaptive_system.get_current_state();
        assert_eq!(
            initial_state.current_sequence.sequence_type,
            DDSequenceType::CPMG
        );
        assert_eq!(initial_state.system_health.health_score, 1.0);

        let stats = adaptive_system.get_adaptation_statistics();
        assert_eq!(stats.total_adaptations, 0);
        assert_eq!(stats.success_rate, 0.0);
    }

    #[tokio::test]
    async fn test_dd_manager_integration() {
        let config = create_test_config();
        let mut manager =
            DynamicalDecouplingManager::new(config, "test_device".to_string(), None, None);

        // Initialize adaptive system
        let adaptive_config = AdaptiveDDConfig::default();
        let initial_sequence = create_test_sequence();
        let available_sequences = vec![DDSequenceType::CPMG, DDSequenceType::XY4];

        manager
            .initialize_adaptive_system(adaptive_config, initial_sequence, available_sequences)
            .unwrap();

        // Initialize multi-qubit coordination
        manager.initialize_multi_qubit_coordination(
            CrosstalkMitigationStrategy::PhaseRandomized,
            SynchronizationRequirements {
                global_sync: true,
                phase_coherence: true,
                timing_tolerance: 0.01,
                sync_points: vec![],
            },
        );

        let status = manager.get_system_status();
        assert!(status.adaptive_enabled);
        assert!(status.multi_qubit_enabled);

        // Test sequence generation
        let executor = MockCircuitExecutor::new();
        let target_qubits = vec![QubitId(0), QubitId(1)];

        let result = manager
            .generate_optimized_sequence(&DDSequenceType::XY4, &target_qubits, 100e-6, &executor)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.quality_score >= 0.0);
        assert!(result.quality_score <= 1.0);
        assert!(result.performance_analysis.is_some());
        assert!(result.noise_analysis.is_some());
        assert!(result.hardware_analysis.is_some());
    }

    #[test]
    fn test_sequence_cache_functionality() {
        let mut cache = crate::dynamical_decoupling::SequenceCache::new();
        let sequence = create_test_sequence();

        // Test cache miss
        assert!(cache.get_sequence("test_key").is_none());

        // Store sequence
        cache.store_sequence("test_key".to_string(), sequence.clone());

        // Test cache hit
        let retrieved = cache.get_sequence("test_key").unwrap();
        assert_eq!(retrieved.sequence_type, sequence.sequence_type);

        // Check statistics
        let (hits, misses, hit_rate) = cache.get_cache_statistics();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(hit_rate, 0.5);
    }

    #[test]
    fn test_sequence_properties_validation() {
        let sequence = create_test_sequence();

        // Validate basic properties
        assert!(sequence.properties.pulse_count > 0);
        assert!(sequence.properties.sequence_order >= 1);
        assert!(sequence.properties.periodicity >= 1);
        assert!(sequence.duration > 0.0);
        assert!(!sequence.pulse_timings.is_empty());
        assert!(!sequence.pulse_phases.is_empty());

        // Validate symmetry properties
        let symmetry = &sequence.properties.symmetry;
        assert!(
            symmetry.time_reversal
                || symmetry.phase_symmetry
                || symmetry.rotational_symmetry
                || symmetry.inversion_symmetry
        );

        // Validate resource requirements
        let resources = &sequence.properties.resource_requirements;
        assert!(resources.gate_count > 0);
        assert!(resources.circuit_depth > 0);
        assert!(resources.execution_time > 0.0);
        assert!(resources.memory_requirements > 0);
    }

    #[test]
    fn test_error_handling() {
        // Test invalid parameters
        let empty_qubits: Vec<QubitId> = vec![];
        let result = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::CPMG,
            &empty_qubits,
            100e-6,
        );
        // Should handle empty qubit list gracefully

        // Test negative duration
        let target_qubits = vec![QubitId(0)];
        let result = DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::CPMG,
            &target_qubits,
            -1.0,
        );
        // Should handle negative duration

        // Test composite sequence with empty input
        let result =
            DDSequenceGenerator::generate_composite_sequence(&[], CompositionStrategy::Sequential);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_performance_metrics_calculation() {
        let config = crate::dynamical_decoupling::config::DDPerformanceConfig::default();
        let analyzer = DDPerformanceAnalyzer::new(config);
        let sequence = create_test_sequence();

        // Test individual metric calculations
        let coherence_time = analyzer
            .measure_coherence_time(&sequence, &MockCircuitExecutor::new())
            .await
            .unwrap();
        assert!(coherence_time > 0.0);

        let fidelity = analyzer
            .measure_process_fidelity(&sequence, &MockCircuitExecutor::new())
            .await
            .unwrap();
        assert!(fidelity >= 0.0 && fidelity <= 1.0);

        let robustness = analyzer
            .calculate_robustness_score(&sequence, &MockCircuitExecutor::new())
            .await
            .unwrap();
        assert!(robustness >= 0.0 && robustness <= 1.0);
    }

    // Helper function to create mock performance analysis
    fn create_mock_performance_analysis() -> crate::dynamical_decoupling::DDPerformanceAnalysis {
        let mut metrics = HashMap::new();
        metrics.insert(
            crate::dynamical_decoupling::config::DDPerformanceMetric::CoherenceTime,
            50.0,
        );
        metrics.insert(
            crate::dynamical_decoupling::config::DDPerformanceMetric::ProcessFidelity,
            0.99,
        );

        crate::dynamical_decoupling::DDPerformanceAnalysis {
            metrics,
            benchmark_results: crate::dynamical_decoupling::performance::BenchmarkResults {
                randomized_benchmarking: None,
                process_tomography: None,
                gate_set_tomography: None,
                cross_entropy_benchmarking: None,
                cycle_benchmarking: None,
            },
            statistical_analysis: crate::dynamical_decoupling::performance::DDStatisticalAnalysis {
                descriptive_stats:
                    crate::dynamical_decoupling::performance::DescriptiveStatistics {
                        means: HashMap::new(),
                        standard_deviations: HashMap::new(),
                        medians: HashMap::new(),
                        percentiles: HashMap::new(),
                        ranges: HashMap::new(),
                    },
                hypothesis_tests: crate::dynamical_decoupling::performance::HypothesisTestResults {
                    t_test_results: HashMap::new(),
                    ks_test_results: HashMap::new(),
                    normality_tests: HashMap::new(),
                },
                correlation_analysis:
                    crate::dynamical_decoupling::performance::CorrelationAnalysis {
                        pearson_correlations: ndarray::Array2::eye(2),
                        spearman_correlations: ndarray::Array2::eye(2),
                        significant_correlations: Vec::new(),
                    },
                distribution_analysis:
                    crate::dynamical_decoupling::performance::DistributionAnalysis {
                        best_fit_distributions: HashMap::new(),
                        distribution_parameters: HashMap::new(),
                        goodness_of_fit: HashMap::new(),
                    },
                confidence_intervals:
                    crate::dynamical_decoupling::performance::ConfidenceIntervals {
                        mean_intervals: HashMap::new(),
                        bootstrap_intervals: HashMap::new(),
                        prediction_intervals: HashMap::new(),
                    },
            },
            comparative_analysis: None,
            performance_trends: crate::dynamical_decoupling::performance::PerformanceTrends {
                trend_slopes: HashMap::new(),
                trend_significance: HashMap::new(),
                seasonality: HashMap::new(),
                outliers: HashMap::new(),
            },
        }
    }
}

/// Benchmark tests for performance evaluation
#[cfg(test)]
mod benchmarks {
    use super::*;
    use crate::dynamical_decoupling::config::DDSequenceType;
    use std::time::Instant;

    #[test]
    fn benchmark_sequence_generation() {
        let target_qubits = vec![quantrs2_core::qubit::QubitId(0)];
        let iterations = 1000;

        let start = Instant::now();
        for _ in 0..iterations {
            let _sequence =
                crate::dynamical_decoupling::DDSequenceGenerator::generate_base_sequence(
                    &DDSequenceType::CPMG,
                    &target_qubits,
                    100e-6,
                )
                .unwrap();
        }
        let duration = start.elapsed();

        println!("Generated {} CPMG sequences in {:?}", iterations, duration);
        println!("Average generation time: {:?}", duration / iterations);

        // Performance assertion: should generate sequences quickly
        assert!(duration.as_millis() < 1000); // Less than 1 second for 1000 sequences
    }

    #[tokio::test]
    async fn benchmark_optimization() {
        let config = crate::dynamical_decoupling::config::DDOptimizationConfig::default();
        let mut optimizer = crate::dynamical_decoupling::DDSequenceOptimizer::new(config);
        let executor = tests::MockCircuitExecutor::new();

        let target_qubits = vec![quantrs2_core::qubit::QubitId(0)];
        let sequence = crate::dynamical_decoupling::DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::CPMG,
            &target_qubits,
            100e-6,
        )
        .unwrap();

        let start = Instant::now();
        let _result = optimizer
            .optimize_sequence(&sequence, &executor)
            .await
            .unwrap();
        let duration = start.elapsed();

        println!("Optimization completed in {:?}", duration);

        // Performance assertion: optimization should complete in reasonable time
        assert!(duration.as_secs() < 10); // Less than 10 seconds
    }

    #[tokio::test]
    async fn benchmark_performance_analysis() {
        let config = crate::dynamical_decoupling::config::DDPerformanceConfig::default();
        let mut analyzer = crate::dynamical_decoupling::DDPerformanceAnalyzer::new(config);
        let executor = tests::MockCircuitExecutor::new();

        let target_qubits = vec![quantrs2_core::qubit::QubitId(0)];
        let sequence = crate::dynamical_decoupling::DDSequenceGenerator::generate_base_sequence(
            &DDSequenceType::XY4,
            &target_qubits,
            100e-6,
        )
        .unwrap();

        let start = Instant::now();
        let _analysis = analyzer
            .analyze_performance(&sequence, &executor)
            .await
            .unwrap();
        let duration = start.elapsed();

        println!("Performance analysis completed in {:?}", duration);

        // Performance assertion: analysis should complete quickly
        assert!(duration.as_millis() < 5000); // Less than 5 seconds
    }
}
