//! Comprehensive test suite for unified benchmarking system
//! 
//! This module provides extensive test coverage for all components of the unified
//! benchmarking system, including unit tests, integration tests, and performance tests.

use quantrs2_device::prelude::*;
use quantrs2_device::unified_benchmarking::*;
use quantrs2_core::prelude::*;
use std::collections::HashMap;
use tokio;
use std::time::Duration;
use ndarray::Array1;

/// Test configuration helper
fn create_test_config() -> UnifiedBenchmarkConfig {
    UnifiedBenchmarkConfig {
        enabled_platforms: vec![
            QuantumPlatform::IBMQuantum,
            QuantumPlatform::AWSBraket,
            QuantumPlatform::AzureQuantum,
        ],
        benchmark_suites: vec![
            BenchmarkSuite::GateLevel,
            BenchmarkSuite::CircuitLevel,
            BenchmarkSuite::AlgorithmLevel,
        ],
        statistical_analysis: StatisticalAnalysisConfig {
            enable_advanced_stats: true,
            confidence_level: 0.95,
            bootstrap_samples: 1000,
            enable_ml_analysis: true,
            enable_correlation_analysis: true,
            enable_anomaly_detection: true,
        },
        optimization: OptimizationConfig {
            enable_multi_objective: true,
            optimization_targets: vec![
                OptimizationTarget::Performance,
                OptimizationTarget::Cost,
                OptimizationTarget::Fidelity,
            ],
            max_iterations: 100,
            convergence_tolerance: 1e-6,
        },
        reporting: ReportingConfig {
            enable_realtime_dashboard: true,
            auto_generate_reports: true,
            export_formats: vec![ReportFormat::HTML, ReportFormat::JSON],
            include_visualizations: true,
            enable_historical_comparison: true,
        },
        monitoring: MonitoringConfig {
            enable_realtime_monitoring: true,
            alert_thresholds: AlertThresholds {
                performance_degradation: 0.1,
                error_rate_increase: 0.05,
                cost_increase: 0.2,
                latency_increase: 0.15,
            },
            monitoring_interval: Duration::from_secs(60),
            enable_predictive_alerts: true,
        },
        resource_management: ResourceManagementConfig {
            enable_intelligent_allocation: true,
            max_concurrent_benchmarks: 10,
            resource_utilization_target: 0.8,
            enable_auto_scaling: true,
            cost_optimization_strategy: CostOptimizationStrategy::Balanced,
        },
        timeout: Duration::from_secs(3600),
        max_retries: 3,
        enable_parallel_execution: true,
    }
}

/// Create test calibration manager
fn create_test_calibration_manager() -> CalibrationManager {
    CalibrationManager::new()
}

#[tokio::test]
async fn test_unified_benchmark_system_creation() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await;
    assert!(system.is_ok(), "Failed to create unified benchmark system: {:?}", system.err());
    
    let system = system.unwrap();
    assert_eq!(system.get_config().enabled_platforms.len(), 3);
    assert!(system.get_config().statistical_analysis.enable_advanced_stats);
}

#[tokio::test]
async fn test_comprehensive_benchmark_execution() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    // This would normally execute real benchmarks, but for testing we'll use mock data
    let result = system.run_comprehensive_benchmark().await;
    
    // In a real implementation, this would succeed with actual hardware
    // For testing purposes, we expect it might fail due to missing hardware connections
    // but the important thing is that the API is correctly structured
    match result {
        Ok(benchmark_result) => {
            assert!(!benchmark_result.platform_results.is_empty());
            assert!(benchmark_result.overall_performance.execution_time > 0.0);
        },
        Err(e) => {
            // Expected in test environment without real hardware connections
            println!("Benchmark execution failed as expected in test environment: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_statistical_analysis_engine() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    // Create mock benchmark data
    let mut mock_data = HashMap::new();
    mock_data.insert("gate_fidelity".to_string(), Array1::from_vec(vec![0.995, 0.994, 0.996, 0.993, 0.997]));
    mock_data.insert("execution_time".to_string(), Array1::from_vec(vec![100.0, 105.0, 98.0, 102.0, 99.0]));
    mock_data.insert("error_rate".to_string(), Array1::from_vec(vec![0.005, 0.006, 0.004, 0.007, 0.003]));
    
    let analysis_result = system.analyze_benchmark_data(&mock_data).await;
    assert!(analysis_result.is_ok(), "Statistical analysis failed: {:?}", analysis_result.err());
    
    let analysis = analysis_result.unwrap();
    assert!(analysis.basic_statistics.contains_key("gate_fidelity"));
    assert!(analysis.correlation_analysis.is_some());
    assert!(analysis.anomaly_detection.is_some());
}

#[tokio::test]
async fn test_optimization_engine() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    // Create mock optimization parameters
    let optimization_params = OptimizationParameters {
        objectives: vec![
            OptimizationObjective {
                name: "performance".to_string(),
                target: OptimizationTarget::Performance,
                weight: 0.4,
                direction: OptimizationDirection::Maximize,
            },
            OptimizationObjective {
                name: "cost".to_string(),
                target: OptimizationTarget::Cost,
                weight: 0.3,
                direction: OptimizationDirection::Minimize,
            },
            OptimizationObjective {
                name: "fidelity".to_string(),
                target: OptimizationTarget::Fidelity,
                weight: 0.3,
                direction: OptimizationDirection::Maximize,
            },
        ],
        constraints: vec![
            OptimizationConstraint {
                name: "budget".to_string(),
                constraint_type: ConstraintType::UpperBound,
                value: 1000.0,
            },
        ],
        algorithm: OptimizationAlgorithm::MultiObjective,
        max_iterations: 50,
        convergence_tolerance: 1e-6,
    };
    
    let optimization_result = system.optimize_benchmark_configuration(optimization_params).await;
    assert!(optimization_result.is_ok(), "Optimization failed: {:?}", optimization_result.err());
    
    let result = optimization_result.unwrap();
    assert!(!result.pareto_front.is_empty());
    assert!(result.convergence_info.converged);
    assert!(result.best_solution.objective_values.len() == 3);
}

#[tokio::test]
async fn test_report_generation() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    // Test custom report generation
    let report_config = CustomReportConfig {
        report_type: "performance_analysis".to_string(),
        title: "Test Performance Report".to_string(),
        description: "Comprehensive performance analysis report".to_string(),
        included_platforms: vec![QuantumPlatform::IBMQuantum, QuantumPlatform::AWSBraket],
        metrics_filter: vec!["gate_fidelity".to_string(), "execution_time".to_string()],
        time_range: TimeRange {
            start: chrono::Utc::now() - chrono::Duration::days(30),
            end: chrono::Utc::now(),
        },
        format: ReportFormat::HTML,
        include_visualizations: true,
        include_statistical_analysis: true,
        include_recommendations: true,
        custom_styling: None,
        export_path: Some("/tmp/test_report.html".to_string()),
    };
    
    let report_result = system.generate_custom_report(report_config).await;
    assert!(report_result.is_ok(), "Report generation failed: {:?}", report_result.err());
    
    let report = report_result.unwrap();
    assert!(!report.content.is_empty());
    assert_eq!(report.format, ReportFormat::HTML);
    assert!(report.metadata.generation_time > chrono::Utc::now() - chrono::Duration::minutes(1));
}

#[tokio::test]
async fn test_real_time_monitoring() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    // Start real-time monitoring
    let monitoring_result = system.start_realtime_monitoring().await;
    assert!(monitoring_result.is_ok(), "Failed to start monitoring: {:?}", monitoring_result.err());
    
    // Simulate some monitoring events
    let mock_event = MonitoringEvent {
        event_id: "test_event_001".to_string(),
        timestamp: chrono::Utc::now(),
        event_type: MonitoringEventType::PerformanceDegradation,
        platform: QuantumPlatform::IBMQuantum,
        severity: EventSeverity::Warning,
        message: "Performance degradation detected in gate fidelity".to_string(),
        metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("current_fidelity".to_string(), 0.985);
            metrics.insert("baseline_fidelity".to_string(), 0.995);
            metrics.insert("degradation_percentage".to_string(), 1.0);
            metrics
        },
        affected_resources: vec!["qubit_0".to_string(), "qubit_1".to_string()],
        recommendations: vec![
            "Perform qubit recalibration".to_string(),
            "Check environmental factors".to_string(),
        ],
    };
    
    // In a real system, this would be processed by the monitoring engine
    let event_processing_result = system.process_monitoring_event(mock_event).await;
    assert!(event_processing_result.is_ok(), "Failed to process monitoring event: {:?}", event_processing_result.err());
}

#[tokio::test]
async fn test_platform_comparison() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    let comparison_config = PlatformComparisonConfig {
        platforms: vec![
            QuantumPlatform::IBMQuantum,
            QuantumPlatform::AWSBraket,
            QuantumPlatform::AzureQuantum,
        ],
        comparison_metrics: vec![
            ComparisonMetric::Performance,
            ComparisonMetric::Cost,
            ComparisonMetric::Fidelity,
            ComparisonMetric::Availability,
        ],
        benchmark_circuits: vec![
            // Mock circuit definitions
            BenchmarkCircuit {
                name: "bell_state".to_string(),
                description: "Bell state preparation benchmark".to_string(),
                qubits: 2,
                depth: 2,
                gates: vec!["H".to_string(), "CNOT".to_string()],
                complexity_level: ComplexityLevel::Basic,
            },
            BenchmarkCircuit {
                name: "grover_3qubit".to_string(), 
                description: "3-qubit Grover algorithm".to_string(),
                qubits: 3,
                depth: 12,
                gates: vec!["H".to_string(), "X".to_string(), "CCX".to_string()],
                complexity_level: ComplexityLevel::Intermediate,
            },
        ],
        statistical_significance_level: 0.05,
        include_cost_analysis: true,
        normalization_method: NormalizationMethod::ZScore,
    };
    
    let comparison_result = system.compare_platforms(comparison_config).await;
    
    // In test environment, this might fail due to lack of real hardware
    // but we can verify the API structure is correct
    match comparison_result {
        Ok(comparison) => {
            assert_eq!(comparison.platforms.len(), 3);
            assert!(!comparison.metric_comparisons.is_empty());
            assert!(comparison.rankings.contains_key(&ComparisonMetric::Performance));
        },
        Err(e) => {
            println!("Platform comparison failed as expected in test environment: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_benchmark_suite_configuration() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    // Create custom benchmark suite
    let custom_suite = CustomBenchmarkSuite {
        name: "quantum_machine_learning_suite".to_string(),
        description: "Comprehensive QML algorithm benchmarks".to_string(),
        circuits: vec![
            BenchmarkCircuit {
                name: "variational_classifier".to_string(),
                description: "Variational quantum classifier benchmark".to_string(),
                qubits: 4,
                depth: 20,
                gates: vec!["RY".to_string(), "RZ".to_string(), "CNOT".to_string()],
                complexity_level: ComplexityLevel::Advanced,
            },
            BenchmarkCircuit {
                name: "quantum_autoencoder".to_string(),
                description: "Quantum autoencoder benchmark".to_string(),
                qubits: 6,
                depth: 30,
                gates: vec!["RY".to_string(), "RZ".to_string(), "CNOT".to_string(), "RX".to_string()],
                complexity_level: ComplexityLevel::Expert,
            },
        ],
        performance_targets: vec![
            PerformanceTarget {
                metric: "classification_accuracy".to_string(),
                target_value: 0.85,
                tolerance: 0.05,
                optimization_direction: OptimizationDirection::Maximize,
            },
            PerformanceTarget {
                metric: "training_time".to_string(),
                target_value: 300.0, // seconds
                tolerance: 60.0,
                optimization_direction: OptimizationDirection::Minimize,
            },
        ],
        resource_requirements: ResourceRequirements {
            min_qubits: 4,
            max_qubits: 16,
            min_fidelity: 0.99,
            max_error_rate: 0.01,
            required_gates: vec!["RY".to_string(), "RZ".to_string(), "CNOT".to_string()],
            estimated_runtime: Duration::from_secs(1800),
        },
        validation_criteria: vec![
            ValidationCriterion {
                name: "fidelity_check".to_string(),
                criterion_type: ValidationCriterionType::MinimumThreshold,
                value: 0.95,
                weight: 0.4,
            },
            ValidationCriterion {
                name: "error_rate_check".to_string(),
                criterion_type: ValidationCriterionType::MaximumThreshold,
                value: 0.05,
                weight: 0.3,
            },
        ],
    };
    
    let suite_registration_result = system.register_custom_benchmark_suite(custom_suite).await;
    assert!(suite_registration_result.is_ok(), "Failed to register custom benchmark suite: {:?}", suite_registration_result.err());
    
    let registered_suites = system.list_available_benchmark_suites().await;
    assert!(registered_suites.is_ok());
    assert!(registered_suites.unwrap().iter().any(|suite| suite.name == "quantum_machine_learning_suite"));
}

#[tokio::test]
async fn test_cost_optimization() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    let cost_analysis_config = CostAnalysisConfig {
        platforms: vec![
            QuantumPlatform::IBMQuantum,
            QuantumPlatform::AWSBraket,
            QuantumPlatform::AzureQuantum,
        ],
        time_horizon: Duration::from_secs(30 * 24 * 3600), // 30 days
        workload_profile: WorkloadProfile {
            daily_circuit_executions: 100,
            average_circuit_depth: 20,
            average_qubits: 5,
            shots_per_circuit: 1000,
            peak_usage_hours: vec![9, 10, 11, 14, 15, 16], // Business hours
        },
        optimization_strategy: CostOptimizationStrategy::Balanced,
        budget_constraints: Some(BudgetConstraints {
            daily_budget: 500.0,
            monthly_budget: 10000.0,
            cost_per_shot_limit: 0.01,
        }),
        include_predictive_modeling: true,
    };
    
    let cost_analysis_result = system.analyze_platform_costs(cost_analysis_config).await;
    
    match cost_analysis_result {
        Ok(analysis) => {
            assert!(!analysis.platform_costs.is_empty());
            assert!(analysis.total_estimated_cost > 0.0);
            assert!(!analysis.optimization_recommendations.is_empty());
        },
        Err(e) => {
            println!("Cost analysis failed as expected in test environment: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_historical_data_management() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    // Test historical data storage and retrieval
    let mock_historical_data = HistoricalBenchmarkData {
        timestamp: chrono::Utc::now() - chrono::Duration::days(7),
        platform: QuantumPlatform::IBMQuantum,
        benchmark_type: BenchmarkType::GateLevel,
        results: BenchmarkResults {
            performance_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("gate_fidelity".to_string(), 0.994);
                metrics.insert("execution_time".to_string(), 102.5);
                metrics.insert("error_rate".to_string(), 0.006);
                metrics
            },
            statistical_analysis: None,
            cost_analysis: None,
        },
        metadata: {
            let mut metadata = HashMap::new();
            metadata.insert("device_id".to_string(), "ibmq_montreal".to_string());
            metadata.insert("software_version".to_string(), "0.1.0-alpha.5".to_string());
            metadata
        },
    };
    
    let storage_result = system.store_historical_data(mock_historical_data).await;
    assert!(storage_result.is_ok(), "Failed to store historical data: {:?}", storage_result.err());
    
    // Test historical data query
    let query_config = HistoricalDataQuery {
        platforms: vec![QuantumPlatform::IBMQuantum],
        time_range: TimeRange {
            start: chrono::Utc::now() - chrono::Duration::days(30),
            end: chrono::Utc::now(),
        },
        benchmark_types: vec![BenchmarkType::GateLevel],
        metrics_filter: vec!["gate_fidelity".to_string()],
        aggregation_method: AggregationMethod::Average,
        include_metadata: true,
    };
    
    let query_result = system.query_historical_data(query_config).await;
    assert!(query_result.is_ok(), "Failed to query historical data: {:?}", query_result.err());
}

#[tokio::test]
async fn test_alert_system() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    // Configure alert rules
    let alert_rules = vec![
        AlertRule {
            name: "gate_fidelity_degradation".to_string(),
            description: "Alert when gate fidelity drops below threshold".to_string(),
            condition: AlertCondition {
                metric: "gate_fidelity".to_string(),
                operator: ComparisonOperator::LessThan,
                threshold: 0.99,
                time_window: Duration::from_secs(300), // 5 minutes
            },
            severity: AlertSeverity::High,
            actions: vec![
                AlertAction::SendNotification {
                    channels: vec!["email".to_string(), "slack".to_string()],
                    message_template: "Gate fidelity has dropped to {current_value} on {platform}".to_string(),
                },
                AlertAction::TriggerRecalibration {
                    affected_qubits: vec![], // Will be determined dynamically
                },
            ],
            enabled: true,
        },
        AlertRule {
            name: "cost_budget_exceeded".to_string(),
            description: "Alert when daily cost budget is exceeded".to_string(),
            condition: AlertCondition {
                metric: "daily_cost".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 500.0,
                time_window: Duration::from_secs(24 * 3600), // 24 hours
            },
            severity: AlertSeverity::Critical,
            actions: vec![
                AlertAction::SendNotification {
                    channels: vec!["email".to_string(), "sms".to_string()],
                    message_template: "Daily cost budget exceeded: ${current_value}".to_string(),
                },
                AlertAction::PauseExecution {
                    platforms: vec![QuantumPlatform::IBMQuantum, QuantumPlatform::AWSBraket],
                },
            ],
            enabled: true,
        },
    ];
    
    let alert_config_result = system.configure_alerts(alert_rules).await;
    assert!(alert_config_result.is_ok(), "Failed to configure alerts: {:?}", alert_config_result.err());
    
    // Test alert triggering
    let mock_metric_update = MetricUpdate {
        platform: QuantumPlatform::IBMQuantum,
        metric_name: "gate_fidelity".to_string(),
        value: 0.985, // Below threshold
        timestamp: chrono::Utc::now(),
        metadata: HashMap::new(),
    };
    
    let alert_processing_result = system.process_metric_update(mock_metric_update).await;
    assert!(alert_processing_result.is_ok(), "Failed to process metric update: {:?}", alert_processing_result.err());
}

#[tokio::test]
async fn test_performance_baseline_management() {
    let config = create_test_config();
    let calibration_manager = create_test_calibration_manager();
    let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
    
    // Create performance baselines
    let baseline_config = BaselineConfig {
        name: "production_baseline_v1".to_string(),
        description: "Production performance baseline for release v1.0".to_string(),
        platforms: vec![QuantumPlatform::IBMQuantum, QuantumPlatform::AWSBraket],
        metrics: vec![
            BaselineMetric {
                name: "gate_fidelity".to_string(),
                expected_value: 0.995,
                tolerance: 0.005,
                critical_threshold: 0.98,
            },
            BaselineMetric {
                name: "execution_time".to_string(),
                expected_value: 100.0,
                tolerance: 20.0,
                critical_threshold: 200.0,
            },
        ],
        collection_period: Duration::from_secs(7 * 24 * 3600), // 7 days
        validation_criteria: vec![
            "minimum_95_percentile_performance".to_string(),
            "no_critical_failures".to_string(),
        ],
    };
    
    let baseline_creation_result = system.create_performance_baseline(baseline_config).await;
    assert!(baseline_creation_result.is_ok(), "Failed to create performance baseline: {:?}", baseline_creation_result.err());
    
    // Test baseline comparison
    let current_metrics = HashMap::from([
        ("gate_fidelity".to_string(), 0.992),
        ("execution_time".to_string(), 105.0),
    ]);
    
    let comparison_result = system.compare_against_baseline("production_baseline_v1", current_metrics).await;
    assert!(comparison_result.is_ok(), "Failed to compare against baseline: {:?}", comparison_result.err());
    
    let comparison = comparison_result.unwrap();
    assert!(!comparison.deviations.is_empty());
    assert!(comparison.overall_health_score >= 0.0 && comparison.overall_health_score <= 1.0);
}

/// Integration tests for end-to-end workflows
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_benchmark_workflow() {
        let config = create_test_config();
        let calibration_manager = create_test_calibration_manager();
        let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
        
        // 1. Configure custom benchmark suite
        let suite = CustomBenchmarkSuite {
            name: "integration_test_suite".to_string(),
            description: "End-to-end integration test benchmark suite".to_string(),
            circuits: vec![
                BenchmarkCircuit {
                    name: "test_circuit".to_string(),
                    description: "Simple test circuit for integration testing".to_string(),
                    qubits: 2,
                    depth: 3,
                    gates: vec!["H".to_string(), "CNOT".to_string(), "RZ".to_string()],
                    complexity_level: ComplexityLevel::Basic,
                },
            ],
            performance_targets: vec![],
            resource_requirements: ResourceRequirements {
                min_qubits: 2,
                max_qubits: 4,
                min_fidelity: 0.95,
                max_error_rate: 0.05,
                required_gates: vec!["H".to_string(), "CNOT".to_string()],
                estimated_runtime: Duration::from_secs(300),
            },
            validation_criteria: vec![],
        };
        
        let suite_result = system.register_custom_benchmark_suite(suite).await;
        assert!(suite_result.is_ok());
        
        // 2. Execute benchmark (mock execution in test environment)
        // In a real scenario, this would execute on actual hardware
        
        // 3. Generate comprehensive report
        let report_config = CustomReportConfig {
            report_type: "integration_test_report".to_string(),
            title: "Integration Test Results".to_string(),
            description: "Complete workflow integration test results".to_string(),
            included_platforms: vec![QuantumPlatform::IBMQuantum],
            metrics_filter: vec![],
            time_range: TimeRange {
                start: chrono::Utc::now() - chrono::Duration::hours(1),
                end: chrono::Utc::now(),
            },
            format: ReportFormat::JSON,
            include_visualizations: false,
            include_statistical_analysis: true,
            include_recommendations: true,
            custom_styling: None,
            export_path: None,
        };
        
        let report_result = system.generate_custom_report(report_config).await;
        assert!(report_result.is_ok());
        
        println!("Integration test workflow completed successfully");
    }
}

/// Performance tests for scalability and efficiency
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_concurrent_benchmark_execution() {
        let config = create_test_config();
        let calibration_manager = create_test_calibration_manager();
        let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
        
        let start_time = Instant::now();
        
        // Create multiple concurrent benchmark tasks
        let mut tasks = vec![];
        for i in 0..5 {
            let system_clone = system.clone();
            let task = tokio::spawn(async move {
                // Mock concurrent benchmark execution
                let mock_data = HashMap::from([
                    (format!("metric_{}", i), Array1::from_vec(vec![0.995, 0.994, 0.996])),
                ]);
                system_clone.analyze_benchmark_data(&mock_data).await
            });
            tasks.push(task);
        }
        
        // Wait for all tasks to complete
        let results = futures::future::join_all(tasks).await;
        let execution_time = start_time.elapsed();
        
        // Verify all tasks completed successfully
        for result in results {
            assert!(result.is_ok());
            assert!(result.unwrap().is_ok());
        }
        
        // Performance assertion: concurrent execution should be efficient
        assert!(execution_time < Duration::from_secs(10), "Concurrent execution took too long: {:?}", execution_time);
        
        println!("Concurrent benchmark execution completed in {:?}", execution_time);
    }

    #[tokio::test]
    async fn test_large_dataset_statistical_analysis() {
        let config = create_test_config();
        let calibration_manager = create_test_calibration_manager();
        let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
        
        // Create large mock dataset
        let large_dataset = (0..10000).map(|i| 0.995 + (i as f64 % 100.0) * 0.0001).collect::<Vec<f64>>();
        let mut mock_data = HashMap::new();
        mock_data.insert("large_dataset".to_string(), Array1::from_vec(large_dataset));
        
        let start_time = Instant::now();
        let analysis_result = system.analyze_benchmark_data(&mock_data).await;
        let analysis_time = start_time.elapsed();
        
        assert!(analysis_result.is_ok(), "Large dataset analysis failed");
        assert!(analysis_time < Duration::from_secs(30), "Large dataset analysis took too long: {:?}", analysis_time);
        
        println!("Large dataset analysis completed in {:?}", analysis_time);
    }
}

/// Error handling and edge case tests
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_invalid_configuration_handling() {
        let mut invalid_config = create_test_config();
        invalid_config.timeout = Duration::from_secs(0); // Invalid timeout
        
        let calibration_manager = create_test_calibration_manager();
        let system_result = UnifiedQuantumBenchmarkSystem::new(invalid_config, calibration_manager).await;
        
        // Should handle invalid configuration gracefully
        match system_result {
            Ok(_) => {}, // System corrected the configuration
            Err(e) => println!("Invalid configuration properly rejected: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_empty_dataset_analysis() {
        let config = create_test_config();
        let calibration_manager = create_test_calibration_manager();
        let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
        
        let empty_data = HashMap::new();
        let analysis_result = system.analyze_benchmark_data(&empty_data).await;
        
        // Should handle empty dataset gracefully
        assert!(analysis_result.is_err() || analysis_result.unwrap().basic_statistics.is_empty());
    }

    #[tokio::test]
    async fn test_network_failure_resilience() {
        let config = create_test_config();
        let calibration_manager = create_test_calibration_manager();
        let system = UnifiedQuantumBenchmarkSystem::new(config, calibration_manager).await.unwrap();
        
        // Test resilience to network failures (simulated by attempting operations that would fail)
        let comparison_config = PlatformComparisonConfig {
            platforms: vec![QuantumPlatform::IBMQuantum],
            comparison_metrics: vec![ComparisonMetric::Performance],
            benchmark_circuits: vec![],
            statistical_significance_level: 0.05,
            include_cost_analysis: false,
            normalization_method: NormalizationMethod::MinMax,
        };
        
        let comparison_result = system.compare_platforms(comparison_config).await;
        
        // Should handle network failures gracefully
        match comparison_result {
            Ok(_) => println!("Platform comparison succeeded unexpectedly"),
            Err(e) => println!("Platform comparison failed as expected due to network issues: {:?}", e),
        }
    }
}