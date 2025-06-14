//! Comprehensive tests for Advanced Performance Analysis module

#[cfg(test)]
mod tests {
    use quantrs2_tytan::advanced_performance_analysis::*;
    // Note: Sampler types not needed for performance analysis tests
    use ndarray::{Array1, Array2, Array3, ArrayD};
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    /// Test analysis configuration
    #[test]
    fn test_analysis_config() {
        let config = AnalysisConfig {
            real_time_monitoring: true,
            monitoring_frequency: 100.0,
            collection_level: MetricsLevel::Detailed,
            analysis_depth: AnalysisDepth::Deep,
            comparative_analysis: true,
            performance_prediction: true,
            statistical_analysis: StatisticalAnalysisConfig {
                confidence_level: 0.95,
                bootstrap_samples: 1000,
                hypothesis_testing: true,
                significance_level: 0.05,
                outlier_detection: true,
                outlier_method: OutlierDetectionMethod::ZScore { threshold: 3.0 },
            },
            visualization: VisualizationConfig {
                real_time_plots: true,
                plot_update_frequency: 1.0,
                export_formats: vec![ExportFormat::PNG],
                dashboard: DashboardConfig {
                    enable_web_dashboard: true,
                    port: 8080,
                    update_interval: 2.0,
                    enable_alerts: true,
                    alert_thresholds: HashMap::new(),
                },
            },
        };

        assert!(config.real_time_monitoring);
        assert_eq!(config.monitoring_frequency, 100.0);
        assert_eq!(config.collection_level, MetricsLevel::Detailed);
        assert_eq!(config.analysis_depth, AnalysisDepth::Deep);
        assert!(config.comparative_analysis);
        assert!(config.performance_prediction);
    }

    /// Test metrics levels
    #[test]
    fn test_metrics_levels() {
        let levels = vec![
            MetricsLevel::Basic,
            MetricsLevel::Detailed,
            MetricsLevel::Comprehensive,
            MetricsLevel::Custom { 
                metrics: vec!["cpu_usage".to_string(), "memory_usage".to_string()] 
            },
        ];

        for level in levels {
            match level {
                MetricsLevel::Basic => assert!(true),
                MetricsLevel::Detailed => assert!(true),
                MetricsLevel::Comprehensive => assert!(true),
                MetricsLevel::Custom { metrics } => {
                    assert_eq!(metrics.len(), 2);
                    assert_eq!(metrics[0], "cpu_usage");
                    assert_eq!(metrics[1], "memory_usage");
                }
            }
        }
    }

    /// Test analysis depth levels
    #[test]
    fn test_analysis_depth_levels() {
        let depths = vec![
            AnalysisDepth::Surface,
            AnalysisDepth::Deep,
            AnalysisDepth::Exhaustive,
            AnalysisDepth::Adaptive,
        ];

        for depth in depths {
            match depth {
                AnalysisDepth::Surface => assert!(true),
                AnalysisDepth::Deep => assert!(true),
                AnalysisDepth::Exhaustive => assert!(true),
                AnalysisDepth::Adaptive => assert!(true),
            }
        }
    }

    /// Test statistical analysis configuration
    #[test]
    fn test_statistical_analysis_config() {
        let config = StatisticalAnalysisConfig {
            confidence_level: 0.99,
            bootstrap_samples: 5000,
            hypothesis_testing: true,
            significance_level: 0.01,
            outlier_detection: true,
            outlier_method: OutlierDetectionMethod::IQR { multiplier: 1.5 },
        };

        assert_eq!(config.confidence_level, 0.99);
        assert_eq!(config.bootstrap_samples, 5000);
        assert!(config.hypothesis_testing);
        assert_eq!(config.significance_level, 0.01);
        assert!(config.outlier_detection);

        match config.outlier_method {
            OutlierDetectionMethod::IQR { multiplier } => {
                assert_eq!(multiplier, 1.5);
            }
            _ => panic!("Wrong outlier detection method"),
        }
    }

    /// Test outlier detection methods
    #[test]
    fn test_outlier_detection_methods() {
        let methods = vec![
            OutlierDetectionMethod::ZScore { threshold: 2.5 },
            OutlierDetectionMethod::IQR { multiplier: 1.5 },
            OutlierDetectionMethod::IsolationForest,
            OutlierDetectionMethod::LocalOutlierFactor,
            OutlierDetectionMethod::StatisticalTests,
        ];

        for method in methods {
            match method {
                OutlierDetectionMethod::ZScore { threshold } => {
                    assert_eq!(threshold, 2.5);
                }
                OutlierDetectionMethod::IQR { multiplier } => {
                    assert_eq!(multiplier, 1.5);
                }
                OutlierDetectionMethod::IsolationForest => {
                    assert!(true);
                }
                OutlierDetectionMethod::LocalOutlierFactor => {
                    assert!(true);
                }
                OutlierDetectionMethod::StatisticalTests => {
                    assert!(true);
                }
            }
        }
    }

    /// Test visualization configuration
    #[test]
    fn test_visualization_config() {
        let config = VisualizationConfig {
            real_time_plots: true,
            plot_update_frequency: 0.5,
            export_formats: vec![ExportFormat::SVG, ExportFormat::PNG],
            dashboard: DashboardConfig {
                enable_web_dashboard: true,
                port: 8080,
                update_interval: 2.0,
                enable_alerts: true,
                alert_thresholds: HashMap::new(),
            },
        };

        assert!(config.real_time_plots);
        assert_eq!(config.plot_update_frequency, 0.5);
        assert_eq!(config.export_formats.len(), 2);
        assert!(config.export_formats.contains(&ExportFormat::SVG));
        assert!(config.export_formats.contains(&ExportFormat::PNG));
        assert!(config.dashboard.enable_web_dashboard);
    }

    // Note: PlotType enum not available in current implementation

    /// Test export formats
    #[test]
    fn test_export_formats() {
        let formats = vec![
            ExportFormat::PNG,
            ExportFormat::SVG,
            ExportFormat::PDF,
            ExportFormat::HTML,
            ExportFormat::JSON,
            ExportFormat::CSV,
        ];

        for format in formats {
            match format {
                ExportFormat::PNG => assert!(true),
                ExportFormat::SVG => assert!(true),
                ExportFormat::PDF => assert!(true),
                ExportFormat::HTML => assert!(true),
                ExportFormat::JSON => assert!(true),
                ExportFormat::CSV => assert!(true),
            }
        }
    }

    // Note: PerformanceMetric, MetricValue, and MetricCategory types not available in current implementation

    /// Test benchmark result
    #[test]
    fn test_benchmark_result() {
        let mut descriptive_stats = HashMap::new();
        descriptive_stats.insert("execution_time".to_string(), DescriptiveStats {
            mean: 50.5,
            std_dev: 5.3,
            min: 42.1,
            max: 65.8,
            median: 48.2,
            quartiles: (46.0, 48.2, 54.1),
            skewness: 0.1,
            kurtosis: 2.8,
        });

        let mut confidence_intervals = HashMap::new();
        confidence_intervals.insert("execution_time".to_string(), (47.8, 53.2));

        let result = BenchmarkResult {
            benchmark_name: "QUBO_Solver_Benchmark".to_string(),
            execution_times: vec![Duration::from_millis(45), Duration::from_millis(52), Duration::from_millis(48)],
            memory_usage: vec![1024, 1152, 1088],
            solution_quality: vec![0.95, 0.97, 0.96],
            convergence_metrics: ConvergenceMetrics {
                time_to_convergence: vec![Duration::from_millis(30)],
                iterations_to_convergence: vec![100],
                convergence_rate: 0.95,
                final_residual: vec![0.001],
                stability_measure: 0.98,
            },
            scaling_analysis: ScalingAnalysis {
                computational_complexity: ComplexityAnalysis {
                    complexity_function: ComplexityFunction::Quadratic,
                    goodness_of_fit: 0.95,
                    confidence_intervals: vec![(1.8, 2.2)],
                    predicted_scaling: HashMap::new(),
                },
                memory_complexity: ComplexityAnalysis {
                    complexity_function: ComplexityFunction::Linear,
                    goodness_of_fit: 0.98,
                    confidence_intervals: vec![(0.9, 1.1)],
                    predicted_scaling: HashMap::new(),
                },
                parallel_efficiency: ParallelEfficiency {
                    strong_scaling: vec![1.0, 0.9, 0.8, 0.7],
                    weak_scaling: vec![1.0, 0.95, 0.9, 0.85],
                    load_balancing: 0.95,
                    communication_overhead: 0.1,
                    optimal_threads: 4,
                },
                scaling_predictions: {
                    let mut predictions = HashMap::new();
                    predictions.insert(1000, 250.0);
                    predictions
                },
            },
            statistical_summary: StatisticalSummary {
                descriptive_stats,
                confidence_intervals,
                hypothesis_tests: vec![],
                effect_sizes: HashMap::new(),
            },
        };

        assert_eq!(result.benchmark_name, "QUBO_Solver_Benchmark");
        assert_eq!(result.execution_times.len(), 3);
        assert_eq!(result.memory_usage.len(), 3);
        assert_eq!(result.solution_quality.len(), 3);
        assert_eq!(result.statistical_summary.descriptive_stats.len(), 1);
    }

    /// Test benchmark configuration
    #[test]
    fn test_benchmark_configuration() {
        let config = BenchmarkConfig {
            iterations: 500,
            warmup_iterations: 50,
            problem_sizes: vec![5, 10, 20, 50, 100],
            time_limit: Duration::from_secs(600),
            memory_limit: 1024 * 1024 * 1024, // 1GB
            detailed_profiling: true,
        };

        assert_eq!(config.iterations, 500);
        assert_eq!(config.warmup_iterations, 50);
        assert_eq!(config.problem_sizes.len(), 5);
        assert_eq!(config.time_limit, Duration::from_secs(600));
        assert_eq!(config.memory_limit, 1024 * 1024 * 1024);
        assert!(config.detailed_profiling);
        assert_eq!(config.problem_sizes[0], 5);
    }

    /// Test statistical summary
    #[test]
    fn test_statistical_summary() {
        let mut descriptive_stats = HashMap::new();
        descriptive_stats.insert("execution_time".to_string(), DescriptiveStats {
            mean: 100.0,
            std_dev: 15.0,
            min: 70.0,
            max: 130.0,
            median: 95.0,
            quartiles: (88.0, 95.0, 112.0),
            skewness: 0.2,
            kurtosis: 2.9,
        });

        let mut confidence_intervals = HashMap::new();
        confidence_intervals.insert("execution_time".to_string(), (92.0, 108.0));
        confidence_intervals.insert("memory_usage".to_string(), (87.0, 113.0));

        let mut effect_sizes = HashMap::new();
        effect_sizes.insert("algorithm_comparison".to_string(), 0.8);

        let summary = StatisticalSummary {
            descriptive_stats,
            confidence_intervals,
            hypothesis_tests: vec![],
            effect_sizes,
        };

        assert_eq!(summary.descriptive_stats.len(), 1);
        assert_eq!(summary.confidence_intervals.len(), 2);
        assert_eq!(summary.effect_sizes.len(), 1);
        assert_eq!(summary.descriptive_stats["execution_time"].mean, 100.0);
        assert_eq!(summary.confidence_intervals["execution_time"], (92.0, 108.0));
        assert_eq!(summary.effect_sizes["algorithm_comparison"], 0.8);
    }

    /// Test bottleneck analysis
    #[test]
    fn test_bottleneck_analysis() {
        let analysis = BottleneckAnalysis {
            bottlenecks: vec![
                Bottleneck {
                    bottleneck_type: BottleneckType::Memory,
                    location: "memory_allocation".to_string(),
                    severity: 35.0,
                    resource: "RAM".to_string(),
                    mitigation_strategies: vec![
                        "Use memory pools".to_string(),
                        "Pre-allocate arrays".to_string(),
                    ],
                },
                Bottleneck {
                    bottleneck_type: BottleneckType::Algorithm,
                    location: "tensor_contraction".to_string(),
                    severity: 20.0,
                    resource: "CPU".to_string(),
                    mitigation_strategies: vec![
                        "Optimize contraction order".to_string(),
                    ],
                },
            ],
            resource_utilization: ResourceUtilizationAnalysis {
                cpu_breakdown: CpuUtilizationBreakdown::default(),
                memory_breakdown: MemoryUtilizationBreakdown::default(),
                io_breakdown: IoUtilizationBreakdown::default(),
                network_breakdown: NetworkUtilizationBreakdown::default(),
            },
            dependency_analysis: DependencyAnalysis {
                critical_path: vec!["memory_allocation".to_string(), "tensor_contraction".to_string()],
                dependency_graph: DependencyGraph {
                    nodes: vec![
                        DependencyNode {
                            operation: "memory_allocation".to_string(),
                            execution_time: Duration::from_millis(100),
                            resource_requirements: ResourceRequirement {
                                memory_mb: 256,
                                cpu_cores: 1,
                                gpu_memory_mb: None,
                            },
                        },
                        DependencyNode {
                            operation: "tensor_contraction".to_string(),
                            execution_time: Duration::from_millis(200),
                            resource_requirements: ResourceRequirement {
                                memory_mb: 512,
                                cpu_cores: 2,
                                gpu_memory_mb: Some(1024),
                            },
                        },
                    ],
                    edges: vec![],
                    properties: GraphProperties {
                        node_count: 2,
                        edge_count: 0,
                        density: 0.0,
                        avg_path_length: 1.0,
                        clustering_coefficient: 0.0,
                    },
                },
                parallelization_opportunities: vec![
                    ParallelizationOpportunity {
                        operations: vec!["independent_calculations".to_string()],
                        potential_speedup: 2.0,
                        strategy: ParallelizationStrategy::DataParallelism,
                        complexity: ComplexityLevel::Medium,
                    }
                ],
                serialization_bottlenecks: vec![],
            },
            optimization_opportunities: vec![
                OptimizationOpportunity {
                    optimization_type: OptimizationType::MemoryOptimization,
                    description: "Reduce memory allocation overhead".to_string(),
                    potential_improvement: 25.0,
                    implementation_effort: EffortLevel::Medium,
                    risk_level: RiskLevel::Low,
                },
            ],
        };

        assert_eq!(analysis.bottlenecks.len(), 2);
        assert_eq!(analysis.bottlenecks[0].bottleneck_type, BottleneckType::Memory);
        assert_eq!(analysis.bottlenecks[0].location, "memory_allocation");
        assert_eq!(analysis.bottlenecks[0].severity, 35.0);
        assert_eq!(analysis.bottlenecks[0].mitigation_strategies.len(), 2);
        assert_eq!(analysis.optimization_opportunities.len(), 1);
    }

    /// Test bottleneck types
    #[test]
    fn test_bottleneck_types() {
        let types = vec![
            BottleneckType::CPU,
            BottleneckType::Memory,
            BottleneckType::IO,
            BottleneckType::Network,
            BottleneckType::Algorithm,
            BottleneckType::Synchronization,
            BottleneckType::Custom { description: "Custom bottleneck".to_string() },
        ];

        for bottleneck_type in types {
            match bottleneck_type {
                BottleneckType::CPU => assert!(true),
                BottleneckType::Memory => assert!(true),
                BottleneckType::IO => assert!(true),
                BottleneckType::Network => assert!(true),
                BottleneckType::Algorithm => assert!(true),
                BottleneckType::Synchronization => assert!(true),
                BottleneckType::Custom { description } => {
                    assert_eq!(description, "Custom bottleneck");
                }
            }
        }
    }

    // Note: PerformanceComparison, ComparisonRecommendation, RealTimeMonitoringData,
    // and PerformancePrediction types not available in current implementation
}