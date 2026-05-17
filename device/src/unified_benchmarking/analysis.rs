//! Analysis utilities for the unified benchmarking system
//!
//! This module provides helper constructors, default-value builders, and
//! pure analysis functions used by `system.rs`. Keeping them here prevents
//! `system.rs` from growing past the 2 000-line limit.

use std::collections::HashMap;
use std::time::Duration;

use scirs2_core::ndarray::Array2;

use super::results::{
    AccuracyComparison, AlgorithmLevelResults, AnomalyDetectionResults, BarrenPlateauAnalysis,
    BreakEvenAnalysis, CapacityPlanningResult, CapacityRecommendation, CentralityAnalysisResult,
    CircuitLevelResults, ClassificationResults, ClusteringResults,
    CommunityDetectionResult, ConnectivityAnalysisResult, ConvergenceAnalysis,
    CorrelationAnalysisResult, CostAnalysisResult, CostMetrics,
    CostOptimizationAnalysisResult, CrossEntropyResult, CrossPlatformAnalysis,
    CrossPlatformComparison, CrossValidationResult, DepthScalingResult, EnsembleResult,
    FeatureImportanceResults, ForecastingResults, GateLevelResults, GraphAnalysisResult,
    HeavyOutputResult, HypothesisTestResult, LinearRegressionResult, MLAnalysisResult,
    MLModelResult, MLRegressionResults, ModelComparisonResult, NISQPerformanceResult,
    NonlinearRegressionResult, OptimizationAnalysisResult, OptimizationResult,
    ParameterSensitivityAnalysis, ParetoAnalysisResult, PerturbationResult,
    PlatformBenchmarkResult, PlatformPerformanceMetrics, PlatformRanking, PolynomialFit,
    ExponentialFit, QuantumAdvantageResult, ROIAnalysis, ROIAnalysisResult,
    RandomizedBenchmarkingResult, RegressionAnalysisResult, ReliabilityMetrics,
    ResourceAnalysisResult, ResourceUtilizationMetrics, RobustnessAnalysisResult,
    ScalingMetric, ScalabilityAnalysis, SciRS2AnalysisResult, SeasonalityAnalysisResult,
    SensitivityAnalysisResult, StabilityAnalysis, StatisticalAnalysisResult,
    StatisticalSummary, SystemCostEfficiency, SystemLevelResults, SystemReliabilityAnalysis,
    SystemResourceUtilization, SystemScalabilityAnalysis, TimeSeriesAnalysisResult,
    TopologyOptimizationResult, TrendAnalysisResult, UncertaintyPropagation,
    VariationalAlgorithmResult, VolumeBenchmarkResult, WidthScalingResult,
    ClassicalComparisonResult, StationarityTestResults,
    FailurePattern,
};
use super::types::QuantumPlatform;

// ─── Primitive builders ───────────────────────────────────────────────────────

/// Build a zero-valued `StatisticalSummary`.
pub fn zero_statistical_summary() -> StatisticalSummary {
    StatisticalSummary {
        mean: 0.0,
        std_dev: 0.0,
        median: 0.0,
        min: 0.0,
        max: 0.0,
        percentiles: HashMap::new(),
        confidence_interval: (0.0, 0.0),
    }
}

/// Build a `StatisticalSummary` from a non-empty slice of `f64` values.
/// If the slice is empty, returns a zero summary.
pub fn statistical_summary_from_slice(values: &[f64]) -> StatisticalSummary {
    if values.is_empty() {
        return zero_statistical_summary();
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / n;
    let std_dev = variance.sqrt();
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = sorted[0];
    let max = *sorted.last().unwrap_or(&0.0);
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let p95_idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
    let p50_idx = ((sorted.len() as f64 * 0.50) as usize).min(sorted.len() - 1);
    let mut percentiles = HashMap::new();
    percentiles.insert(50u8, sorted[p50_idx]);
    percentiles.insert(95u8, sorted[p95_idx]);
    // 95 % confidence interval (approximate, normal assumption)
    let ci_half = 1.96 * std_dev / n.sqrt();
    StatisticalSummary {
        mean,
        std_dev,
        median,
        min,
        max,
        percentiles,
        confidence_interval: (mean - ci_half, mean + ci_half),
    }
}

// ─── Default result constructors ─────────────────────────────────────────────

/// Build a minimal, valid `GateLevelResults` representing a device that
/// has been characterised with generic default values.
pub fn default_gate_level_results() -> GateLevelResults {
    GateLevelResults {
        single_qubit_results: HashMap::new(),
        two_qubit_results: HashMap::new(),
        multi_qubit_results: HashMap::new(),
        randomized_benchmarking: RandomizedBenchmarkingResult {
            clifford_fidelity: 0.99,
            decay_parameter: 0.001,
            confidence_interval: (0.985, 0.995),
            sequence_lengths: vec![1, 2, 4, 8, 16],
            survival_probabilities: vec![1.0, 0.998, 0.992, 0.984, 0.968],
        },
        process_tomography: None,
    }
}

/// Build a minimal, valid `CircuitLevelResults`.
pub fn default_circuit_level_results() -> CircuitLevelResults {
    CircuitLevelResults {
        depth_scaling: DepthScalingResult {
            depth_vs_fidelity: vec![(1, 0.99), (10, 0.90), (50, 0.70)],
            depth_vs_execution_time: vec![
                (1, Duration::from_micros(50)),
                (10, Duration::from_micros(500)),
                (50, Duration::from_millis(3)),
            ],
            scaling_exponent: 1.2,
            coherence_limited_depth: 100,
        },
        width_scaling: WidthScalingResult {
            width_vs_fidelity: vec![(1, 0.99), (5, 0.95), (20, 0.80)],
            width_vs_execution_time: vec![
                (1, Duration::from_micros(50)),
                (5, Duration::from_micros(150)),
                (20, Duration::from_millis(1)),
            ],
            scaling_exponent: 0.8,
            connectivity_limited_width: 50,
        },
        circuit_type_results: HashMap::new(),
        parametric_results: HashMap::new(),
        volume_benchmarks: VolumeBenchmarkResult {
            heavy_output: HeavyOutputResult {
                heavy_output_probability: 0.66,
                theoretical_threshold: 0.5,
                statistical_significance: 0.95,
            },
            cross_entropy: CrossEntropyResult {
                cross_entropy_benchmarking_fidelity: 0.95,
                linear_xeb_fidelity: 0.94,
                log_xeb_fidelity: 0.93,
            },
            quantum_volume: 32,
        },
    }
}

/// Build a minimal, valid `AlgorithmLevelResults`.
pub fn default_algorithm_level_results() -> AlgorithmLevelResults {
    AlgorithmLevelResults {
        algorithm_results: HashMap::new(),
        nisq_performance: NISQPerformanceResult {
            noise_resilience: 0.85,
            error_mitigation_effectiveness: 0.70,
            depth_limited_performance: {
                let mut m = HashMap::new();
                m.insert(10_usize, 0.95_f64);
                m.insert(50, 0.80);
                m.insert(100, 0.60);
                m
            },
            variational_optimization_convergence: ConvergenceAnalysis {
                convergence_achieved: true,
                iterations_to_convergence: Some(150),
                final_cost: -1.0,
                cost_history: vec![-0.1, -0.5, -0.8, -1.0],
                gradient_norms: vec![0.5, 0.2, 0.05, 0.001],
            },
        },
        quantum_advantage: QuantumAdvantageResult {
            advantage_demonstrated: false,
            speedup_factor: None,
            confidence_level: 0.0,
            problem_instances_tested: 0,
        },
        classical_comparison: ClassicalComparisonResult {
            classical_runtime: Duration::from_secs(1),
            quantum_runtime: Duration::from_millis(100),
            speedup_ratio: 10.0,
            accuracy_comparison: AccuracyComparison {
                classical_accuracy: 1.0,
                quantum_accuracy: 0.95,
                relative_error: 0.05,
            },
        },
        variational_algorithm_performance: VariationalAlgorithmResult {
            optimization_landscapes: HashMap::new(),
            convergence_analysis: ConvergenceAnalysis {
                convergence_achieved: true,
                iterations_to_convergence: Some(200),
                final_cost: -0.9,
                cost_history: vec![-0.1, -0.5, -0.8, -0.9],
                gradient_norms: vec![0.5, 0.2, 0.05, 0.01],
            },
            parameter_sensitivity: ParameterSensitivityAnalysis {
                sensitivity_matrix: Array2::eye(2),
                most_sensitive_parameters: vec![0],
                robustness_score: 0.75,
            },
            barren_plateau_analysis: BarrenPlateauAnalysis {
                plateau_detected: false,
                gradient_variance: 0.1,
                effective_dimension: 4.0,
                mitigation_strategies: vec![],
            },
        },
    }
}

/// Build a minimal, valid `SystemLevelResults`.
pub fn default_system_level_results(platform: &QuantumPlatform) -> SystemLevelResults {
    SystemLevelResults {
        cross_platform_comparison: CrossPlatformComparison {
            platform_rankings: vec![PlatformRanking {
                platform: platform.clone(),
                overall_score: 0.85,
                category_scores: {
                    let mut m = HashMap::new();
                    m.insert("fidelity".to_string(), 0.90);
                    m.insert("speed".to_string(), 0.80);
                    m
                },
                rank: 1,
            }],
            relative_performance: {
                let mut m = HashMap::new();
                m.insert(format!("{platform:?}"), 1.0);
                m
            },
            statistical_significance: HashMap::new(),
        },
        resource_utilization: SystemResourceUtilization {
            average_queue_time: Duration::from_secs(60),
            throughput: 10.0,
            utilization_rate: 0.75,
            peak_usage_times: vec![],
        },
        reliability_analysis: SystemReliabilityAnalysis {
            uptime: 0.995,
            error_frequency: 0.001,
            recovery_time: Duration::from_secs(300),
            failure_patterns: vec![],
        },
        scalability_analysis: SystemScalabilityAnalysis {
            max_supported_qubits: 127,
            max_circuit_depth: 1000,
            performance_scaling: HashMap::new(),
        },
        cost_efficiency: SystemCostEfficiency {
            cost_per_shot: 0.0001,
            cost_per_gate: 0.000001,
            cost_efficiency_score: 0.80,
            roi_analysis: ROIAnalysis {
                investment_cost: 1000.0,
                operational_cost: 100.0,
                performance_benefit: 5000.0,
                roi_ratio: 4.5,
            },
        },
    }
}

// ─── Aggregate metrics calculators ───────────────────────────────────────────

/// Compute `PlatformPerformanceMetrics` from the four benchmark result sets.
pub fn compute_performance_metrics(
    gate: &GateLevelResults,
    circuit: &CircuitLevelResults,
    algo: &AlgorithmLevelResults,
    system: &SystemLevelResults,
) -> PlatformPerformanceMetrics {
    // Fidelity: use RB clifford fidelity as the primary signal.
    let rb_fidelity = gate.randomized_benchmarking.clifford_fidelity;

    // Error rate: 1 − fidelity is a simple lower bound.
    let error_rate = (1.0 - rb_fidelity).max(0.0);

    // Throughput: take from system-level resource utilisation.
    let throughput = system.resource_utilization.throughput;

    // Availability: taken from system reliability analysis.
    let availability = system.reliability_analysis.uptime.clamp(0.0, 1.0);

    // Average execution time: use the smallest depth point in depth-vs-exec-time.
    let avg_execution_time = circuit
        .depth_scaling
        .depth_vs_execution_time
        .first()
        .map(|(_, d)| *d)
        .unwrap_or(Duration::from_millis(100));

    // Blend algo fidelity if available.
    let fidelity = if !algo.nisq_performance.depth_limited_performance.is_empty() {
        let vals: Vec<f64> = algo
            .nisq_performance
            .depth_limited_performance
            .values()
            .copied()
            .collect();
        let algo_fidelity = vals.iter().sum::<f64>() / vals.len() as f64;
        (rb_fidelity + algo_fidelity) / 2.0
    } else {
        rb_fidelity
    };

    PlatformPerformanceMetrics {
        overall_fidelity: fidelity.clamp(0.0, 1.0),
        average_execution_time: avg_execution_time,
        throughput,
        error_rate,
        availability,
    }
}

/// Compute `ReliabilityMetrics` from the three benchmark result sets.
pub fn compute_reliability_metrics(
    gate: &GateLevelResults,
    _circuit: &CircuitLevelResults,
    _algo: &AlgorithmLevelResults,
) -> ReliabilityMetrics {
    let error_rate = (1.0 - gate.randomized_benchmarking.clifford_fidelity).max(0.0);
    // MTBF: rough heuristic — 1 / error_rate hours if error_rate > 0.
    let mtbf_hours = if error_rate > 1e-9 {
        (1.0 / error_rate).min(876_000.0) // cap at 100 years in hours
    } else {
        876_000.0
    };
    let mtbf = Duration::from_secs_f64(mtbf_hours * 3600.0);
    ReliabilityMetrics {
        uptime: (1.0 - error_rate).clamp(0.0, 1.0),
        mtbf,
        mttr: Duration::from_secs(300), // default 5-minute recovery
        availability: (1.0 - error_rate).clamp(0.0, 1.0),
    }
}

/// Compute `CostMetrics` from the three benchmark result sets.
pub fn compute_cost_metrics(
    _gate: &GateLevelResults,
    circuit: &CircuitLevelResults,
    _algo: &AlgorithmLevelResults,
) -> CostMetrics {
    // Use the volume benchmark quantum-volume score as a cost proxy:
    // higher QV → more capable but also higher cost.
    let qv = circuit.volume_benchmarks.quantum_volume as f64;
    let cost_per_shot = (0.0001 * qv / 32.0).max(0.00001);
    let cost_per_hour = cost_per_shot * 3600.0;
    let total_cost = cost_per_hour; // 1-hour default window
    let mut breakdown = HashMap::new();
    breakdown.insert("gate_operations".to_string(), total_cost * 0.6);
    breakdown.insert("readout".to_string(), total_cost * 0.2);
    breakdown.insert("qubit_time".to_string(), total_cost * 0.2);
    CostMetrics {
        total_cost,
        cost_per_shot,
        cost_per_hour,
        cost_breakdown: breakdown,
    }
}

// ─── Cross-platform analysis ──────────────────────────────────────────────────

/// Rank platforms and produce a `CrossPlatformAnalysis`.
pub fn compute_cross_platform_analysis(
    platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
) -> CrossPlatformAnalysis {
    if platform_results.is_empty() {
        return CrossPlatformAnalysis {
            platform_comparison: HashMap::new(),
            best_platform_per_metric: HashMap::new(),
            statistical_significance_tests: HashMap::new(),
        };
    }

    let mut platform_comparison: HashMap<String, f64> = HashMap::new();
    let mut fidelity_scores: Vec<(QuantumPlatform, f64)> = Vec::new();
    let mut error_scores: Vec<(QuantumPlatform, f64)> = Vec::new();
    let mut throughput_scores: Vec<(QuantumPlatform, f64)> = Vec::new();

    for (platform, result) in platform_results {
        let m = &result.performance_metrics;
        let label = format!("{platform:?}");
        // Composite score: higher fidelity, lower error_rate, higher throughput.
        let composite = m.overall_fidelity * 0.5
            + (1.0 - m.error_rate).clamp(0.0, 1.0) * 0.3
            + (m.throughput / 100.0).clamp(0.0, 1.0) * 0.2;
        platform_comparison.insert(label, composite);
        fidelity_scores.push((platform.clone(), m.overall_fidelity));
        error_scores.push((platform.clone(), m.error_rate));
        throughput_scores.push((platform.clone(), m.throughput));
    }

    let best_fidelity = fidelity_scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(p, _)| p.clone());

    let best_error = error_scores
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(p, _)| p.clone());

    let best_throughput = throughput_scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(p, _)| p.clone());

    let mut best_platform_per_metric: HashMap<String, QuantumPlatform> = HashMap::new();
    if let Some(p) = best_fidelity {
        best_platform_per_metric.insert("fidelity".to_string(), p);
    }
    if let Some(p) = best_error {
        best_platform_per_metric.insert("error_rate".to_string(), p);
    }
    if let Some(p) = best_throughput {
        best_platform_per_metric.insert("throughput".to_string(), p);
    }

    // Simple significance placeholder — all at 0.05.
    let statistical_significance_tests: HashMap<String, f64> = platform_comparison
        .keys()
        .map(|k| (k.clone(), 0.05))
        .collect();

    CrossPlatformAnalysis {
        platform_comparison,
        best_platform_per_metric,
        statistical_significance_tests,
    }
}

// ─── SciRS2 analysis ──────────────────────────────────────────────────────────

/// Produce a fully-structured but analytically trivial `SciRS2AnalysisResult`.
/// This is the fallback when the scirs2 feature is not available or the
/// per-platform data is too sparse for meaningful analysis.
pub fn default_scirs2_analysis() -> SciRS2AnalysisResult {
    let hypothesis_test = HypothesisTestResult {
        test_name: "baseline_t_test".to_string(),
        p_value: 1.0,
        statistic: 0.0,
        critical_value: 1.96,
        significant: false,
        effect_size: 0.0,
    };

    let stationarity_test = HypothesisTestResult {
        test_name: "adf".to_string(),
        p_value: 0.5,
        statistic: -1.0,
        critical_value: -2.86,
        significant: false,
        effect_size: 0.0,
    };

    SciRS2AnalysisResult {
        statistical_analysis: StatisticalAnalysisResult {
            hypothesis_tests: vec![hypothesis_test],
            correlation_analysis: CorrelationAnalysisResult {
                correlationmatrix: Array2::eye(1),
                significant_correlations: vec![],
                partial_correlations: Array2::eye(1),
            },
            regression_analysis: RegressionAnalysisResult {
                linear_regression: LinearRegressionResult {
                    coefficients: vec![0.0],
                    r_squared: 0.0,
                    adjusted_r_squared: 0.0,
                    p_values: vec![1.0],
                    residuals: vec![],
                },
                nonlinear_regression: NonlinearRegressionResult {
                    model_type: "none".to_string(),
                    parameters: vec![],
                    r_squared: 0.0,
                    mse: 0.0,
                    convergence_achieved: false,
                },
                model_comparison: ModelComparisonResult {
                    aic_scores: HashMap::new(),
                    bic_scores: HashMap::new(),
                    cross_validation_scores: HashMap::new(),
                    best_model: "none".to_string(),
                },
            },
            time_series_analysis: TimeSeriesAnalysisResult {
                trend_analysis: TrendAnalysisResult {
                    trend_detected: false,
                    trend_direction: "flat".to_string(),
                    trend_strength: 0.0,
                    trend_coefficients: vec![0.0],
                    change_points: vec![],
                },
                seasonality_analysis: SeasonalityAnalysisResult {
                    seasonal_components: vec![],
                    seasonal_period: 0,
                    seasonal_strength: 0.0,
                },
                stationarity_tests: StationarityTestResults {
                    adf_test: stationarity_test.clone(),
                    kpss_test: stationarity_test,
                    is_stationary: true,
                },
                forecasting: ForecastingResults {
                    forecasts: vec![],
                    confidence_intervals: vec![],
                    forecast_horizon: 0,
                    model_performance: HashMap::new(),
                },
            },
        },
        ml_analysis: MLAnalysisResult {
            clustering_results: ClusteringResults {
                cluster_assignments: vec![],
                cluster_centers: Array2::zeros((0, 0)),
                silhouette_score: 0.0,
                inertia: 0.0,
                optimal_clusters: 1,
            },
            classification_results: ClassificationResults {
                model_accuracy: 0.0,
                precision: vec![],
                recall: vec![],
                f1_score: vec![],
                confusion_matrix: Array2::zeros((0, 0)),
                feature_importance: vec![],
            },
            regression_results: MLRegressionResults {
                models: vec![],
                ensemble_result: EnsembleResult {
                    ensemble_mse: 0.0,
                    ensemble_mae: 0.0,
                    ensemble_r_squared: 0.0,
                    model_weights: vec![],
                },
                cross_validation: CrossValidationResult {
                    cv_scores: vec![],
                    mean_cv_score: 0.0,
                    std_cv_score: 0.0,
                },
            },
            anomaly_detection: AnomalyDetectionResults {
                anomaly_scores: vec![],
                anomaly_labels: vec![],
                anomaly_count: 0,
                feature_importance: FeatureImportanceResults {
                    importance_scores: vec![],
                    feature_names: vec![],
                    ranked_features: vec![],
                },
            },
        },
        optimization_analysis: OptimizationAnalysisResult {
            optimization_results: vec![],
            pareto_analysis: ParetoAnalysisResult {
                pareto_front: vec![],
                pareto_solutions: vec![],
                hypervolume: 0.0,
                spread_metric: 0.0,
            },
            sensitivity_analysis: SensitivityAnalysisResult {
                sensitivity_indices: vec![],
                total_sensitivity_indices: vec![],
                interaction_effects: Array2::zeros((0, 0)),
            },
            robustness_analysis: RobustnessAnalysisResult {
                robustness_score: 0.0,
                stability_analysis: StabilityAnalysis {
                    stability_score: 0.0,
                    perturbation_analysis: vec![],
                },
                uncertainty_propagation: UncertaintyPropagation {
                    input_uncertainties: vec![],
                    output_uncertainty: 0.0,
                    uncertainty_contributions: vec![],
                },
            },
        },
        graph_analysis: GraphAnalysisResult {
            connectivity_analysis: ConnectivityAnalysisResult {
                connectivity_matrix: Array2::zeros((0, 0)),
                path_lengths: Array2::zeros((0, 0)),
                clustering_coefficient: 0.0,
                graph_density: 0.0,
            },
            centrality_analysis: CentralityAnalysisResult {
                betweenness_centrality: vec![],
                closeness_centrality: vec![],
                eigenvector_centrality: vec![],
                pagerank: vec![],
            },
            community_detection: CommunityDetectionResult {
                community_assignments: vec![],
                modularity: 0.0,
                num_communities: 0,
                community_sizes: vec![],
            },
            topology_optimization: TopologyOptimizationResult {
                optimal_topology: Array2::zeros((0, 0)),
                optimization_objective: 0.0,
                improvement_factor: 1.0,
            },
        },
    }
}

// ─── Resource and cost analysis ───────────────────────────────────────────────

/// Aggregate platform results into a `ResourceAnalysisResult`.
pub fn compute_resource_analysis(
    platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
) -> ResourceAnalysisResult {
    let throughputs: Vec<f64> = platform_results
        .values()
        .map(|r| r.performance_metrics.throughput)
        .collect();
    let utilizations: Vec<f64> = platform_results
        .values()
        .map(|r| r.performance_metrics.availability)
        .collect();

    let avg_throughput = if throughputs.is_empty() {
        0.0
    } else {
        throughputs.iter().sum::<f64>() / throughputs.len() as f64
    };
    let avg_utilization = if utilizations.is_empty() {
        0.0
    } else {
        utilizations.iter().sum::<f64>() / utilizations.len() as f64
    };
    let peak_utilization = utilizations
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);

    let util_metric = |avg: f64, peak: f64| ResourceUtilizationMetrics {
        average_utilization: avg,
        peak_utilization: peak,
        utilization_distribution: vec![avg],
        efficiency_score: if peak > 0.0 { avg / peak } else { 1.0 },
    };

    ResourceAnalysisResult {
        cpu_utilization: util_metric(avg_utilization * 0.6, peak_utilization * 0.7),
        memory_utilization: util_metric(avg_utilization * 0.4, peak_utilization * 0.5),
        network_utilization: util_metric(avg_throughput / 100.0, avg_throughput / 50.0),
        storage_utilization: util_metric(0.2, 0.4),
        capacity_planning: CapacityPlanningResult {
            current_capacity: avg_throughput,
            projected_demand: vec![avg_throughput * 1.1, avg_throughput * 1.2],
            capacity_recommendations: vec![CapacityRecommendation {
                resource_type: "qubit_count".to_string(),
                recommended_capacity: 256.0,
                timeline: Duration::from_secs(7_776_000), // 90 days
                cost_estimate: 50_000.0,
            }],
            scaling_timeline: vec![],
        },
    }
}

/// Aggregate platform results into a `CostAnalysisResult`.
pub fn compute_cost_analysis(
    platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
) -> CostAnalysisResult {
    let total_cost: f64 = platform_results
        .values()
        .map(|r| r.cost_metrics.total_cost)
        .sum();

    let mut cost_breakdown: HashMap<String, f64> = HashMap::new();
    let mut cost_per_metric: HashMap<String, f64> = HashMap::new();
    for (platform, result) in platform_results {
        let label = format!("{platform:?}");
        cost_breakdown.insert(label.clone(), result.cost_metrics.total_cost);
        cost_per_metric.insert(
            format!("{label}.cost_per_shot"),
            result.cost_metrics.cost_per_shot,
        );
    }

    let potential_savings = total_cost * 0.15; // assume 15% optimisation headroom
    CostAnalysisResult {
        total_cost,
        cost_breakdown,
        cost_per_metric,
        cost_optimization: CostOptimizationAnalysisResult {
            potential_savings,
            optimization_strategies: vec![],
            implementation_roadmap: vec![],
        },
        roi_analysis: ROIAnalysisResult {
            roi_percentage: 250.0,
            payback_period: Duration::from_secs(365 * 24 * 3600),
            net_present_value: total_cost * 2.5,
            break_even_analysis: BreakEvenAnalysis {
                break_even_point: Duration::from_secs(180 * 24 * 3600),
                break_even_volume: total_cost,
                sensitivity_analysis: vec![],
            },
        },
    }
}

// ─── Fidelity statistics ───────────────────────────────────────────────────────

/// Simple fidelity statistics aggregated from a slice of raw fidelity values.
#[derive(Debug, Clone)]
pub struct FidelityStats {
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub std_dev: f64,
    pub n: usize,
}

/// Compute fidelity statistics from a slice of values in `[0, 1]`.
///
/// Returns `None` if the slice is empty.
pub fn compute_fidelity_statistics(results: &[f64]) -> Option<FidelityStats> {
    if results.is_empty() {
        return None;
    }
    let summary = statistical_summary_from_slice(results);
    let p95 = *summary.percentiles.get(&95u8).unwrap_or(&summary.max);
    Some(FidelityStats {
        mean: summary.mean,
        median: summary.median,
        p95,
        std_dev: summary.std_dev,
        n: results.len(),
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_summary_from_slice() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = statistical_summary_from_slice(&values);
        assert!((s.mean - 3.0).abs() < 1e-9);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
    }

    #[test]
    fn test_statistical_summary_empty() {
        let s = statistical_summary_from_slice(&[]);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn test_compute_fidelity_statistics() {
        let values = vec![0.99, 0.98, 0.97, 0.96, 0.95];
        let stats = compute_fidelity_statistics(&values).expect("should have stats");
        assert!(stats.mean > 0.96 && stats.mean < 0.99);
        assert_eq!(stats.n, 5);
    }

    #[test]
    fn test_compute_fidelity_statistics_empty() {
        assert!(compute_fidelity_statistics(&[]).is_none());
    }

    #[test]
    fn test_default_gate_level_results() {
        let g = default_gate_level_results();
        assert!(g.randomized_benchmarking.clifford_fidelity > 0.9);
    }

    #[test]
    fn test_default_scirs2_analysis_is_valid() {
        let a = default_scirs2_analysis();
        assert!(!a.statistical_analysis.hypothesis_tests.is_empty());
    }

    #[test]
    fn test_cross_platform_analysis_empty() {
        let cpa = compute_cross_platform_analysis(&HashMap::new());
        assert!(cpa.platform_comparison.is_empty());
    }
}
