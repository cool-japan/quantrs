//! Comprehensive Device Performance Analytics Dashboard
//!
//! This module provides a comprehensive real-time performance analytics dashboard
//! that unifies monitoring, visualization, and intelligent insights across all quantum
//! device components using SciRS2's advanced analytics and machine learning capabilities.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use serde::{Deserialize, Serialize};

// SciRS2 dependencies for advanced analytics
#[cfg(feature = "scirs2")]
use scirs2_graph::{
    betweenness_centrality, closeness_centrality, dijkstra_path, minimum_spanning_tree,
    strongly_connected_components, Graph,
};
#[cfg(feature = "scirs2")]
use scirs2_linalg::{det, eig, inv, matrix_norm, prelude::*, svd, LinalgError, LinalgResult};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{minimize, OptimizeResult};
use scirs2_stats::ttest::Alternative;
#[cfg(feature = "scirs2")]
use scirs2_stats::{corrcoef, distributions, mean, pearsonr, spearmanr, std, var};

// Fallback implementations when SciRS2 is not available
#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2;
#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::prelude::*;

use crate::{
    adaptive_compilation::AdaptiveCompilationConfig,
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    integrated_device_manager::IntegratedQuantumDeviceManager,
    noise_model::CalibrationNoiseModel,
    topology::HardwareTopology,
    CircuitResult, DeviceError, DeviceResult,
};

// Module declarations
pub mod alerting;
pub mod config;
pub mod data_collection;
pub mod ml_analytics;
pub mod optimization;
pub mod reporting;
pub mod visualization;

// Re-exports for public API
pub use alerting::*;
pub use config::*;
pub use data_collection::*;
pub use ml_analytics::*;
pub use optimization::*;
pub use reporting::*;
pub use visualization::*;

#[cfg(not(feature = "scirs2"))]
pub use fallback_scirs2::*;

// ── PerformanceDashboard ──────────────────────────────────────────────────

/// Metrics captured for a single circuit execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Wall-clock execution time in milliseconds
    pub execution_time_ms: f64,
    /// Measured circuit fidelity (0.0 – 1.0)
    pub fidelity: f64,
    /// Whether the execution succeeded
    pub success: bool,
    /// Number of gates in the circuit
    pub gate_count: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Two-qubit gate count
    pub two_qubit_gate_count: usize,
    /// Error rate observed during execution
    pub error_rate: f64,
    /// Timestamp of this execution (UNIX milliseconds)
    pub timestamp_ms: u128,
}

impl ExecutionMetrics {
    /// Create a new `ExecutionMetrics` snapshot.
    pub fn new(
        execution_time_ms: f64,
        fidelity: f64,
        success: bool,
        gate_count: usize,
        circuit_depth: usize,
        two_qubit_gate_count: usize,
        error_rate: f64,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        Self {
            execution_time_ms,
            fidelity,
            success,
            gate_count,
            circuit_depth,
            two_qubit_gate_count,
            error_rate,
            timestamp_ms,
        }
    }
}

/// Aggregated statistics returned by `PerformanceDashboard::get_summary`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    /// Total number of recorded executions
    pub total_executions: usize,
    /// Number of successful executions
    pub successful_executions: usize,
    /// Success rate (0.0 – 1.0)
    pub success_rate: f64,
    /// Mean fidelity across all executions
    pub mean_fidelity: f64,
    /// Minimum fidelity observed
    pub min_fidelity: f64,
    /// Maximum fidelity observed
    pub max_fidelity: f64,
    /// Mean execution time in milliseconds
    pub mean_execution_time_ms: f64,
    /// 95th-percentile execution time in milliseconds
    pub p95_execution_time_ms: f64,
    /// Mean error rate
    pub mean_error_rate: f64,
    /// Mean gate count
    pub mean_gate_count: f64,
    /// Mean circuit depth
    pub mean_circuit_depth: f64,
    /// Number of distinct circuit IDs tracked
    pub distinct_circuits: usize,
}

/// Per-circuit accumulated statistics (running totals for Welford's algorithm).
#[derive(Debug, Clone)]
struct CircuitStats {
    count: usize,
    total_execution_time_ms: f64,
    total_fidelity: f64,
    total_error_rate: f64,
    total_gate_count: usize,
    total_circuit_depth: usize,
    successes: usize,
    /// Sorted list of execution times kept for percentile computation
    sorted_times: Vec<f64>,
}

impl CircuitStats {
    fn new() -> Self {
        Self {
            count: 0,
            total_execution_time_ms: 0.0,
            total_fidelity: 0.0,
            total_error_rate: 0.0,
            total_gate_count: 0,
            total_circuit_depth: 0,
            successes: 0,
            sorted_times: Vec::new(),
        }
    }

    fn record(&mut self, m: &ExecutionMetrics) {
        self.count += 1;
        self.total_execution_time_ms += m.execution_time_ms;
        self.total_fidelity += m.fidelity;
        self.total_error_rate += m.error_rate;
        self.total_gate_count += m.gate_count;
        self.total_circuit_depth += m.circuit_depth;
        if m.success {
            self.successes += 1;
        }
        // Insert in sorted order for percentile queries
        let pos = self
            .sorted_times
            .partition_point(|&t| t <= m.execution_time_ms);
        self.sorted_times.insert(pos, m.execution_time_ms);
    }

    fn percentile_time(&self, p: f64) -> f64 {
        if self.sorted_times.is_empty() {
            return 0.0;
        }
        let idx = ((p / 100.0) * (self.sorted_times.len() as f64 - 1.0)).round() as usize;
        self.sorted_times
            .get(idx.min(self.sorted_times.len() - 1))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Real-time performance analytics dashboard.
///
/// Records execution metrics per circuit ID, computes aggregate statistics,
/// and can export a Markdown performance report.
///
/// # Example
/// ```rust,ignore
/// let mut dashboard = PerformanceDashboard::new(DashboardConfig::default());
/// dashboard.record_execution("bell_state", ExecutionMetrics::new(...));
/// let summary = dashboard.get_summary();
/// println!("{}", dashboard.export_report());
/// ```
pub struct PerformanceDashboard {
    /// Per-circuit statistics
    stats: HashMap<String, CircuitStats>,
    /// Global history (most recent first, bounded by config buffer_size)
    history: VecDeque<(String, ExecutionMetrics)>,
    /// Dashboard configuration
    config: DashboardConfig,
    /// Creation timestamp
    created_at: SystemTime,
}

impl PerformanceDashboard {
    /// Create a new dashboard with the given configuration.
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            stats: HashMap::new(),
            history: VecDeque::new(),
            config,
            created_at: SystemTime::now(),
        }
    }

    /// Record one circuit execution.
    ///
    /// The metrics are associated with `circuit_id` and added to the global
    /// rolling history buffer.
    pub fn record_execution(&mut self, circuit_id: &str, metrics: ExecutionMetrics) {
        // Update per-circuit stats
        self.stats
            .entry(circuit_id.to_string())
            .or_insert_with(CircuitStats::new)
            .record(&metrics);

        // Add to rolling history
        self.history.push_front((circuit_id.to_string(), metrics));
        let buffer = self.config.data_config.buffer_size;
        while self.history.len() > buffer {
            self.history.pop_back();
        }
    }

    /// Compute aggregated statistics across all recorded executions.
    pub fn get_summary(&self) -> DashboardSummary {
        if self.stats.is_empty() {
            return DashboardSummary {
                total_executions: 0,
                successful_executions: 0,
                success_rate: 0.0,
                mean_fidelity: 0.0,
                min_fidelity: 0.0,
                max_fidelity: 0.0,
                mean_execution_time_ms: 0.0,
                p95_execution_time_ms: 0.0,
                mean_error_rate: 0.0,
                mean_gate_count: 0.0,
                mean_circuit_depth: 0.0,
                distinct_circuits: 0,
            };
        }

        let total_executions: usize = self.stats.values().map(|s| s.count).sum();
        let successful_executions: usize = self.stats.values().map(|s| s.successes).sum();

        let sum_fidelity: f64 = self.stats.values().map(|s| s.total_fidelity).sum();
        let sum_time: f64 = self.stats.values().map(|s| s.total_execution_time_ms).sum();
        let sum_error: f64 = self.stats.values().map(|s| s.total_error_rate).sum();
        let sum_gates: usize = self.stats.values().map(|s| s.total_gate_count).sum();
        let sum_depth: usize = self.stats.values().map(|s| s.total_circuit_depth).sum();

        // Min/max fidelity from per-execution history
        let (min_fidelity, max_fidelity) = self
            .history
            .iter()
            .fold((f64::MAX, f64::MIN), |(mn, mx), (_, m)| {
                (mn.min(m.fidelity), mx.max(m.fidelity))
            });
        let min_fidelity = if min_fidelity == f64::MAX {
            0.0
        } else {
            min_fidelity
        };
        let max_fidelity = if max_fidelity == f64::MIN {
            0.0
        } else {
            max_fidelity
        };

        // Global p95 execution time from history
        let mut all_times: Vec<f64> = self
            .history
            .iter()
            .map(|(_, m)| m.execution_time_ms)
            .collect();
        all_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_idx = ((0.95 * (all_times.len() as f64 - 1.0)).round() as usize)
            .min(all_times.len().saturating_sub(1));
        let p95_execution_time_ms = all_times.get(p95_idx).copied().unwrap_or(0.0);

        let n = total_executions as f64;
        DashboardSummary {
            total_executions,
            successful_executions,
            success_rate: if total_executions > 0 {
                successful_executions as f64 / n
            } else {
                0.0
            },
            mean_fidelity: sum_fidelity / n,
            min_fidelity,
            max_fidelity,
            mean_execution_time_ms: sum_time / n,
            p95_execution_time_ms,
            mean_error_rate: sum_error / n,
            mean_gate_count: sum_gates as f64 / n,
            mean_circuit_depth: sum_depth as f64 / n,
            distinct_circuits: self.stats.len(),
        }
    }

    /// Export a Markdown-formatted performance report.
    pub fn export_report(&self) -> String {
        let summary = self.get_summary();
        let uptime = self.created_at.elapsed().unwrap_or_default().as_secs();

        let mut report = String::new();
        report.push_str("# QuantRS2 Device Performance Dashboard\n\n");
        report.push_str(&format!("**Dashboard uptime:** {}s\n\n", uptime));

        report.push_str("## Summary\n\n");
        report.push_str("| Metric | Value |\n");
        report.push_str("|--------|-------|\n");
        report.push_str(&format!(
            "| Total Executions | {} |\n",
            summary.total_executions
        ));
        report.push_str(&format!(
            "| Successful Executions | {} |\n",
            summary.successful_executions
        ));
        report.push_str(&format!(
            "| Success Rate | {:.2}% |\n",
            summary.success_rate * 100.0
        ));
        report.push_str(&format!(
            "| Mean Fidelity | {:.4} |\n",
            summary.mean_fidelity
        ));
        report.push_str(&format!("| Min Fidelity | {:.4} |\n", summary.min_fidelity));
        report.push_str(&format!("| Max Fidelity | {:.4} |\n", summary.max_fidelity));
        report.push_str(&format!(
            "| Mean Execution Time | {:.2} ms |\n",
            summary.mean_execution_time_ms
        ));
        report.push_str(&format!(
            "| P95 Execution Time | {:.2} ms |\n",
            summary.p95_execution_time_ms
        ));
        report.push_str(&format!(
            "| Mean Error Rate | {:.6} |\n",
            summary.mean_error_rate
        ));
        report.push_str(&format!(
            "| Mean Gate Count | {:.1} |\n",
            summary.mean_gate_count
        ));
        report.push_str(&format!(
            "| Mean Circuit Depth | {:.1} |\n",
            summary.mean_circuit_depth
        ));
        report.push_str(&format!(
            "| Distinct Circuits Tracked | {} |\n",
            summary.distinct_circuits
        ));

        // Per-circuit breakdown
        if !self.stats.is_empty() {
            report.push_str("\n## Per-Circuit Breakdown\n\n");
            report.push_str(
                "| Circuit ID | Executions | Success Rate | Mean Fidelity | Mean Time (ms) |\n",
            );
            report.push_str(
                "|-----------|-----------|-------------|--------------|---------------|\n",
            );

            let mut circuit_ids: Vec<&String> = self.stats.keys().collect();
            circuit_ids.sort();
            for id in circuit_ids {
                let s = match self.stats.get(id) {
                    Some(s) => s,
                    None => continue,
                };
                let mean_f = if s.count > 0 {
                    s.total_fidelity / s.count as f64
                } else {
                    0.0
                };
                let mean_t = if s.count > 0 {
                    s.total_execution_time_ms / s.count as f64
                } else {
                    0.0
                };
                let sr = if s.count > 0 {
                    s.successes as f64 / s.count as f64
                } else {
                    0.0
                };
                report.push_str(&format!(
                    "| {} | {} | {:.1}% | {:.4} | {:.2} |\n",
                    id,
                    s.count,
                    sr * 100.0,
                    mean_f,
                    mean_t
                ));
            }
        }

        report
    }

    /// Return the number of executions recorded for a specific circuit.
    pub fn execution_count(&self, circuit_id: &str) -> usize {
        self.stats.get(circuit_id).map_or(0, |s| s.count)
    }

    /// Clear all recorded data.
    pub fn reset(&mut self) {
        self.stats.clear();
        self.history.clear();
    }
}

impl Default for DashboardConfig {
    fn default() -> Self {
        use crate::performance_dashboard::{
            alerting::{AlertingConfig, AnomalyDetectionAlgorithm, AnomalyDetectionConfig},
            data_collection::{
                AggregationConfig, AggregationFunction, DataCollectionConfig, MetricsConfig,
                PerformanceMetric, QualityMetric, ResourceMetric, SamplingConfig, SamplingStrategy,
                TimeWindow,
            },
            ml_analytics::{
                EvaluationConfig, EvaluationMetric, FeatureConfig, MLAnalyticsConfig,
                ModelSelectionCriteria, TrainingConfig,
            },
            optimization::{DashboardOptimizationConfig, OptimizationObjective},
            reporting::{
                DistributionConfig, ReportFormat, ReportFrequency, ReportSchedule, ReportingConfig,
            },
            visualization::{
                ColorScheme, GridLayout, InteractiveConfig, LayoutConfig, ThemeConfig,
                VisualizationConfig,
            },
        };
        use std::collections::HashMap;
        use std::time::Duration;

        DashboardConfig {
            enable_realtime_monitoring: false,
            data_config: DataCollectionConfig {
                collection_interval: 60,
                buffer_size: 1000,
                retention_days: 30,
                metrics_config: MetricsConfig {
                    performance_metrics: vec![
                        PerformanceMetric::Fidelity,
                        PerformanceMetric::Latency,
                        PerformanceMetric::ErrorRate,
                    ],
                    resource_metrics: vec![ResourceMetric::CpuUtilization],
                    quality_metrics: vec![QualityMetric::GateFidelity],
                    custom_metrics: vec![],
                },
                aggregation_config: AggregationConfig {
                    aggregation_functions: vec![
                        AggregationFunction::Mean,
                        AggregationFunction::Percentile(95.0),
                    ],
                    time_windows: vec![TimeWindow::Minutes(5), TimeWindow::Hours(1)],
                    grouping_dimensions: vec!["circuit_id".to_string()],
                },
                sampling_config: SamplingConfig {
                    sampling_strategy: SamplingStrategy::Fixed,
                    sample_rate: 1.0,
                    adaptive_sampling: false,
                    quality_based_sampling: false,
                },
            },
            visualization_config: VisualizationConfig {
                refresh_rate: 60,
                chart_types: vec![],
                layout_config: LayoutConfig {
                    grid_layout: GridLayout {
                        rows: 4,
                        columns: 3,
                        gap_size: 8,
                    },
                    responsive_design: true,
                    panel_configuration: vec![],
                },
                theme_config: ThemeConfig {
                    color_scheme: ColorScheme::Default,
                    dark_mode: false,
                    custom_styling: HashMap::new(),
                },
                interactive_config: InteractiveConfig {
                    enable_drill_down: true,
                    enable_filtering: true,
                    enable_zooming: true,
                    enable_real_time_updates: false,
                },
            },
            alerting_config: AlertingConfig {
                enable_alerting: false,
                alert_thresholds: HashMap::new(),
                notification_channels: vec![],
                escalation_rules: vec![],
                anomaly_detection: AnomalyDetectionConfig {
                    detection_algorithms: vec![AnomalyDetectionAlgorithm::StatisticalOutlier],
                    sensitivity: 0.95,
                    baseline_window: Duration::from_secs(3600),
                    detection_window: Duration::from_secs(300),
                },
            },
            ml_config: MLAnalyticsConfig {
                enable_ml_analytics: false,
                prediction_models: vec![],
                feature_config: FeatureConfig {
                    feature_selection_methods: vec![],
                    feature_engineering_rules: vec![],
                    dimensionality_reduction: None,
                },
                training_config: TrainingConfig {
                    training_data_size: 1000,
                    validation_split: 0.2,
                    cross_validation_folds: 5,
                    hyperparameter_tuning: false,
                    model_selection_criteria: ModelSelectionCriteria::CrossValidationScore,
                },
                evaluation_config: EvaluationConfig {
                    evaluation_metrics: vec![EvaluationMetric::RMSE],
                    test_data_size: 200,
                    evaluation_frequency: Duration::from_secs(3600),
                    performance_tracking: true,
                },
            },
            optimization_config: DashboardOptimizationConfig {
                enable_auto_recommendations: false,
                optimization_objectives: vec![OptimizationObjective::BalancedPerformance],
                confidence_threshold: 0.8,
                priority_weighting: HashMap::new(),
            },
            reporting_config: ReportingConfig {
                enable_automated_reports: false,
                report_schedule: ReportSchedule {
                    frequency: ReportFrequency::Daily,
                    time_of_day: "00:00".to_string(),
                    time_zone: "UTC".to_string(),
                    custom_schedule: None,
                },
                report_formats: vec![ReportFormat::HTML],
                distribution_config: DistributionConfig {
                    email_recipients: vec![],
                    file_storage_locations: vec![],
                    api_endpoints: vec![],
                },
            },
        }
    }
}

#[cfg(test)]
mod dashboard_tests {
    use super::*;

    fn sample_metrics(fidelity: f64, success: bool, time_ms: f64) -> ExecutionMetrics {
        ExecutionMetrics::new(time_ms, fidelity, success, 5, 3, 2, 1.0 - fidelity)
    }

    #[test]
    fn test_dashboard_creation() {
        let dashboard = PerformanceDashboard::new(DashboardConfig::default());
        let summary = dashboard.get_summary();
        assert_eq!(summary.total_executions, 0);
        assert_eq!(summary.distinct_circuits, 0);
    }

    #[test]
    fn test_record_and_summary() {
        let mut dashboard = PerformanceDashboard::new(DashboardConfig::default());
        dashboard.record_execution("bell_state", sample_metrics(0.99, true, 10.0));
        dashboard.record_execution("bell_state", sample_metrics(0.97, true, 12.0));
        dashboard.record_execution("ghz", sample_metrics(0.95, false, 20.0));

        let summary = dashboard.get_summary();
        assert_eq!(summary.total_executions, 3);
        assert_eq!(summary.successful_executions, 2);
        assert_eq!(summary.distinct_circuits, 2);
        assert!((summary.success_rate - 2.0 / 3.0).abs() < 1e-6);
        assert!(summary.mean_fidelity > 0.0);
    }

    #[test]
    fn test_execution_count() {
        let mut dashboard = PerformanceDashboard::new(DashboardConfig::default());
        dashboard.record_execution("circ_a", sample_metrics(0.9, true, 5.0));
        dashboard.record_execution("circ_a", sample_metrics(0.88, true, 6.0));

        assert_eq!(dashboard.execution_count("circ_a"), 2);
        assert_eq!(dashboard.execution_count("missing"), 0);
    }

    #[test]
    fn test_export_report_contains_sections() {
        let mut dashboard = PerformanceDashboard::new(DashboardConfig::default());
        dashboard.record_execution("test_circuit", sample_metrics(0.99, true, 8.0));

        let report = dashboard.export_report();
        assert!(report.contains("# QuantRS2 Device Performance Dashboard"));
        assert!(report.contains("## Summary"));
        assert!(report.contains("## Per-Circuit Breakdown"));
        assert!(report.contains("test_circuit"));
    }

    #[test]
    fn test_reset() {
        let mut dashboard = PerformanceDashboard::new(DashboardConfig::default());
        dashboard.record_execution("circ", sample_metrics(0.9, true, 5.0));
        assert_eq!(dashboard.get_summary().total_executions, 1);
        dashboard.reset();
        assert_eq!(dashboard.get_summary().total_executions, 0);
    }
}
