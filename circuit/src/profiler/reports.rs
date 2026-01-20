//! Profiling report types and formats
//!
//! This module provides comprehensive report types for profiling results
//! including various analysis reports, export formats, and report generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

// Import types from sibling modules
use super::analyzers::*;
use super::benchmarks::*;
use super::collectors::*;
use super::metrics::*;

pub enum ExportFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// Binary format
    Binary,
    /// HTML report
    HTML,
    /// PDF report
    PDF,
}

/// Real-time metrics snapshot
#[derive(Debug, Clone)]
pub struct RealtimeMetrics {
    /// Current performance metrics
    pub current_metrics: Vec<PerformanceMetric>,
    /// Gate performance data
    pub gate_performance: HashMap<String, GateProfile>,
    /// Memory usage snapshot
    pub memory_usage: Option<MemorySnapshot>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Resource utilization summary
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization
    pub cpu: f64,
    /// Memory utilization
    pub memory: f64,
    /// GPU utilization
    pub gpu: Option<f64>,
    /// I/O utilization
    pub io: f64,
    /// Network utilization
    pub network: f64,
}

/// Comprehensive profiling report
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    /// Session ID
    pub session_id: String,
    /// Profiling start time
    pub start_time: SystemTime,
    /// Profiling end time
    pub end_time: SystemTime,
    /// Total profiling duration
    pub total_duration: Duration,
    /// Performance summary
    pub performance_summary: PerformanceSummary,
    /// Detailed analysis
    pub detailed_analysis: DetailedAnalysis,
    /// Report metadata
    pub metadata: HashMap<String, String>,
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Overall performance score
    pub overall_score: f64,
    /// Gate performance summary
    pub gate_performance: HashMap<String, f64>,
    /// Memory efficiency score
    pub memory_efficiency: f64,
    /// Resource utilization score
    pub resource_utilization: f64,
    /// Identified bottlenecks
    pub bottlenecks: Vec<String>,
    /// Performance recommendations
    pub recommendations: Vec<String>,
}

/// Detailed performance analysis
#[derive(Debug, Clone)]
pub struct DetailedAnalysis {
    /// Gate-level analysis
    pub gate_analysis: HashMap<String, GateAnalysisReport>,
    /// Memory analysis
    pub memory_analysis: MemoryAnalysisReport,
    /// Resource analysis
    pub resource_analysis: ResourceAnalysisReport,
    /// Anomaly detection results
    pub anomaly_detection: AnomalyDetectionReport,
    /// Regression analysis
    pub regression_analysis: RegressionReport,
}

/// Gate analysis report
#[derive(Debug, Clone)]
pub struct GateAnalysisReport {
    /// Gate performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Timing analysis
    pub timing_analysis: TimingAnalysisReport,
    /// Resource usage analysis
    pub resource_analysis: GateResourceAnalysis,
    /// Error analysis
    pub error_analysis: GateErrorAnalysis,
}

/// Timing analysis report
#[derive(Debug, Clone)]
pub struct TimingAnalysisReport {
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Timing variance
    pub timing_variance: f64,
    /// Timing trends
    pub timing_trends: TrendDirection,
    /// Performance anomalies
    pub timing_anomalies: Vec<String>,
}

/// Gate resource analysis
#[derive(Debug, Clone)]
pub struct GateResourceAnalysis {
    /// CPU usage patterns
    pub cpu_patterns: HashMap<String, f64>,
    /// Memory usage patterns
    pub memory_patterns: HashMap<String, f64>,
    /// I/O patterns
    pub io_patterns: HashMap<String, f64>,
}

/// Gate error analysis
#[derive(Debug, Clone)]
pub struct GateErrorAnalysis {
    /// Error rates
    pub error_rates: HashMap<String, f64>,
    /// Error patterns
    pub error_patterns: Vec<String>,
    /// Error correlations
    pub error_correlations: HashMap<String, f64>,
}

/// Memory analysis report
#[derive(Debug, Clone)]
pub struct MemoryAnalysisReport {
    /// Peak memory usage
    pub peak_usage: usize,
    /// Average memory usage
    pub average_usage: f64,
    /// Memory efficiency score
    pub efficiency_score: f64,
    /// Detected memory leaks
    pub leak_detection: Vec<String>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
}

/// Resource analysis report
#[derive(Debug, Clone)]
pub struct ResourceAnalysisReport {
    /// CPU analysis
    pub cpu_analysis: CpuAnalysisReport,
    /// Memory resource analysis
    pub memory_analysis: MemoryResourceAnalysis,
    /// I/O analysis
    pub io_analysis: IoAnalysisReport,
    /// Network analysis
    pub network_analysis: NetworkAnalysisReport,
}

/// CPU analysis report
#[derive(Debug, Clone)]
pub struct CpuAnalysisReport {
    /// Average CPU utilization
    pub average_utilization: f64,
    /// Peak CPU utilization
    pub peak_utilization: f64,
    /// Cache efficiency
    pub cache_efficiency: f64,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
}

/// Memory resource analysis
#[derive(Debug, Clone)]
pub struct MemoryResourceAnalysis {
    /// Utilization patterns
    pub utilization_patterns: HashMap<String, f64>,
    /// Allocation efficiency
    pub allocation_efficiency: f64,
    /// Fragmentation analysis
    pub fragmentation_analysis: f64,
}

/// I/O analysis report
#[derive(Debug, Clone)]
pub struct IoAnalysisReport {
    /// Throughput analysis
    pub throughput_analysis: ThroughputAnalysisReport,
    /// Latency analysis
    pub latency_analysis: LatencyAnalysisReport,
}

/// Throughput analysis report
#[derive(Debug, Clone)]
pub struct ThroughputAnalysisReport {
    /// Read throughput
    pub read_throughput: f64,
    /// Write throughput
    pub write_throughput: f64,
    /// Throughput efficiency
    pub throughput_efficiency: f64,
}

/// Latency analysis report
#[derive(Debug, Clone)]
pub struct LatencyAnalysisReport {
    /// Average latency
    pub average_latency: Duration,
    /// Latency distribution
    pub latency_distribution: HashMap<String, f64>,
    /// Latency trends
    pub latency_trends: TrendDirection,
}

/// Network analysis report
#[derive(Debug, Clone)]
pub struct NetworkAnalysisReport {
    /// Bandwidth efficiency
    pub bandwidth_efficiency: f64,
    /// Connection analysis
    pub connection_analysis: ConnectionAnalysisReport,
    /// Latency characteristics
    pub latency_characteristics: Duration,
}

/// Connection analysis report
#[derive(Debug, Clone)]
pub struct ConnectionAnalysisReport {
    /// Connection reliability
    pub connection_reliability: f64,
    /// Connection efficiency
    pub connection_efficiency: f64,
}

/// Anomaly detection report
#[derive(Debug, Clone)]
pub struct AnomalyDetectionReport {
    /// Detected anomalies
    pub detected_anomalies: Vec<PerformanceAnomaly>,
    /// Anomaly patterns
    pub anomaly_patterns: Vec<String>,
    /// Severity distribution
    pub severity_distribution: HashMap<AnomySeverity, usize>,
}

/// Regression analysis report
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Detected regressions
    pub detected_regressions: Vec<PerformanceRegression>,
    /// Regression trends
    pub regression_trends: HashMap<String, TrendDirection>,
    /// Impact assessment
    pub impact_assessment: HashMap<String, f64>,
}

/// Performance data container
#[derive(Debug, Clone)]
pub struct PerformanceData {
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// System state
    pub system_state: SystemState,
    /// Environment information
    pub environment: EnvironmentInfo,
}

/// Performance analysis report
#[derive(Debug, Clone)]
pub struct PerformanceAnalysisReport {
    /// Analysis timestamp
    pub analysis_timestamp: SystemTime,
    /// Overall performance score
    pub overall_performance_score: f64,
    /// Performance trends
    pub performance_trends: HashMap<String, TrendDirection>,
    /// Bottleneck analysis
    pub bottleneck_analysis: BottleneckAnalysisReport,
    /// Optimization recommendations
    pub optimization_recommendations: Vec<String>,
    /// Predictive analysis
    pub predictive_analysis: PredictiveAnalysisReport,
    /// Statistical analysis
    pub statistical_analysis: StatisticalAnalysisReport,
}

/// Bottleneck analysis report
#[derive(Debug, Clone)]
pub struct BottleneckAnalysisReport {
    /// Identified bottlenecks
    pub identified_bottlenecks: Vec<ResourceBottleneck>,
    /// Bottleneck impact
    pub bottleneck_impact: HashMap<String, f64>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Predictive analysis report
#[derive(Debug, Clone)]
pub struct PredictiveAnalysisReport {
    /// Performance forecasts
    pub performance_forecasts: HashMap<String, PredictionResult>,
    /// Capacity planning
    pub capacity_planning: CapacityPlanningReport,
    /// Risk assessment
    pub risk_assessment: RiskAssessmentReport,
}

/// Capacity planning report
#[derive(Debug, Clone)]
pub struct CapacityPlanningReport {
    /// Current capacity utilization
    pub current_capacity: f64,
    /// Projected capacity needs
    pub projected_capacity_needs: HashMap<String, f64>,
    /// Scaling recommendations
    pub scaling_recommendations: Vec<String>,
}

/// Risk assessment report
#[derive(Debug, Clone)]
pub struct RiskAssessmentReport {
    /// Performance risks
    pub performance_risks: Vec<String>,
    /// Risk mitigation strategies
    pub risk_mitigation: Vec<String>,
}

/// Statistical analysis report
#[derive(Debug, Clone)]
pub struct StatisticalAnalysisReport {
    /// Descriptive statistics
    pub descriptive_statistics: HashMap<String, f64>,
    /// Correlation analysis
    pub correlation_analysis: HashMap<String, f64>,
    /// Hypothesis test results
    pub hypothesis_tests: HashMap<String, f64>,
}
