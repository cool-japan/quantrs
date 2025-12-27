//! Performance monitoring configuration and metrics.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudPerformanceMonitoringConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Metrics to collect
    pub metrics: Vec<PerformanceMetric>,
    /// Collection frequency
    pub frequency: Duration,
    /// Data retention
    pub retention: Duration,
    /// Real-time monitoring
    pub real_time: RealTimeMonitoringConfig,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
    /// Benchmarking configuration
    pub benchmarking: BenchmarkingConfig,
}

/// Performance metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerformanceMetric {
    Latency,
    Throughput,
    ResponseTime,
    ErrorRate,
    Availability,
    CPUUtilization,
    MemoryUtilization,
    NetworkUtilization,
    DiskUtilization,
    QuantumCircuitDepth,
    QuantumGateErrors,
    QuantumCoherence,
    Custom(String),
}

/// Real-time monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMonitoringConfig {
    /// Enable real-time monitoring
    pub enabled: bool,
    /// Streaming frequency
    pub streaming_frequency: Duration,
    /// Buffer size
    pub buffer_size: usize,
    /// Aggregation window
    pub aggregation_window: Duration,
    /// Real-time dashboards
    pub dashboards: Vec<DashboardConfig>,
}

/// Performance threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Warning thresholds
    pub warning: ThresholdLevels,
    /// Critical thresholds
    pub critical: ThresholdLevels,
    /// Custom thresholds per metric
    pub custom_thresholds: std::collections::HashMap<PerformanceMetric, ThresholdLevels>,
}

/// Threshold levels for performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdLevels {
    /// CPU utilization threshold (percentage)
    pub cpu_utilization: f64,
    /// Memory utilization threshold (percentage)
    pub memory_utilization: f64,
    /// Network utilization threshold (percentage)
    pub network_utilization: f64,
    /// Disk utilization threshold (percentage)
    pub disk_utilization: f64,
    /// Response time threshold (milliseconds)
    pub response_time: f64,
    /// Error rate threshold (percentage)
    pub error_rate: f64,
    /// Availability threshold (percentage)
    pub availability: f64,
}

/// Benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingConfig {
    /// Enable automated benchmarking
    pub enabled: bool,
    /// Benchmark frequency
    pub frequency: Duration,
    /// Benchmark types
    pub benchmark_types: Vec<BenchmarkType>,
    /// Baseline comparison
    pub baseline_comparison: BaselineComparisonConfig,
}

/// Benchmark types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkType {
    PerformanceStress,
    LoadTesting,
    CapacityTesting,
    EnduranceTesting,
    ScalabilityTesting,
    QuantumBenchmark,
    Custom(String),
}

/// Baseline comparison configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparisonConfig {
    /// Enable baseline comparison
    pub enabled: bool,
    /// Baseline update frequency
    pub update_frequency: Duration,
    /// Comparison tolerance (percentage)
    pub tolerance: f64,
    /// Historical baselines to maintain
    pub historical_count: usize,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Dashboard name
    pub name: String,
    /// Dashboard type
    pub dashboard_type: DashboardType,
    /// Metrics to display
    pub metrics: Vec<PerformanceMetric>,
    /// Refresh rate
    pub refresh_rate: Duration,
    /// Layout configuration
    pub layout: DashboardLayout,
}

/// Dashboard types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardType {
    Overview,
    Detailed,
    RealTime,
    Historical,
    Comparative,
    Custom(String),
}

/// Dashboard layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    /// Number of columns
    pub columns: usize,
    /// Widget configurations
    pub widgets: Vec<WidgetConfig>,
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    /// Widget type
    pub widget_type: WidgetType,
    /// Position (row, column)
    pub position: (usize, usize),
    /// Size (width, height)
    pub size: (usize, usize),
    /// Metric to display
    pub metric: PerformanceMetric,
}

/// Widget types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetType {
    LineChart,
    BarChart,
    Gauge,
    Counter,
    Table,
    Heatmap,
    Custom(String),
}

impl Default for CloudPerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: vec![
                PerformanceMetric::Latency,
                PerformanceMetric::Throughput,
                PerformanceMetric::CPUUtilization,
                PerformanceMetric::MemoryUtilization,
            ],
            frequency: Duration::from_secs(30),
            retention: Duration::from_secs(86400 * 30), // 30 days
            real_time: RealTimeMonitoringConfig::default(),
            thresholds: PerformanceThresholds::default(),
            benchmarking: BenchmarkingConfig::default(),
        }
    }
}

impl Default for RealTimeMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            streaming_frequency: Duration::from_secs(5),
            buffer_size: 1000,
            aggregation_window: Duration::from_secs(60),
            dashboards: vec![],
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            warning: ThresholdLevels::warning_defaults(),
            critical: ThresholdLevels::critical_defaults(),
            custom_thresholds: std::collections::HashMap::new(),
        }
    }
}

impl ThresholdLevels {
    pub const fn warning_defaults() -> Self {
        Self {
            cpu_utilization: 70.0,
            memory_utilization: 75.0,
            network_utilization: 70.0,
            disk_utilization: 80.0,
            response_time: 1000.0, // 1 second
            error_rate: 5.0,
            availability: 95.0,
        }
    }

    pub const fn critical_defaults() -> Self {
        Self {
            cpu_utilization: 85.0,
            memory_utilization: 90.0,
            network_utilization: 85.0,
            disk_utilization: 95.0,
            response_time: 5000.0, // 5 seconds
            error_rate: 10.0,
            availability: 90.0,
        }
    }
}

impl Default for BenchmarkingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            frequency: Duration::from_secs(86400), // daily
            benchmark_types: vec![BenchmarkType::PerformanceStress],
            baseline_comparison: BaselineComparisonConfig::default(),
        }
    }
}

impl Default for BaselineComparisonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_frequency: Duration::from_secs(86400 * 7), // weekly
            tolerance: 10.0,
            historical_count: 10,
        }
    }
}
