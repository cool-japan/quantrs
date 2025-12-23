//! Performance metrics collection types and utilities
//!
//! This module provides types and functionality for collecting and aggregating
//! performance metrics during quantum circuit profiling.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

pub struct MetricsCollector {
    /// Collected performance metrics
    pub metrics: VecDeque<PerformanceMetric>,
    /// Metric aggregation rules
    pub aggregation_rules: HashMap<String, AggregationRule>,
    /// Real-time metric streams
    pub metric_streams: HashMap<String, MetricStream>,
    /// Collection statistics
    pub collection_stats: CollectionStatistics,
}

/// Individual performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Measurement timestamp
    pub timestamp: SystemTime,
    /// Metric category
    pub category: MetricCategory,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Confidence score
    pub confidence: f64,
    /// Statistical significance
    pub significance: Option<f64>,
}

/// Categories of performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricCategory {
    /// Execution timing metrics
    Timing,
    /// Memory usage metrics
    Memory,
    /// Resource utilization metrics
    Resource,
    /// Gate operation metrics
    Gate,
    /// Circuit complexity metrics
    Complexity,
    /// Error rate metrics
    Error,
    /// Throughput metrics
    Throughput,
    /// Latency metrics
    Latency,
    /// Custom metric category
    Custom { name: String },
}

/// Metric aggregation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationRule {
    /// Aggregation function type
    pub function: AggregationFunction,
    /// Time window for aggregation
    pub window: Duration,
    /// Minimum samples required
    pub min_samples: usize,
    /// Statistical confidence level
    pub confidence_level: f64,
}

/// Aggregation function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    /// Mean value
    Mean,
    /// Median value
    Median,
    /// Maximum value
    Maximum,
    /// Minimum value
    Minimum,
    /// Standard deviation
    StandardDeviation,
    /// Percentile value
    Percentile { percentile: f64 },
    /// Moving average
    MovingAverage { window_size: usize },
    /// Exponential moving average
    ExponentialMovingAverage { alpha: f64 },
}

/// Real-time metric stream
#[derive(Debug, Clone)]
pub struct MetricStream {
    /// Stream name
    pub name: String,
    /// Current value
    pub current_value: f64,
    /// Value history
    pub history: VecDeque<f64>,
    /// Stream statistics
    pub statistics: StreamStatistics,
    /// Anomaly detection threshold
    pub anomaly_threshold: f64,
}

/// Stream statistics
#[derive(Debug, Clone)]
pub struct StreamStatistics {
    /// Sample count
    pub sample_count: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Trend direction
    pub trend: TrendDirection,
}

/// Trend direction analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable trend
    Stable,
    /// Oscillating trend
    Oscillating,
    /// Unknown trend
    Unknown,
}

/// Collection statistics
#[derive(Debug, Clone)]
pub struct CollectionStatistics {
    /// Total metrics collected
    pub total_metrics: usize,
    /// Collection duration
    pub collection_duration: Duration,
    /// Average collection rate
    pub average_rate: f64,
    /// Collection errors
    pub collection_errors: usize,
    /// Memory usage for collection
    pub memory_usage: usize,
}
