//! Data Collection Configuration Types

use serde::{Deserialize, Serialize};

/// Data collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCollectionConfig {
    /// Collection interval in seconds
    pub collection_interval: u64,
    /// Buffer size for historical data
    pub buffer_size: usize,
    /// Data retention period in days
    pub retention_days: u32,
    /// Metrics to collect
    pub metrics_config: MetricsConfig,
    /// Aggregation settings
    pub aggregation_config: AggregationConfig,
    /// Sampling configuration
    pub sampling_config: SamplingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub performance_metrics: Vec<PerformanceMetric>,
    pub resource_metrics: Vec<ResourceMetric>,
    pub quality_metrics: Vec<QualityMetric>,
    pub custom_metrics: Vec<CustomMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    pub aggregation_functions: Vec<AggregationFunction>,
    pub time_windows: Vec<TimeWindow>,
    pub grouping_dimensions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub sampling_strategy: SamplingStrategy,
    pub sample_rate: f64,
    pub adaptive_sampling: bool,
    pub quality_based_sampling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PerformanceMetric {
    Fidelity,
    Throughput,
    Latency,
    ErrorRate,
    SuccessRate,
    CircuitDepth,
    GateCount,
    SwapCount,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResourceMetric {
    CpuUtilization,
    MemoryUtilization,
    NetworkUtilization,
    StorageUtilization,
    QuantumUtilization,
    QueueLength,
    ActiveConnections,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QualityMetric {
    ProcessFidelity,
    MeasurementFidelity,
    GateFidelity,
    StatePreparationFidelity,
    ReadoutFidelity,
    CoherenceTime,
    CalibrationDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CustomMetric {
    UserDefined(String),
    ApplicationSpecific(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AggregationFunction {
    Mean,
    Median,
    Min,
    Max,
    Sum,
    Count,
    Percentile(f64),
    StandardDeviation,
    Variance,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TimeWindow {
    Seconds(u64),
    Minutes(u64),
    Hours(u64),
    Days(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SamplingStrategy {
    Fixed,
    Adaptive,
    QualityBased,
    EventDriven,
    Hybrid,
}
