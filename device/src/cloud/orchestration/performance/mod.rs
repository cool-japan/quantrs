//! Performance optimization configurations for cloud orchestration
//!
//! This module contains comprehensive performance optimization settings including
//! latency optimization, throughput optimization, network configurations,
//! resource scaling, and performance prediction capabilities.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

pub mod latency;
pub mod network;
pub mod prediction;
pub mod scaling;
pub mod throughput;

pub use latency::*;
pub use network::*;
pub use prediction::*;
pub use scaling::*;
pub use throughput::*;

/// Cloud performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudPerformanceConfig {
    /// Performance optimization strategies
    pub optimization_strategies: Vec<PerformanceOptimizationStrategy>,
    /// Latency optimization
    pub latency_optimization: LatencyOptimizationConfig,
    /// Throughput optimization
    pub throughput_optimization: ThroughputOptimizationConfig,
    /// QoS requirements
    pub qos_requirements: QoSRequirements,
    /// Performance prediction
    pub performance_prediction: PerformancePredictionConfig,
}

/// Performance optimization strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceOptimizationStrategy {
    LatencyMinimization,
    ThroughputMaximization,
    ResourceUtilizationOptimization,
    EnergyEfficiencyOptimization,
    CostPerformanceOptimization,
    QoSOptimization,
    CustomStrategy(String),
}

/// QoS requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSRequirements {
    /// Latency requirements
    pub latency: LatencyRequirements,
    /// Throughput requirements
    pub throughput: ThroughputRequirements,
    /// Availability requirements
    pub availability: AvailabilityRequirements,
    /// Reliability requirements
    pub reliability: ReliabilityRequirements,
}

/// Latency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRequirements {
    /// Maximum latency
    pub max_latency: Duration,
    /// Target latency
    pub target_latency: Duration,
    /// Percentile requirements
    pub percentiles: BTreeMap<String, Duration>,
}

/// Throughput requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputRequirements {
    /// Minimum throughput
    pub min_throughput: f64,
    /// Target throughput
    pub target_throughput: f64,
    /// Peak throughput
    pub peak_throughput: f64,
}

/// Availability requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityRequirements {
    /// Target availability
    pub target_availability: f64,
    /// Maximum downtime
    pub max_downtime: Duration,
    /// Recovery time objective
    pub rto: Duration,
    /// Recovery point objective
    pub rpo: Duration,
}

/// Reliability requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityRequirements {
    /// Mean time between failures
    pub mtbf: Duration,
    /// Mean time to repair
    pub mttr: Duration,
    /// Error rate threshold
    pub error_rate_threshold: f64,
}
