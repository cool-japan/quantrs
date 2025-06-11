//! Latency optimization configurations

use super::network::NetworkOptimizationConfig;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Latency optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyOptimizationConfig {
    /// Target latency
    pub target_latency: Duration,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Optimization techniques
    pub techniques: Vec<LatencyOptimizationTechnique>,
    /// Network optimization
    pub network_optimization: NetworkOptimizationConfig,
}

/// Latency optimization techniques
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LatencyOptimizationTechnique {
    GeographicProximity,
    Caching,
    ConnectionPooling,
    RequestBatching,
    PreemptiveScheduling,
    Custom(String),
}
