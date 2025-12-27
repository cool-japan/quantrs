//! Network optimization algorithms and configurations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthAllocationConfig {
    pub total_bandwidth: f64,
    pub priority_based_allocation: bool,
    pub dynamic_allocation: bool,
    pub oversubscription_ratio: f64,
}

impl Default for BandwidthAllocationConfig {
    fn default() -> Self {
        Self {
            total_bandwidth: 1_000_000.0, // 1 Mbps
            priority_based_allocation: true,
            dynamic_allocation: true,
            oversubscription_ratio: 1.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyOptimizationConfig {
    pub target_latency: Duration,
    pub route_optimization: bool,
    pub caching_enabled: bool,
}

impl Default for LatencyOptimizationConfig {
    fn default() -> Self {
        Self {
            target_latency: Duration::from_millis(10),
            route_optimization: true,
            caching_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveNetworkConfig {
    pub objective_weights: HashMap<String, f64>,
}

impl Default for MultiObjectiveNetworkConfig {
    fn default() -> Self {
        let mut objective_weights = HashMap::new();
        objective_weights.insert("latency".to_string(), 0.3);
        objective_weights.insert("throughput".to_string(), 0.3);
        objective_weights.insert("fidelity".to_string(), 0.4);

        Self { objective_weights }
    }
}
