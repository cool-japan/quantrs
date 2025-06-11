//! Configuration structures for quantum network protocols

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// Core configuration structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionNegotiationConfig {
    pub supported_versions: Vec<String>,
    pub preferred_version: String,
    pub compatibility_matrix: HashMap<String, Vec<String>>,
    pub negotiation_timeout: Duration,
    pub fallback_version: String,
}

impl Default for VersionNegotiationConfig {
    fn default() -> Self {
        Self {
            supported_versions: vec!["1.0".to_string(), "1.1".to_string()],
            preferred_version: "1.1".to_string(),
            compatibility_matrix: HashMap::new(),
            negotiation_timeout: Duration::from_secs(30),
            fallback_version: "1.0".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionManagementConfig {
    pub max_connections: u32,
    pub connection_timeout: Duration,
    pub keepalive_interval: Duration,
    pub retry_attempts: u32,
    pub backoff_strategy: String,
    pub connection_pooling: bool,
}

impl Default for ConnectionManagementConfig {
    fn default() -> Self {
        Self {
            max_connections: 100,
            connection_timeout: Duration::from_secs(60),
            keepalive_interval: Duration::from_secs(30),
            retry_attempts: 3,
            backoff_strategy: "exponential".to_string(),
            connection_pooling: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowControlConfig {
    pub window_size: u32,
    pub max_packet_size: u32,
    pub buffer_size: u32,
    pub flow_control_algorithm: String,
    pub congestion_avoidance: bool,
}

impl Default for FlowControlConfig {
    fn default() -> Self {
        Self {
            window_size: 64000,
            max_packet_size: 1500,
            buffer_size: 128000,
            flow_control_algorithm: "sliding_window".to_string(),
            congestion_avoidance: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionControlConfig {
    pub algorithm: String,
    pub initial_window: u32,
    pub max_window: u32,
    pub congestion_threshold: f64,
    pub recovery_strategy: String,
}

impl Default for CongestionControlConfig {
    fn default() -> Self {
        Self {
            algorithm: "TCP_Cubic".to_string(),
            initial_window: 10,
            max_window: 65536,
            congestion_threshold: 0.8,
            recovery_strategy: "fast_recovery".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumQoSConfig {
    pub priority_levels: Vec<String>,
    pub bandwidth_allocation: HashMap<String, f64>,
    pub latency_requirements: HashMap<String, Duration>,
    pub reliability_requirements: HashMap<String, f64>,
}

impl Default for QuantumQoSConfig {
    fn default() -> Self {
        let mut bandwidth_allocation = HashMap::new();
        bandwidth_allocation.insert("high".to_string(), 0.5);
        bandwidth_allocation.insert("medium".to_string(), 0.3);
        bandwidth_allocation.insert("low".to_string(), 0.2);

        let mut latency_requirements = HashMap::new();
        latency_requirements.insert("high".to_string(), Duration::from_millis(10));
        latency_requirements.insert("medium".to_string(), Duration::from_millis(50));
        latency_requirements.insert("low".to_string(), Duration::from_millis(100));

        let mut reliability_requirements = HashMap::new();
        reliability_requirements.insert("high".to_string(), 0.999);
        reliability_requirements.insert("medium".to_string(), 0.99);
        reliability_requirements.insert("low".to_string(), 0.95);

        Self {
            priority_levels: vec!["high".to_string(), "medium".to_string(), "low".to_string()],
            bandwidth_allocation,
            latency_requirements,
            reliability_requirements,
        }
    }
}
