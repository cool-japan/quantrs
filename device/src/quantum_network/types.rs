//! Common types and data structures for quantum network protocols

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// Manager placeholder types
#[derive(Debug, Clone)]
pub struct QuantumTopologyManager;

#[derive(Debug, Clone)]
pub struct QuantumRoutingEngine;

#[derive(Debug, Clone)]
pub struct NetworkPerformanceAnalyzer;

#[derive(Debug, Clone)]
pub struct NetworkOptimizer;

#[derive(Debug, Clone)]
pub struct NetworkErrorCorrector;

#[derive(Debug, Clone)]
pub struct NetworkFaultDetector;

#[derive(Debug, Clone)]
pub struct QuantumNetworkMonitor;

#[derive(Debug, Clone)]
pub struct NetworkAnalyticsEngine;

#[derive(Debug, Clone)]
pub struct QuantumNetworkState;

#[derive(Debug, Clone)]
pub struct NetworkSessionManager;

// Supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwappingResources {
    pub entanglement_pairs: u32,
    pub classical_communication: u32,
    pub storage_time: Duration,
}

impl Default for SwappingResources {
    fn default() -> Self {
        Self {
            entanglement_pairs: 2,
            classical_communication: 2,
            storage_time: Duration::from_millis(100),
        }
    }
}

// Network Quality and Performance types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkQualityMetrics {
    pub latency: Duration,
    pub throughput: f64,
    pub fidelity: f64,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkPerformanceMetrics {
    pub bandwidth_utilization: f64,
    pub packet_loss_rate: f64,
    pub jitter: Duration,
    pub availability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkSecurityAnalysis {
    pub security_level: String,
    pub vulnerabilities: Vec<String>,
    pub threat_assessment: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkOptimizationResult {
    pub improvements: HashMap<String, f64>,
    pub cost_reduction: f64,
    pub performance_gain: f64,
}
