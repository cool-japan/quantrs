//! Core types for the unified benchmarking system

use serde::{Deserialize, Serialize};

/// Quantum computing platforms
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumPlatform {
    IBMQuantum {
        device_name: String,
        hub: Option<String>,
    },
    AWSBraket {
        device_arn: String,
        region: String,
    },
    AzureQuantum {
        target_id: String,
        workspace: String,
    },
    IonQ {
        device_name: String,
    },
    Rigetti {
        device_name: String,
    },
    GoogleQuantumAI {
        device_name: String,
    },
    Custom {
        platform_id: String,
        endpoint: String,
    },
}

/// Performance baseline metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BaselineMetric {
    Fidelity,
    ExecutionTime,
    ErrorRate,
    Cost,
    Throughput,
}

/// Baseline metric values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetricValue {
    pub metric: BaselineMetric,
    pub value: f64,
    pub confidence_interval: (f64, f64),
    pub measurement_count: usize,
}

/// Performance baseline data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub platform: QuantumPlatform,
    pub metrics: Vec<BaselineMetricValue>,
    pub last_updated: std::time::SystemTime,
    pub version: String,
}
