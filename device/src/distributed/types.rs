//! Common types and data structures for distributed orchestration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use super::config::DistributedOrchestratorConfig;

// Main orchestrator type
//
// Holds the real, locally-computable state needed for node registration and
// workflow scheduling: a node registry, a workflow registry, and a monotonic
// execution-id counter. Actual network dispatch to remote nodes is not
// implemented (it requires a real inter-node transport), so
// `execute_distributed` reports that limitation honestly instead of
// fabricating a successful result.
#[derive(Debug)]
pub struct DistributedQuantumOrchestrator {
    pub(crate) config: DistributedOrchestratorConfig,
    pub(crate) nodes: Arc<Mutex<HashMap<String, NodeInfo>>>,
    pub(crate) workflows: Arc<Mutex<HashMap<String, DistributedWorkflow>>>,
    pub(crate) execution_counter: Arc<Mutex<u64>>,
}

// Core execution types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedExecutionResult {
    pub execution_id: String,
    pub status: DistributedExecutionStatus,
    pub results: HashMap<String, String>,
    pub performance_metrics: DistributedPerformanceAnalytics,
    pub execution_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedExecutionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: SocketAddr,
    pub capabilities: NodeCapabilities,
    pub status: NodeStatus,
    #[serde(skip)]
    pub last_heartbeat: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub max_qubits: u32,
    pub supported_gates: Vec<String>,
    pub connectivity: HashMap<u32, Vec<u32>>,
    pub error_rates: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Available,
    Busy,
    Offline,
    Maintenance,
    Error,
}

// Workflow types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedWorkflow {
    pub workflow_id: String,
    pub workflow_type: DistributedWorkflowType,
    pub steps: Vec<String>,
    pub dependencies: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedWorkflowType {
    Sequential,
    Parallel,
    ConditionalBranching,
    IterativeLoop,
    EventDriven,
}

// Event and command types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedEvent {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: SystemTime,
    pub data: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCommand {
    pub command_id: String,
    pub command_type: String,
    pub target_node: String,
    pub parameters: HashMap<String, String>,
}

// Circuit decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitDecompositionResult {
    pub subcircuits: Vec<String>,
    pub dependencies: HashMap<String, Vec<String>>,
    pub resource_requirements: HashMap<String, u32>,
}

// Default implementations
impl Default for DistributedExecutionResult {
    fn default() -> Self {
        Self {
            execution_id: "default".to_string(),
            status: DistributedExecutionStatus::Pending,
            results: HashMap::new(),
            performance_metrics: DistributedPerformanceAnalytics::default(),
            execution_time: Duration::from_secs(0),
        }
    }
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            max_qubits: 5,
            supported_gates: vec![
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "CNOT".to_string(),
            ],
            connectivity: HashMap::new(),
            error_rates: HashMap::new(),
        }
    }
}

// Placeholder for analytics type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DistributedPerformanceAnalytics {
    pub throughput: f64,
    pub latency: Duration,
    pub error_rate: f64,
    pub resource_utilization: f64,
}
