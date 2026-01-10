//! Distributed computing types for scientific performance optimization.
//!
//! This module contains cluster management, communication,
//! fault tolerance, and distributed coordination.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::config::{
    ClusterConfig, CommunicationProtocol, DistributedComputingConfig, NodeResources,
};

/// Distributed coordinator for cluster computing
pub struct DistributedCoordinator {
    /// Configuration
    pub config: DistributedComputingConfig,
    /// Cluster manager
    pub cluster_manager: ClusterManager,
    /// Communication manager
    pub communication_manager: CommunicationManager,
    /// Fault tolerance manager
    pub fault_tolerance_manager: FaultToleranceManager,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator
    #[must_use]
    pub fn new(config: DistributedComputingConfig) -> Self {
        Self {
            config,
            cluster_manager: ClusterManager::new(),
            communication_manager: CommunicationManager::new(),
            fault_tolerance_manager: FaultToleranceManager::new(),
        }
    }

    /// Check if distributed computing is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enable_distributed
    }

    /// Get cluster size
    #[must_use]
    pub fn cluster_size(&self) -> usize {
        self.cluster_manager.active_nodes.len()
    }
}

/// Cluster manager for node coordination
#[derive(Debug)]
pub struct ClusterManager {
    /// Cluster configuration
    pub config: ClusterConfig,
    /// Active nodes
    pub active_nodes: HashMap<String, ClusterNode>,
    /// Node statistics
    pub node_statistics: HashMap<String, NodeStatistics>,
}

impl ClusterManager {
    /// Create a new cluster manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ClusterConfig::default(),
            active_nodes: HashMap::new(),
            node_statistics: HashMap::new(),
        }
    }

    /// Add a node to the cluster
    pub fn add_node(&mut self, address: String, resources: NodeResources) {
        let node = ClusterNode {
            address: address.clone(),
            resources,
            status: NodeStatus::Active,
            current_workload: NodeWorkload::default(),
        };
        self.active_nodes.insert(address.clone(), node);
        self.node_statistics
            .insert(address, NodeStatistics::default());
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, address: &str) -> Option<ClusterNode> {
        self.node_statistics.remove(address);
        self.active_nodes.remove(address)
    }

    /// Get available nodes
    #[must_use]
    pub fn available_nodes(&self) -> Vec<&ClusterNode> {
        self.active_nodes
            .values()
            .filter(|n| n.status == NodeStatus::Active)
            .collect()
    }

    /// Update node status
    pub fn update_node_status(&mut self, address: &str, status: NodeStatus) {
        if let Some(node) = self.active_nodes.get_mut(address) {
            node.status = status;
        }
    }

    /// Get node by address
    #[must_use]
    pub fn get_node(&self, address: &str) -> Option<&ClusterNode> {
        self.active_nodes.get(address)
    }
}

impl Default for ClusterManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Cluster node representation
#[derive(Debug)]
pub struct ClusterNode {
    /// Node address
    pub address: String,
    /// Node resources
    pub resources: NodeResources,
    /// Node status
    pub status: NodeStatus,
    /// Current workload
    pub current_workload: NodeWorkload,
}

impl ClusterNode {
    /// Check if node is available for work
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.status == NodeStatus::Active && self.current_workload.cpu_utilization < 0.9
    }

    /// Get available capacity
    #[must_use]
    pub fn available_capacity(&self) -> f64 {
        1.0 - self.current_workload.cpu_utilization
    }
}

/// Node status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is active and available
    Active,
    /// Node is busy
    Busy,
    /// Node is temporarily unavailable
    Unavailable,
    /// Node has failed
    Failed,
    /// Node is in maintenance
    Maintenance,
}

/// Node workload information
#[derive(Debug, Clone)]
pub struct NodeWorkload {
    /// Active tasks
    pub active_tasks: Vec<String>,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
}

impl Default for NodeWorkload {
    fn default() -> Self {
        Self {
            active_tasks: Vec::new(),
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            network_utilization: 0.0,
        }
    }
}

impl NodeWorkload {
    /// Calculate overall load
    #[must_use]
    pub fn overall_load(&self) -> f64 {
        (self.cpu_utilization + self.memory_utilization + self.network_utilization) / 3.0
    }
}

/// Communication manager for inter-node communication
#[derive(Debug)]
pub struct CommunicationManager {
    /// Communication protocol
    pub protocol: CommunicationProtocol,
    /// Active connections
    pub connections: HashMap<String, Connection>,
    /// Message queues
    pub message_queues: HashMap<String, VecDeque<Message>>,
    /// Communication statistics
    pub statistics: CommunicationStatistics,
}

impl CommunicationManager {
    /// Create a new communication manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            protocol: CommunicationProtocol::TCP,
            connections: HashMap::new(),
            message_queues: HashMap::new(),
            statistics: CommunicationStatistics::default(),
        }
    }

    /// Establish connection to a node
    pub fn connect(&mut self, address: &str) -> Result<(), String> {
        if self.connections.contains_key(address) {
            return Ok(());
        }

        let connection = Connection {
            id: format!("conn_{}", self.connections.len()),
            remote_address: address.to_string(),
            status: ConnectionStatus::Active,
            statistics: ConnectionStatistics::default(),
        };

        self.connections.insert(address.to_string(), connection);
        self.message_queues
            .insert(address.to_string(), VecDeque::new());
        self.statistics.connections_established += 1;

        Ok(())
    }

    /// Disconnect from a node
    pub fn disconnect(&mut self, address: &str) {
        if let Some(mut conn) = self.connections.remove(address) {
            conn.status = ConnectionStatus::Disconnected;
            self.statistics.connections_closed += 1;
        }
        self.message_queues.remove(address);
    }

    /// Send a message
    pub fn send(&mut self, destination: &str, message: Message) -> Result<(), String> {
        if let Some(queue) = self.message_queues.get_mut(destination) {
            queue.push_back(message);
            self.statistics.messages_sent += 1;
            Ok(())
        } else {
            Err(format!("No connection to {destination}"))
        }
    }

    /// Receive messages from a node
    pub fn receive(&mut self, source: &str) -> Vec<Message> {
        let mut messages = Vec::new();
        if let Some(queue) = self.message_queues.get_mut(source) {
            while let Some(msg) = queue.pop_front() {
                messages.push(msg);
                self.statistics.messages_received += 1;
            }
        }
        messages
    }
}

impl Default for CommunicationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection representation
#[derive(Debug)]
pub struct Connection {
    /// Connection identifier
    pub id: String,
    /// Remote address
    pub remote_address: String,
    /// Connection status
    pub status: ConnectionStatus,
    /// Connection statistics
    pub statistics: ConnectionStatistics,
}

/// Connection status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// Connection is active
    Active,
    /// Connection is being established
    Connecting,
    /// Connection is temporarily disconnected
    Disconnected,
    /// Connection has failed
    Failed,
}

/// Message for inter-node communication
#[derive(Debug, Clone)]
pub struct Message {
    /// Message identifier
    pub id: String,
    /// Source node
    pub source: String,
    /// Destination node
    pub destination: String,
    /// Message type
    pub message_type: MessageType,
    /// Message payload
    pub payload: Vec<u8>,
    /// Timestamp
    pub timestamp: Instant,
}

impl Message {
    /// Create a new message
    #[must_use]
    pub fn new(
        source: String,
        destination: String,
        message_type: MessageType,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            id: uuid_simple(),
            source,
            destination,
            message_type,
            payload,
            timestamp: Instant::now(),
        }
    }
}

/// Message types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageType {
    /// Task assignment
    TaskAssignment,
    /// Task result
    TaskResult,
    /// Heartbeat
    Heartbeat,
    /// Status update
    StatusUpdate,
    /// Error notification
    Error,
    /// Control message
    Control,
}

/// Fault tolerance manager
#[derive(Debug, Clone, Default)]
pub struct FaultToleranceManager {
    /// Failed nodes
    pub failed_nodes: Vec<String>,
    /// Recovery attempts
    pub recovery_attempts: HashMap<String, u32>,
    /// Checkpoints
    pub checkpoints: HashMap<String, Checkpoint>,
}

impl FaultToleranceManager {
    /// Create a new fault tolerance manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            failed_nodes: Vec::new(),
            recovery_attempts: HashMap::new(),
            checkpoints: HashMap::new(),
        }
    }

    /// Record node failure
    pub fn record_failure(&mut self, node_address: &str) {
        if !self.failed_nodes.contains(&node_address.to_string()) {
            self.failed_nodes.push(node_address.to_string());
        }
        *self
            .recovery_attempts
            .entry(node_address.to_string())
            .or_insert(0) += 1;
    }

    /// Record node recovery
    pub fn record_recovery(&mut self, node_address: &str) {
        self.failed_nodes.retain(|n| n != node_address);
    }

    /// Create checkpoint
    pub fn create_checkpoint(&mut self, task_id: &str, data: Vec<u8>) {
        let checkpoint = Checkpoint {
            task_id: task_id.to_string(),
            data,
            timestamp: Instant::now(),
        };
        self.checkpoints.insert(task_id.to_string(), checkpoint);
    }

    /// Get checkpoint
    #[must_use]
    pub fn get_checkpoint(&self, task_id: &str) -> Option<&Checkpoint> {
        self.checkpoints.get(task_id)
    }
}

/// Checkpoint for fault recovery
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Task identifier
    pub task_id: String,
    /// Checkpoint data
    pub data: Vec<u8>,
    /// Timestamp
    pub timestamp: Instant,
}

// Statistics types

/// Node statistics
#[derive(Debug, Clone, Default)]
pub struct NodeStatistics {
    /// Tasks completed
    pub tasks_completed: u64,
    /// Tasks failed
    pub tasks_failed: u64,
    /// Total execution time
    pub total_execution_time: std::time::Duration,
    /// Average task time
    pub avg_task_time: std::time::Duration,
}

/// Connection statistics
#[derive(Debug, Clone, Default)]
pub struct ConnectionStatistics {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Errors
    pub errors: u64,
}

/// Communication statistics
#[derive(Debug, Clone, Default)]
pub struct CommunicationStatistics {
    /// Connections established
    pub connections_established: u64,
    /// Connections closed
    pub connections_closed: u64,
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Total bytes transferred
    pub total_bytes: u64,
}

/// Generate a simple UUID-like string
fn uuid_simple() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("msg_{now:x}")
}
