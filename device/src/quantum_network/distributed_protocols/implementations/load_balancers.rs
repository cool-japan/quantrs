//! Load balancer implementations for distributed quantum computation

use super::super::types::*;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use uuid::Uuid;

#[async_trait]
impl LoadBalancer for RoundRobinBalancer {
    fn select_nodes(
        &self,
        partitions: &[CircuitPartition],
        available_nodes: &HashMap<NodeId, NodeInfo>,
        _requirements: &ExecutionRequirements,
    ) -> Result<HashMap<Uuid, NodeId>> {
        let mut assignments = HashMap::new();
        let nodes: Vec<_> = available_nodes.keys().cloned().collect();

        if nodes.is_empty() {
            return Err(DistributedComputationError::ResourceAllocation(
                "No available nodes".to_string(),
            ));
        }

        for partition in partitions {
            let mut index = self
                .current_index
                .lock()
                .expect("Round-robin index mutex poisoned");
            let selected_node = nodes[*index % nodes.len()].clone();
            *index += 1;
            assignments.insert(partition.partition_id, selected_node);
        }

        Ok(assignments)
    }

    fn rebalance_load(
        &self,
        _current_allocation: &HashMap<Uuid, NodeId>,
        _nodes: &HashMap<NodeId, NodeInfo>,
    ) -> Option<HashMap<Uuid, NodeId>> {
        None // Round robin doesn't need rebalancing
    }

    fn predict_execution_time(&self, partition: &CircuitPartition, _node: &NodeInfo) -> Duration {
        partition.estimated_execution_time
    }

    async fn select_node(
        &self,
        available_nodes: &[NodeInfo],
        _requirements: &ResourceRequirements,
    ) -> Result<NodeId> {
        if available_nodes.is_empty() {
            return Err(DistributedComputationError::ResourceAllocation(
                "No available nodes".to_string(),
            ));
        }

        let mut index = self
            .current_index
            .lock()
            .expect("Round-robin index mutex poisoned");
        let selected_node = available_nodes[*index % available_nodes.len()]
            .node_id
            .clone();
        *index += 1;
        Ok(selected_node)
    }

    async fn update_node_metrics(
        &self,
        _node_id: &NodeId,
        _metrics: &PerformanceMetrics,
    ) -> Result<()> {
        Ok(()) // Round robin doesn't use metrics
    }

    fn get_balancer_metrics(&self) -> LoadBalancerMetrics {
        LoadBalancerMetrics {
            total_decisions: 0,
            average_decision_time: Duration::from_millis(1),
            prediction_accuracy: 1.0,
            load_distribution_variance: 0.0,
            total_requests: 0,
            successful_allocations: 0,
            failed_allocations: 0,
            average_response_time: Duration::from_millis(0),
            node_utilization: HashMap::new(),
        }
    }
}

impl Default for RoundRobinBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRobinBalancer {
    pub fn new() -> Self {
        Self {
            current_index: Arc::new(Mutex::new(0)),
        }
    }
}

/// Capability-based load balancer
#[derive(Debug)]
pub struct CapabilityBasedBalancer {
    pub capability_weights: HashMap<String, f64>,
    pub performance_history: Arc<RwLock<HashMap<NodeId, PerformanceHistory>>>,
}

/// ML-optimized load balancer
#[derive(Debug)]
pub struct MLOptimizedBalancer {
    pub model_path: String,
    pub feature_extractor: Arc<FeatureExtractor>,
    pub prediction_cache: Arc<Mutex<HashMap<String, NodeId>>>,
    pub training_data_collector: Arc<TrainingDataCollector>,
}

/// Training data collector for ML models
#[derive(Debug)]
pub struct TrainingDataCollector {
    pub data_buffer: Arc<Mutex<std::collections::VecDeque<TrainingDataPoint>>>,
    pub collection_interval: Duration,
    pub max_buffer_size: usize,
}

impl Default for CapabilityBasedBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilityBasedBalancer {
    pub fn new() -> Self {
        let mut capability_weights = HashMap::new();
        capability_weights.insert("qubit_count".to_string(), 0.3);
        capability_weights.insert("gate_fidelity".to_string(), 0.4);
        capability_weights.insert("connectivity".to_string(), 0.3);

        Self {
            capability_weights,
            performance_history: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl LoadBalancer for CapabilityBasedBalancer {
    fn select_nodes(
        &self,
        partitions: &[CircuitPartition],
        available_nodes: &HashMap<NodeId, NodeInfo>,
        _requirements: &ExecutionRequirements,
    ) -> Result<HashMap<Uuid, NodeId>> {
        let mut allocation = HashMap::new();

        for partition in partitions {
            if let Some((node_id, _)) = available_nodes.iter().next() {
                allocation.insert(partition.partition_id, node_id.clone());
            }
        }

        Ok(allocation)
    }

    fn rebalance_load(
        &self,
        _current_allocation: &HashMap<Uuid, NodeId>,
        _nodes: &HashMap<NodeId, NodeInfo>,
    ) -> Option<HashMap<Uuid, NodeId>> {
        None // No rebalancing needed in simplified implementation
    }

    fn predict_execution_time(&self, partition: &CircuitPartition, _node: &NodeInfo) -> Duration {
        Duration::from_millis(partition.gates.len() as u64 * 10)
    }

    async fn select_node(
        &self,
        available_nodes: &[NodeInfo],
        requirements: &ResourceRequirements,
    ) -> Result<NodeId> {
        // Select the first available node that meets requirements
        available_nodes
            .iter()
            .find(|node| {
                node.capabilities.max_qubits >= requirements.qubits_needed
                    && node
                        .capabilities
                        .gate_fidelities
                        .values()
                        .all(|&fidelity| fidelity >= 0.999) // Default threshold (equivalent to error rate <= 0.001)
            })
            .map(|node| node.node_id.clone())
            .ok_or_else(|| {
                DistributedComputationError::NodeSelectionFailed(
                    "No suitable node found".to_string(),
                )
            })
    }

    async fn update_node_metrics(
        &self,
        _node_id: &NodeId,
        _metrics: &PerformanceMetrics,
    ) -> Result<()> {
        // Update metrics for the specified node
        // In a real implementation, this would update internal state
        Ok(())
    }

    fn get_balancer_metrics(&self) -> LoadBalancerMetrics {
        LoadBalancerMetrics {
            total_decisions: 0,
            average_decision_time: Duration::from_millis(1),
            prediction_accuracy: 1.0,
            load_distribution_variance: 0.0,
            total_requests: 0,
            successful_allocations: 0,
            failed_allocations: 0,
            average_response_time: Duration::from_millis(0),
            node_utilization: HashMap::new(),
        }
    }
}
