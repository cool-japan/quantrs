//! # RoundRobinBalancer - Trait Implementations
//!
//! This module contains trait implementations for `RoundRobinBalancer`.
//!
//! ## Implemented Traits
//!
//! - `LoadBalancer`
//! - `LoadBalancer`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

use super::type_definitions::*;
use crate::quantum_network::distributed_protocols::{NodeId, NodeInfo, PerformanceMetrics};

#[async_trait]
impl LoadBalancer for RoundRobinBalancer {
    async fn select_node(
        &self,
        available_nodes: &[NodeInfo],
        _requirements: &ResourceRequirements,
    ) -> Result<NodeId> {
        if available_nodes.is_empty() {
            return Err(NetworkOptimizationError::TopologyOptimizationFailed(
                "No available nodes".to_string(),
            ));
        }
        let index = self
            .current_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % available_nodes.len();
        Ok(available_nodes[index].node_id.clone())
    }
    async fn update_node_metrics(
        &self,
        _node_id: &NodeId,
        _metrics: &PerformanceMetrics,
    ) -> Result<()> {
        Ok(())
    }
    fn get_balancer_metrics(&self) -> LoadBalancerMetrics {
        LoadBalancerMetrics {
            total_decisions: 0,
            average_decision_time: Duration::from_millis(1),
            prediction_accuracy: 1.0,
            load_distribution_variance: 0.0,
        }
    }
}

#[async_trait]
impl crate::quantum_network::distributed_protocols::LoadBalancer for RoundRobinBalancer {
    fn select_nodes(
        &self,
        partitions: &[crate::quantum_network::distributed_protocols::CircuitPartition],
        available_nodes: &HashMap<NodeId, crate::quantum_network::distributed_protocols::NodeInfo>,
        _requirements: &crate::quantum_network::distributed_protocols::ExecutionRequirements,
    ) -> std::result::Result<
        HashMap<Uuid, NodeId>,
        crate::quantum_network::distributed_protocols::DistributedComputationError,
    > {
        let mut allocation = HashMap::new();
        let nodes: Vec<_> = available_nodes.keys().cloned().collect();
        if nodes.is_empty() {
            return Err(
                crate::quantum_network::distributed_protocols::DistributedComputationError::ResourceAllocation(
                    "No available nodes".to_string(),
                ),
            );
        }
        for partition in partitions {
            let index = self
                .current_index
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                % nodes.len();
            allocation.insert(partition.partition_id, nodes[index].clone());
        }
        Ok(allocation)
    }
    fn rebalance_load(
        &self,
        _current_allocation: &HashMap<Uuid, NodeId>,
        _nodes: &HashMap<NodeId, crate::quantum_network::distributed_protocols::NodeInfo>,
    ) -> Option<HashMap<Uuid, NodeId>> {
        None
    }
    fn predict_execution_time(
        &self,
        partition: &crate::quantum_network::distributed_protocols::CircuitPartition,
        _node: &crate::quantum_network::distributed_protocols::NodeInfo,
    ) -> Duration {
        partition.estimated_execution_time
    }
    async fn select_node(
        &self,
        available_nodes: &[crate::quantum_network::distributed_protocols::NodeInfo],
        _requirements: &crate::quantum_network::distributed_protocols::ResourceRequirements,
    ) -> std::result::Result<
        NodeId,
        crate::quantum_network::distributed_protocols::DistributedComputationError,
    > {
        if available_nodes.is_empty() {
            return Err(
                crate::quantum_network::distributed_protocols::DistributedComputationError::ResourceAllocation(
                    "No available nodes".to_string(),
                ),
            );
        }
        let index = self
            .current_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % available_nodes.len();
        Ok(available_nodes[index].node_id.clone())
    }
    async fn update_node_metrics(
        &self,
        _node_id: &NodeId,
        _metrics: &crate::quantum_network::distributed_protocols::PerformanceMetrics,
    ) -> std::result::Result<
        (),
        crate::quantum_network::distributed_protocols::DistributedComputationError,
    > {
        Ok(())
    }
    fn get_balancer_metrics(
        &self,
    ) -> crate::quantum_network::distributed_protocols::LoadBalancerMetrics {
        crate::quantum_network::distributed_protocols::LoadBalancerMetrics {
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
