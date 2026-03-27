//! # AutoParallelEngine - distribute_tasks_across_nodes_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::distributed_simulator::{DistributedQuantumSimulator, DistributedSimulatorConfig};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use uuid::Uuid;

use super::types::{NodeCapacity, ParallelTask};

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Distribute tasks across cluster nodes
    pub(super) fn distribute_tasks_across_nodes(
        &self,
        tasks: &[ParallelTask],
        distributed_sim: &DistributedQuantumSimulator,
    ) -> QuantRS2Result<Vec<Vec<ParallelTask>>> {
        let cluster_status = distributed_sim.get_cluster_status();
        let num_nodes = cluster_status.len();
        if num_nodes == 0 {
            return Ok(vec![tasks.to_vec()]);
        }
        let node_capacities = Self::analyze_node_capabilities(&cluster_status);
        let mut sorted_tasks: Vec<_> = tasks.to_vec();
        sorted_tasks.sort_by(|a, b| {
            b.cost
                .partial_cmp(&a.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut distributed_tasks = vec![Vec::new(); num_nodes];
        let mut node_loads = vec![0.0; num_nodes];
        for task in sorted_tasks {
            let best_node = Self::select_best_node_for_task(&task, &node_capacities, &node_loads);
            distributed_tasks[best_node].push(task.clone());
            node_loads[best_node] += task.cost;
        }
        Self::rebalance_node_distribution(
            &mut distributed_tasks,
            &node_capacities,
            &mut node_loads,
        )?;
        Ok(distributed_tasks)
    }
    /// Analyze node capabilities from cluster status
    pub(super) fn analyze_node_capabilities(
        cluster_status: &HashMap<Uuid, crate::distributed_simulator::NodeInfo>,
    ) -> Vec<NodeCapacity> {
        cluster_status
            .values()
            .map(|info| NodeCapacity {
                cpu_cores: 4,
                memory_gb: 16.0,
                gpu_available: false,
                network_bandwidth_gbps: 10.0,
                relative_performance: 1.0,
            })
            .collect()
    }
    /// Select the best node for a given task
    pub(super) fn select_best_node_for_task(
        task: &ParallelTask,
        node_capacities: &[NodeCapacity],
        node_loads: &[f64],
    ) -> usize {
        let mut best_node = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (idx, capacity) in node_capacities.iter().enumerate() {
            let load_factor = 1.0 - (node_loads[idx] / capacity.relative_performance).min(1.0);
            let memory_factor = if task.memory_requirement
                < (capacity.memory_gb * 1024.0 * 1024.0 * 1024.0) as usize
            {
                1.0
            } else {
                0.5
            };
            let score = load_factor * capacity.relative_performance * memory_factor;
            if score > best_score {
                best_score = score;
                best_node = idx;
            }
        }
        best_node
    }
    /// Rebalance task distribution if needed
    pub(super) fn rebalance_node_distribution(
        distributed_tasks: &mut [Vec<ParallelTask>],
        node_capacities: &[NodeCapacity],
        node_loads: &mut [f64],
    ) -> QuantRS2Result<()> {
        let total_load: f64 = node_loads.iter().sum();
        let avg_load = total_load / node_loads.len() as f64;
        const IMBALANCE_THRESHOLD: f64 = 0.3;
        for _ in 0..5 {
            let mut rebalanced = false;
            let heavy_nodes: Vec<usize> = node_loads
                .iter()
                .enumerate()
                .filter(|(_, load)| **load > avg_load * (1.0 + IMBALANCE_THRESHOLD))
                .map(|(idx, _)| idx)
                .collect();
            let light_nodes: Vec<usize> = node_loads
                .iter()
                .enumerate()
                .filter(|(_, load)| **load < avg_load * (1.0 - IMBALANCE_THRESHOLD))
                .map(|(idx, _)| idx)
                .collect();
            for &heavy_idx in &heavy_nodes {
                for &light_idx in &light_nodes {
                    if heavy_idx != light_idx {
                        if let Some(task) = distributed_tasks[heavy_idx].pop() {
                            node_loads[heavy_idx] -= task.cost;
                            distributed_tasks[light_idx].push(task.clone());
                            node_loads[light_idx] += task.cost;
                            rebalanced = true;
                            break;
                        }
                    }
                }
                if rebalanced {
                    break;
                }
            }
            if !rebalanced {
                break;
            }
        }
        Ok(())
    }
}
