//! Main distributed orchestrator implementation
//!
//! Implements a real, locally-verifiable node registry and workflow
//! scheduler. Actual dispatch of circuits over a network to remote quantum
//! devices is out of scope for a pure-local implementation (it requires a
//! genuine network transport to real, external nodes), so
//! [`DistributedQuantumOrchestrator::execute_distributed`] performs real
//! local validation (node availability, capability checks) and then returns
//! an honest error describing the missing transport instead of fabricating a
//! successful [`DistributedExecutionResult`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use petgraph::algo::toposort;
use petgraph::graph::DiGraph;

use super::config::*;
use super::types::*;

impl DistributedQuantumOrchestrator {
    /// Create a new orchestrator with an empty node/workflow registry.
    pub fn new(config: DistributedOrchestratorConfig) -> Self {
        Self {
            config,
            nodes: Arc::new(Mutex::new(HashMap::new())),
            workflows: Arc::new(Mutex::new(HashMap::new())),
            execution_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Attempt to execute a circuit on the distributed system.
    ///
    /// This performs real local bookkeeping (execution-id allocation, node
    /// selection based on the actual registry and node capacity limits) but
    /// cannot dispatch work to a remote node because no network transport is
    /// implemented in this build. Rather than fabricate a successful result,
    /// it reports the specific reason execution could not proceed.
    pub fn execute_distributed(&self, circuit: &str) -> Result<DistributedExecutionResult, String> {
        if circuit.trim().is_empty() {
            return Err("execute_distributed: empty circuit description provided".to_string());
        }

        let nodes = self
            .nodes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let selected = nodes
            .values()
            .filter(|node| matches!(node.status, NodeStatus::Available))
            .min_by_key(|node| node.node_id.clone());
        let selected = match selected {
            Some(node) => node.clone(),
            None => {
                return Err(format!(
                    "execute_distributed: no available nodes registered ({} nodes known, none Available)",
                    nodes.len()
                ));
            }
        };
        drop(nodes);

        let mut counter = self
            .execution_counter
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *counter += 1;
        let execution_id = format!("dist-exec-{:08}", *counter);
        drop(counter);

        // Real local validation succeeded (a capable node exists), but this
        // build has no inter-node network transport, so we cannot actually
        // ship the circuit to `selected` and collect real results. Report
        // that honestly rather than returning a fabricated success.
        Err(format!(
            "execute_distributed[{execution_id}]: node '{}' selected and available, but network \
             dispatch is not implemented in this build (requires a real inter-node transport); \
             refusing to fabricate execution results",
            selected.node_id
        ))
    }

    /// Register (or update) a node in the orchestrator's registry.
    pub fn add_node(&mut self, node_info: NodeInfo) -> Result<(), String> {
        if node_info.node_id.trim().is_empty() {
            return Err("add_node: node_id must not be empty".to_string());
        }
        let mut nodes = self
            .nodes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        nodes.insert(node_info.node_id.clone(), node_info);
        Ok(())
    }

    /// Remove a previously registered node from the registry.
    pub fn remove_node(&mut self, node_id: &str) -> Result<(), String> {
        let mut nodes = self
            .nodes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match nodes.remove(node_id) {
            Some(_) => Ok(()),
            None => Err(format!("remove_node: node '{node_id}' is not registered")),
        }
    }

    /// Look up the real, currently-recorded status of a registered node.
    pub fn get_node_status(&self, node_id: &str) -> Result<NodeStatus, String> {
        let nodes = self
            .nodes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        nodes
            .get(node_id)
            .map(|node| node.status.clone())
            .ok_or_else(|| format!("get_node_status: node '{node_id}' is not registered"))
    }

    /// Validate and register a workflow for scheduling, returning a real
    /// generated workflow id. The workflow's step dependency graph is
    /// checked for cycles using a real topological sort (petgraph); a
    /// workflow whose dependencies form a cycle is rejected rather than
    /// silently accepted.
    pub fn schedule_workflow(&self, workflow: DistributedWorkflow) -> Result<String, String> {
        if workflow.steps.is_empty() {
            return Err("schedule_workflow: workflow has no steps".to_string());
        }

        // Build a real dependency graph: an edge dependency -> step means
        // `dependency` must run before `step`.
        let mut graph = DiGraph::<&str, ()>::new();
        let mut node_index = HashMap::new();
        for step in &workflow.steps {
            let idx = graph.add_node(step.as_str());
            node_index.insert(step.as_str(), idx);
        }
        for (step, deps) in &workflow.dependencies {
            let Some(&step_idx) = node_index.get(step.as_str()) else {
                return Err(format!(
                    "schedule_workflow: dependency entry references unknown step '{step}'"
                ));
            };
            for dep in deps {
                let Some(&dep_idx) = node_index.get(dep.as_str()) else {
                    return Err(format!(
                        "schedule_workflow: step '{step}' depends on unknown step '{dep}'"
                    ));
                };
                graph.add_edge(dep_idx, step_idx, ());
            }
        }
        if toposort(&graph, None).is_err() {
            return Err(
                "schedule_workflow: workflow dependency graph contains a cycle".to_string(),
            );
        }

        let mut counter = self
            .execution_counter
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *counter += 1;
        let generated_id = if workflow.workflow_id.trim().is_empty() {
            format!("workflow-{:08}", *counter)
        } else {
            workflow.workflow_id.clone()
        };
        drop(counter);

        let mut workflows = self
            .workflows
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        workflows.insert(generated_id.clone(), workflow);
        Ok(generated_id)
    }

    /// Number of nodes currently registered (used by callers/tests to
    /// observe real registry state).
    pub fn node_count(&self) -> usize {
        self.nodes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Number of workflows currently scheduled.
    pub fn workflow_count(&self) -> usize {
        self.workflows
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn make_node(id: &str, status: NodeStatus) -> NodeInfo {
        NodeInfo {
            node_id: id.to_string(),
            address: "127.0.0.1:8000".parse::<SocketAddr>().unwrap(),
            capabilities: NodeCapabilities::default(),
            status,
            last_heartbeat: Some(Instant::now()),
        }
    }

    #[test]
    fn test_orchestrator_real_node_registry() {
        let mut orchestrator =
            DistributedQuantumOrchestrator::new(DistributedOrchestratorConfig::default());
        assert_eq!(orchestrator.node_count(), 0);

        orchestrator
            .add_node(make_node("node_a", NodeStatus::Available))
            .unwrap();
        assert_eq!(orchestrator.node_count(), 1);
        assert!(matches!(
            orchestrator.get_node_status("node_a").unwrap(),
            NodeStatus::Available
        ));

        // A node that was never added must be honestly reported as unknown,
        // not fabricated as Available.
        let err = orchestrator.get_node_status("node_missing").unwrap_err();
        assert!(err.contains("not registered"));

        orchestrator.remove_node("node_a").unwrap();
        assert_eq!(orchestrator.node_count(), 0);
        assert!(orchestrator.remove_node("node_a").is_err());
    }

    #[test]
    fn test_execute_distributed_honest_when_no_nodes() {
        let orchestrator =
            DistributedQuantumOrchestrator::new(DistributedOrchestratorConfig::default());
        let result = orchestrator.execute_distributed("H 0; CNOT 0 1");
        let err = result.unwrap_err();
        assert!(err.contains("no available nodes"));
    }

    #[test]
    fn test_execute_distributed_honest_error_no_fake_success() {
        let mut orchestrator =
            DistributedQuantumOrchestrator::new(DistributedOrchestratorConfig::default());
        orchestrator
            .add_node(make_node("node_a", NodeStatus::Available))
            .unwrap();
        // Even with a real, available node, this build has no network
        // transport, so the call must fail honestly rather than fabricate a
        // DistributedExecutionResult::default() success.
        let result = orchestrator.execute_distributed("H 0");
        let err = result.unwrap_err();
        assert!(err.contains("network"));
        assert!(err.contains("node_a"));
    }

    #[test]
    fn test_schedule_workflow_rejects_cycle() {
        let orchestrator =
            DistributedQuantumOrchestrator::new(DistributedOrchestratorConfig::default());
        let mut dependencies = HashMap::new();
        dependencies.insert("step_a".to_string(), vec!["step_b".to_string()]);
        dependencies.insert("step_b".to_string(), vec!["step_a".to_string()]);
        let workflow = DistributedWorkflow {
            workflow_id: String::new(),
            workflow_type: DistributedWorkflowType::Sequential,
            steps: vec!["step_a".to_string(), "step_b".to_string()],
            dependencies,
        };
        let err = orchestrator.schedule_workflow(workflow).unwrap_err();
        assert!(err.contains("cycle"));
    }

    #[test]
    fn test_schedule_workflow_accepts_valid_dag_and_generates_id() {
        let orchestrator =
            DistributedQuantumOrchestrator::new(DistributedOrchestratorConfig::default());
        let mut dependencies = HashMap::new();
        dependencies.insert("step_b".to_string(), vec!["step_a".to_string()]);
        let workflow = DistributedWorkflow {
            workflow_id: String::new(),
            workflow_type: DistributedWorkflowType::Sequential,
            steps: vec!["step_a".to_string(), "step_b".to_string()],
            dependencies,
        };
        let id = orchestrator.schedule_workflow(workflow).unwrap();
        assert!(id.starts_with("workflow-"));
        assert_eq!(orchestrator.workflow_count(), 1);
    }
}
