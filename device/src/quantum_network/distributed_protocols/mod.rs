//! Distributed Quantum Computation Protocols

pub mod implementations;
pub mod types;

// Re-export everything
pub use implementations::*;
pub use types::*;
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration as ChronoDuration, Utc};
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};
    use std::time::Duration;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_distributed_orchestrator_creation() {
        let config = DistributedComputationConfig::default();
        let orchestrator = DistributedQuantumOrchestrator::new(config);

        let status = orchestrator.get_system_status().await;
        assert_eq!(status.total_nodes, 0);
        assert_eq!(status.active_nodes, 0);
    }

    #[tokio::test]
    async fn test_node_registration() {
        let config = DistributedComputationConfig::default();
        let orchestrator = DistributedQuantumOrchestrator::new(config);

        let node_info = NodeInfo {
            node_id: NodeId("test_node".to_string()),
            capabilities: NodeCapabilities {
                max_qubits: 10,
                supported_gates: vec!["H".to_string(), "CNOT".to_string()],
                connectivity_graph: vec![(0, 1), (1, 2)],
                gate_fidelities: HashMap::new(),
                readout_fidelity: 0.95,
                coherence_times: HashMap::new(),
                classical_compute_power: 1000.0,
                memory_capacity_gb: 8,
                network_bandwidth_mbps: 1000.0,
            },
            current_load: NodeLoad {
                qubits_in_use: 0,
                active_circuits: 0,
                cpu_utilization: 0.1,
                memory_utilization: 0.2,
                network_utilization: 0.05,
                queue_length: 0,
                estimated_completion_time: Duration::from_secs(0),
            },
            network_info: NetworkInfo {
                ip_address: "192.168.1.100".to_string(),
                port: 8080,
                latency_to_nodes: HashMap::new(),
                bandwidth_to_nodes: HashMap::new(),
                connection_quality: HashMap::new(),
            },
            status: NodeStatus::Active,
            last_heartbeat: Utc::now(),
        };

        orchestrator
            .register_node(node_info)
            .await
            .expect("Node registration should succeed");

        let status = orchestrator.get_system_status().await;
        assert_eq!(status.total_nodes, 1);
        assert_eq!(status.active_nodes, 1);
        assert_eq!(status.total_qubits, 10);
    }

    #[tokio::test]
    async fn test_circuit_partitioning() {
        let circuit = QuantumCircuit {
            circuit_id: Uuid::new_v4(),
            gates: vec![QuantumGate {
                gate_type: "H".to_string(),
                target_qubits: vec![QubitId {
                    node_id: NodeId("node1".to_string()),
                    local_id: 0,
                    global_id: Uuid::new_v4(),
                }],
                parameters: vec![],
                control_qubits: vec![],
                classical_controls: vec![],
            }],
            qubit_count: 2,
            classical_bit_count: 2,
            measurements: vec![],
            metadata: HashMap::new(),
        };

        let mut nodes = HashMap::new();
        nodes.insert(
            NodeId("node1".to_string()),
            NodeInfo {
                node_id: NodeId("node1".to_string()),
                capabilities: NodeCapabilities {
                    max_qubits: 10,
                    supported_gates: vec!["H".to_string()],
                    connectivity_graph: vec![(0, 1)],
                    gate_fidelities: HashMap::new(),
                    readout_fidelity: 0.95,
                    coherence_times: HashMap::new(),
                    classical_compute_power: 1000.0,
                    memory_capacity_gb: 8,
                    network_bandwidth_mbps: 1000.0,
                },
                current_load: NodeLoad {
                    qubits_in_use: 0,
                    active_circuits: 0,
                    cpu_utilization: 0.1,
                    memory_utilization: 0.2,
                    network_utilization: 0.05,
                    queue_length: 0,
                    estimated_completion_time: Duration::from_secs(0),
                },
                network_info: NetworkInfo {
                    ip_address: "192.168.1.100".to_string(),
                    port: 8080,
                    latency_to_nodes: HashMap::new(),
                    bandwidth_to_nodes: HashMap::new(),
                    connection_quality: HashMap::new(),
                },
                status: NodeStatus::Active,
                last_heartbeat: Utc::now(),
            },
        );

        let config = DistributedComputationConfig::default();
        let partitioner = CircuitPartitioner::new();

        let partitions = partitioner
            .partition_circuit(&circuit, &nodes, &config)
            .expect("Circuit partitioning should succeed");
        assert!(!partitions.is_empty());
        assert_eq!(partitions[0].gates.len(), 1);
    }

    #[test]
    fn test_load_balancer() {
        let balancer = CapabilityBasedBalancer::new();

        let partition = CircuitPartition {
            partition_id: Uuid::new_v4(),
            node_id: NodeId("test".to_string()),
            gates: vec![
                QuantumGate {
                    gate_type: "H".to_string(),
                    target_qubits: vec![QubitId {
                        node_id: NodeId("test".to_string()),
                        local_id: 0,
                        global_id: Uuid::new_v4(),
                    }],
                    control_qubits: vec![],
                    parameters: vec![],
                    classical_controls: vec![],
                },
                QuantumGate {
                    gate_type: "CNOT".to_string(),
                    target_qubits: vec![QubitId {
                        node_id: NodeId("test".to_string()),
                        local_id: 1,
                        global_id: Uuid::new_v4(),
                    }],
                    control_qubits: vec![QubitId {
                        node_id: NodeId("test".to_string()),
                        local_id: 0,
                        global_id: Uuid::new_v4(),
                    }],
                    parameters: vec![],
                    classical_controls: vec![],
                },
            ],
            dependencies: vec![],
            input_qubits: vec![],
            output_qubits: vec![],
            classical_inputs: vec![],
            estimated_execution_time: Duration::from_millis(100),
            resource_requirements: ResourceRequirements {
                qubits_needed: 5,
                gates_count: 10,
                memory_mb: 50,
                execution_time_estimate: Duration::from_millis(100),
                entanglement_pairs_needed: 0,
                classical_communication_bits: 0,
            },
        };

        let node_info = NodeInfo {
            node_id: NodeId("node1".to_string()),
            capabilities: NodeCapabilities {
                max_qubits: 10,
                supported_gates: vec![],
                connectivity_graph: vec![],
                gate_fidelities: HashMap::new(),
                readout_fidelity: 0.95,
                coherence_times: HashMap::new(),
                classical_compute_power: 1000.0,
                memory_capacity_gb: 8,
                network_bandwidth_mbps: 1000.0,
            },
            current_load: NodeLoad {
                qubits_in_use: 2,
                active_circuits: 1,
                cpu_utilization: 0.3,
                memory_utilization: 0.4,
                network_utilization: 0.1,
                queue_length: 2,
                estimated_completion_time: Duration::from_secs(30),
            },
            network_info: NetworkInfo {
                ip_address: "192.168.1.100".to_string(),
                port: 8080,
                latency_to_nodes: HashMap::new(),
                bandwidth_to_nodes: HashMap::new(),
                connection_quality: HashMap::new(),
            },
            status: NodeStatus::Active,
            last_heartbeat: Utc::now(),
        };

        let execution_time = balancer.predict_execution_time(&partition, &node_info);
        assert!(execution_time > Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_state_synchronization() {
        let protocol = BasicSynchronizationProtocol::new();

        let nodes = vec![NodeId("node1".to_string()), NodeId("node2".to_string())];
        let result = protocol
            .synchronize_states(&nodes, 0.95)
            .await
            .expect("State synchronization should succeed");

        assert!(result.success);
        assert_eq!(result.consistency_level, 0.95);
        assert_eq!(result.synchronized_nodes.len(), 2);
    }

    #[tokio::test]
    async fn test_checkpoint_storage() {
        let storage = InMemoryCheckpointStorage::new();

        let checkpoint_data = CheckpointData {
            timestamp: Utc::now(),
            system_state: SystemState {
                nodes: HashMap::new(),
                active_computations: HashMap::new(),
                distributed_states: HashMap::new(),
                network_topology: NetworkTopology {
                    nodes: vec![],
                    edges: vec![],
                    edge_weights: HashMap::new(),
                    clustering_coefficient: 0.0,
                    diameter: 0,
                },
                resource_allocation: HashMap::new(),
            },
            computation_progress: HashMap::new(),
            quantum_states: HashMap::new(),
            metadata: HashMap::new(),
        };

        let checkpoint_id = Uuid::new_v4();
        storage
            .store_checkpoint(checkpoint_id, &checkpoint_data)
            .await
            .expect("Checkpoint storage should succeed");

        let loaded_data = storage
            .load_checkpoint(checkpoint_id)
            .await
            .expect("Checkpoint loading should succeed");
        assert_eq!(loaded_data.timestamp, checkpoint_data.timestamp);

        let checkpoints = storage
            .list_checkpoints()
            .await
            .expect("Listing checkpoints should succeed");
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0], checkpoint_id);
    }

    #[tokio::test]
    async fn test_metrics_storage() {
        let storage = InMemoryMetricsStorage::new();

        let metric = Metric {
            metric_name: "cpu_utilization".to_string(),
            value: 0.75,
            timestamp: Utc::now(),
            tags: HashMap::new(),
            node_id: Some(NodeId("node1".to_string())),
        };

        storage
            .store_metric(&metric)
            .await
            .expect("Metric storage should succeed");

        let query = MetricsQuery {
            metric_names: vec!["cpu_utilization".to_string()],
            time_range: (Utc::now() - ChronoDuration::seconds(60), Utc::now()),
            filters: HashMap::new(),
            limit: None,
        };

        let results = storage
            .query_metrics(&query)
            .await
            .expect("Metrics query should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metric_name, "cpu_utilization");
        assert_eq!(results[0].value, 0.75);
    }

    #[tokio::test]
    async fn test_consensus_engine() {
        let consensus = RaftConsensus::new();

        let proposal = "test_proposal".to_string();
        let participants = vec![NodeId("node1".to_string()), NodeId("node2".to_string())];

        let result = consensus
            .reach_consensus(proposal.clone(), &participants, Duration::from_secs(30))
            .await
            .expect("Consensus should be reached");

        assert!(result.consensus_achieved);
        assert_eq!(result.decision, proposal);
        assert_eq!(result.participating_nodes.len(), 2);
        assert!(result.confidence > 0.9);
    }
}
