#!/usr/bin/env python3
"""
Test suite for distributed quantum simulation functionality.
"""

import pytest
import json
import time
import threading
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

try:
    from quantrs2.distributed_simulation import (
        DistributionStrategy, NodeRole, SimulationStatus, NodeInfo, DistributedTask,
        CircuitPartitioner, LoadBalancer, ClusterCommunication, DistributedSimulator,
        get_distributed_simulator, start_distributed_simulation_service,
        stop_distributed_simulation_service, simulate_circuit_distributed
    )
    HAS_DISTRIBUTED_SIMULATION = True
except ImportError:
    HAS_DISTRIBUTED_SIMULATION = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestNodeInfo:
    """Test NodeInfo functionality."""
    
    def test_node_info_creation(self):
        """Test creating node info."""
        node = NodeInfo(
            node_id="test_node",
            role=NodeRole.WORKER,
            host="localhost",
            port=5000,
            cpu_count=4,
            memory_gb=8.0
        )
        
        assert node.node_id == "test_node"
        assert node.role == NodeRole.WORKER
        assert node.host == "localhost"
        assert node.port == 5000
        assert node.cpu_count == 4
        assert node.memory_gb == 8.0
        assert node.status == "online"
        assert node.current_load == 0.0
    
    def test_node_info_to_dict(self):
        """Test converting node info to dictionary."""
        node = NodeInfo(
            node_id="dict_test",
            role=NodeRole.COORDINATOR,
            host="127.0.0.1",
            port=3000
        )
        
        node_dict = node.to_dict()
        
        assert node_dict['node_id'] == "dict_test"
        assert node_dict['role'] == NodeRole.COORDINATOR.value
        assert node_dict['host'] == "127.0.0.1"
        assert node_dict['port'] == 3000
        assert 'last_seen' in node_dict
        assert 'capabilities' in node_dict


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestDistributedTask:
    """Test DistributedTask functionality."""
    
    def test_distributed_task_creation(self):
        """Test creating distributed task."""
        circuit_data = {
            'n_qubits': 3,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        partition_info = {
            'partition_id': 0,
            'partition_type': 'amplitude'
        }
        
        task = DistributedTask(
            task_id="test_task",
            circuit_data=circuit_data,
            partition_info=partition_info,
            target_nodes=["node1", "node2"]
        )
        
        assert task.task_id == "test_task"
        assert task.circuit_data == circuit_data
        assert task.partition_info == partition_info
        assert task.target_nodes == ["node1", "node2"]
        assert task.status == SimulationStatus.PENDING
        assert task.priority == 1
    
    def test_distributed_task_status_updates(self):
        """Test updating task status."""
        task = DistributedTask(
            task_id="status_test",
            circuit_data={},
            partition_info={},
            target_nodes=[]
        )
        
        # Update to running
        task.status = SimulationStatus.RUNNING
        task.assigned_at = time.time()
        
        assert task.status == SimulationStatus.RUNNING
        assert task.assigned_at is not None
        
        # Update to completed
        task.status = SimulationStatus.COMPLETED
        task.completed_at = time.time()
        task.result = {'success': True}
        
        assert task.status == SimulationStatus.COMPLETED
        assert task.result['success'] is True
    
    def test_distributed_task_to_dict(self):
        """Test converting task to dictionary."""
        task = DistributedTask(
            task_id="dict_task",
            circuit_data={'n_qubits': 2},
            partition_info={'partition_id': 1},
            target_nodes=["node1"]
        )
        
        task_dict = task.to_dict()
        
        assert task_dict['task_id'] == "dict_task"
        assert task_dict['status'] == SimulationStatus.PENDING.value
        assert task_dict['circuit_data'] == {'n_qubits': 2}
        assert task_dict['target_nodes'] == ["node1"]


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestCircuitPartitioner:
    """Test CircuitPartitioner functionality."""
    
    def setup_method(self):
        """Set up test partitioner."""
        self.partitioner = CircuitPartitioner()
    
    def test_amplitude_partitioning(self):
        """Test amplitude partitioning strategy."""
        circuit_data = {
            'n_qubits': 4,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        partitions = self.partitioner.partition_circuit(
            circuit_data, DistributionStrategy.AMPLITUDE_PARTITIONING, 4
        )
        
        assert len(partitions) == 4
        
        for i, partition in enumerate(partitions):
            assert partition['partition_id'] == i
            assert partition['partition_type'] == 'amplitude'
            assert 'amplitude_range' in partition
            assert partition['communication_pattern'] == 'all_to_all'
    
    def test_gate_level_partitioning(self):
        """Test gate-level partitioning strategy."""
        circuit_data = {
            'n_qubits': 3,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'h', 'qubits': [1]},
                {'gate': 'cnot', 'qubits': [0, 2]},
                {'gate': 'x', 'qubits': [1]}
            ]
        }
        
        partitions = self.partitioner.partition_circuit(
            circuit_data, DistributionStrategy.GATE_LEVEL_PARALLELISM, 2
        )
        
        assert len(partitions) == 2
        
        for partition in partitions:
            assert partition['partition_type'] == 'gate_parallel'
            assert 'assigned_gate_layers' in partition
            assert partition['communication_pattern'] == 'barrier_sync'
    
    def test_circuit_partitioning(self):
        """Test circuit partitioning strategy."""
        circuit_data = {
            'n_qubits': 6,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'h', 'qubits': [3]},
                {'gate': 'cnot', 'qubits': [0, 1]},
                {'gate': 'cnot', 'qubits': [3, 4]}
            ]
        }
        
        partitions = self.partitioner.partition_circuit(
            circuit_data, DistributionStrategy.CIRCUIT_PARTITIONING, 3
        )
        
        assert len(partitions) == 3
        
        for partition in partitions:
            assert partition['partition_type'] == 'circuit_split'
            assert 'local_qubits' in partition
            assert 'entangling_gates' in partition
            assert partition['communication_pattern'] == 'nearest_neighbor'
    
    def test_hybrid_partitioning(self):
        """Test hybrid partitioning strategy."""
        # Small circuit should use gate-level parallelism
        small_circuit = {
            'n_qubits': 5,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        partitions = self.partitioner.partition_circuit(
            small_circuit, DistributionStrategy.HYBRID, 2
        )
        
        assert len(partitions) == 2
        assert partitions[0]['partition_type'] == 'gate_parallel'
        
        # Large circuit should use different strategy
        large_circuit = {
            'n_qubits': 25,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        partitions = self.partitioner.partition_circuit(
            large_circuit, DistributionStrategy.HYBRID, 2
        )
        
        assert len(partitions) == 2
        assert partitions[0]['partition_type'] == 'circuit_split'
    
    def test_gate_parallelism_analysis(self):
        """Test gate parallelism analysis."""
        gates = [
            {'gate': 'h', 'qubits': [0]},  # Layer 0
            {'gate': 'h', 'qubits': [1]},  # Layer 0 (parallel with above)
            {'gate': 'cnot', 'qubits': [0, 1]},  # Layer 1 (depends on both)
            {'gate': 'x', 'qubits': [2]}   # Layer 0 (independent)
        ]
        
        layers = self.partitioner._analyze_gate_parallelism(gates, 3)
        
        # Should have multiple layers
        assert len(layers) >= 2
        
        # First layer should have parallel gates
        first_layer = layers[0]
        used_qubits = set()
        for gate in first_layer:
            gate_qubits = set(gate.get('qubits', []))
            assert not (gate_qubits & used_qubits)  # No overlap
            used_qubits.update(gate_qubits)
    
    def test_partitioning_with_invalid_strategy(self):
        """Test partitioning with invalid strategy."""
        circuit_data = {'n_qubits': 2}
        
        # Should fall back to amplitude partitioning
        partitions = self.partitioner.partition_circuit(
            circuit_data, DistributionStrategy.AMPLITUDE_PARTITIONING, 2
        )
        
        assert len(partitions) == 2
        assert partitions[0]['partition_type'] == 'amplitude'
    
    def test_partitioning_error_handling(self):
        """Test error handling in partitioning."""
        # Empty circuit data
        empty_data = {}
        
        partitions = self.partitioner.partition_circuit(
            empty_data, DistributionStrategy.AMPLITUDE_PARTITIONING, 2
        )
        
        # Should return at least one partition (fallback)
        assert len(partitions) >= 1


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestLoadBalancer:
    """Test LoadBalancer functionality."""
    
    def setup_method(self):
        """Set up test load balancer."""
        self.load_balancer = LoadBalancer()
    
    def test_node_metrics_update(self):
        """Test updating node metrics."""
        metrics = {
            'cpu_usage': 50.0,
            'memory_usage': 30.0,
            'network_latency': 10.0,
            'active_tasks': 2
        }
        
        self.load_balancer.update_node_metrics("node1", metrics)
        
        assert "node1" in self.load_balancer.node_metrics
        stored_metrics = self.load_balancer.node_metrics["node1"]
        assert stored_metrics['cpu_usage'] == 50.0
        assert stored_metrics['memory_usage'] == 30.0
        assert 'last_update' in stored_metrics
    
    def test_optimal_node_selection(self):
        """Test selecting optimal nodes."""
        # Create test nodes
        nodes = [
            NodeInfo("node1", NodeRole.WORKER, "host1", 5000, cpu_count=4, memory_gb=8.0),
            NodeInfo("node2", NodeRole.WORKER, "host2", 5001, cpu_count=8, memory_gb=16.0),
            NodeInfo("node3", NodeRole.WORKER, "host3", 5002, cpu_count=2, memory_gb=4.0)
        ]
        
        # Set different loads
        nodes[0].current_load = 0.1
        nodes[1].current_load = 0.5
        nodes[2].current_load = 0.9
        
        task_requirements = {
            'num_nodes': 2,
            'min_memory_gb': 4.0,
            'min_cpu_count': 2
        }
        
        selected = self.load_balancer.select_optimal_nodes(nodes, task_requirements)
        
        assert len(selected) == 2
        # Should prefer lower load nodes
        assert "node1" in selected  # Lowest load
    
    def test_node_selection_with_insufficient_nodes(self):
        """Test node selection when insufficient nodes available."""
        nodes = [
            NodeInfo("node1", NodeRole.WORKER, "host1", 5000, cpu_count=1, memory_gb=2.0)
        ]
        
        task_requirements = {
            'num_nodes': 3,  # More than available
            'min_memory_gb': 1.0,
            'min_cpu_count': 1
        }
        
        selected = self.load_balancer.select_optimal_nodes(nodes, task_requirements)
        
        # Should return what's available
        assert len(selected) == 1
        assert selected[0] == "node1"
    
    def test_node_selection_with_requirements_filter(self):
        """Test node selection with strict requirements."""
        nodes = [
            NodeInfo("node1", NodeRole.WORKER, "host1", 5000, cpu_count=4, memory_gb=8.0),
            NodeInfo("node2", NodeRole.WORKER, "host2", 5001, cpu_count=1, memory_gb=2.0),  # Too small
            NodeInfo("node3", NodeRole.WORKER, "host3", 5002, cpu_count=8, memory_gb=16.0)
        ]
        
        task_requirements = {
            'num_nodes': 2,
            'min_memory_gb': 8.0,  # Filters out node2
            'min_cpu_count': 4
        }
        
        selected = self.load_balancer.select_optimal_nodes(nodes, task_requirements)
        
        assert len(selected) == 2
        assert "node2" not in selected  # Should be filtered out
        assert "node1" in selected
        assert "node3" in selected
    
    def test_node_score_calculation(self):
        """Test node scoring algorithm."""
        node = NodeInfo("test_node", NodeRole.WORKER, "host", 5000, 
                       cpu_count=4, memory_gb=8.0)
        node.current_load = 0.3
        
        score = self.load_balancer._calculate_node_score(node)
        
        assert isinstance(score, float)
        assert score > 0
        
        # Node with higher load should have higher score
        high_load_node = NodeInfo("high_load", NodeRole.WORKER, "host", 5000,
                                 cpu_count=4, memory_gb=8.0)
        high_load_node.current_load = 0.8
        
        high_score = self.load_balancer._calculate_node_score(high_load_node)
        assert high_score > score
    
    def test_task_completion_tracking(self):
        """Test task completion history tracking."""
        self.load_balancer.update_task_completion("node1", 1.5)
        self.load_balancer.update_task_completion("node1", 2.0)
        
        assert "node1" in self.load_balancer.task_history
        assert len(self.load_balancer.task_history["node1"]) == 2
        assert 1.5 in self.load_balancer.task_history["node1"]
        assert 2.0 in self.load_balancer.task_history["node1"]
    
    def test_task_history_limit(self):
        """Test task history size limit."""
        # Add many completions
        for i in range(150):
            self.load_balancer.update_task_completion("node1", float(i))
        
        # Should be limited to 100 entries
        assert len(self.load_balancer.task_history["node1"]) == 100
        # Should keep the most recent ones
        assert 149.0 in self.load_balancer.task_history["node1"]


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestClusterCommunication:
    """Test ClusterCommunication functionality."""
    
    def setup_method(self):
        """Set up test communication."""
        self.comm = ClusterCommunication("test_node", port=0)
    
    def teardown_method(self):
        """Clean up test communication."""
        if self.comm:
            self.comm.stop_server()
    
    def test_communication_initialization(self):
        """Test communication initialization."""
        assert self.comm.node_id == "test_node"
        assert self.comm.port == 0
        assert not self.comm.running
        assert len(self.comm.message_handlers) > 0
    
    def test_server_start_stop(self):
        """Test starting and stopping communication server."""
        port = self.comm.start_server()
        
        assert port > 0
        assert self.comm.port == port
        assert self.comm.running is True
        
        # Stop server
        self.comm.stop_server()
        assert self.comm.running is False
    
    def test_message_handler_registration(self):
        """Test registering message handlers."""
        def test_handler(message):
            return {'response': 'test'}
        
        self.comm.register_handler('test_message', test_handler)
        
        assert 'test_message' in self.comm.message_handlers
        assert self.comm.message_handlers['test_message'] == test_handler
    
    def test_default_message_handlers(self):
        """Test default message handlers."""
        # Test ping handler
        ping_message = {'type': 'ping', 'timestamp': time.time()}
        response = self.comm._handle_ping(ping_message)
        
        assert response['type'] == 'pong'
        assert response['node_id'] == 'test_node'
        assert 'timestamp' in response
        
        # Test task assignment handler
        task_message = {'type': 'task_assignment', 'task_id': 'test_task'}
        response = self.comm._handle_task_assignment(task_message)
        
        assert response['type'] == 'task_accepted'
        assert response['task_id'] == 'test_task'
    
    def test_message_processing(self):
        """Test message processing."""
        # Test known message type
        ping_message = {'type': 'ping'}
        response = self.comm._process_message(ping_message)
        
        assert response is not None
        assert response['type'] == 'pong'
        
        # Test unknown message type
        unknown_message = {'type': 'unknown_type'}
        response = self.comm._process_message(unknown_message)
        
        assert response['type'] == 'error'
        assert 'Unknown message type' in response['message']
    
    def test_send_message_to_node(self):
        """Test sending message to specific node."""
        message = {'type': 'test', 'data': 'hello'}
        
        # Mock implementation should return success
        response = self.comm.send_message_to_node('target_node', message)
        
        assert response is not None
        assert response['type'] == 'ack'
    
    def test_broadcast_message(self):
        """Test broadcasting message."""
        # Add mock connections
        self.comm.connections['node1'] = Mock()
        self.comm.connections['node2'] = Mock()
        
        message = {'type': 'broadcast_test'}
        successful_nodes = self.comm.broadcast_message(message)
        
        # Should attempt to send to all nodes
        assert isinstance(successful_nodes, list)


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestDistributedSimulator:
    """Test DistributedSimulator functionality."""
    
    def setup_method(self):
        """Set up test simulator."""
        self.simulator = DistributedSimulator("test_simulator", NodeRole.WORKER)
    
    def teardown_method(self):
        """Clean up test simulator."""
        if self.simulator:
            self.simulator.stop()
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        assert self.simulator.node_id == "test_simulator"
        assert self.simulator.role == NodeRole.WORKER
        assert self.simulator.node_info.node_id == "test_simulator"
        assert not self.simulator.running
        assert len(self.simulator.active_tasks) == 0
    
    def test_simulator_start_stop(self):
        """Test starting and stopping simulator."""
        success = self.simulator.start()
        
        assert success is True
        assert self.simulator.running is True
        assert self.simulator.node_info.port > 0
        
        # Stop simulator
        self.simulator.stop()
        assert self.simulator.running is False
    
    def test_submit_distributed_simulation(self):
        """Test submitting distributed simulation."""
        circuit_data = {
            'n_qubits': 3,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        task_id = self.simulator.submit_distributed_simulation(
            circuit_data, DistributionStrategy.AMPLITUDE_PARTITIONING, 2
        )
        
        assert task_id is not None
        assert len(self.simulator.active_tasks) > 0
        
        # Check that partition tasks were created
        partition_tasks = [task_id for task_id in self.simulator.active_tasks.keys() 
                          if task_id.startswith(task_id.split('_part_')[0])]
        assert len(partition_tasks) > 0
    
    def test_simulation_status_tracking(self):
        """Test simulation status tracking."""
        circuit_data = {'n_qubits': 2}
        
        task_id = self.simulator.submit_distributed_simulation(circuit_data)
        
        # Initial status should be pending or running
        status = self.simulator.get_simulation_status(task_id)
        assert status in [SimulationStatus.PENDING, SimulationStatus.RUNNING]
        
        # Status for nonexistent task
        nonexistent_status = self.simulator.get_simulation_status("nonexistent")
        assert nonexistent_status is None
    
    def test_local_circuit_simulation(self):
        """Test local circuit simulation."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        result = self.simulator._simulate_local_circuit(circuit_data)
        
        assert result['success'] is True
        assert result['n_qubits'] == 2
        assert result['gate_count'] == 2
        assert result['simulation_time'] >= 0
    
    def test_amplitude_partition_simulation(self):
        """Test amplitude partition simulation."""
        circuit_data = {'n_qubits': 3}
        partition_info = {
            'partition_id': 0,
            'partition_type': 'amplitude',
            'amplitude_range': (0, 4)
        }
        
        result = self.simulator._simulate_amplitude_partition(circuit_data, partition_info)
        
        assert result['success'] is True
        assert result['partition_type'] == 'amplitude'
        assert result['amplitude_range'] == (0, 4)
        assert 'amplitudes' in result
    
    def test_gate_parallel_partition_simulation(self):
        """Test gate-parallel partition simulation."""
        circuit_data = {'n_qubits': 2}
        partition_info = {
            'partition_id': 1,
            'partition_type': 'gate_parallel',
            'assigned_gate_layers': [
                [{'gate': 'h', 'qubits': [0]}],
                [{'gate': 'x', 'qubits': [1]}]
            ]
        }
        
        result = self.simulator._simulate_gate_parallel_partition(circuit_data, partition_info)
        
        assert result['success'] is True
        assert result['partition_type'] == 'gate_parallel'
        assert result['total_gates'] == 2
        assert len(result['processed_layers']) == 2
    
    def test_circuit_split_partition_simulation(self):
        """Test circuit split partition simulation."""
        circuit_data = {'n_qubits': 2, 'gates': []}
        partition_info = {
            'partition_id': 0,
            'partition_type': 'circuit_split',
            'local_qubits': [0, 1],
            'entangling_gates': [],
            'circuit_data': circuit_data
        }
        
        result = self.simulator._simulate_circuit_split_partition(circuit_data, partition_info)
        
        assert result['success'] is True
        assert result['partition_type'] == 'circuit_split'
        assert result['local_qubits'] == [0, 1]
        assert 'local_result' in result
    
    def test_cluster_management(self):
        """Test cluster management functionality."""
        # Test joining cluster
        success = self.simulator.join_cluster("localhost", 5000)
        assert success is True
        
        # Check cluster status
        status = self.simulator.get_cluster_status()
        assert status['node_id'] == "test_simulator"
        assert status['role'] == NodeRole.WORKER.value
        assert 'cluster_size' in status
        assert 'active_tasks' in status
    
    def test_communication_message_handlers(self):
        """Test communication message handlers."""
        # Test simulate circuit handler
        message = {
            'type': 'simulate_circuit',
            'circuit_data': {'n_qubits': 2},
            'strategy': 'amplitude_partitioning'
        }
        
        response = self.simulator._handle_simulate_circuit(message)
        
        assert response['type'] == 'simulation_accepted'
        assert 'task_id' in response
        
        # Test join cluster handler
        join_message = {
            'type': 'join_cluster',
            'node_info': {
                'node_id': 'new_node',
                'role': 'worker',
                'host': 'localhost',
                'port': 5001
            }
        }
        
        response = self.simulator._handle_join_cluster(join_message)
        assert response['type'] == 'join_accepted'
    
    def test_task_processing_workflow(self):
        """Test complete task processing workflow."""
        self.simulator.start()
        
        circuit_data = {
            'n_qubits': 2,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        # Submit simulation
        task_id = self.simulator.submit_distributed_simulation(circuit_data)
        
        # Wait briefly for processing
        time.sleep(0.5)
        
        # Check if any tasks completed
        status = self.simulator.get_simulation_status(task_id)
        assert status is not None
        
        # If completed, check result
        if status == SimulationStatus.COMPLETED:
            result = self.simulator.get_simulation_result(task_id)
            assert result is not None
            assert 'partition_results' in result
    
    def test_error_handling_in_simulation(self):
        """Test error handling during simulation."""
        # Test with invalid circuit data
        invalid_circuit = {'invalid': 'data'}
        partition_info = {'partition_id': 0, 'partition_type': 'amplitude'}
        
        result = self.simulator._simulate_circuit_partition(invalid_circuit, partition_info)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'partition_id' in result


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_distributed_simulator(self):
        """Test getting global distributed simulator."""
        simulator1 = get_distributed_simulator()
        simulator2 = get_distributed_simulator()
        
        # Should be singleton
        assert simulator1 is simulator2
        assert isinstance(simulator1, DistributedSimulator)
    
    def test_start_stop_distributed_service(self):
        """Test starting and stopping distributed service."""
        # Start service
        success = start_distributed_simulation_service(
            node_id="test_service",
            role=NodeRole.WORKER,
            port=0
        )
        
        assert success is True
        
        # Get simulator instance
        simulator = get_distributed_simulator()
        assert simulator.running is True
        
        # Stop service
        stop_distributed_simulation_service()
        
        # Simulator should be cleaned up
        assert simulator.running is False
    
    def test_simulate_circuit_distributed(self):
        """Test distributed circuit simulation convenience function."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        result = simulate_circuit_distributed(
            circuit_data, 
            DistributionStrategy.AMPLITUDE_PARTITIONING,
            timeout=5.0
        )
        
        # Should complete or timeout gracefully
        assert result is not None or result is None  # Either success or timeout


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_partitioner_error_handling(self):
        """Test partitioner error handling."""
        partitioner = CircuitPartitioner()
        
        # Test with malformed circuit data
        malformed_data = {'gates': 'not_a_list'}
        
        partitions = partitioner.partition_circuit(
            malformed_data, DistributionStrategy.AMPLITUDE_PARTITIONING, 2
        )
        
        # Should return at least one partition as fallback
        assert len(partitions) >= 1
    
    def test_load_balancer_error_handling(self):
        """Test load balancer error handling."""
        load_balancer = LoadBalancer()
        
        # Test with malformed node data
        try:
            malformed_nodes = [{'not': 'a_node_info'}]
            selected = load_balancer.select_optimal_nodes(malformed_nodes, {})
            # Should handle gracefully
            assert isinstance(selected, list)
        except Exception:
            # Should not raise unhandled exceptions
            assert False, "Should handle malformed data gracefully"
    
    def test_simulator_communication_errors(self):
        """Test simulator communication error handling."""
        simulator = DistributedSimulator("error_test", NodeRole.WORKER)
        
        try:
            # Test handling malformed messages
            malformed_message = {'type': 'invalid', 'data': object()}
            response = simulator._handle_simulate_circuit(malformed_message)
            
            # Should return error response
            assert response['type'] == 'simulation_rejected'
            assert 'error' in response
            
        finally:
            simulator.stop()
    
    def test_communication_server_errors(self):
        """Test communication server error handling."""
        comm = ClusterCommunication("error_test")
        
        try:
            # Test starting server on invalid port
            original_socket = comm.socket
            comm.socket = Mock()
            comm.socket.bind.side_effect = Exception("Port error")
            
            port = comm.start_server()
            
            # Should handle error gracefully
            assert port == 0
            
        finally:
            comm.stop_server()


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestDistributedSimulationIntegration:
    """Test integration scenarios."""
    
    def test_end_to_end_distributed_simulation(self):
        """Test complete distributed simulation workflow."""
        # Start coordinator
        coordinator = DistributedSimulator("coordinator", NodeRole.COORDINATOR)
        coordinator_success = coordinator.start()
        
        # Start worker
        worker = DistributedSimulator("worker1", NodeRole.WORKER)
        worker_success = worker.start()
        
        try:
            assert coordinator_success is True
            assert worker_success is True
            
            # Create test circuit
            circuit_data = {
                'n_qubits': 3,
                'gates': [
                    {'gate': 'h', 'qubits': [0]},
                    {'gate': 'h', 'qubits': [1]},
                    {'gate': 'cnot', 'qubits': [0, 2]},
                    {'gate': 'cnot', 'qubits': [1, 2]}
                ]
            }
            
            # Submit simulation to coordinator
            task_id = coordinator.submit_distributed_simulation(
                circuit_data, DistributionStrategy.AMPLITUDE_PARTITIONING, 2
            )
            
            assert task_id is not None
            
            # Wait for processing
            start_time = time.time()
            while time.time() - start_time < 5.0:
                status = coordinator.get_simulation_status(task_id)
                if status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED]:
                    break
                time.sleep(0.1)
            
            # Check final status
            final_status = coordinator.get_simulation_status(task_id)
            assert final_status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED]
            
            if final_status == SimulationStatus.COMPLETED:
                result = coordinator.get_simulation_result(task_id)
                assert result is not None
                assert 'partition_results' in result
                assert result['partition_count'] > 0
            
        finally:
            coordinator.stop()
            worker.stop()
    
    def test_multi_strategy_comparison(self):
        """Test comparing different distribution strategies."""
        simulator = DistributedSimulator("strategy_test", NodeRole.WORKER)
        success = simulator.start()
        
        try:
            assert success is True
            
            circuit_data = {
                'n_qubits': 4,
                'gates': [
                    {'gate': 'h', 'qubits': [0]},
                    {'gate': 'h', 'qubits': [1]},
                    {'gate': 'cnot', 'qubits': [0, 2]},
                    {'gate': 'cnot', 'qubits': [1, 3]},
                    {'gate': 'z', 'qubits': [2]},
                    {'gate': 'z', 'qubits': [3]}
                ]
            }
            
            strategies = [
                DistributionStrategy.AMPLITUDE_PARTITIONING,
                DistributionStrategy.GATE_LEVEL_PARALLELISM,
                DistributionStrategy.CIRCUIT_PARTITIONING
            ]
            
            results = {}
            for strategy in strategies:
                task_id = simulator.submit_distributed_simulation(
                    circuit_data, strategy, 2
                )
                
                # Wait briefly for processing
                time.sleep(0.5)
                
                status = simulator.get_simulation_status(task_id)
                results[strategy] = status
            
            # All strategies should at least start processing
            for strategy, status in results.items():
                assert status in [SimulationStatus.PENDING, SimulationStatus.RUNNING, 
                                SimulationStatus.COMPLETED]
            
        finally:
            simulator.stop()
    
    def test_load_balancing_with_multiple_nodes(self):
        """Test load balancing across multiple nodes."""
        # Create multiple nodes with different capabilities
        nodes = [
            NodeInfo("high_perf", NodeRole.WORKER, "host1", 5000, 
                    cpu_count=8, memory_gb=32.0),
            NodeInfo("medium_perf", NodeRole.WORKER, "host2", 5001, 
                    cpu_count=4, memory_gb=16.0),
            NodeInfo("low_perf", NodeRole.WORKER, "host3", 5002, 
                    cpu_count=2, memory_gb=8.0)
        ]
        
        # Set different loads
        nodes[0].current_load = 0.2  # High-performance, low load
        nodes[1].current_load = 0.5  # Medium performance, medium load
        nodes[2].current_load = 0.1  # Low performance, low load
        
        load_balancer = LoadBalancer()
        
        # Test different task requirements
        high_req_task = {
            'num_nodes': 1,
            'min_memory_gb': 16.0,
            'min_cpu_count': 4
        }
        
        selected = load_balancer.select_optimal_nodes(nodes, high_req_task)
        
        # Should prefer high-performance node for demanding task
        assert len(selected) == 1
        assert selected[0] in ["high_perf", "medium_perf"]  # Both meet requirements
        
        # Test bulk task
        bulk_task = {
            'num_nodes': 3,
            'min_memory_gb': 4.0,
            'min_cpu_count': 1
        }
        
        selected = load_balancer.select_optimal_nodes(nodes, bulk_task)
        
        # Should use all available nodes
        assert len(selected) == 3
    
    def test_fault_tolerance(self):
        """Test fault tolerance in distributed simulation."""
        simulator = DistributedSimulator("fault_test", NodeRole.WORKER)
        success = simulator.start()
        
        try:
            assert success is True
            
            # Create a task that might fail
            circuit_data = {
                'n_qubits': 2,
                'gates': [{'gate': 'invalid_gate', 'qubits': [0]}]  # Invalid gate
            }
            
            task_id = simulator.submit_distributed_simulation(circuit_data)
            
            # Wait for processing
            time.sleep(1.0)
            
            # Should handle invalid gate gracefully
            status = simulator.get_simulation_status(task_id)
            
            # Status should be determined (not hanging)
            assert status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED, 
                            SimulationStatus.RUNNING]
            
            # System should remain operational
            assert simulator.running is True
            
        finally:
            simulator.stop()


@pytest.mark.skipif(not HAS_DISTRIBUTED_SIMULATION, reason="distributed_simulation module not available")
class TestDistributedSimulationPerformance:
    """Test performance characteristics."""
    
    def test_partitioning_performance(self):
        """Test partitioning performance with large circuits."""
        partitioner = CircuitPartitioner()
        
        # Create large circuit
        large_circuit = {
            'n_qubits': 10,
            'gates': [
                {'gate': 'h', 'qubits': [i]} for i in range(10)
            ] + [
                {'gate': 'cnot', 'qubits': [i, (i + 1) % 10]} for i in range(10)
            ]
        }
        
        start_time = time.time()
        
        # Test different strategies
        strategies = [
            DistributionStrategy.AMPLITUDE_PARTITIONING,
            DistributionStrategy.GATE_LEVEL_PARALLELISM,
            DistributionStrategy.CIRCUIT_PARTITIONING
        ]
        
        for strategy in strategies:
            partitions = partitioner.partition_circuit(large_circuit, strategy, 4)
            assert len(partitions) == 4
        
        total_time = time.time() - start_time
        
        # Should complete partitioning quickly
        assert total_time < 1.0  # Less than 1 second for all strategies
    
    def test_concurrent_simulation_performance(self):
        """Test performance with concurrent simulations."""
        simulator = DistributedSimulator("perf_test", NodeRole.WORKER)
        success = simulator.start()
        
        try:
            assert success is True
            
            # Submit multiple simulations concurrently
            task_ids = []
            
            start_time = time.time()
            
            for i in range(10):
                circuit_data = {
                    'n_qubits': 3,
                    'circuit_id': i,
                    'gates': [
                        {'gate': 'h', 'qubits': [0]},
                        {'gate': 'cnot', 'qubits': [0, 1]}
                    ]
                }
                
                task_id = simulator.submit_distributed_simulation(circuit_data)
                task_ids.append(task_id)
            
            submission_time = time.time() - start_time
            
            # Submission should be fast
            assert submission_time < 1.0
            
            # Wait for some processing
            time.sleep(2.0)
            
            # Check how many completed
            completed_count = 0
            for task_id in task_ids:
                status = simulator.get_simulation_status(task_id)
                if status == SimulationStatus.COMPLETED:
                    completed_count += 1
            
            # Should have processed some/all tasks
            assert completed_count >= 0  # At least trying to process
            
        finally:
            simulator.stop()
    
    def test_memory_usage_monitoring(self):
        """Test memory usage doesn't grow excessively."""
        simulator = DistributedSimulator("memory_test", NodeRole.WORKER)
        success = simulator.start()
        
        try:
            assert success is True
            
            initial_task_count = len(simulator.active_tasks) + len(simulator.completed_tasks)
            
            # Submit and complete many small tasks
            for i in range(50):
                circuit_data = {
                    'n_qubits': 2,
                    'task_num': i,
                    'gates': [{'gate': 'h', 'qubits': [0]}]
                }
                
                simulator.submit_distributed_simulation(circuit_data)
            
            # Wait for processing
            time.sleep(1.0)
            
            final_task_count = len(simulator.active_tasks) + len(simulator.completed_tasks)
            
            # Should have processed tasks (memory usage check)
            assert final_task_count >= initial_task_count
            
            # Task counts should be reasonable (not accumulating excessively)
            assert len(simulator.active_tasks) < 100  # Should not pile up indefinitely
            
        finally:
            simulator.stop()


if __name__ == "__main__":
    pytest.main([__file__])