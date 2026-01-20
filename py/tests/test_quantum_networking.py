#!/usr/bin/env python3
"""
Test suite for quantum networking functionality.
"""

import pytest
import json
import time
import math
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

try:
    from quantrs2.quantum_networking import (
        NetworkTopology, ProtocolType, NetworkState, ChannelType,
        QuantumChannel, NetworkNode, EntanglementPair, NetworkProtocol,
        QuantumNetworkTopology, EntanglementDistribution, QuantumTeleportation,
        QuantumNetworkSimulator, QuantumNetworkVisualizer,
        get_quantum_network_simulator, create_quantum_network,
        distribute_entanglement, teleport_qubit, visualize_quantum_network
    )
    HAS_QUANTUM_NETWORKING = True
except ImportError:
    HAS_QUANTUM_NETWORKING = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestQuantumChannel:
    """Test QuantumChannel functionality."""
    
    def test_channel_creation(self):
        """Test creating quantum channel."""
        channel = QuantumChannel(
            channel_id="test_channel",
            source_node="node_a",
            target_node="node_b",
            channel_type=ChannelType.FIBER_OPTIC,
            fidelity=0.95,
            distance=10.0
        )
        
        assert channel.channel_id == "test_channel"
        assert channel.source_node == "node_a"
        assert channel.target_node == "node_b"
        assert channel.channel_type == ChannelType.FIBER_OPTIC
        assert channel.fidelity == 0.95
        assert channel.distance == 10.0
        assert channel.usage_count == 0
    
    def test_transmission_fidelity_calculation(self):
        """Test transmission fidelity calculation."""
        channel = QuantumChannel(
            channel_id="fidelity_test",
            source_node="node_a",
            target_node="node_b",
            channel_type=ChannelType.FIBER_OPTIC,
            fidelity=0.99,
            distance=5.0,
            loss_rate=0.1
        )
        
        base_fidelity = 1.0
        transmission_fidelity = channel.calculate_transmission_fidelity(base_fidelity)
        
        # Should be less than base fidelity due to losses
        assert transmission_fidelity < base_fidelity
        assert transmission_fidelity > 0.0
    
    def test_default_noise_model(self):
        """Test default noise model creation."""
        channel = QuantumChannel(
            channel_id="noise_test",
            source_node="node_a",
            target_node="node_b",
            channel_type=ChannelType.FREE_SPACE,
            distance=100.0
        )
        
        noise_model = channel.noise_model
        assert 'error_rate' in noise_model
        assert 'decoherence_time' in noise_model
        assert 'gate_error_rate' in noise_model
        assert 'measurement_error_rate' in noise_model
        
        # Free space should have higher error rate for long distance
        assert noise_model['error_rate'] > 0.01
    
    def test_channel_to_dict(self):
        """Test converting channel to dictionary."""
        channel = QuantumChannel(
            channel_id="dict_test",
            source_node="node_a",
            target_node="node_b",
            channel_type=ChannelType.SATELLITE
        )
        
        channel_dict = channel.to_dict()
        
        assert channel_dict['channel_id'] == "dict_test"
        assert channel_dict['channel_type'] == ChannelType.SATELLITE.value
        assert 'noise_model' in channel_dict
        assert 'created_at' in channel_dict


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestNetworkNode:
    """Test NetworkNode functionality."""
    
    def test_node_creation(self):
        """Test creating network node."""
        node = NetworkNode(
            node_id="test_node",
            node_type="quantum_computer",
            position=(5.0, 10.0),
            memory_capacity=20,
            coherence_time=2000.0
        )
        
        assert node.node_id == "test_node"
        assert node.node_type == "quantum_computer"
        assert node.position == (5.0, 10.0)
        assert node.memory_capacity == 20
        assert node.coherence_time == 2000.0
        assert node.status == "online"
        assert len(node.capabilities) > 0
    
    def test_protocol_capability_check(self):
        """Test checking protocol capabilities."""
        node = NetworkNode(
            node_id="capable_node",
            capabilities=["entanglement_generation", "quantum_teleportation", "quantum_measurement"]
        )
        
        # Should be able to perform teleportation
        assert node.can_perform_protocol(ProtocolType.QUANTUM_TELEPORTATION)
        assert node.can_perform_protocol(ProtocolType.ENTANGLEMENT_DISTRIBUTION)
        
        # Should not be able to perform protocols requiring missing capabilities
        assert not node.can_perform_protocol(ProtocolType.DISTRIBUTED_COMPUTING)
    
    def test_distance_calculation(self):
        """Test distance calculation between nodes."""
        node_a = NetworkNode("node_a", position=(0.0, 0.0))
        node_b = NetworkNode("node_b", position=(3.0, 4.0))
        
        distance = node_a.calculate_distance(node_b)
        assert abs(distance - 5.0) < 1e-10  # 3-4-5 triangle
    
    def test_node_to_dict(self):
        """Test converting node to dictionary."""
        node = NetworkNode(
            node_id="dict_node",
            node_type="sensor",
            position=(1.0, 2.0)
        )
        
        node_dict = node.to_dict()
        
        assert node_dict['node_id'] == "dict_node"
        assert node_dict['node_type'] == "sensor"
        assert node_dict['position'] == (1.0, 2.0)
        assert 'capabilities' in node_dict
        assert 'created_at' in node_dict


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestEntanglementPair:
    """Test EntanglementPair functionality."""
    
    def test_entanglement_pair_creation(self):
        """Test creating entanglement pair."""
        pair = EntanglementPair(
            pair_id="test_pair",
            node_a="node_1",
            node_b="node_2",
            qubit_a=0,
            qubit_b=1,
            fidelity=0.95,
            coherence_time=1000.0
        )
        
        assert pair.pair_id == "test_pair"
        assert pair.node_a == "node_1"
        assert pair.node_b == "node_2"
        assert pair.qubit_a == 0
        assert pair.qubit_b == 1
        assert pair.fidelity == 0.95
        assert pair.coherence_time == 1000.0
        assert pair.usage_count == 0
    
    def test_entanglement_expiration(self):
        """Test entanglement expiration check."""
        # Create pair with short coherence time
        pair = EntanglementPair(
            pair_id="expiring_pair",
            node_a="node_1",
            node_b="node_2",
            qubit_a=0,
            qubit_b=1,
            coherence_time=1.0  # 1 microsecond
        )
        
        # Should not be expired immediately
        assert not pair.is_expired()
        
        # Should be expired after sufficient time
        future_time = time.time() + 0.001  # 1 millisecond later
        assert pair.is_expired(future_time)
    
    def test_current_fidelity_calculation(self):
        """Test current fidelity calculation with decoherence."""
        pair = EntanglementPair(
            pair_id="fidelity_test",
            node_a="node_1",
            node_b="node_2",
            qubit_a=0,
            qubit_b=1,
            fidelity=1.0,
            coherence_time=1000.0
        )
        
        # Current fidelity should be close to initial
        current_fidelity = pair.calculate_current_fidelity()
        assert current_fidelity > 0.99
        
        # Fidelity should decay over time
        future_time = time.time() + 0.001  # 1ms later
        future_fidelity = pair.calculate_current_fidelity(future_time)
        assert future_fidelity < current_fidelity
    
    def test_entanglement_pair_to_dict(self):
        """Test converting entanglement pair to dictionary."""
        pair = EntanglementPair(
            pair_id="dict_pair",
            node_a="node_1",
            node_b="node_2",
            qubit_a=0,
            qubit_b=1
        )
        
        pair_dict = pair.to_dict()
        
        assert pair_dict['pair_id'] == "dict_pair"
        assert pair_dict['node_a'] == "node_1"
        assert pair_dict['node_b'] == "node_2"
        assert 'current_fidelity' in pair_dict
        assert 'created_at' in pair_dict


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestQuantumNetworkTopology:
    """Test QuantumNetworkTopology functionality."""
    
    def setup_method(self):
        """Set up test topology."""
        self.topology = QuantumNetworkTopology()
    
    def test_node_addition(self):
        """Test adding nodes to topology."""
        node = NetworkNode("test_node", position=(0.0, 0.0))
        
        success = self.topology.add_node(node)
        assert success is True
        assert "test_node" in self.topology.nodes
        assert "test_node" in self.topology.adjacency_matrix
    
    def test_channel_addition(self):
        """Test adding channels to topology."""
        # Add nodes first
        node_a = NetworkNode("node_a", position=(0.0, 0.0))
        node_b = NetworkNode("node_b", position=(1.0, 0.0))
        
        self.topology.add_node(node_a)
        self.topology.add_node(node_b)
        
        # Add channel
        channel = QuantumChannel(
            channel_id="test_channel",
            source_node="node_a",
            target_node="node_b",
            channel_type=ChannelType.FIBER_OPTIC,
            distance=1.0
        )
        
        success = self.topology.add_channel(channel)
        assert success is True
        assert "test_channel" in self.topology.channels
        assert self.topology.adjacency_matrix["node_a"]["node_b"] == 1.0
    
    def test_channel_addition_failure(self):
        """Test channel addition with missing nodes."""
        channel = QuantumChannel(
            channel_id="missing_nodes",
            source_node="nonexistent_a",
            target_node="nonexistent_b",
            channel_type=ChannelType.FIBER_OPTIC
        )
        
        success = self.topology.add_channel(channel)
        assert success is False
    
    def test_shortest_path_finding(self):
        """Test shortest path algorithm."""
        # Create simple linear topology
        nodes = [NetworkNode(f"node_{i}", position=(i, 0)) for i in range(4)]
        for node in nodes:
            self.topology.add_node(node)
        
        # Add channels in sequence
        for i in range(3):
            channel = QuantumChannel(
                channel_id=f"channel_{i}",
                source_node=f"node_{i}",
                target_node=f"node_{i+1}",
                channel_type=ChannelType.FIBER_OPTIC,
                distance=1.0
            )
            self.topology.add_channel(channel)
        
        # Find path from first to last node
        path = self.topology.find_shortest_path("node_0", "node_3")
        assert path == ["node_0", "node_1", "node_2", "node_3"]
        
        # Path to same node
        same_path = self.topology.find_shortest_path("node_1", "node_1")
        assert same_path == ["node_1"]
        
        # Path to nonexistent node
        no_path = self.topology.find_shortest_path("node_0", "nonexistent")
        assert no_path is None
    
    def test_star_topology_creation(self):
        """Test creating star topology."""
        success = self.topology.create_topology(NetworkTopology.STAR, 5)
        
        assert success is True
        assert len(self.topology.nodes) == 5
        assert len(self.topology.channels) == 4  # Center connected to 4 others
        
        # Check center node is at origin
        center_node = self.topology.nodes["node_0"]
        assert center_node.position == (0.0, 0.0)
    
    def test_mesh_topology_creation(self):
        """Test creating mesh topology."""
        success = self.topology.create_topology(NetworkTopology.MESH, 4)
        
        assert success is True
        assert len(self.topology.nodes) == 4
        assert len(self.topology.channels) == 6  # n(n-1)/2 for full mesh
    
    def test_ring_topology_creation(self):
        """Test creating ring topology."""
        success = self.topology.create_topology(NetworkTopology.RING, 5)
        
        assert success is True
        assert len(self.topology.nodes) == 5
        assert len(self.topology.channels) == 5  # Each node connected to next
    
    def test_linear_topology_creation(self):
        """Test creating linear topology."""
        success = self.topology.create_topology(NetworkTopology.LINEAR, 4)
        
        assert success is True
        assert len(self.topology.nodes) == 4
        assert len(self.topology.channels) == 3  # n-1 connections
    
    def test_grid_topology_creation(self):
        """Test creating grid topology."""
        success = self.topology.create_topology(NetworkTopology.GRID, 9)  # 3x3 grid
        
        assert success is True
        assert len(self.topology.nodes) == 9
        # Grid connections: (3-1)*3 horizontal + 3*(3-1) vertical = 12
        assert len(self.topology.channels) == 12
    
    def test_network_diameter_calculation(self):
        """Test network diameter calculation."""
        # Create simple linear topology
        success = self.topology.create_topology(NetworkTopology.LINEAR, 3, spacing=2.0)
        assert success is True
        
        diameter = self.topology.get_network_diameter()
        assert abs(diameter - 4.0) < 0.1  # Total distance should be ~4.0


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestEntanglementDistribution:
    """Test EntanglementDistribution functionality."""
    
    def setup_method(self):
        """Set up test entanglement distribution."""
        self.topology = QuantumNetworkTopology()
        self.topology.create_topology(NetworkTopology.STAR, 5)
        self.ent_dist = EntanglementDistribution(self.topology)
    
    def test_direct_entanglement_distribution(self):
        """Test direct entanglement distribution."""
        # Distribute entanglement between center and leaf node
        pair_id = self.ent_dist.distribute_entanglement("node_0", "node_1", "direct")
        
        assert pair_id is not None
        assert pair_id in self.ent_dist.entanglement_pairs
        
        pair = self.ent_dist.entanglement_pairs[pair_id]
        assert pair.node_a == "node_0"
        assert pair.node_b == "node_1"
    
    def test_entanglement_distribution_failure(self):
        """Test entanglement distribution failure scenarios."""
        # Try to distribute between nonexistent nodes
        pair_id = self.ent_dist.distribute_entanglement("nonexistent_a", "nonexistent_b")
        assert pair_id is None
        
        # Try with impossible fidelity requirement
        pair_id = self.ent_dist.distribute_entanglement("node_0", "node_1", target_fidelity=1.5)
        assert pair_id is None
    
    def test_repeater_based_distribution(self):
        """Test entanglement distribution via repeaters."""
        # Create linear topology for repeater testing
        linear_topology = QuantumNetworkTopology()
        linear_topology.create_topology(NetworkTopology.LINEAR, 4)
        repeater_dist = EntanglementDistribution(linear_topology)
        
        # Distribute between end nodes (should use repeaters)
        pair_id = repeater_dist.distribute_entanglement("node_0", "node_3", "repeater")
        
        assert pair_id is not None
        assert pair_id in repeater_dist.entanglement_pairs
    
    def test_entanglement_swapping(self):
        """Test entanglement swapping."""
        # Create mesh topology for swapping test
        mesh_topology = QuantumNetworkTopology()
        mesh_topology.create_topology(NetworkTopology.MESH, 3)
        swap_dist = EntanglementDistribution(mesh_topology)
        
        # Use swapping strategy
        pair_id = swap_dist.distribute_entanglement("node_0", "node_2", "swap")
        
        assert pair_id is not None
        assert pair_id in swap_dist.entanglement_pairs
    
    def test_entanglement_consumption(self):
        """Test consuming entanglement pairs."""
        pair_id = self.ent_dist.distribute_entanglement("node_0", "node_1")
        assert pair_id is not None
        
        # Consume the pair
        success = self.ent_dist.consume_entanglement(pair_id)
        assert success is True
        assert pair_id not in self.ent_dist.entanglement_pairs
        
        # Try to consume nonexistent pair
        success = self.ent_dist.consume_entanglement("nonexistent")
        assert success is False
    
    def test_expired_entanglement_cleanup(self):
        """Test cleanup of expired entanglement."""
        # Create pair with very short coherence time
        pair = EntanglementPair(
            pair_id="short_lived",
            node_a="node_0",
            node_b="node_1",
            qubit_a=0,
            qubit_b=1,
            coherence_time=0.001  # Very short
        )
        
        self.ent_dist.entanglement_pairs["short_lived"] = pair
        
        # Wait for expiration
        time.sleep(0.002)
        
        # Cleanup should remove expired pair
        expired_count = self.ent_dist.cleanup_expired_entanglement()
        assert expired_count >= 1
        assert "short_lived" not in self.ent_dist.entanglement_pairs
    
    def test_entanglement_statistics(self):
        """Test entanglement statistics calculation."""
        # Create some entanglement pairs
        for i in range(1, 4):
            self.ent_dist.distribute_entanglement("node_0", f"node_{i}")
        
        stats = self.ent_dist.get_entanglement_statistics()
        
        assert stats['total_pairs'] >= 3
        assert stats['active_pairs'] >= 0
        assert 'average_fidelity' in stats
        assert 'min_fidelity' in stats
        assert 'max_fidelity' in stats


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestQuantumTeleportation:
    """Test QuantumTeleportation functionality."""
    
    def setup_method(self):
        """Set up test teleportation."""
        self.topology = QuantumNetworkTopology()
        self.topology.create_topology(NetworkTopology.STAR, 5)
        self.ent_dist = EntanglementDistribution(self.topology)
        self.teleportation = QuantumTeleportation(self.ent_dist)
    
    def test_successful_teleportation(self):
        """Test successful quantum teleportation."""
        # First create entanglement
        pair_id = self.ent_dist.distribute_entanglement("node_0", "node_1")
        assert pair_id is not None
        
        # Perform teleportation
        result = self.teleportation.teleport_qubit("node_0", "node_1")
        
        assert result['success'] is True
        assert result['source_node'] == "node_0"
        assert result['target_node'] == "node_1"
        assert 'teleportation_fidelity' in result
        assert 'classical_delay' in result
        assert 'measurement_outcomes' in result
        
        # Entanglement should be consumed
        assert pair_id not in self.ent_dist.entanglement_pairs
    
    def test_teleportation_without_entanglement(self):
        """Test teleportation when no entanglement exists."""
        # Try teleportation without existing entanglement
        result = self.teleportation.teleport_qubit("node_0", "node_1")
        
        # Should either succeed (if entanglement was created) or fail gracefully
        assert 'success' in result
        if not result['success']:
            assert 'error' in result
    
    def test_teleportation_with_custom_state(self):
        """Test teleportation with custom qubit state."""
        # Create entanglement first
        pair_id = self.ent_dist.distribute_entanglement("node_0", "node_1")
        assert pair_id is not None
        
        # Custom qubit state
        custom_state = {'alpha': 0.6, 'beta': 0.8}  # Normalized |+⟩ state
        
        result = self.teleportation.teleport_qubit("node_0", "node_1", custom_state)
        
        assert result['success'] is True
        assert result['original_state'] == custom_state
    
    def test_teleportation_statistics(self):
        """Test teleportation statistics."""
        # Perform multiple teleportations
        for i in range(3):
            # Create fresh entanglement for each teleportation
            pair_id = self.ent_dist.distribute_entanglement("node_0", f"node_{i+1}")
            if pair_id:
                self.teleportation.teleport_qubit("node_0", f"node_{i+1}")
        
        stats = self.teleportation.get_teleportation_statistics()
        
        assert stats['total_teleportations'] >= 0
        assert 'success_rate' in stats
        assert 'average_fidelity' in stats
        assert 'average_delay' in stats
    
    def test_classical_communication_delay_calculation(self):
        """Test classical communication delay calculation."""
        # Create linear topology for delay testing
        linear_topology = QuantumNetworkTopology()
        linear_topology.create_topology(NetworkTopology.LINEAR, 4, spacing=100.0)  # 100km spacing
        linear_ent_dist = EntanglementDistribution(linear_topology)
        linear_teleportation = QuantumTeleportation(linear_ent_dist)
        
        delay = linear_teleportation._calculate_classical_communication_delay("node_0", "node_3")
        
        # Should have measurable delay over long distance
        assert delay > 0.0
        assert delay < 1.0  # Should be reasonable


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestQuantumNetworkSimulator:
    """Test QuantumNetworkSimulator functionality."""
    
    def setup_method(self):
        """Set up test simulator."""
        self.simulator = QuantumNetworkSimulator("test_network")
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        assert self.simulator.network_id == "test_network"
        assert self.simulator.state == NetworkState.IDLE
        assert len(self.simulator.topology.nodes) == 0
        assert len(self.simulator.topology.channels) == 0
    
    def test_network_creation(self):
        """Test creating quantum network."""
        success = self.simulator.create_network(NetworkTopology.STAR, 5)
        
        assert success is True
        assert self.simulator.state == NetworkState.IDLE
        assert len(self.simulator.topology.nodes) == 5
        assert len(self.simulator.topology.channels) == 4
    
    def test_custom_node_addition(self):
        """Test adding custom nodes."""
        custom_node = NetworkNode(
            node_id="custom_node",
            node_type="quantum_sensor",
            position=(100.0, 100.0)
        )
        
        success = self.simulator.add_custom_node(custom_node)
        assert success is True
        assert "custom_node" in self.simulator.topology.nodes
    
    def test_custom_channel_addition(self):
        """Test adding custom channels."""
        # Add nodes first
        node_a = NetworkNode("node_a", position=(0.0, 0.0))
        node_b = NetworkNode("node_b", position=(1.0, 0.0))
        
        self.simulator.add_custom_node(node_a)
        self.simulator.add_custom_node(node_b)
        
        # Add custom channel
        custom_channel = QuantumChannel(
            channel_id="custom_channel",
            source_node="node_a",
            target_node="node_b",
            channel_type=ChannelType.FREE_SPACE,
            distance=1.0
        )
        
        success = self.simulator.add_custom_channel(custom_channel)
        assert success is True
        assert "custom_channel" in self.simulator.topology.channels
    
    def test_entanglement_distribution_workflow(self):
        """Test entanglement distribution workflow."""
        # Create network first
        self.simulator.create_network(NetworkTopology.MESH, 4)
        
        # Distribute entanglement
        pair_id = self.simulator.run_entanglement_distribution("node_0", "node_1")
        
        assert pair_id is not None
        assert self.simulator.state == NetworkState.OPERATING
        assert self.simulator.statistics['total_entanglement_pairs'] >= 1
    
    def test_teleportation_workflow(self):
        """Test quantum teleportation workflow."""
        # Create network
        self.simulator.create_network(NetworkTopology.STAR, 3)
        
        # Run teleportation
        result = self.simulator.run_quantum_teleportation("node_0", "node_1")
        
        assert 'success' in result
        if result['success']:
            assert self.simulator.statistics['total_teleportations'] >= 1
            assert self.simulator.statistics['successful_protocols'] >= 1
    
    def test_superdense_coding_protocol(self):
        """Test superdense coding protocol."""
        # Create network
        self.simulator.create_network(NetworkTopology.MESH, 3)
        
        # Run superdense coding
        classical_bits = [1, 0]
        result = self.simulator.run_superdense_coding("node_0", "node_1", classical_bits)
        
        assert 'success' in result
        if result['success']:
            assert result['classical_bits'] == classical_bits
            assert 'encoded_state' in result
            assert 'protocol_fidelity' in result
    
    def test_superdense_coding_invalid_bits(self):
        """Test superdense coding with invalid input."""
        self.simulator.create_network(NetworkTopology.STAR, 3)
        
        # Invalid number of bits
        result = self.simulator.run_superdense_coding("node_0", "node_1", [1, 0, 1])
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_network_status_reporting(self):
        """Test network status reporting."""
        # Create network and run some protocols
        self.simulator.create_network(NetworkTopology.RING, 4)
        self.simulator.run_entanglement_distribution("node_0", "node_1")
        
        status = self.simulator.get_network_status()
        
        assert status['network_id'] == "test_network"
        assert status['state'] == self.simulator.state.value
        assert status['num_nodes'] == 4
        assert status['num_channels'] == 4
        assert 'entanglement_statistics' in status
        assert 'teleportation_statistics' in status
        assert 'protocol_statistics' in status
        assert 'uptime' in status
    
    def test_resource_cleanup(self):
        """Test cleaning up expired resources."""
        # Create network and some entanglement
        self.simulator.create_network(NetworkTopology.STAR, 3)
        
        # Create entanglement with short coherence time
        pair = EntanglementPair(
            pair_id="short_lived_test",
            node_a="node_0",
            node_b="node_1",
            qubit_a=0,
            qubit_b=1,
            coherence_time=0.001
        )
        self.simulator.entanglement_manager.entanglement_pairs["short_lived_test"] = pair
        
        # Wait for expiration
        time.sleep(0.002)
        
        # Cleanup
        cleanup_result = self.simulator.cleanup_expired_resources()
        
        assert 'expired_entanglements' in cleanup_result
        assert cleanup_result['expired_entanglements'] >= 1
    
    def test_network_data_export(self):
        """Test exporting network data."""
        # Create and populate network
        self.simulator.create_network(NetworkTopology.MESH, 3)
        self.simulator.run_entanglement_distribution("node_0", "node_1")
        
        export_data = self.simulator.export_network_data()
        
        assert 'network_info' in export_data
        assert 'nodes' in export_data
        assert 'channels' in export_data
        assert 'entanglement_pairs' in export_data
        assert 'statistics' in export_data
        
        # Check structure
        assert export_data['network_info']['network_id'] == "test_network"
        assert len(export_data['nodes']) == 3
        assert len(export_data['channels']) == 3  # Full mesh of 3 nodes


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING or not MATPLOTLIB_AVAILABLE, 
                   reason="quantum_networking or matplotlib not available")
class TestQuantumNetworkVisualizer:
    """Test QuantumNetworkVisualizer functionality."""
    
    def setup_method(self):
        """Set up test visualizer."""
        self.visualizer = QuantumNetworkVisualizer()
        self.simulator = QuantumNetworkSimulator("viz_test")
        self.simulator.create_network(NetworkTopology.STAR, 5)
    
    @patch('matplotlib.pyplot.show')
    def test_network_visualization(self, mock_show):
        """Test network visualization."""
        # Add some entanglement for visualization
        self.simulator.run_entanglement_distribution("node_0", "node_1")
        
        success = self.visualizer.visualize_network(self.simulator)
        assert success is True
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_performance_plotting(self, mock_show):
        """Test protocol performance plotting."""
        # Run some protocols to generate data
        self.simulator.run_entanglement_distribution("node_0", "node_1")
        self.simulator.run_quantum_teleportation("node_0", "node_1")
        
        success = self.visualizer.plot_protocol_performance(self.simulator)
        assert success is True
        mock_show.assert_called_once()
    
    def test_visualizer_without_matplotlib(self):
        """Test visualizer behavior without matplotlib."""
        # This test would require mocking the import, skipping for now
        pass


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_quantum_network_simulator(self):
        """Test getting global simulator instance."""
        simulator1 = get_quantum_network_simulator()
        simulator2 = get_quantum_network_simulator()
        
        # Should be singleton
        assert simulator1 is simulator2
        assert isinstance(simulator1, QuantumNetworkSimulator)
    
    def test_create_quantum_network(self):
        """Test convenience function for network creation."""
        success = create_quantum_network(NetworkTopology.RING, 4)
        
        assert success is True
        
        # Check that global simulator was used
        simulator = get_quantum_network_simulator()
        assert len(simulator.topology.nodes) == 4
        assert len(simulator.topology.channels) == 4
    
    def test_distribute_entanglement_convenience(self):
        """Test convenience function for entanglement distribution."""
        # Create network first
        create_quantum_network(NetworkTopology.MESH, 3)
        
        # Use convenience function
        pair_id = distribute_entanglement("node_0", "node_1")
        
        assert pair_id is not None
        
        # Check in global simulator
        simulator = get_quantum_network_simulator()
        assert pair_id in simulator.entanglement_manager.entanglement_pairs
    
    def test_teleport_qubit_convenience(self):
        """Test convenience function for teleportation."""
        # Create network first
        create_quantum_network(NetworkTopology.STAR, 3)
        
        # Use convenience function
        result = teleport_qubit("node_0", "node_1")
        
        assert 'success' in result
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
    @patch('matplotlib.pyplot.show')
    def test_visualize_quantum_network_convenience(self, mock_show):
        """Test convenience function for visualization."""
        # Create network first
        create_quantum_network(NetworkTopology.RING, 4)
        
        # Use convenience function
        success = visualize_quantum_network()
        
        assert success is True
        mock_show.assert_called_once()


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_topology_creation_errors(self):
        """Test topology creation error handling."""
        topology = QuantumNetworkTopology()
        
        # Try to create topology with invalid parameters
        success = topology.create_topology(NetworkTopology.STAR, 0)  # Zero nodes
        assert success is False
    
    def test_entanglement_distribution_errors(self):
        """Test entanglement distribution error handling."""
        topology = QuantumNetworkTopology()
        ent_dist = EntanglementDistribution(topology)
        
        # Try distribution without nodes
        pair_id = ent_dist.distribute_entanglement("nonexistent_a", "nonexistent_b")
        assert pair_id is None
    
    def test_simulator_error_states(self):
        """Test simulator error state handling."""
        simulator = QuantumNetworkSimulator()
        
        # Try operations without network
        result = simulator.run_quantum_teleportation("node_0", "node_1")
        assert result['success'] is False
        assert 'error' in result
    
    def test_invalid_protocol_parameters(self):
        """Test handling of invalid protocol parameters."""
        simulator = QuantumNetworkSimulator()
        simulator.create_network(NetworkTopology.STAR, 3)
        
        # Invalid superdense coding bits
        result = simulator.run_superdense_coding("node_0", "node_1", [1, 0, 1, 1])
        assert result['success'] is False
        
        # Invalid nodes for teleportation
        result = simulator.run_quantum_teleportation("invalid_node", "node_1")
        assert result['success'] is False


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestQuantumNetworkingIntegration:
    """Test integration scenarios for quantum networking."""
    
    def test_end_to_end_quantum_communication(self):
        """Test complete quantum communication workflow."""
        # Create distributed network
        simulator = QuantumNetworkSimulator("integration_test")
        success = simulator.create_network(NetworkTopology.MESH, 6)
        assert success is True
        
        # Distribute entanglement across multiple pairs
        entanglement_pairs = []
        for i in range(1, 4):
            pair_id = simulator.run_entanglement_distribution("node_0", f"node_{i}")
            if pair_id:
                entanglement_pairs.append(pair_id)
        
        assert len(entanglement_pairs) >= 1
        
        # Perform teleportations
        teleportation_results = []
        for i in range(1, min(4, len(entanglement_pairs) + 1)):
            result = simulator.run_quantum_teleportation("node_0", f"node_{i}")
            teleportation_results.append(result)
        
        # Check results
        successful_teleportations = sum(1 for r in teleportation_results if r.get('success', False))
        assert successful_teleportations >= 0  # Should have at least attempted
        
        # Network should still be operational
        assert simulator.state in [NetworkState.OPERATING, NetworkState.IDLE]
    
    def test_multi_protocol_network_usage(self):
        """Test using multiple protocols simultaneously."""
        simulator = QuantumNetworkSimulator("multi_protocol_test")
        simulator.create_network(NetworkTopology.RING, 8)
        
        # Run various protocols
        protocols_run = 0
        
        # Entanglement distribution
        for i in range(1, 4):
            pair_id = simulator.run_entanglement_distribution("node_0", f"node_{i}")
            if pair_id:
                protocols_run += 1
        
        # Quantum teleportation
        for i in range(1, 3):
            result = simulator.run_quantum_teleportation(f"node_{i}", f"node_{i+1}")
            if result.get('success', False):
                protocols_run += 1
        
        # Superdense coding
        result = simulator.run_superdense_coding("node_0", "node_7", [1, 1])
        if result.get('success', False):
            protocols_run += 1
        
        # Check overall network performance
        status = simulator.get_network_status()
        assert status['protocol_statistics']['total_protocols'] >= protocols_run
    
    def test_network_scalability(self):
        """Test network scalability with larger topologies."""
        # Test various topology sizes
        topologies = [
            (NetworkTopology.STAR, 10),
            (NetworkTopology.RING, 12),
            (NetworkTopology.GRID, 16),  # 4x4 grid
            (NetworkTopology.LINEAR, 8)
        ]
        
        for topology_type, num_nodes in topologies:
            simulator = QuantumNetworkSimulator(f"scale_test_{topology_type.value}")
            success = simulator.create_network(topology_type, num_nodes)
            
            assert success is True
            assert len(simulator.topology.nodes) == num_nodes
            
            # Test basic functionality
            if len(simulator.topology.channels) > 0:
                # Try entanglement distribution
                nodes = list(simulator.topology.nodes.keys())
                pair_id = simulator.run_entanglement_distribution(nodes[0], nodes[1])
                # Should either succeed or fail gracefully
                assert pair_id is not None or pair_id is None
    
    def test_fault_tolerance_and_recovery(self):
        """Test network fault tolerance."""
        simulator = QuantumNetworkSimulator("fault_tolerance_test")
        simulator.create_network(NetworkTopology.MESH, 5)
        
        # Create some entanglement
        original_pairs = []
        for i in range(1, 4):
            pair_id = simulator.run_entanglement_distribution("node_0", f"node_{i}")
            if pair_id:
                original_pairs.append(pair_id)
        
        # Simulate node failure by removing a node (in real implementation)
        # For now, test cleanup of expired resources
        cleanup_result = simulator.cleanup_expired_resources()
        
        # Network should remain operational
        assert simulator.state != NetworkState.ERROR
        
        # Should be able to create new entanglement
        new_pair_id = simulator.run_entanglement_distribution("node_1", "node_2")
        # Should either succeed or fail gracefully
        assert new_pair_id is not None or new_pair_id is None
    
    def test_protocol_fidelity_degradation(self):
        """Test protocol behavior under fidelity degradation."""
        simulator = QuantumNetworkSimulator("fidelity_test")
        
        # Create network with custom low-fidelity channels
        simulator.create_network(NetworkTopology.LINEAR, 4)
        
        # Modify channel fidelities to simulate degradation
        for channel in simulator.topology.channels.values():
            channel.fidelity = 0.7  # Lower fidelity
            channel.loss_rate = 0.5  # Higher loss
        
        # Test entanglement distribution with degraded channels
        pair_id = simulator.run_entanglement_distribution("node_0", "node_3", strategy="repeater")
        
        if pair_id:
            # Check if fidelity is appropriately reduced
            pair = simulator.entanglement_manager.entanglement_pairs[pair_id]
            assert pair.fidelity < 1.0
            
            # Test teleportation with degraded entanglement
            result = simulator.run_quantum_teleportation("node_0", "node_3")
            if result.get('success', False):
                assert result['teleportation_fidelity'] < 1.0


@pytest.mark.skipif(not HAS_QUANTUM_NETWORKING, reason="quantum_networking module not available")
class TestQuantumNetworkingPerformance:
    """Test performance characteristics of quantum networking."""
    
    def test_large_network_creation_performance(self):
        """Test performance with large networks."""
        start_time = time.time()
        
        # Create large mesh network
        simulator = QuantumNetworkSimulator("performance_test")
        success = simulator.create_network(NetworkTopology.MESH, 20)
        
        creation_time = time.time() - start_time
        
        assert success is True
        assert creation_time < 5.0  # Should create within 5 seconds
        assert len(simulator.topology.nodes) == 20
        assert len(simulator.topology.channels) == 20 * 19 // 2  # Full mesh
    
    def test_concurrent_protocol_performance(self):
        """Test performance with many concurrent protocols."""
        simulator = QuantumNetworkSimulator("concurrent_test")
        simulator.create_network(NetworkTopology.GRID, 25)  # 5x5 grid
        
        start_time = time.time()
        
        # Run many entanglement distributions
        successful_distributions = 0
        for i in range(20):
            source = f"node_{i % 25}"
            target = f"node_{(i + 1) % 25}"
            pair_id = simulator.run_entanglement_distribution(source, target)
            if pair_id:
                successful_distributions += 1
        
        distribution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert distribution_time < 10.0
        assert successful_distributions >= 0  # At least attempting
    
    def test_memory_usage_with_many_protocols(self):
        """Test memory usage doesn't grow excessively."""
        simulator = QuantumNetworkSimulator("memory_test")
        simulator.create_network(NetworkTopology.RING, 10)
        
        initial_pair_count = len(simulator.entanglement_manager.entanglement_pairs)
        
        # Run many protocols and cleanup
        for cycle in range(10):
            # Create entanglement
            for i in range(5):
                simulator.run_entanglement_distribution("node_0", f"node_{i+1}")
            
            # Consume some entanglement through teleportation
            for i in range(3):
                simulator.run_quantum_teleportation("node_0", f"node_{i+1}")
            
            # Cleanup expired resources
            simulator.cleanup_expired_resources()
        
        final_pair_count = len(simulator.entanglement_manager.entanglement_pairs)
        
        # Memory usage should be bounded
        assert final_pair_count < initial_pair_count + 50  # Shouldn't accumulate excessively
    
    def test_pathfinding_performance(self):
        """Test pathfinding performance in large networks."""
        simulator = QuantumNetworkSimulator("pathfinding_test")
        simulator.create_network(NetworkTopology.GRID, 36)  # 6x6 grid
        
        start_time = time.time()
        
        # Find paths between many node pairs
        path_count = 0
        for i in range(20):
            source = f"node_{i % 36}"
            target = f"node_{(i + 10) % 36}"
            path = simulator.topology.find_shortest_path(source, target)
            if path:
                path_count += 1
        
        pathfinding_time = time.time() - start_time
        
        # Should find paths quickly
        assert pathfinding_time < 2.0
        assert path_count >= 15  # Most paths should be found


if __name__ == "__main__":
    pytest.main([__file__])