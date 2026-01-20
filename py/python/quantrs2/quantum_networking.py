"""
Quantum Networking Protocols

This module provides comprehensive support for quantum networking protocols,
including quantum communication, entanglement distribution, and network simulation.
"""

import json
import time
import math
import random
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import socket
import uuid

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import networkx as nx
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False


class NetworkTopology(Enum):
    """Network topology types."""
    STAR = "star"
    MESH = "mesh"
    RING = "ring"
    LINEAR = "linear"
    TREE = "tree"
    GRID = "grid"
    CUSTOM = "custom"


class ProtocolType(Enum):
    """Quantum networking protocol types."""
    ENTANGLEMENT_DISTRIBUTION = "entanglement_distribution"
    QUANTUM_TELEPORTATION = "quantum_teleportation"
    QUANTUM_SUPERDENSE_CODING = "quantum_superdense_coding"
    QUANTUM_REPEATER = "quantum_repeater"
    QUANTUM_INTERNET = "quantum_internet"
    DISTRIBUTED_COMPUTING = "distributed_computing"
    QUANTUM_SENSING_NETWORK = "quantum_sensing_network"


class NetworkState(Enum):
    """Network states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    DISTRIBUTING = "distributing"
    OPERATING = "operating"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ChannelType(Enum):
    """Quantum channel types."""
    DIRECT = "direct"
    FIBER_OPTIC = "fiber_optic"
    FREE_SPACE = "free_space"
    SATELLITE = "satellite"
    MICROWAVE = "microwave"


@dataclass
class QuantumChannel:
    """Represents a quantum communication channel."""
    channel_id: str
    source_node: str
    target_node: str
    channel_type: ChannelType
    fidelity: float = 0.99
    transmission_rate: float = 1000.0  # Hz
    distance: float = 1.0  # km
    loss_rate: float = 0.0  # dB/km
    noise_model: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    
    def __post_init__(self):
        """Initialize channel properties."""
        if self.noise_model is None:
            self.noise_model = self._default_noise_model()
    
    def _default_noise_model(self) -> Dict[str, Any]:
        """Create default noise model for channel."""
        base_error_rate = 0.01
        
        # Adjust error rate based on channel type and distance
        if self.channel_type == ChannelType.FIBER_OPTIC:
            error_rate = base_error_rate + self.distance * 0.001
        elif self.channel_type == ChannelType.FREE_SPACE:
            error_rate = base_error_rate + self.distance * 0.005
        elif self.channel_type == ChannelType.SATELLITE:
            error_rate = base_error_rate + self.distance * 0.0001
        else:
            error_rate = base_error_rate
        
        return {
            'error_rate': min(error_rate, 0.5),
            'decoherence_time': 1000.0 / (1 + self.distance),  # microseconds
            'gate_error_rate': 0.001,
            'measurement_error_rate': 0.01
        }
    
    def calculate_transmission_fidelity(self, base_fidelity: float = 1.0) -> float:
        """Calculate transmission fidelity accounting for channel effects."""
        # Account for distance-based loss
        distance_factor = math.exp(-self.loss_rate * self.distance / 10)
        
        # Account for noise
        noise_factor = 1.0 - self.noise_model['error_rate']
        
        # Combined fidelity
        effective_fidelity = base_fidelity * self.fidelity * distance_factor * noise_factor
        return max(effective_fidelity, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'channel_id': self.channel_id,
            'source_node': self.source_node,
            'target_node': self.target_node,
            'channel_type': self.channel_type.value,
            'fidelity': self.fidelity,
            'transmission_rate': self.transmission_rate,
            'distance': self.distance,
            'loss_rate': self.loss_rate,
            'noise_model': self.noise_model,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'usage_count': self.usage_count
        }


@dataclass
class NetworkNode:
    """Represents a node in a quantum network."""
    node_id: str
    node_type: str = "quantum_computer"
    position: Tuple[float, float] = (0.0, 0.0)
    capabilities: List[str] = field(default_factory=list)
    memory_capacity: int = 10  # Number of qubits
    coherence_time: float = 1000.0  # microseconds
    gate_fidelity: float = 0.999
    measurement_fidelity: float = 0.99
    processing_power: float = 1.0  # Relative processing capacity
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    current_load: float = 0.0
    status: str = "online"
    
    def __post_init__(self):
        """Initialize node capabilities."""
        if not self.capabilities:
            self.capabilities = [
                "entanglement_generation",
                "quantum_teleportation",
                "quantum_measurement",
                "classical_communication"
            ]
    
    def can_perform_protocol(self, protocol_type: ProtocolType) -> bool:
        """Check if node can perform specific protocol."""
        required_capabilities = {
            ProtocolType.ENTANGLEMENT_DISTRIBUTION: ["entanglement_generation"],
            ProtocolType.QUANTUM_TELEPORTATION: ["quantum_teleportation", "quantum_measurement"],
            ProtocolType.QUANTUM_SUPERDENSE_CODING: ["quantum_measurement"],
            ProtocolType.QUANTUM_REPEATER: ["entanglement_generation", "quantum_teleportation"],
            ProtocolType.QUANTUM_INTERNET: ["entanglement_generation", "quantum_teleportation"],
            ProtocolType.DISTRIBUTED_COMPUTING: ["quantum_computation"],
            ProtocolType.QUANTUM_SENSING_NETWORK: ["quantum_sensing"]
        }
        
        required = required_capabilities.get(protocol_type, [])
        return all(cap in self.capabilities for cap in required)
    
    def calculate_distance(self, other_node: 'NetworkNode') -> float:
        """Calculate distance to another node."""
        dx = self.position[0] - other_node.position[0]
        dy = self.position[1] - other_node.position[1]
        return math.sqrt(dx**2 + dy**2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'position': self.position,
            'capabilities': self.capabilities,
            'memory_capacity': self.memory_capacity,
            'coherence_time': self.coherence_time,
            'gate_fidelity': self.gate_fidelity,
            'measurement_fidelity': self.measurement_fidelity,
            'processing_power': self.processing_power,
            'created_at': self.created_at,
            'last_active': self.last_active,
            'current_load': self.current_load,
            'status': self.status
        }


@dataclass
class EntanglementPair:
    """Represents an entangled qubit pair."""
    pair_id: str
    node_a: str
    node_b: str
    qubit_a: int
    qubit_b: int
    fidelity: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    coherence_time: float = 1000.0  # microseconds
    usage_count: int = 0
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if entanglement has expired due to decoherence."""
        if current_time is None:
            current_time = time.time()
        
        time_elapsed = (current_time - self.created_at) * 1e6  # Convert to microseconds
        return time_elapsed > self.coherence_time
    
    def calculate_current_fidelity(self, current_time: Optional[float] = None) -> float:
        """Calculate current fidelity accounting for decoherence."""
        if self.is_expired(current_time):
            return 0.0
        
        if current_time is None:
            current_time = time.time()
        
        time_elapsed = (current_time - self.created_at) * 1e6  # microseconds
        decay_factor = math.exp(-time_elapsed / self.coherence_time)
        return self.fidelity * decay_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pair_id': self.pair_id,
            'node_a': self.node_a,
            'node_b': self.node_b,
            'qubit_a': self.qubit_a,
            'qubit_b': self.qubit_b,
            'fidelity': self.fidelity,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'coherence_time': self.coherence_time,
            'usage_count': self.usage_count,
            'current_fidelity': self.calculate_current_fidelity()
        }


@dataclass
class NetworkProtocol:
    """Base class for quantum network protocols."""
    protocol_id: str
    protocol_type: ProtocolType
    participants: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = "initialized"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'protocol_id': self.protocol_id,
            'protocol_type': self.protocol_type.value,
            'participants': self.participants,
            'parameters': self.parameters,
            'created_at': self.created_at,
            'status': self.status
        }


class QuantumNetworkTopology:
    """Manages quantum network topology and routing."""
    
    def __init__(self):
        self.topology_type = NetworkTopology.CUSTOM
        self.nodes: Dict[str, NetworkNode] = {}
        self.channels: Dict[str, QuantumChannel] = {}
        self.adjacency_matrix: Dict[str, Dict[str, float]] = {}
        
    def add_node(self, node: NetworkNode) -> bool:
        """Add a node to the network."""
        try:
            self.nodes[node.node_id] = node
            self.adjacency_matrix[node.node_id] = {}
            return True
        except Exception:
            return False
    
    def add_channel(self, channel: QuantumChannel) -> bool:
        """Add a channel to the network."""
        try:
            # Check if both nodes exist
            if channel.source_node not in self.nodes or channel.target_node not in self.nodes:
                return False
            
            self.channels[channel.channel_id] = channel
            
            # Update adjacency matrix
            distance = channel.distance
            self.adjacency_matrix[channel.source_node][channel.target_node] = distance
            self.adjacency_matrix[channel.target_node][channel.source_node] = distance
            
            return True
        except Exception:
            return False
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using Dijkstra's algorithm."""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        if source == target:
            return [source]
        
        # Dijkstra's algorithm
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[source] = 0
        previous = {}
        unvisited = set(self.nodes.keys())
        
        while unvisited:
            # Find unvisited node with minimum distance
            current = min(unvisited, key=lambda x: distances[x])
            
            if distances[current] == float('inf'):
                break
            
            if current == target:
                # Reconstruct path
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(source)
                return path[::-1]
            
            unvisited.remove(current)
            
            # Update distances to neighbors
            for neighbor, distance in self.adjacency_matrix[current].items():
                if neighbor in unvisited:
                    new_distance = distances[current] + distance
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        return None
    
    def get_network_diameter(self) -> float:
        """Calculate network diameter (longest shortest path)."""
        max_distance = 0.0
        node_ids = list(self.nodes.keys())
        
        for i, source in enumerate(node_ids):
            for target in node_ids[i+1:]:
                path = self.find_shortest_path(source, target)
                if path:
                    distance = self._calculate_path_distance(path)
                    max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _calculate_path_distance(self, path: List[str]) -> float:
        """Calculate total distance of a path."""
        total_distance = 0.0
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # Find channel between nodes
            for channel in self.channels.values():
                if ((channel.source_node == current and channel.target_node == next_node) or
                    (channel.source_node == next_node and channel.target_node == current)):
                    total_distance += channel.distance
                    break
        
        return total_distance
    
    def create_topology(self, topology_type: NetworkTopology, 
                       num_nodes: int, **kwargs) -> bool:
        """Create a standard network topology."""
        try:
            self.topology_type = topology_type
            self.nodes.clear()
            self.channels.clear()
            self.adjacency_matrix.clear()
            
            # Create nodes
            for i in range(num_nodes):
                node = NetworkNode(
                    node_id=f"node_{i}",
                    position=self._generate_node_position(i, num_nodes, topology_type, **kwargs)
                )
                self.add_node(node)
            
            # Create channels based on topology
            if topology_type == NetworkTopology.STAR:
                self._create_star_topology(**kwargs)
            elif topology_type == NetworkTopology.MESH:
                self._create_mesh_topology(**kwargs)
            elif topology_type == NetworkTopology.RING:
                self._create_ring_topology(**kwargs)
            elif topology_type == NetworkTopology.LINEAR:
                self._create_linear_topology(**kwargs)
            elif topology_type == NetworkTopology.TREE:
                self._create_tree_topology(**kwargs)
            elif topology_type == NetworkTopology.GRID:
                self._create_grid_topology(**kwargs)
            
            return True
            
        except Exception:
            return False
    
    def _generate_node_position(self, index: int, total_nodes: int, 
                               topology_type: NetworkTopology, **kwargs) -> Tuple[float, float]:
        """Generate node position based on topology."""
        if topology_type == NetworkTopology.RING:
            angle = 2 * math.pi * index / total_nodes
            radius = kwargs.get('radius', 10.0)
            return (radius * math.cos(angle), radius * math.sin(angle))
        
        elif topology_type == NetworkTopology.LINEAR:
            spacing = kwargs.get('spacing', 1.0)
            return (index * spacing, 0.0)
        
        elif topology_type == NetworkTopology.GRID:
            grid_size = int(math.ceil(math.sqrt(total_nodes)))
            row = index // grid_size
            col = index % grid_size
            spacing = kwargs.get('spacing', 1.0)
            return (col * spacing, row * spacing)
        
        elif topology_type == NetworkTopology.STAR:
            if index == 0:  # Central node
                return (0.0, 0.0)
            else:
                angle = 2 * math.pi * (index - 1) / (total_nodes - 1)
                radius = kwargs.get('radius', 5.0)
                return (radius * math.cos(angle), radius * math.sin(angle))
        
        else:
            # Random positioning for other topologies
            return (random.uniform(-10, 10), random.uniform(-10, 10))
    
    def _create_star_topology(self, **kwargs):
        """Create star topology channels."""
        center_node = "node_0"
        channel_type = kwargs.get('channel_type', ChannelType.FIBER_OPTIC)
        
        for i in range(1, len(self.nodes)):
            node_id = f"node_{i}"
            channel = QuantumChannel(
                channel_id=f"channel_{center_node}_{node_id}",
                source_node=center_node,
                target_node=node_id,
                channel_type=channel_type,
                distance=self.nodes[center_node].calculate_distance(self.nodes[node_id])
            )
            self.add_channel(channel)
    
    def _create_mesh_topology(self, **kwargs):
        """Create full mesh topology channels."""
        channel_type = kwargs.get('channel_type', ChannelType.FIBER_OPTIC)
        node_ids = list(self.nodes.keys())
        
        for i, source in enumerate(node_ids):
            for target in node_ids[i+1:]:
                channel = QuantumChannel(
                    channel_id=f"channel_{source}_{target}",
                    source_node=source,
                    target_node=target,
                    channel_type=channel_type,
                    distance=self.nodes[source].calculate_distance(self.nodes[target])
                )
                self.add_channel(channel)
    
    def _create_ring_topology(self, **kwargs):
        """Create ring topology channels."""
        channel_type = kwargs.get('channel_type', ChannelType.FIBER_OPTIC)
        node_ids = list(self.nodes.keys())
        
        for i in range(len(node_ids)):
            source = node_ids[i]
            target = node_ids[(i + 1) % len(node_ids)]
            
            channel = QuantumChannel(
                channel_id=f"channel_{source}_{target}",
                source_node=source,
                target_node=target,
                channel_type=channel_type,
                distance=self.nodes[source].calculate_distance(self.nodes[target])
            )
            self.add_channel(channel)
    
    def _create_linear_topology(self, **kwargs):
        """Create linear topology channels."""
        channel_type = kwargs.get('channel_type', ChannelType.FIBER_OPTIC)
        node_ids = list(self.nodes.keys())
        
        for i in range(len(node_ids) - 1):
            source = node_ids[i]
            target = node_ids[i + 1]
            
            channel = QuantumChannel(
                channel_id=f"channel_{source}_{target}",
                source_node=source,
                target_node=target,
                channel_type=channel_type,
                distance=self.nodes[source].calculate_distance(self.nodes[target])
            )
            self.add_channel(channel)
    
    def _create_tree_topology(self, **kwargs):
        """Create tree topology channels."""
        channel_type = kwargs.get('channel_type', ChannelType.FIBER_OPTIC)
        branching_factor = kwargs.get('branching_factor', 2)
        node_ids = list(self.nodes.keys())
        
        for i, node_id in enumerate(node_ids):
            # Connect to parent (except root)
            if i > 0:
                parent_index = (i - 1) // branching_factor
                parent_id = node_ids[parent_index]
                
                channel = QuantumChannel(
                    channel_id=f"channel_{parent_id}_{node_id}",
                    source_node=parent_id,
                    target_node=node_id,
                    channel_type=channel_type,
                    distance=self.nodes[parent_id].calculate_distance(self.nodes[node_id])
                )
                self.add_channel(channel)
    
    def _create_grid_topology(self, **kwargs):
        """Create grid topology channels."""
        channel_type = kwargs.get('channel_type', ChannelType.FIBER_OPTIC)
        node_ids = list(self.nodes.keys())
        grid_size = int(math.ceil(math.sqrt(len(node_ids))))
        
        for i, node_id in enumerate(node_ids):
            row = i // grid_size
            col = i % grid_size
            
            # Connect to right neighbor
            if col < grid_size - 1 and i + 1 < len(node_ids):
                right_neighbor = node_ids[i + 1]
                channel = QuantumChannel(
                    channel_id=f"channel_{node_id}_{right_neighbor}",
                    source_node=node_id,
                    target_node=right_neighbor,
                    channel_type=channel_type,
                    distance=self.nodes[node_id].calculate_distance(self.nodes[right_neighbor])
                )
                self.add_channel(channel)
            
            # Connect to bottom neighbor
            if row < grid_size - 1 and i + grid_size < len(node_ids):
                bottom_neighbor = node_ids[i + grid_size]
                channel = QuantumChannel(
                    channel_id=f"channel_{node_id}_{bottom_neighbor}",
                    source_node=node_id,
                    target_node=bottom_neighbor,
                    channel_type=channel_type,
                    distance=self.nodes[node_id].calculate_distance(self.nodes[bottom_neighbor])
                )
                self.add_channel(channel)


class EntanglementDistribution:
    """Manages entanglement distribution across quantum networks."""
    
    def __init__(self, topology: QuantumNetworkTopology):
        self.topology = topology
        self.entanglement_pairs: Dict[str, EntanglementPair] = {}
        self.distribution_strategies = {
            'direct': self._distribute_direct,
            'repeater': self._distribute_via_repeaters,
            'swap': self._distribute_via_swapping
        }
        
    def distribute_entanglement(self, source_node: str, target_node: str,
                              strategy: str = 'direct', 
                              target_fidelity: float = 0.9) -> Optional[str]:
        """Distribute entanglement between two nodes."""
        if source_node not in self.topology.nodes or target_node not in self.topology.nodes:
            return None
        
        if strategy not in self.distribution_strategies:
            strategy = 'direct'
        
        try:
            return self.distribution_strategies[strategy](source_node, target_node, target_fidelity)
        except Exception:
            return None
    
    def _distribute_direct(self, source_node: str, target_node: str, 
                          target_fidelity: float) -> Optional[str]:
        """Direct entanglement distribution."""
        # Find direct channel between nodes
        channel = self._find_channel(source_node, target_node)
        if not channel:
            return None
        
        # Calculate transmission fidelity
        transmission_fidelity = channel.calculate_transmission_fidelity()
        
        if transmission_fidelity < target_fidelity:
            return None
        
        # Create entanglement pair
        pair_id = f"ent_{source_node}_{target_node}_{int(time.time() * 1000)}"
        
        # Find available qubits
        source_qubit = self._find_available_qubit(source_node)
        target_qubit = self._find_available_qubit(target_node)
        
        if source_qubit is None or target_qubit is None:
            return None
        
        entanglement_pair = EntanglementPair(
            pair_id=pair_id,
            node_a=source_node,
            node_b=target_node,
            qubit_a=source_qubit,
            qubit_b=target_qubit,
            fidelity=transmission_fidelity,
            coherence_time=min(self.topology.nodes[source_node].coherence_time,
                              self.topology.nodes[target_node].coherence_time)
        )
        
        self.entanglement_pairs[pair_id] = entanglement_pair
        channel.usage_count += 1
        channel.last_used = time.time()
        
        return pair_id
    
    def _distribute_via_repeaters(self, source_node: str, target_node: str,
                                 target_fidelity: float) -> Optional[str]:
        """Entanglement distribution via quantum repeaters."""
        path = self.topology.find_shortest_path(source_node, target_node)
        if not path or len(path) < 3:  # Need at least one intermediate node
            return self._distribute_direct(source_node, target_node, target_fidelity)
        
        # Create entanglement segments
        segments = []
        for i in range(len(path) - 1):
            segment_id = self._distribute_direct(path[i], path[i + 1], target_fidelity)
            if not segment_id:
                # Clean up created segments
                for seg_id in segments:
                    self.consume_entanglement(seg_id)
                return None
            segments.append(segment_id)
        
        # Perform entanglement swapping
        return self._perform_entanglement_swapping(segments, source_node, target_node)
    
    def _distribute_via_swapping(self, source_node: str, target_node: str,
                                target_fidelity: float) -> Optional[str]:
        """Entanglement distribution via entanglement swapping."""
        # Find intermediate node
        intermediate_nodes = []
        for node_id in self.topology.nodes:
            if (node_id != source_node and node_id != target_node and
                self._find_channel(source_node, node_id) and
                self._find_channel(node_id, target_node)):
                intermediate_nodes.append(node_id)
        
        if not intermediate_nodes:
            return self._distribute_direct(source_node, target_node, target_fidelity)
        
        # Use first available intermediate node
        intermediate = intermediate_nodes[0]
        
        # Create two entanglement pairs
        pair1_id = self._distribute_direct(source_node, intermediate, target_fidelity)
        pair2_id = self._distribute_direct(intermediate, target_node, target_fidelity)
        
        if not pair1_id or not pair2_id:
            if pair1_id:
                self.consume_entanglement(pair1_id)
            if pair2_id:
                self.consume_entanglement(pair2_id)
            return None
        
        return self._perform_entanglement_swapping([pair1_id, pair2_id], source_node, target_node)
    
    def _perform_entanglement_swapping(self, segment_ids: List[str], 
                                     final_source: str, final_target: str) -> Optional[str]:
        """Perform entanglement swapping operation."""
        if len(segment_ids) < 2:
            return None
        
        # For simplicity, simulate perfect swapping
        # In reality, this would involve quantum operations
        
        # Calculate combined fidelity
        combined_fidelity = 1.0
        min_coherence_time = float('inf')
        
        for segment_id in segment_ids:
            pair = self.entanglement_pairs[segment_id]
            combined_fidelity *= pair.fidelity
            min_coherence_time = min(min_coherence_time, pair.coherence_time)
        
        # Consume intermediate segments
        for segment_id in segment_ids:
            self.consume_entanglement(segment_id)
        
        # Create final entanglement pair
        final_pair_id = f"ent_{final_source}_{final_target}_{int(time.time() * 1000)}"
        
        final_pair = EntanglementPair(
            pair_id=final_pair_id,
            node_a=final_source,
            node_b=final_target,
            qubit_a=self._find_available_qubit(final_source),
            qubit_b=self._find_available_qubit(final_target),
            fidelity=combined_fidelity,
            coherence_time=min_coherence_time
        )
        
        self.entanglement_pairs[final_pair_id] = final_pair
        return final_pair_id
    
    def _find_channel(self, source: str, target: str) -> Optional[QuantumChannel]:
        """Find channel between two nodes."""
        for channel in self.topology.channels.values():
            if ((channel.source_node == source and channel.target_node == target) or
                (channel.source_node == target and channel.target_node == source)):
                return channel
        return None
    
    def _find_available_qubit(self, node_id: str) -> Optional[int]:
        """Find available qubit in a node."""
        node = self.topology.nodes[node_id]
        used_qubits = set()
        
        # Check which qubits are already entangled
        for pair in self.entanglement_pairs.values():
            if pair.node_a == node_id:
                used_qubits.add(pair.qubit_a)
            elif pair.node_b == node_id:
                used_qubits.add(pair.qubit_b)
        
        # Find first available qubit
        for i in range(node.memory_capacity):
            if i not in used_qubits:
                return i
        
        return None
    
    def consume_entanglement(self, pair_id: str) -> bool:
        """Consume an entanglement pair (e.g., for teleportation)."""
        if pair_id in self.entanglement_pairs:
            del self.entanglement_pairs[pair_id]
            return True
        return False
    
    def cleanup_expired_entanglement(self) -> int:
        """Remove expired entanglement pairs."""
        current_time = time.time()
        expired_pairs = []
        
        for pair_id, pair in self.entanglement_pairs.items():
            if pair.is_expired(current_time):
                expired_pairs.append(pair_id)
        
        for pair_id in expired_pairs:
            del self.entanglement_pairs[pair_id]
        
        return len(expired_pairs)
    
    def get_entanglement_statistics(self) -> Dict[str, Any]:
        """Get entanglement distribution statistics."""
        current_time = time.time()
        active_pairs = 0
        total_fidelity = 0.0
        min_fidelity = 1.0
        max_fidelity = 0.0
        
        for pair in self.entanglement_pairs.values():
            if not pair.is_expired(current_time):
                active_pairs += 1
                fidelity = pair.calculate_current_fidelity(current_time)
                total_fidelity += fidelity
                min_fidelity = min(min_fidelity, fidelity)
                max_fidelity = max(max_fidelity, fidelity)
        
        return {
            'total_pairs': len(self.entanglement_pairs),
            'active_pairs': active_pairs,
            'expired_pairs': len(self.entanglement_pairs) - active_pairs,
            'average_fidelity': total_fidelity / active_pairs if active_pairs > 0 else 0.0,
            'min_fidelity': min_fidelity if active_pairs > 0 else 0.0,
            'max_fidelity': max_fidelity if active_pairs > 0 else 0.0
        }


class QuantumTeleportation:
    """Implements quantum teleportation protocol."""
    
    def __init__(self, entanglement_manager: EntanglementDistribution):
        self.entanglement_manager = entanglement_manager
        self.teleportation_history: List[Dict[str, Any]] = []
        
    def teleport_qubit(self, source_node: str, target_node: str, 
                      qubit_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform quantum teleportation between two nodes."""
        # Find available entanglement
        entanglement_pair = None
        for pair in self.entanglement_manager.entanglement_pairs.values():
            if ((pair.node_a == source_node and pair.node_b == target_node) or
                (pair.node_a == target_node and pair.node_b == source_node)):
                if not pair.is_expired():
                    entanglement_pair = pair
                    break
        
        if not entanglement_pair:
            # Try to create entanglement
            pair_id = self.entanglement_manager.distribute_entanglement(source_node, target_node)
            if not pair_id:
                return {
                    'success': False,
                    'error': 'No entanglement available',
                    'timestamp': time.time()
                }
            entanglement_pair = self.entanglement_manager.entanglement_pairs[pair_id]
        
        # Simulate teleportation process
        result = self._simulate_teleportation(entanglement_pair, qubit_state)
        
        # Consume entanglement
        self.entanglement_manager.consume_entanglement(entanglement_pair.pair_id)
        
        # Record history
        self.teleportation_history.append(result)
        
        return result
    
    def _simulate_teleportation(self, entanglement_pair: EntanglementPair,
                               qubit_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate quantum teleportation process."""
        # Default qubit state if not provided
        if qubit_state is None:
            qubit_state = {'alpha': 1.0, 'beta': 0.0}  # |0⟩ state
        
        # Calculate fidelity of teleportation
        current_fidelity = entanglement_pair.calculate_current_fidelity()
        
        # Simulate Bell measurement on source side
        measurement_outcomes = self._simulate_bell_measurement()
        
        # Calculate teleportation fidelity
        # Includes entanglement fidelity and measurement errors
        source_node = self.entanglement_manager.topology.nodes[entanglement_pair.node_a]
        target_node = self.entanglement_manager.topology.nodes[entanglement_pair.node_b]
        
        measurement_fidelity = min(source_node.measurement_fidelity, target_node.measurement_fidelity)
        teleportation_fidelity = current_fidelity * measurement_fidelity
        
        # Simulate classical communication delay
        classical_delay = self._calculate_classical_communication_delay(
            entanglement_pair.node_a, entanglement_pair.node_b
        )
        
        return {
            'success': True,
            'source_node': entanglement_pair.node_a,
            'target_node': entanglement_pair.node_b,
            'measurement_outcomes': measurement_outcomes,
            'teleportation_fidelity': teleportation_fidelity,
            'classical_delay': classical_delay,
            'original_state': qubit_state,
            'entanglement_pair_id': entanglement_pair.pair_id,
            'timestamp': time.time()
        }
    
    def _simulate_bell_measurement(self) -> Dict[str, int]:
        """Simulate Bell state measurement outcomes."""
        # Random Bell state measurement outcomes
        return {
            'measurement_x': random.randint(0, 1),
            'measurement_z': random.randint(0, 1)
        }
    
    def _calculate_classical_communication_delay(self, source: str, target: str) -> float:
        """Calculate classical communication delay between nodes."""
        # Find shortest path for classical communication
        path = self.entanglement_manager.topology.find_shortest_path(source, target)
        if not path:
            return float('inf')
        
        # Calculate total distance
        total_distance = self.entanglement_manager.topology._calculate_path_distance(path)
        
        # Assume speed of light propagation (200,000 km/s in fiber)
        speed_of_light_fiber = 200000  # km/s
        propagation_delay = total_distance / speed_of_light_fiber
        
        # Add processing delays
        processing_delay = len(path) * 0.001  # 1ms per hop
        
        return propagation_delay + processing_delay
    
    def get_teleportation_statistics(self) -> Dict[str, Any]:
        """Get teleportation protocol statistics."""
        if not self.teleportation_history:
            return {
                'total_teleportations': 0,
                'success_rate': 0.0,
                'average_fidelity': 0.0,
                'average_delay': 0.0
            }
        
        successful = [t for t in self.teleportation_history if t['success']]
        
        total_fidelity = sum(t['teleportation_fidelity'] for t in successful)
        total_delay = sum(t['classical_delay'] for t in successful)
        
        return {
            'total_teleportations': len(self.teleportation_history),
            'successful_teleportations': len(successful),
            'success_rate': len(successful) / len(self.teleportation_history),
            'average_fidelity': total_fidelity / len(successful) if successful else 0.0,
            'average_delay': total_delay / len(successful) if successful else 0.0
        }


class QuantumNetworkSimulator:
    """Main quantum network simulator."""
    
    def __init__(self, network_id: str = None):
        self.network_id = network_id or f"qnet_{int(time.time())}"
        self.topology = QuantumNetworkTopology()
        self.entanglement_manager = EntanglementDistribution(self.topology)
        self.teleportation_protocol = QuantumTeleportation(self.entanglement_manager)
        self.protocols: Dict[str, NetworkProtocol] = {}
        self.state = NetworkState.IDLE
        self.created_at = time.time()
        self.statistics = {
            'total_protocols': 0,
            'successful_protocols': 0,
            'total_entanglement_pairs': 0,
            'total_teleportations': 0
        }
        
    def create_network(self, topology_type: NetworkTopology, num_nodes: int, **kwargs) -> bool:
        """Create a quantum network with specified topology."""
        try:
            self.state = NetworkState.INITIALIZING
            success = self.topology.create_topology(topology_type, num_nodes, **kwargs)
            
            if success:
                self.state = NetworkState.IDLE
                return True
            else:
                self.state = NetworkState.ERROR
                return False
                
        except Exception:
            self.state = NetworkState.ERROR
            return False
    
    def add_custom_node(self, node: NetworkNode) -> bool:
        """Add a custom node to the network."""
        return self.topology.add_node(node)
    
    def add_custom_channel(self, channel: QuantumChannel) -> bool:
        """Add a custom channel to the network."""
        return self.topology.add_channel(channel)
    
    def run_entanglement_distribution(self, source: str, target: str,
                                    strategy: str = 'direct',
                                    target_fidelity: float = 0.9) -> Optional[str]:
        """Run entanglement distribution protocol."""
        self.state = NetworkState.DISTRIBUTING
        
        try:
            pair_id = self.entanglement_manager.distribute_entanglement(
                source, target, strategy, target_fidelity
            )
            
            if pair_id:
                self.statistics['total_entanglement_pairs'] += 1
                
            self.state = NetworkState.OPERATING
            return pair_id
            
        except Exception:
            self.state = NetworkState.ERROR
            return None
    
    def run_quantum_teleportation(self, source: str, target: str,
                                 qubit_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run quantum teleportation protocol."""
        self.state = NetworkState.OPERATING
        
        try:
            result = self.teleportation_protocol.teleport_qubit(source, target, qubit_state)
            
            if result['success']:
                self.statistics['total_teleportations'] += 1
                self.statistics['successful_protocols'] += 1
            
            self.statistics['total_protocols'] += 1
            return result
            
        except Exception as e:
            self.state = NetworkState.ERROR
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def run_superdense_coding(self, source: str, target: str, 
                             classical_bits: List[int]) -> Dict[str, Any]:
        """Run quantum superdense coding protocol."""
        # Find entanglement pair
        entanglement_pair = None
        for pair in self.entanglement_manager.entanglement_pairs.values():
            if ((pair.node_a == source and pair.node_b == target) or
                (pair.node_a == target and pair.node_b == source)):
                if not pair.is_expired():
                    entanglement_pair = pair
                    break
        
        if not entanglement_pair:
            # Create entanglement
            pair_id = self.run_entanglement_distribution(source, target)
            if not pair_id:
                return {
                    'success': False,
                    'error': 'No entanglement available',
                    'timestamp': time.time()
                }
            entanglement_pair = self.entanglement_manager.entanglement_pairs[pair_id]
        
        # Simulate superdense coding
        result = self._simulate_superdense_coding(entanglement_pair, classical_bits)
        
        # Consume entanglement
        self.entanglement_manager.consume_entanglement(entanglement_pair.pair_id)
        
        return result
    
    def _simulate_superdense_coding(self, entanglement_pair: EntanglementPair,
                                   classical_bits: List[int]) -> Dict[str, Any]:
        """Simulate superdense coding protocol."""
        # Encode two classical bits in quantum state
        if len(classical_bits) != 2:
            return {
                'success': False,
                'error': 'Superdense coding requires exactly 2 classical bits',
                'timestamp': time.time()
            }
        
        # Determine Bell state based on classical bits
        bell_states = {
            (0, 0): '|Φ+⟩',  # No operation
            (0, 1): '|Ψ+⟩',  # X gate
            (1, 0): '|Φ-⟩',  # Z gate
            (1, 1): '|Ψ-⟩'   # XZ gates
        }
        
        bell_state = bell_states[tuple(classical_bits)]
        
        # Calculate protocol fidelity
        current_fidelity = entanglement_pair.calculate_current_fidelity()
        source_node = self.topology.nodes[entanglement_pair.node_a]
        target_node = self.topology.nodes[entanglement_pair.node_b]
        
        gate_fidelity = min(source_node.gate_fidelity, target_node.gate_fidelity)
        measurement_fidelity = target_node.measurement_fidelity
        
        protocol_fidelity = current_fidelity * gate_fidelity * measurement_fidelity
        
        return {
            'success': True,
            'source_node': entanglement_pair.node_a,
            'target_node': entanglement_pair.node_b,
            'classical_bits': classical_bits,
            'encoded_state': bell_state,
            'protocol_fidelity': protocol_fidelity,
            'entanglement_pair_id': entanglement_pair.pair_id,
            'timestamp': time.time()
        }
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status."""
        ent_stats = self.entanglement_manager.get_entanglement_statistics()
        tel_stats = self.teleportation_protocol.get_teleportation_statistics()
        
        return {
            'network_id': self.network_id,
            'state': self.state.value,
            'topology_type': self.topology.topology_type.value,
            'num_nodes': len(self.topology.nodes),
            'num_channels': len(self.topology.channels),
            'network_diameter': self.topology.get_network_diameter(),
            'entanglement_statistics': ent_stats,
            'teleportation_statistics': tel_stats,
            'protocol_statistics': self.statistics,
            'created_at': self.created_at,
            'uptime': time.time() - self.created_at
        }
    
    def cleanup_expired_resources(self) -> Dict[str, int]:
        """Clean up expired network resources."""
        expired_entanglements = self.entanglement_manager.cleanup_expired_entanglement()
        
        return {
            'expired_entanglements': expired_entanglements
        }
    
    def export_network_data(self) -> Dict[str, Any]:
        """Export complete network data."""
        return {
            'network_info': {
                'network_id': self.network_id,
                'created_at': self.created_at,
                'state': self.state.value,
                'topology_type': self.topology.topology_type.value
            },
            'nodes': {node_id: node.to_dict() for node_id, node in self.topology.nodes.items()},
            'channels': {ch_id: ch.to_dict() for ch_id, ch in self.topology.channels.items()},
            'entanglement_pairs': {pair_id: pair.to_dict() 
                                  for pair_id, pair in self.entanglement_manager.entanglement_pairs.items()},
            'statistics': self.statistics
        }


class QuantumNetworkVisualizer:
    """Visualizes quantum networks and protocols."""
    
    def __init__(self):
        if not MATPLOTLIB_AVAILABLE:
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Matplotlib/NetworkX not available, visualization disabled")
    
    def visualize_network(self, simulator: QuantumNetworkSimulator, 
                         show_entanglement: bool = True,
                         show_labels: bool = True) -> bool:
        """Visualize quantum network topology."""
        if not MATPLOTLIB_AVAILABLE:
            return False
        
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for node_id, node in simulator.topology.nodes.items():
                G.add_node(node_id, pos=node.position, 
                          node_type=node.node_type, 
                          load=node.current_load)
            
            # Add edges (channels)
            for channel in simulator.topology.channels.values():
                G.add_edge(channel.source_node, channel.target_node,
                          distance=channel.distance,
                          fidelity=channel.fidelity,
                          channel_type=channel.channel_type.value)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get positions
            pos = nx.get_node_attributes(G, 'pos')
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=1000, alpha=0.7, ax=ax)
            
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 width=2, alpha=0.5, ax=ax)
            
            if show_labels:
                nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
            
            # Draw entanglement pairs
            if show_entanglement:
                self._draw_entanglement_pairs(ax, simulator, pos)
            
            # Add legend and title
            ax.set_title(f'Quantum Network: {simulator.network_id}')
            ax.set_aspect('equal')
            
            # Add network statistics as text
            stats = simulator.get_network_status()
            stats_text = f"""Network Statistics:
Nodes: {stats['num_nodes']}
Channels: {stats['num_channels']}
Active Entanglements: {stats['entanglement_statistics']['active_pairs']}
Teleportations: {stats['teleportation_statistics']['total_teleportations']}
Diameter: {stats['network_diameter']:.2f} km"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            return True
            
        except Exception as e:
            logging.error(f"Network visualization error: {e}")
            return False
    
    def _draw_entanglement_pairs(self, ax, simulator: QuantumNetworkSimulator, pos: Dict):
        """Draw entanglement pairs on the network."""
        for pair in simulator.entanglement_manager.entanglement_pairs.values():
            if not pair.is_expired():
                node_a_pos = pos[pair.node_a]
                node_b_pos = pos[pair.node_b]
                
                # Draw entanglement as colored line
                fidelity = pair.calculate_current_fidelity()
                color = plt.cm.Reds(fidelity)
                
                ax.plot([node_a_pos[0], node_b_pos[0]], 
                       [node_a_pos[1], node_b_pos[1]], 
                       color=color, linewidth=4, alpha=0.7, linestyle='--')
    
    def plot_protocol_performance(self, simulator: QuantumNetworkSimulator) -> bool:
        """Plot protocol performance metrics."""
        if not MATPLOTLIB_AVAILABLE:
            return False
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Entanglement statistics
            self._plot_entanglement_stats(axes[0, 0], simulator)
            
            # Plot 2: Teleportation performance
            self._plot_teleportation_stats(axes[0, 1], simulator)
            
            # Plot 3: Network topology metrics
            self._plot_topology_metrics(axes[1, 0], simulator)
            
            # Plot 4: Protocol timeline
            self._plot_protocol_timeline(axes[1, 1], simulator)
            
            plt.tight_layout()
            plt.show()
            
            return True
            
        except Exception as e:
            logging.error(f"Performance visualization error: {e}")
            return False
    
    def _plot_entanglement_stats(self, ax, simulator: QuantumNetworkSimulator):
        """Plot entanglement statistics."""
        stats = simulator.entanglement_manager.get_entanglement_statistics()
        
        labels = ['Active', 'Expired']
        sizes = [stats['active_pairs'], stats['expired_pairs']]
        colors = ['lightgreen', 'lightcoral']
        
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        
        ax.set_title('Entanglement Pairs Status')
    
    def _plot_teleportation_stats(self, ax, simulator: QuantumNetworkSimulator):
        """Plot teleportation statistics."""
        stats = simulator.teleportation_protocol.get_teleportation_statistics()
        
        if stats['total_teleportations'] > 0:
            metrics = ['Success Rate', 'Avg Fidelity', 'Avg Delay (s)']
            values = [stats['success_rate'], stats['average_fidelity'], 
                     stats['average_delay']]
            
            bars = ax.bar(metrics, values, color=['green', 'blue', 'orange'])
            ax.set_ylabel('Value')
            ax.set_title('Teleportation Performance')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No teleportation data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Teleportation Performance')
    
    def _plot_topology_metrics(self, ax, simulator: QuantumNetworkSimulator):
        """Plot network topology metrics."""
        # Calculate connectivity metrics
        node_degrees = []
        for node_id in simulator.topology.nodes:
            degree = len(simulator.topology.adjacency_matrix[node_id])
            node_degrees.append(degree)
        
        if node_degrees:
            ax.hist(node_degrees, bins=max(1, len(set(node_degrees))), 
                   alpha=0.7, color='skyblue')
            ax.set_xlabel('Node Degree')
            ax.set_ylabel('Frequency')
            ax.set_title('Node Connectivity Distribution')
        else:
            ax.text(0.5, 0.5, 'No topology data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Node Connectivity Distribution')
    
    def _plot_protocol_timeline(self, ax, simulator: QuantumNetworkSimulator):
        """Plot protocol execution timeline."""
        # For now, show simple statistics
        protocols = ['Entanglement', 'Teleportation', 'Total']
        counts = [
            simulator.statistics['total_entanglement_pairs'],
            simulator.statistics['total_teleportations'],
            simulator.statistics['total_protocols']
        ]
        
        bars = ax.bar(protocols, counts, color=['purple', 'green', 'blue'])
        ax.set_ylabel('Count')
        ax.set_title('Protocol Usage')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')


# Global simulator instance
_quantum_network_simulator: Optional[QuantumNetworkSimulator] = None


def get_quantum_network_simulator() -> QuantumNetworkSimulator:
    """Get global quantum network simulator instance."""
    global _quantum_network_simulator
    if _quantum_network_simulator is None:
        _quantum_network_simulator = QuantumNetworkSimulator()
    return _quantum_network_simulator


def create_quantum_network(topology_type: NetworkTopology, num_nodes: int, **kwargs) -> bool:
    """Convenience function to create a quantum network."""
    simulator = get_quantum_network_simulator()
    return simulator.create_network(topology_type, num_nodes, **kwargs)


def distribute_entanglement(source: str, target: str, strategy: str = 'direct') -> Optional[str]:
    """Convenience function to distribute entanglement."""
    simulator = get_quantum_network_simulator()
    return simulator.run_entanglement_distribution(source, target, strategy)


def teleport_qubit(source: str, target: str, qubit_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to perform quantum teleportation."""
    simulator = get_quantum_network_simulator()
    return simulator.run_quantum_teleportation(source, target, qubit_state)


def visualize_quantum_network(show_entanglement: bool = True) -> bool:
    """Convenience function to visualize quantum network."""
    simulator = get_quantum_network_simulator()
    visualizer = QuantumNetworkVisualizer()
    return visualizer.visualize_network(simulator, show_entanglement)


# CLI interface
def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Quantum Networking")
    parser.add_argument("--topology", choices=["star", "mesh", "ring", "linear", "tree", "grid"],
                       default="star", help="Network topology")
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes")
    parser.add_argument("--visualize", action="store_true", help="Visualize network")
    parser.add_argument("--demo", action="store_true", help="Run demo protocols")
    
    args = parser.parse_args()
    
    # Create network
    topology_map = {
        'star': NetworkTopology.STAR,
        'mesh': NetworkTopology.MESH,
        'ring': NetworkTopology.RING,
        'linear': NetworkTopology.LINEAR,
        'tree': NetworkTopology.TREE,
        'grid': NetworkTopology.GRID
    }
    
    topology = topology_map.get(args.topology, NetworkTopology.STAR)
    
    success = create_quantum_network(topology, args.nodes)
    if not success:
        print("Failed to create quantum network")
        return 1
    
    print(f"Created quantum network with {args.nodes} nodes using {args.topology} topology")
    
    if args.demo:
        # Run demo protocols
        print("\nRunning demo protocols...")
        
        # Distribute entanglement
        pair_id = distribute_entanglement("node_0", "node_1")
        if pair_id:
            print(f"Successfully distributed entanglement: {pair_id}")
        else:
            print("Failed to distribute entanglement")
        
        # Perform teleportation
        result = teleport_qubit("node_0", "node_1")
        if result['success']:
            print(f"Quantum teleportation successful with fidelity: {result['teleportation_fidelity']:.3f}")
        else:
            print(f"Quantum teleportation failed: {result['error']}")
        
        # Show network status
        simulator = get_quantum_network_simulator()
        status = simulator.get_network_status()
        print(f"\nNetwork Status:")
        print(f"  State: {status['state']}")
        print(f"  Active entanglements: {status['entanglement_statistics']['active_pairs']}")
        print(f"  Total teleportations: {status['teleportation_statistics']['total_teleportations']}")
    
    if args.visualize:
        print("\nVisualizing network...")
        visualize_quantum_network()
    
    return 0


if __name__ == "__main__":
    exit(main())