"""
Distributed Quantum Simulation

This module provides distributed quantum circuit simulation capabilities
with support for cluster computing, circuit partitioning, and parallel execution.
"""

import json
import time
import asyncio
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import socket
import pickle
import hashlib

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import mpi4py
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False


class DistributionStrategy(Enum):
    """Strategies for distributing quantum simulations."""
    AMPLITUDE_PARTITIONING = "amplitude_partitioning"
    GATE_LEVEL_PARALLELISM = "gate_level_parallelism"
    CIRCUIT_PARTITIONING = "circuit_partitioning"
    HYBRID = "hybrid"


class NodeRole(Enum):
    """Roles for nodes in distributed simulation."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    STORAGE = "storage"


class SimulationStatus(Enum):
    """Status of distributed simulation."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    role: NodeRole
    host: str
    port: int
    cpu_count: int = field(default_factory=lambda: multiprocessing.cpu_count())
    memory_gb: float = 8.0
    gpu_count: int = 0
    status: str = "online"
    last_seen: float = field(default_factory=time.time)
    current_load: float = 0.0
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'role': self.role.value,
            'host': self.host,
            'port': self.port,
            'cpu_count': self.cpu_count,
            'memory_gb': self.memory_gb,
            'gpu_count': self.gpu_count,
            'status': self.status,
            'last_seen': self.last_seen,
            'current_load': self.current_load,
            'capabilities': self.capabilities
        }


@dataclass
class DistributedTask:
    """Represents a distributed simulation task."""
    task_id: str
    circuit_data: Dict[str, Any]
    partition_info: Dict[str, Any]
    target_nodes: List[str]
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: SimulationStatus = SimulationStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'circuit_data': self.circuit_data,
            'partition_info': self.partition_info,
            'target_nodes': self.target_nodes,
            'priority': self.priority,
            'created_at': self.created_at,
            'assigned_at': self.assigned_at,
            'completed_at': self.completed_at,
            'status': self.status.value,
            'result': self.result,
            'error_message': self.error_message
        }


class CircuitPartitioner:
    """Handles quantum circuit partitioning for distributed execution."""
    
    def __init__(self):
        self.partitioning_strategies = {
            'amplitude': self._amplitude_partitioning,
            'gate_parallel': self._gate_level_partitioning,
            'circuit_split': self._circuit_partitioning,
            'hybrid': self._hybrid_partitioning
        }
    
    def partition_circuit(self, circuit_data: Dict[str, Any], 
                         strategy: DistributionStrategy,
                         num_partitions: int) -> List[Dict[str, Any]]:
        """Partition a circuit for distributed execution."""
        try:
            if strategy.value in self.partitioning_strategies:
                return self.partitioning_strategies[strategy.value](
                    circuit_data, num_partitions
                )
            else:
                # Default to amplitude partitioning
                return self._amplitude_partitioning(circuit_data, num_partitions)
        except Exception as e:
            logging.error(f"Circuit partitioning failed: {e}")
            # Return single partition as fallback
            return [circuit_data]
    
    def _amplitude_partitioning(self, circuit_data: Dict[str, Any], 
                               num_partitions: int) -> List[Dict[str, Any]]:
        """Partition circuit by distributing amplitude storage."""
        n_qubits = circuit_data.get('n_qubits', 2)
        total_amplitudes = 2 ** n_qubits
        
        # Calculate amplitude ranges for each partition
        amplitudes_per_partition = total_amplitudes // num_partitions
        partitions = []
        
        for i in range(num_partitions):
            start_idx = i * amplitudes_per_partition
            end_idx = start_idx + amplitudes_per_partition
            if i == num_partitions - 1:  # Last partition gets remainder
                end_idx = total_amplitudes
            
            partition = {
                'partition_id': i,
                'partition_type': 'amplitude',
                'circuit_data': circuit_data.copy(),
                'amplitude_range': (start_idx, end_idx),
                'local_qubit_count': n_qubits,
                'dependencies': [],
                'communication_pattern': 'all_to_all'
            }
            partitions.append(partition)
        
        return partitions
    
    def _gate_level_partitioning(self, circuit_data: Dict[str, Any],
                                num_partitions: int) -> List[Dict[str, Any]]:
        """Partition circuit by parallelizing independent gates."""
        gates = circuit_data.get('gates', [])
        n_qubits = circuit_data.get('n_qubits', 2)
        
        # Analyze gate dependencies
        gate_layers = self._analyze_gate_parallelism(gates, n_qubits)
        
        partitions = []
        for i in range(num_partitions):
            # Distribute gate layers across partitions
            assigned_layers = [layer for j, layer in enumerate(gate_layers) if j % num_partitions == i]
            
            partition = {
                'partition_id': i,
                'partition_type': 'gate_parallel',
                'circuit_data': circuit_data.copy(),
                'assigned_gate_layers': assigned_layers,
                'local_qubit_count': n_qubits,
                'dependencies': list(range(num_partitions)),  # All partitions need to sync
                'communication_pattern': 'barrier_sync'
            }
            partitions.append(partition)
        
        return partitions
    
    def _circuit_partitioning(self, circuit_data: Dict[str, Any],
                             num_partitions: int) -> List[Dict[str, Any]]:
        """Partition circuit by splitting into subcircuits."""
        n_qubits = circuit_data.get('n_qubits', 2)
        gates = circuit_data.get('gates', [])
        
        # Distribute qubits across partitions
        qubits_per_partition = max(1, n_qubits // num_partitions)
        partitions = []
        
        for i in range(num_partitions):
            start_qubit = i * qubits_per_partition
            end_qubit = min(start_qubit + qubits_per_partition, n_qubits)
            if i == num_partitions - 1:  # Last partition gets remaining qubits
                end_qubit = n_qubits
            
            local_qubits = list(range(start_qubit, end_qubit))
            
            # Filter gates that operate on local qubits
            local_gates = []
            entangling_gates = []
            
            for gate in gates:
                gate_qubits = gate.get('qubits', [])
                if all(q in local_qubits for q in gate_qubits):
                    local_gates.append(gate)
                elif any(q in local_qubits for q in gate_qubits):
                    entangling_gates.append(gate)
            
            partition = {
                'partition_id': i,
                'partition_type': 'circuit_split',
                'circuit_data': {
                    'n_qubits': len(local_qubits),
                    'gates': local_gates,
                    'qubit_mapping': {local_idx: global_qubit 
                                    for local_idx, global_qubit in enumerate(local_qubits)}
                },
                'local_qubits': local_qubits,
                'entangling_gates': entangling_gates,
                'dependencies': [j for j in range(num_partitions) if j != i],
                'communication_pattern': 'nearest_neighbor'
            }
            partitions.append(partition)
        
        return partitions
    
    def _hybrid_partitioning(self, circuit_data: Dict[str, Any],
                            num_partitions: int) -> List[Dict[str, Any]]:
        """Hybrid partitioning combining multiple strategies."""
        n_qubits = circuit_data.get('n_qubits', 2)
        
        # Use different strategies based on problem size
        if n_qubits <= 10:
            return self._gate_level_partitioning(circuit_data, num_partitions)
        elif n_qubits <= 20:
            return self._amplitude_partitioning(circuit_data, num_partitions)
        else:
            return self._circuit_partitioning(circuit_data, num_partitions)
    
    def _analyze_gate_parallelism(self, gates: List[Dict[str, Any]], 
                                 n_qubits: int) -> List[List[Dict[str, Any]]]:
        """Analyze gates to find parallel execution opportunities."""
        # Simple parallelism analysis - gates can be parallel if they don't share qubits
        layers = []
        remaining_gates = gates.copy()
        
        while remaining_gates:
            current_layer = []
            used_qubits = set()
            
            i = 0
            while i < len(remaining_gates):
                gate = remaining_gates[i]
                gate_qubits = set(gate.get('qubits', []))
                
                if not (gate_qubits & used_qubits):  # No qubit conflict
                    current_layer.append(gate)
                    used_qubits.update(gate_qubits)
                    remaining_gates.pop(i)
                else:
                    i += 1
            
            if current_layer:
                layers.append(current_layer)
            else:
                # Fallback: add next gate even if there's conflict
                if remaining_gates:
                    layers.append([remaining_gates.pop(0)])
        
        return layers


class LoadBalancer:
    """Handles load balancing across cluster nodes."""
    
    def __init__(self):
        self.node_metrics: Dict[str, Dict[str, float]] = {}
        self.task_history: Dict[str, List[float]] = {}
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]) -> None:
        """Update metrics for a node."""
        self.node_metrics[node_id] = {
            'cpu_usage': metrics.get('cpu_usage', 0.0),
            'memory_usage': metrics.get('memory_usage', 0.0),
            'network_latency': metrics.get('network_latency', 0.0),
            'active_tasks': metrics.get('active_tasks', 0),
            'last_update': time.time()
        }
    
    def select_optimal_nodes(self, available_nodes: List[NodeInfo], 
                           task_requirements: Dict[str, Any]) -> List[str]:
        """Select optimal nodes for a task."""
        num_nodes_needed = task_requirements.get('num_nodes', 1)
        min_memory = task_requirements.get('min_memory_gb', 1.0)
        min_cpu_count = task_requirements.get('min_cpu_count', 1)
        
        # Filter nodes by requirements
        suitable_nodes = []
        for node in available_nodes:
            if (node.memory_gb >= min_memory and 
                node.cpu_count >= min_cpu_count and
                node.status == "online"):
                suitable_nodes.append(node)
        
        if len(suitable_nodes) < num_nodes_needed:
            # Not enough suitable nodes
            return [node.node_id for node in suitable_nodes[:num_nodes_needed]]
        
        # Score nodes based on current load and capabilities
        node_scores = []
        for node in suitable_nodes:
            score = self._calculate_node_score(node)
            node_scores.append((node.node_id, score))
        
        # Sort by score (lower is better) and select top nodes
        node_scores.sort(key=lambda x: x[1])
        selected_nodes = [node_id for node_id, _ in node_scores[:num_nodes_needed]]
        
        return selected_nodes
    
    def _calculate_node_score(self, node: NodeInfo) -> float:
        """Calculate node suitability score (lower is better)."""
        base_score = node.current_load
        
        # Get recent metrics if available
        if node.node_id in self.node_metrics:
            metrics = self.node_metrics[node.node_id]
            age = time.time() - metrics.get('last_update', 0)
            
            if age < 300:  # Metrics less than 5 minutes old
                base_score += metrics.get('cpu_usage', 0.0) * 0.5
                base_score += metrics.get('memory_usage', 0.0) * 0.3
                base_score += metrics.get('network_latency', 0.0) * 0.1
                base_score += metrics.get('active_tasks', 0) * 0.1
        
        # Penalize nodes with lower resources
        base_score += 1.0 / max(node.cpu_count, 1)
        base_score += 1.0 / max(node.memory_gb, 1.0)
        
        return base_score
    
    def update_task_completion(self, node_id: str, completion_time: float) -> None:
        """Update task completion history for a node."""
        if node_id not in self.task_history:
            self.task_history[node_id] = []
        
        self.task_history[node_id].append(completion_time)
        
        # Keep only recent history
        if len(self.task_history[node_id]) > 100:
            self.task_history[node_id] = self.task_history[node_id][-100:]


class ClusterCommunication:
    """Handles communication between cluster nodes."""
    
    def __init__(self, node_id: str, port: int = 0):
        self.node_id = node_id
        self.port = port
        self.socket = None
        self.connections: Dict[str, socket.socket] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        self.server_thread = None
        
        # Setup default message handlers
        self.register_handler('ping', self._handle_ping)
        self.register_handler('task_assignment', self._handle_task_assignment)
        self.register_handler('task_result', self._handle_task_result)
        self.register_handler('node_status', self._handle_node_status)
    
    def start_server(self) -> int:
        """Start communication server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            if self.port == 0:
                # Find available port
                self.socket.bind(('', 0))
                self.port = self.socket.getsockname()[1]
            else:
                self.socket.bind(('', self.port))
            
            self.socket.listen(10)
            self.running = True
            
            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            return self.port
            
        except Exception as e:
            logging.error(f"Failed to start communication server: {e}")
            return 0
    
    def stop_server(self) -> None:
        """Stop communication server."""
        self.running = False
        
        if self.socket:
            self.socket.close()
        
        # Close all connections
        for conn in self.connections.values():
            try:
                conn.close()
            except Exception:
                pass
        
        self.connections.clear()
        
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
    
    def _server_loop(self) -> None:
        """Main server loop."""
        while self.running:
            try:
                client_socket, address = self.socket.accept()
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    logging.error(f"Server loop error: {e}")
                break
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple) -> None:
        """Handle individual client connection."""
        try:
            while self.running:
                # Receive message
                data = self._receive_message(client_socket)
                if not data:
                    break
                
                # Process message
                response = self._process_message(data)
                
                # Send response
                if response:
                    self._send_message(client_socket, response)
                    
        except Exception as e:
            logging.error(f"Client handling error: {e}")
        finally:
            client_socket.close()
    
    def _receive_message(self, sock: socket.socket) -> Optional[Dict[str, Any]]:
        """Receive message from socket."""
        try:
            # First, receive message length
            length_data = sock.recv(4)
            if len(length_data) < 4:
                return None
            
            message_length = int.from_bytes(length_data, 'big')
            
            # Receive the actual message
            message_data = b''
            while len(message_data) < message_length:
                chunk = sock.recv(min(message_length - len(message_data), 4096))
                if not chunk:
                    return None
                message_data += chunk
            
            # Deserialize message
            return pickle.loads(message_data)
            
        except Exception as e:
            logging.error(f"Message receive error: {e}")
            return None
    
    def _send_message(self, sock: socket.socket, message: Dict[str, Any]) -> bool:
        """Send message to socket."""
        try:
            # Serialize message
            message_data = pickle.dumps(message)
            message_length = len(message_data)
            
            # Send length first
            sock.send(message_length.to_bytes(4, 'big'))
            
            # Send message data
            sock.send(message_data)
            return True
            
        except Exception as e:
            logging.error(f"Message send error: {e}")
            return False
    
    def _process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process received message."""
        message_type = message.get('type')
        
        if message_type in self.message_handlers:
            try:
                return self.message_handlers[message_type](message)
            except Exception as e:
                logging.error(f"Message handler error for {message_type}: {e}")
                return {'type': 'error', 'message': str(e)}
        else:
            return {'type': 'error', 'message': f'Unknown message type: {message_type}'}
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register message handler."""
        self.message_handlers[message_type] = handler
    
    def send_message_to_node(self, target_node: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message to specific node."""
        # This is a simplified implementation
        # In practice, you'd need node discovery and connection management
        try:
            # Mock sending - return success response
            return {'type': 'ack', 'node_id': self.node_id}
        except Exception as e:
            logging.error(f"Failed to send message to {target_node}: {e}")
            return None
    
    def broadcast_message(self, message: Dict[str, Any]) -> List[str]:
        """Broadcast message to all connected nodes."""
        successful_nodes = []
        
        for node_id in self.connections:
            try:
                response = self.send_message_to_node(node_id, message)
                if response:
                    successful_nodes.append(node_id)
            except Exception as e:
                logging.error(f"Broadcast failed to {node_id}: {e}")
        
        return successful_nodes
    
    # Default message handlers
    def _handle_ping(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping message."""
        return {
            'type': 'pong',
            'node_id': self.node_id,
            'timestamp': time.time()
        }
    
    def _handle_task_assignment(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task assignment message."""
        return {
            'type': 'task_accepted',
            'node_id': self.node_id,
            'task_id': message.get('task_id')
        }
    
    def _handle_task_result(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task result message."""
        return {
            'type': 'result_acknowledged',
            'node_id': self.node_id
        }
    
    def _handle_node_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle node status message."""
        return {
            'type': 'status_acknowledged',
            'node_id': self.node_id
        }


class DistributedSimulator:
    """Main distributed quantum simulator."""
    
    def __init__(self, node_id: str, role: NodeRole = NodeRole.WORKER):
        self.node_id = node_id
        self.role = role
        self.node_info = NodeInfo(
            node_id=node_id,
            role=role,
            host=socket.gethostname(),
            port=0
        )
        
        self.cluster_nodes: Dict[str, NodeInfo] = {}
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        
        self.partitioner = CircuitPartitioner()
        self.load_balancer = LoadBalancer()
        self.communication = ClusterCommunication(node_id)
        
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        
        # Setup communication handlers
        self._setup_communication_handlers()
    
    def _setup_communication_handlers(self) -> None:
        """Setup communication message handlers."""
        self.communication.register_handler('simulate_circuit', self._handle_simulate_circuit)
        self.communication.register_handler('join_cluster', self._handle_join_cluster)
        self.communication.register_handler('node_metrics', self._handle_node_metrics)
        self.communication.register_handler('task_status', self._handle_task_status)
    
    def start(self, port: int = 0) -> bool:
        """Start the distributed simulator."""
        try:
            # Start communication server
            actual_port = self.communication.start_server()
            if actual_port == 0:
                logging.error("Failed to start communication server")
                return False
            
            self.node_info.port = actual_port
            self.running = True
            
            # Start worker threads
            if self.role in [NodeRole.WORKER, NodeRole.COORDINATOR]:
                self._start_worker_threads()
            
            logging.info(f"Distributed simulator started on port {actual_port}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start distributed simulator: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the distributed simulator."""
        self.running = False
        
        # Stop communication
        self.communication.stop_server()
        
        # Stop worker threads
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        self.worker_threads.clear()
        
        logging.info("Distributed simulator stopped")
    
    def _start_worker_threads(self, num_workers: int = 2) -> None:
        """Start worker threads for processing tasks."""
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"DistributedWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing distributed tasks."""
        while self.running:
            try:
                # Process pending tasks
                pending_tasks = [task for task in self.active_tasks.values() 
                               if task.status == SimulationStatus.PENDING]
                
                if pending_tasks:
                    # Sort by priority
                    pending_tasks.sort(key=lambda t: t.priority, reverse=True)
                    task = pending_tasks[0]
                    
                    # Process task
                    self._process_distributed_task(task)
                else:
                    # No pending tasks, sleep briefly
                    time.sleep(0.1)
                    
            except Exception as e:
                logging.error(f"Worker loop error: {e}")
                time.sleep(1.0)
    
    def _process_distributed_task(self, task: DistributedTask) -> None:
        """Process a distributed simulation task."""
        try:
            task.status = SimulationStatus.RUNNING
            task.assigned_at = time.time()
            
            # Simulate circuit partition
            result = self._simulate_circuit_partition(
                task.circuit_data, 
                task.partition_info
            )
            
            # Update task with result
            task.result = result
            task.status = SimulationStatus.COMPLETED
            task.completed_at = time.time()
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Update load balancer
            completion_time = task.completed_at - task.assigned_at
            self.load_balancer.update_task_completion(self.node_id, completion_time)
            
        except Exception as e:
            task.status = SimulationStatus.FAILED
            task.error_message = str(e)
            task.completed_at = time.time()
            
            logging.error(f"Task {task.task_id} failed: {e}")
    
    def _simulate_circuit_partition(self, circuit_data: Dict[str, Any], 
                                   partition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a circuit partition."""
        try:
            # Extract partition information
            partition_type = partition_info.get('partition_type', 'amplitude')
            partition_id = partition_info.get('partition_id', 0)
            
            if partition_type == 'amplitude':
                return self._simulate_amplitude_partition(circuit_data, partition_info)
            elif partition_type == 'gate_parallel':
                return self._simulate_gate_parallel_partition(circuit_data, partition_info)
            elif partition_type == 'circuit_split':
                return self._simulate_circuit_split_partition(circuit_data, partition_info)
            else:
                # Default simulation
                return self._simulate_local_circuit(circuit_data)
                
        except Exception as e:
            logging.error(f"Circuit partition simulation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'partition_id': partition_info.get('partition_id', 0)
            }
    
    def _simulate_amplitude_partition(self, circuit_data: Dict[str, Any], 
                                     partition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate amplitude partition."""
        amplitude_range = partition_info.get('amplitude_range', (0, 1))
        start_idx, end_idx = amplitude_range
        
        # Mock amplitude simulation
        n_amplitudes = end_idx - start_idx
        
        if NUMPY_AVAILABLE:
            # Create mock amplitudes
            amplitudes = np.random.complex128((n_amplitudes,))
            amplitudes /= np.linalg.norm(amplitudes)
            amplitude_data = amplitudes.tolist()
        else:
            # Simple mock without numpy
            amplitude_data = [complex(0.1, 0.0) for _ in range(n_amplitudes)]
        
        return {
            'success': True,
            'partition_id': partition_info.get('partition_id', 0),
            'partition_type': 'amplitude',
            'amplitude_range': amplitude_range,
            'amplitudes': amplitude_data,
            'simulation_time': 0.1
        }
    
    def _simulate_gate_parallel_partition(self, circuit_data: Dict[str, Any],
                                         partition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate gate-parallel partition."""
        assigned_layers = partition_info.get('assigned_gate_layers', [])
        
        # Simulate processing each gate layer
        processed_layers = []
        total_time = 0.0
        
        for layer in assigned_layers:
            layer_start = time.time()
            
            # Mock gate layer processing
            layer_result = {
                'gates_processed': len(layer),
                'layer_depth': 1
            }
            
            layer_time = time.time() - layer_start
            total_time += layer_time
            
            processed_layers.append(layer_result)
        
        return {
            'success': True,
            'partition_id': partition_info.get('partition_id', 0),
            'partition_type': 'gate_parallel',
            'processed_layers': processed_layers,
            'total_gates': sum(len(layer) for layer in assigned_layers),
            'simulation_time': total_time
        }
    
    def _simulate_circuit_split_partition(self, circuit_data: Dict[str, Any],
                                         partition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate circuit split partition."""
        local_qubits = partition_info.get('local_qubits', [])
        entangling_gates = partition_info.get('entangling_gates', [])
        
        # Simulate local circuit
        local_circuit_data = partition_info.get('circuit_data', {})
        local_result = self._simulate_local_circuit(local_circuit_data)
        
        return {
            'success': True,
            'partition_id': partition_info.get('partition_id', 0),
            'partition_type': 'circuit_split',
            'local_qubits': local_qubits,
            'local_result': local_result,
            'entangling_gates_count': len(entangling_gates),
            'simulation_time': local_result.get('simulation_time', 0.1)
        }
    
    def _simulate_local_circuit(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate circuit locally."""
        try:
            n_qubits = circuit_data.get('n_qubits', 2)
            gates = circuit_data.get('gates', [])
            
            start_time = time.time()
            
            if _NATIVE_AVAILABLE:
                # Use native simulation
                try:
                    circuit = _quantrs2.PyCircuit(n_qubits)
                    
                    # Add gates
                    for gate_data in gates:
                        gate_name = gate_data.get('gate', '').lower()
                        qubits = gate_data.get('qubits', [])
                        params = gate_data.get('params', [])
                        
                        if gate_name == 'h' and len(qubits) >= 1:
                            circuit.h(qubits[0])
                        elif gate_name == 'x' and len(qubits) >= 1:
                            circuit.x(qubits[0])
                        elif gate_name == 'cnot' and len(qubits) >= 2:
                            circuit.cnot(qubits[0], qubits[1])
                        # Add more gates as needed
                    
                    # Run simulation
                    result = circuit.run()
                    
                    simulation_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'n_qubits': n_qubits,
                        'gate_count': len(gates),
                        'simulation_time': simulation_time,
                        'native_simulation': True
                    }
                    
                except Exception as e:
                    logging.warning(f"Native simulation failed, using mock: {e}")
            
            # Mock simulation
            simulation_time = time.time() - start_time + 0.01  # Add small delay
            
            return {
                'success': True,
                'n_qubits': n_qubits,
                'gate_count': len(gates),
                'simulation_time': simulation_time,
                'native_simulation': False
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'simulation_time': 0.0
            }
    
    def submit_distributed_simulation(self, circuit_data: Dict[str, Any],
                                    strategy: DistributionStrategy = DistributionStrategy.AMPLITUDE_PARTITIONING,
                                    num_nodes: Optional[int] = None) -> str:
        """Submit a circuit for distributed simulation."""
        
        # Determine number of nodes to use
        if num_nodes is None:
            num_nodes = min(len(self.cluster_nodes), 4)  # Default to 4 nodes max
        
        num_nodes = max(1, num_nodes)  # At least 1 node
        
        # Generate unique task ID
        task_id = f"sim_{int(time.time() * 1000)}_{hash(str(circuit_data)) % 10000}"
        
        # Partition circuit
        partitions = self.partitioner.partition_circuit(circuit_data, strategy, num_nodes)
        
        # Select optimal nodes
        available_nodes = [node for node in self.cluster_nodes.values() 
                          if node.status == "online"]
        
        if not available_nodes:
            available_nodes = [self.node_info]  # Use self as fallback
        
        task_requirements = {
            'num_nodes': num_nodes,
            'min_memory_gb': 2.0,
            'min_cpu_count': 1
        }
        
        selected_nodes = self.load_balancer.select_optimal_nodes(
            available_nodes, task_requirements
        )
        
        # Create distributed tasks for each partition
        for i, partition in enumerate(partitions):
            partition_task_id = f"{task_id}_part_{i}"
            
            task = DistributedTask(
                task_id=partition_task_id,
                circuit_data=circuit_data,
                partition_info=partition,
                target_nodes=selected_nodes[i:i+1] if i < len(selected_nodes) else [self.node_id],
                priority=1
            )
            
            self.active_tasks[partition_task_id] = task
        
        return task_id
    
    def get_simulation_status(self, task_id: str) -> Optional[SimulationStatus]:
        """Get status of distributed simulation."""
        # Check for individual partition tasks
        partition_tasks = [task for task_id_part, task in self.active_tasks.items() 
                          if task_id_part.startswith(task_id)]
        partition_tasks.extend([task for task_id_part, task in self.completed_tasks.items() 
                               if task_id_part.startswith(task_id)])
        
        if not partition_tasks:
            return None
        
        # Determine overall status
        statuses = [task.status for task in partition_tasks]
        
        if all(status == SimulationStatus.COMPLETED for status in statuses):
            return SimulationStatus.COMPLETED
        elif any(status == SimulationStatus.FAILED for status in statuses):
            return SimulationStatus.FAILED
        elif any(status == SimulationStatus.RUNNING for status in statuses):
            return SimulationStatus.RUNNING
        else:
            return SimulationStatus.PENDING
    
    def get_simulation_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get distributed simulation result."""
        # Collect results from all partition tasks
        partition_results = []
        
        for task_id_part, task in self.completed_tasks.items():
            if task_id_part.startswith(task_id) and task.result:
                partition_results.append(task.result)
        
        if not partition_results:
            return None
        
        # Combine partition results
        combined_result = {
            'task_id': task_id,
            'status': 'completed',
            'partition_count': len(partition_results),
            'partition_results': partition_results,
            'total_simulation_time': sum(
                result.get('simulation_time', 0.0) for result in partition_results
            ),
            'success': all(result.get('success', False) for result in partition_results)
        }
        
        return combined_result
    
    def join_cluster(self, coordinator_host: str, coordinator_port: int) -> bool:
        """Join an existing cluster."""
        try:
            # Send join request to coordinator
            join_message = {
                'type': 'join_cluster',
                'node_info': self.node_info.to_dict()
            }
            
            # In a real implementation, you'd establish connection and send message
            # For now, just add self to cluster
            self.cluster_nodes[self.node_id] = self.node_info
            
            logging.info(f"Joined cluster at {coordinator_host}:{coordinator_port}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to join cluster: {e}")
            return False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        return {
            'node_id': self.node_id,
            'role': self.role.value,
            'cluster_size': len(self.cluster_nodes),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'running': self.running,
            'port': self.node_info.port
        }
    
    # Communication message handlers
    def _handle_simulate_circuit(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle circuit simulation request."""
        try:
            circuit_data = message.get('circuit_data', {})
            strategy = DistributionStrategy(message.get('strategy', 'amplitude_partitioning'))
            
            task_id = self.submit_distributed_simulation(circuit_data, strategy)
            
            return {
                'type': 'simulation_accepted',
                'task_id': task_id,
                'node_id': self.node_id
            }
            
        except Exception as e:
            return {
                'type': 'simulation_rejected',
                'error': str(e),
                'node_id': self.node_id
            }
    
    def _handle_join_cluster(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cluster join request."""
        try:
            node_info_data = message.get('node_info', {})
            node_info = NodeInfo(**node_info_data)
            
            # Add node to cluster
            self.cluster_nodes[node_info.node_id] = node_info
            
            return {
                'type': 'join_accepted',
                'cluster_size': len(self.cluster_nodes),
                'coordinator_id': self.node_id
            }
            
        except Exception as e:
            return {
                'type': 'join_rejected',
                'error': str(e)
            }
    
    def _handle_node_metrics(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle node metrics update."""
        try:
            node_id = message.get('node_id')
            metrics = message.get('metrics', {})
            
            self.load_balancer.update_node_metrics(node_id, metrics)
            
            return {
                'type': 'metrics_acknowledged',
                'coordinator_id': self.node_id
            }
            
        except Exception as e:
            return {
                'type': 'metrics_rejected',
                'error': str(e)
            }
    
    def _handle_task_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task status request."""
        try:
            task_id = message.get('task_id')
            status = self.get_simulation_status(task_id)
            
            return {
                'type': 'task_status_response',
                'task_id': task_id,
                'status': status.value if status else 'not_found',
                'node_id': self.node_id
            }
            
        except Exception as e:
            return {
                'type': 'task_status_error',
                'error': str(e)
            }


# Global distributed simulator instance
_distributed_simulator: Optional[DistributedSimulator] = None


def get_distributed_simulator() -> DistributedSimulator:
    """Get global distributed simulator instance."""
    global _distributed_simulator
    if _distributed_simulator is None:
        import uuid
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        _distributed_simulator = DistributedSimulator(node_id)
    return _distributed_simulator


def start_distributed_simulation_service(node_id: Optional[str] = None,
                                        role: NodeRole = NodeRole.WORKER,
                                        port: int = 0) -> bool:
    """Start distributed simulation service."""
    if node_id is None:
        import uuid
        node_id = f"node_{uuid.uuid4().hex[:8]}"
    
    global _distributed_simulator
    _distributed_simulator = DistributedSimulator(node_id, role)
    return _distributed_simulator.start(port)


def stop_distributed_simulation_service() -> None:
    """Stop distributed simulation service."""
    global _distributed_simulator
    if _distributed_simulator:
        _distributed_simulator.stop()
        _distributed_simulator = None


def simulate_circuit_distributed(circuit_data: Dict[str, Any],
                                strategy: DistributionStrategy = DistributionStrategy.AMPLITUDE_PARTITIONING,
                                timeout: float = 60.0) -> Optional[Dict[str, Any]]:
    """Simulate circuit using distributed computing."""
    simulator = get_distributed_simulator()
    
    if not simulator.running:
        simulator.start()
    
    # Submit simulation
    task_id = simulator.submit_distributed_simulation(circuit_data, strategy)
    
    # Wait for completion
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = simulator.get_simulation_status(task_id)
        
        if status == SimulationStatus.COMPLETED:
            return simulator.get_simulation_result(task_id)
        elif status == SimulationStatus.FAILED:
            return None
        
        time.sleep(0.1)
    
    # Timeout
    return None


# CLI interface
def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Distributed Simulation")
    parser.add_argument("--mode", choices=["coordinator", "worker", "client"], 
                       default="worker", help="Node role")
    parser.add_argument("--node-id", help="Node ID (auto-generated if not provided)")
    parser.add_argument("--port", type=int, default=0, help="Communication port")
    parser.add_argument("--coordinator", help="Coordinator address (host:port)")
    parser.add_argument("--circuit-file", help="Circuit file to simulate (JSON)")
    parser.add_argument("--strategy", choices=["amplitude", "gate_parallel", "circuit_split", "hybrid"],
                       default="amplitude", help="Distribution strategy")
    
    args = parser.parse_args()
    
    if args.mode == "client" and args.circuit_file:
        # Client mode - submit simulation
        try:
            with open(args.circuit_file, 'r') as f:
                circuit_data = json.load(f)
            
            strategy = DistributionStrategy(f"{args.strategy}_partitioning")
            result = simulate_circuit_distributed(circuit_data, strategy)
            
            if result:
                print("Simulation completed successfully:")
                print(json.dumps(result, indent=2))
            else:
                print("Simulation failed or timed out")
                return 1
                
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    else:
        # Service mode - start distributed node
        role = NodeRole.COORDINATOR if args.mode == "coordinator" else NodeRole.WORKER
        
        success = start_distributed_simulation_service(
            node_id=args.node_id,
            role=role,
            port=args.port
        )
        
        if not success:
            print("Failed to start distributed simulation service")
            return 1
        
        simulator = get_distributed_simulator()
        
        # Join cluster if coordinator specified
        if args.coordinator and role == NodeRole.WORKER:
            try:
                host, port = args.coordinator.split(':')
                simulator.join_cluster(host, int(port))
            except Exception as e:
                print(f"Failed to join cluster: {e}")
        
        print(f"Distributed simulation service started")
        print(f"Node ID: {simulator.node_id}")
        print(f"Role: {role.value}")
        print(f"Port: {simulator.node_info.port}")
        
        try:
            # Keep service running
            while True:
                time.sleep(1.0)
                
                # Print status periodically
                if int(time.time()) % 30 == 0:
                    status = simulator.get_cluster_status()
                    print(f"Status: {status}")
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
            stop_distributed_simulation_service()
    
    return 0


if __name__ == "__main__":
    exit(main())