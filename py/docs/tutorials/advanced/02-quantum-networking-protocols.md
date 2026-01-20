# Advanced Tutorial: Quantum Networking and Communication Protocols

## Overview

This tutorial covers advanced quantum networking concepts and their implementation in QuantRS2-Py, including quantum teleportation, entanglement distribution, quantum repeaters, and network topologies.

## Prerequisites

- Understanding of quantum entanglement
- Familiarity with quantum measurements
- Knowledge of classical communication protocols
- Completion of intermediate tutorials

## Topics Covered

1. Quantum Teleportation Protocol
2. Entanglement Distribution and Swapping
3. Quantum Repeaters and Purification
4. Quantum Key Distribution (QKD) Networks
5. Quantum Network Topologies
6. Distributed Quantum Computing

## 1. Quantum Teleportation

### Basic Teleportation Protocol

```python
import quantrs2 as qr
import numpy as np

def quantum_teleportation():
    """
    Implement the quantum teleportation protocol.

    Alice wants to send an unknown quantum state to Bob using
    entanglement and classical communication.
    """
    # 3 qubits: 0 (Alice's state), 1 (Alice's entangled), 2 (Bob's entangled)
    circuit = qr.PyCircuit(3)

    # Step 1: Prepare the state to teleport (qubit 0)
    # Arbitrary state: α|0⟩ + β|1⟩
    theta = np.pi / 3
    phi = np.pi / 4
    circuit.ry(0, theta)
    circuit.rz(0, phi)

    # Step 2: Create entangled pair between Alice (qubit 1) and Bob (qubit 2)
    circuit.h(1)
    circuit.cnot(1, 2)

    # Step 3: Alice performs Bell measurement
    circuit.cnot(0, 1)
    circuit.h(0)

    # Step 4: Alice measures her qubits
    circuit.measure(0)
    circuit.measure(1)

    # Step 5: Bob applies corrections based on measurement results
    # (This would be done conditionally based on classical bits)
    # If measurement results are 00, no correction needed
    # If 01, apply X gate
    # If 10, apply Z gate
    # If 11, apply XZ gates

    return circuit

# Run teleportation
teleport_circuit = quantum_teleportation()
result = teleport_circuit.run(shots=1000)

print("Teleportation measurements:", result.measurements())
```

### Teleportation with Correction

```python
class QuantumTeleporter:
    """Complete quantum teleportation with automatic corrections."""

    def __init__(self):
        self.circuit = None
        self.measurement_results = None

    def teleport_state(self, state_params):
        """
        Teleport a quantum state.

        Args:
            state_params: (theta, phi) parameters for state preparation

        Returns:
            Teleported state at Bob's location
        """
        theta, phi = state_params

        # Create circuit for teleportation
        circuit = qr.PyCircuit(3)

        # Prepare state to teleport
        circuit.ry(0, theta)
        circuit.rz(0, phi)

        # Create Bell pair
        circuit.h(1)
        circuit.cnot(1, 2)

        # Bell measurement
        circuit.cnot(0, 1)
        circuit.h(0)

        # Measure Alice's qubits
        circuit.measure(0)
        circuit.measure(1)

        # Run circuit
        result = circuit.run(shots=1)
        measurements = result.measurements()[0]

        # Apply corrections to Bob's qubit based on measurements
        bob_circuit = qr.PyCircuit(1)
        if measurements[1] == '1':
            bob_circuit.x(0)
        if measurements[0] == '1':
            bob_circuit.z(0)

        return bob_circuit

    def verify_teleportation(self, original_params, teleported_circuit):
        """Verify that teleportation was successful."""
        # Create original state
        orig_circuit = qr.PyCircuit(1)
        theta, phi = original_params
        orig_circuit.ry(0, theta)
        orig_circuit.rz(0, phi)

        # Calculate fidelity between original and teleported states
        orig_result = orig_circuit.run()
        teleport_result = teleported_circuit.run()

        # Get state vectors
        orig_state = orig_result.state_probabilities()
        teleport_state = teleport_result.state_probabilities()

        # Calculate fidelity
        fidelity = np.abs(np.sum(np.sqrt(orig_state * teleport_state)))**2
        return fidelity

# Example usage
teleporter = QuantumTeleporter()
state_params = (np.pi/3, np.pi/4)

teleported = teleporter.teleport_state(state_params)
fidelity = teleporter.verify_teleportation(state_params, teleported)

print(f"Teleportation fidelity: {fidelity:.4f}")
```

## 2. Entanglement Distribution

### Direct Entanglement Distribution

```python
from quantrs2 import quantum_networking as qn

def create_entanglement_distribution_network(num_nodes):
    """
    Create a network for distributing entanglement between nodes.

    Args:
        num_nodes: Number of nodes in the network

    Returns:
        QuantumNetwork instance
    """
    # Create network topology
    network = qn.QuantumNetwork()

    # Add nodes
    for i in range(num_nodes):
        network.add_node(f"node_{i}", capabilities={
            'memory_qubits': 10,
            'processing_qubits': 5,
            'coherence_time': 1.0  # seconds
        })

    # Create quantum channels between nodes
    for i in range(num_nodes - 1):
        network.add_channel(
            f"node_{i}",
            f"node_{i+1}",
            channel_params={
                'loss_rate': 0.2,  # dB/km
                'distance': 10,     # km
                'noise_rate': 0.01
            }
        )

    return network

# Create network
network = create_entanglement_distribution_network(5)

# Distribute entanglement between first and last nodes
network.distribute_entanglement("node_0", "node_4", strategy="direct")

# Check entanglement fidelity
fidelity = network.get_entanglement_fidelity("node_0", "node_4")
print(f"Entanglement fidelity: {fidelity:.4f}")
```

### Entanglement Swapping

```python
def entanglement_swapping_protocol():
    """
    Demonstrate entanglement swapping to extend entanglement range.

    Alice has qubit A, Bob has qubit B, Charlie has qubits C1 and C2.
    Alice-C1 and Bob-C2 are entangled. After swapping, Alice-Bob become entangled.
    """
    circuit = qr.PyCircuit(4)  # A, C1, C2, B

    # Create entangled pairs: A-C1 and C2-B
    # A-C1 pair
    circuit.h(0)
    circuit.cnot(0, 1)

    # C2-B pair
    circuit.h(2)
    circuit.cnot(2, 3)

    # Charlie performs Bell measurement on C1 and C2
    circuit.cnot(1, 2)
    circuit.h(1)
    circuit.measure(1)
    circuit.measure(2)

    # Now A and B should be entangled
    return circuit

# Run entanglement swapping
swap_circuit = entanglement_swapping_protocol()
result = swap_circuit.run(shots=1000)

# Analyze results
measurements = result.measurements()
print("Swapping measurements:", measurements)
```

### Multi-Hop Entanglement Distribution

```python
class EntanglementRouter:
    """Route entanglement through multiple hops in a quantum network."""

    def __init__(self, network):
        self.network = network
        self.entanglement_map = {}

    def find_path(self, source, destination):
        """Find path between source and destination nodes."""
        # Simple BFS pathfinding
        from collections import deque

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if current == destination:
                return path

            for neighbor in self.network.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def distribute_multi_hop(self, source, destination):
        """Distribute entanglement using multiple hops."""
        path = self.find_path(source, destination)

        if not path:
            raise ValueError(f"No path between {source} and {destination}")

        # Create entanglement between adjacent nodes
        for i in range(len(path) - 1):
            self.network.create_entanglement(path[i], path[i+1])

        # Perform entanglement swapping at intermediate nodes
        for i in range(1, len(path) - 1):
            self.network.perform_entanglement_swapping(
                path[i-1], path[i], path[i+1]
            )

        return self.network.get_entanglement_fidelity(source, destination)

# Example usage
network = create_entanglement_distribution_network(10)
router = EntanglementRouter(network)

# Distribute entanglement across 10 nodes
fidelity = router.distribute_multi_hop("node_0", "node_9")
print(f"Multi-hop entanglement fidelity: {fidelity:.4f}")
```

## 3. Quantum Repeaters and Purification

### Entanglement Purification

```python
def entanglement_purification_protocol(fidelity1, fidelity2):
    """
    Purify two low-fidelity entangled pairs into one higher-fidelity pair.

    Args:
        fidelity1: Fidelity of first entangled pair
        fidelity2: Fidelity of second entangled pair

    Returns:
        Fidelity of purified pair
    """
    # Purification circuit for two Bell pairs
    circuit = qr.PyCircuit(4)

    # Pairs: (0,1) and (2,3)
    # Assume they start in noisy Bell states

    # Purification protocol
    # 1. Perform bilateral CNOT operations
    circuit.cnot(0, 2)
    circuit.cnot(1, 3)

    # 2. Measure qubits 2 and 3
    circuit.measure(2)
    circuit.measure(3)

    # 3. Keep pair (0,1) if measurements agree, discard otherwise

    result = circuit.run(shots=1000)
    measurements = result.measurements()

    # Calculate success probability and resulting fidelity
    success_count = sum(1 for m in measurements if m[2] == m[3])
    success_prob = success_count / len(measurements)

    # Theoretical fidelity after purification
    # F' = (F1*F2 + (1-F1)*(1-F2)/3) / (F1*F2 + (1-F1)*(1-F2))
    numerator = fidelity1 * fidelity2 + (1-fidelity1) * (1-fidelity2) / 3
    denominator = fidelity1 * fidelity2 + (1-fidelity1) * (1-fidelity2)
    purified_fidelity = numerator / denominator

    return {
        'purified_fidelity': purified_fidelity,
        'success_probability': success_prob
    }

# Example purification
result = entanglement_purification_protocol(0.8, 0.75)
print(f"Purified fidelity: {result['purified_fidelity']:.4f}")
print(f"Success probability: {result['success_probability']:.4f}")
```

### Quantum Repeater Chain

```python
class QuantumRepeater:
    """Implement quantum repeater for long-distance entanglement."""

    def __init__(self, num_segments):
        self.num_segments = num_segments
        self.segment_length = 100  # km per segment
        self.base_fidelity = 0.9  # Initial fidelity

    def generate_segment_entanglement(self, segment_id):
        """Generate entanglement for one segment."""
        # Simulate distance-dependent fidelity degradation
        loss_factor = np.exp(-0.02 * self.segment_length)
        fidelity = self.base_fidelity * loss_factor
        return fidelity

    def repeater_protocol(self):
        """Execute full repeater protocol."""
        # Step 1: Generate entanglement for all segments
        segment_fidelities = [
            self.generate_segment_entanglement(i)
            for i in range(self.num_segments)
        ]

        print(f"Initial segment fidelities: {segment_fidelities}")

        # Step 2: Perform entanglement purification on each segment
        purified_fidelities = []
        for i in range(0, len(segment_fidelities), 2):
            if i + 1 < len(segment_fidelities):
                result = entanglement_purification_protocol(
                    segment_fidelities[i],
                    segment_fidelities[i+1]
                )
                purified_fidelities.append(result['purified_fidelity'])
            else:
                purified_fidelities.append(segment_fidelities[i])

        print(f"Purified fidelities: {purified_fidelities}")

        # Step 3: Perform entanglement swapping to connect segments
        current_fidelities = purified_fidelities
        while len(current_fidelities) > 1:
            next_fidelities = []
            for i in range(0, len(current_fidelities), 2):
                if i + 1 < len(current_fidelities):
                    # Swapping reduces fidelity
                    swapped_fidelity = current_fidelities[i] * current_fidelities[i+1] * 0.95
                    next_fidelities.append(swapped_fidelity)
                else:
                    next_fidelities.append(current_fidelities[i])
            current_fidelities = next_fidelities

        final_fidelity = current_fidelities[0]
        total_distance = self.num_segments * self.segment_length

        return {
            'final_fidelity': final_fidelity,
            'total_distance_km': total_distance
        }

# Example repeater chain
repeater = QuantumRepeater(num_segments=8)
result = repeater.repeater_protocol()

print(f"\nQuantum Repeater Results:")
print(f"Total distance: {result['total_distance_km']} km")
print(f"Final fidelity: {result['final_fidelity']:.4f}")
```

## 4. QKD Network Implementation

```python
from quantrs2 import crypto

class QKDNetwork:
    """Quantum Key Distribution network for secure communication."""

    def __init__(self):
        self.nodes = {}
        self.keys = {}

    def add_node(self, node_id, qkd_protocol='BB84'):
        """Add a node to the QKD network."""
        self.nodes[node_id] = {
            'protocol': qkd_protocol,
            'keys': {}
        }

    def establish_key(self, alice_id, bob_id, key_length=256):
        """Establish shared key between two nodes."""
        if alice_id not in self.nodes or bob_id not in self.nodes:
            raise ValueError("Both nodes must be in the network")

        # Use BB84 protocol
        bb84 = crypto.BB84QKD(key_length=key_length)

        # Alice prepares qubits
        alice_bits, alice_bases = bb84.alice_prepare()

        # Bob measures qubits
        bob_results, bob_bases = bb84.bob_measure()

        # Classical communication to compare bases
        key = bb84.sift_key(alice_bits, alice_bases, bob_bases, bob_results)

        # Store keys
        key_id = f"{alice_id}-{bob_id}"
        self.keys[key_id] = key
        self.nodes[alice_id]['keys'][bob_id] = key
        self.nodes[bob_id]['keys'][alice_id] = key

        return key

    def relay_key(self, source, intermediary, destination):
        """Relay key through trusted intermediary node."""
        # Establish keys: source-intermediary and intermediary-destination
        key1 = self.establish_key(source, intermediary)
        key2 = self.establish_key(intermediary, destination)

        # Intermediary XORs keys to create relay key
        relay_key = bytes(a ^ b for a, b in zip(key1, key2))

        return relay_key

# Example QKD network
qkd_net = QKDNetwork()
qkd_net.add_node("Alice")
qkd_net.add_node("Bob")
qkd_net.add_node("Charlie")

# Establish direct key
key_ab = qkd_net.establish_key("Alice", "Bob", key_length=256)
print(f"Key established between Alice and Bob: {len(key_ab)} bits")

# Relay key through Charlie
relay_key = qkd_net.relay_key("Alice", "Charlie", "Bob")
print(f"Relay key length: {len(relay_key)} bits")
```

## 5. Network Topologies and Routing

```python
def create_quantum_network_topology(topology_type, num_nodes):
    """
    Create quantum network with specified topology.

    Args:
        topology_type: 'star', 'mesh', 'ring', 'tree', 'grid'
        num_nodes: Number of nodes in the network
    """
    network = qn.QuantumNetwork()

    # Add nodes
    for i in range(num_nodes):
        network.add_node(f"node_{i}")

    if topology_type == 'star':
        # Central hub connected to all other nodes
        hub = "node_0"
        for i in range(1, num_nodes):
            network.add_channel(hub, f"node_{i}")

    elif topology_type == 'mesh':
        # Fully connected network
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                network.add_channel(f"node_{i}", f"node_{j}")

    elif topology_type == 'ring':
        # Ring topology
        for i in range(num_nodes):
            next_node = (i + 1) % num_nodes
            network.add_channel(f"node_{i}", f"node_{next_node}")

    elif topology_type == 'tree':
        # Binary tree structure
        for i in range((num_nodes - 1) // 2):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            if left_child < num_nodes:
                network.add_channel(f"node_{i}", f"node_{left_child}")
            if right_child < num_nodes:
                network.add_channel(f"node_{i}", f"node_{right_child}")

    elif topology_type == 'grid':
        # 2D grid topology
        grid_size = int(np.sqrt(num_nodes))
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                # Connect to right neighbor
                if j < grid_size - 1:
                    network.add_channel(f"node_{node_id}", f"node_{node_id+1}")
                # Connect to bottom neighbor
                if i < grid_size - 1:
                    network.add_channel(f"node_{node_id}", f"node_{node_id+grid_size}")

    return network

# Compare topologies
topologies = ['star', 'mesh', 'ring', 'tree', 'grid']
num_nodes = 16

for topology in topologies:
    network = create_quantum_network_topology(topology, num_nodes)

    # Calculate network metrics
    avg_path_length = network.average_path_length()
    connectivity = network.connectivity_measure()

    print(f"\nTopology: {topology}")
    print(f"Average path length: {avg_path_length:.2f}")
    print(f"Connectivity: {connectivity:.2f}")
```

## 6. Distributed Quantum Computing

```python
from quantrs2 import distributed_simulation as ds

def distributed_quantum_algorithm(algorithm_type, num_nodes):
    """
    Execute quantum algorithm across distributed nodes.

    Args:
        algorithm_type: Type of algorithm ('vqe', 'qaoa', 'grover')
        num_nodes: Number of computing nodes
    """
    # Create distributed cluster
    cluster = ds.QuantumCluster()

    # Add worker nodes
    for i in range(num_nodes):
        cluster.add_worker(f"worker_{i}", qubits=10, capabilities={
            'gpu': True if i % 2 == 0 else False,
            'memory_gb': 16
        })

    if algorithm_type == 'vqe':
        # Distribute VQE computation
        from quantrs2 import ml

        hamiltonian = create_distributed_hamiltonian()
        ansatz = create_distributed_ansatz()

        vqe = ml.VQE(hamiltonian, ansatz)
        result = cluster.run_distributed(vqe)

        return result

    elif algorithm_type == 'qaoa':
        # Distribute QAOA computation
        problem = create_maxcut_problem()
        qaoa = ml.QAOA(problem, layers=3)

        result = cluster.run_distributed(qaoa)
        return result

    elif algorithm_type == 'grover':
        # Distribute Grover's search
        search_space = 2**20  # Large search space
        target = 0b10101010

        grover = create_distributed_grover(search_space, target)
        result = cluster.run_distributed(grover)

        return result

# Example distributed computation
result = distributed_quantum_algorithm('vqe', num_nodes=4)
print(f"Distributed computation result: {result}")
```

## Exercises

1. Implement complete teleportation with noise and error correction
2. Create a quantum repeater simulator with realistic parameters
3. Design an optimal network topology for a specific use case
4. Implement entanglement routing algorithm for arbitrary topologies
5. Build a complete QKD network with key relay capability

## Further Reading

- Quantum Internet: https://arxiv.org/abs/1810.06539
- Quantum Repeaters: https://arxiv.org/abs/0906.2699
- Entanglement Distribution: https://arxiv.org/abs/1701.01062
- Quantum Networks: https://arxiv.org/abs/1803.02118

## Next Steps

- Advanced Tutorial: Quantum Cloud Computing
- Advanced Tutorial: Hybrid Quantum-Classical Algorithms
- Research: Practical Quantum Network Implementations
