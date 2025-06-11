# Quick Start Guide

Get up and running with QuantRS2 in just 5 minutes! This guide will walk you through the basics of quantum circuit creation, simulation, and analysis.

## Your First Quantum Circuit

Let's start with the most famous quantum circuit - the Bell state:

```python
import quantrs2

# Create a 2-qubit circuit
circuit = quantrs2.Circuit(2)

# Add Hadamard gate to qubit 0
circuit.h(0)

# Add CNOT gate (control: qubit 0, target: qubit 1)
circuit.cx(0, 1)

# Add measurements
circuit.measure_all()

# Run the simulation
result = circuit.run()

# Display results
print("Bell state probabilities:")
print(result.state_probabilities())
```

**Output:**
```
Bell state probabilities:
{'00': 0.5, '11': 0.5}
```

Congratulations! You've created a quantum entangled state where measuring both qubits gives either `00` or `11` with equal probability.

## Understanding the Code

Let's break down what happened:

1. **Circuit Creation**: `Circuit(2)` creates a 2-qubit quantum circuit
2. **Hadamard Gate**: `h(0)` puts qubit 0 in superposition (|0⟩ + |1⟩)/√2
3. **CNOT Gate**: `cx(0, 1)` entangles the qubits
4. **Measurement**: `measure_all()` measures all qubits
5. **Simulation**: `run()` executes the circuit on a quantum simulator

## Visualization

Visualize your circuit:

```python
# Visualize the circuit structure
quantrs2.visualize_circuit(circuit)

# Create a more detailed visualization
quantrs2.visualize_circuit(circuit, show_measurements=True, style='detailed')
```

## Basic Quantum Gates

QuantRS2 supports all standard quantum gates:

```python
import quantrs2

circuit = quantrs2.Circuit(3)

# Single-qubit gates
circuit.h(0)        # Hadamard
circuit.x(1)        # Pauli-X (NOT)
circuit.y(2)        # Pauli-Y
circuit.z(0)        # Pauli-Z
circuit.s(1)        # S gate
circuit.t(2)        # T gate

# Rotation gates
circuit.rx(0, 1.57)  # X-rotation (π/2)
circuit.ry(1, 0.785) # Y-rotation (π/4)
circuit.rz(2, 3.14)  # Z-rotation (π)

# Two-qubit gates
circuit.cx(0, 1)     # CNOT
circuit.cy(1, 2)     # Controlled-Y
circuit.cz(0, 2)     # Controlled-Z
circuit.swap(0, 1)   # SWAP

# Run and analyze
result = circuit.run()
print(f"Final state amplitudes: {result.amplitudes}")
```

## Parameterized Circuits

Create circuits with parameters for variational algorithms:

```python
import numpy as np

def create_variational_circuit(parameters):
    """Create a parameterized quantum circuit."""
    circuit = quantrs2.Circuit(4)
    
    # Initial layer - superposition
    for qubit in range(4):
        circuit.h(qubit)
    
    # Parameterized layers
    for layer in range(3):
        # Rotation layer
        for qubit in range(4):
            circuit.ry(qubit, parameters[layer * 4 + qubit])
        
        # Entangling layer
        for qubit in range(3):
            circuit.cx(qubit, qubit + 1)
    
    return circuit

# Generate random parameters
params = np.random.random(12) * 2 * np.pi

# Create and run the circuit
var_circuit = create_variational_circuit(params)
result = var_circuit.run()

print(f"Variational circuit created with {len(params)} parameters")
print(f"Final state vector norm: {np.linalg.norm(result.amplitudes):.6f}")
```

## Quantum Algorithms

### Grover's Algorithm

Implement a simple Grover search:

```python
def grovers_algorithm(num_qubits, target_state):
    """Grover's algorithm for searching."""
    circuit = quantrs2.Circuit(num_qubits)
    
    # Initialize superposition
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    # Number of iterations
    iterations = int(np.pi * np.sqrt(2**num_qubits) / 4)
    
    for _ in range(iterations):
        # Oracle (mark target state)
        oracle(circuit, target_state)
        
        # Diffusion operator
        diffusion_operator(circuit, num_qubits)
    
    circuit.measure_all()
    return circuit

def oracle(circuit, target):
    """Oracle function to mark target state."""
    # Simple oracle for demonstration
    if target == "11":  # Target state |11⟩
        circuit.cz(0, 1)

def diffusion_operator(circuit, num_qubits):
    """Grover diffusion operator."""
    for qubit in range(num_qubits):
        circuit.h(qubit)
        circuit.x(qubit)
    
    # Multi-controlled Z
    circuit.h(num_qubits - 1)
    for i in range(num_qubits - 1):
        circuit.cx(i, num_qubits - 1)
    circuit.h(num_qubits - 1)
    
    for qubit in range(num_qubits):
        circuit.x(qubit)
        circuit.h(qubit)

# Run Grover's algorithm
grover_circuit = grovers_algorithm(2, "11")
result = grover_circuit.run()
print("Grover's algorithm results:")
print(result.state_probabilities())
```

## Machine Learning Integration

Use QuantRS2 for quantum machine learning:

```python
from quantrs2.ml import QNN, VQE

# Create a Quantum Neural Network
qnn = QNN(num_qubits=4, num_layers=3)

# Generate training data
X_train = np.random.random((100, 4))
y_train = np.random.randint(0, 2, 100)

# Train the QNN
qnn.fit(X_train, y_train, epochs=10)

# Make predictions
X_test = np.random.random((10, 4))
predictions = qnn.predict(X_test)

print(f"QNN trained on {len(X_train)} samples")
print(f"Test predictions: {predictions}")

# Variational Quantum Eigensolver (VQE)
vqe = VQE(num_qubits=4, ansatz_depth=3)

# Define a simple Hamiltonian (Pauli-Z on all qubits)
hamiltonian = "Z0 + Z1 + Z2 + Z3"

# Find ground state energy
ground_energy = vqe.find_ground_state(hamiltonian)
print(f"Ground state energy: {ground_energy:.6f}")
```

## Performance Analysis

Analyze circuit performance with the built-in profiler:

```python
from quantrs2.profiler import profile_circuit

# Create a complex circuit
complex_circuit = quantrs2.Circuit(8)

# Add many gates
for layer in range(10):
    for qubit in range(8):
        complex_circuit.rx(qubit, np.random.random())
    for qubit in range(7):
        complex_circuit.cx(qubit, qubit + 1)

# Profile the circuit
profile_result = profile_circuit(complex_circuit)

print("Circuit Performance Analysis:")
print(f"Total gates: {profile_result.total_gates}")
print(f"Circuit depth: {profile_result.circuit_depth}")
print(f"Execution time: {profile_result.execution_time:.4f}s")
print(f"Memory usage: {profile_result.memory_usage:.2f} MB")
```

## Quantum State Analysis

Analyze quantum states in detail:

```python
# Create an interesting quantum state
analysis_circuit = quantrs2.Circuit(3)
analysis_circuit.h(0)
analysis_circuit.cx(0, 1)
analysis_circuit.ry(2, np.pi/3)
analysis_circuit.cx(1, 2)

result = analysis_circuit.run()

# Analyze the state
print("Quantum State Analysis:")
print(f"State vector: {result.amplitudes}")
print(f"Probabilities: {result.state_probabilities()}")

# Calculate quantum metrics
from quantrs2.utils import calculate_entanglement, calculate_purity

entanglement = calculate_entanglement(result)
purity = calculate_purity(result)

print(f"Entanglement measure: {entanglement:.4f}")
print(f"State purity: {purity:.4f}")
```

## Visualization and Debugging

Use QuantRS2's powerful visualization tools:

```python
# Create a circuit for visualization
viz_circuit = quantrs2.Circuit(4)
viz_circuit.h(0)
viz_circuit.cx(0, 1)
viz_circuit.ry(2, np.pi/4)
viz_circuit.cx(2, 3)
viz_circuit.measure_all()

# Visualize circuit
quantrs2.visualize_circuit(viz_circuit, style='modern')

# Visualize probabilities
result = viz_circuit.run()
quantrs2.visualize_probabilities(result, style='histogram')

# Debug step by step
from quantrs2.debugging import step_through_circuit

debugger = step_through_circuit(viz_circuit)
for step, state in enumerate(debugger):
    print(f"Step {step}: {state.amplitudes[:4]}...")  # Show first 4 amplitudes
```

## Working with Real Hardware

QuantRS2 integrates with real quantum hardware:

```python
from quantrs2.cloud import get_quantum_backend

# Connect to IBM Quantum
backend = get_quantum_backend('ibm', device='ibmq_qasm_simulator')

# Optimize circuit for hardware
optimized_circuit = backend.optimize_circuit(circuit)

# Submit job
job = backend.run_circuit(optimized_circuit, shots=1024)

# Get results
hardware_result = job.result()
print(f"Hardware execution results: {hardware_result.counts}")
```

## Next Steps

Now that you've mastered the basics, explore more advanced features:

1. **[Build Complex Circuits](first-circuit.md)**: Learn advanced circuit construction
2. **[Quantum Algorithms](../user-guide/quantum-algorithms.md)**: Implement famous algorithms
3. **[Machine Learning](../advanced/machine-learning.md)**: Quantum ML applications
4. **[Performance Optimization](../user-guide/performance.md)**: Optimize your quantum code
5. **[Real Hardware](../advanced/hardware-integration.md)**: Run on actual quantum computers

## Common Patterns

### Error Handling

```python
try:
    circuit = quantrs2.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    result = circuit.run()
except quantrs2.QuantumError as e:
    print(f"Quantum computation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Batch Processing

```python
# Process multiple circuits efficiently
circuits = [create_variational_circuit(np.random.random(12)) for _ in range(10)]

# Run in parallel
from quantrs2.parallel import run_circuits_parallel
results = run_circuits_parallel(circuits)

print(f"Processed {len(circuits)} circuits in parallel")
```

### Circuit Composition

```python
# Combine circuits
def create_encoder(qubits):
    circuit = quantrs2.Circuit(qubits)
    for i in range(qubits):
        circuit.h(i)
    return circuit

def create_decoder(qubits):
    circuit = quantrs2.Circuit(qubits)
    for i in range(qubits-1):
        circuit.cx(i, i+1)
    return circuit

# Compose circuits
full_circuit = create_encoder(4) + create_decoder(4)
result = full_circuit.run()
```

## Tips for Success

1. **Start Simple**: Begin with small circuits and gradually increase complexity
2. **Use Visualization**: Always visualize your circuits to understand their structure
3. **Profile Performance**: Use the profiler to optimize your quantum code
4. **Test Thoroughly**: Use QuantRS2's testing tools to validate your algorithms
5. **Read the Docs**: Explore the comprehensive API documentation

**Ready for more?** Continue to [Building Your First Circuit](first-circuit.md) for a deeper dive!