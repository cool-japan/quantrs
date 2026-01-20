# Building Your First Circuit

This guide will teach you the fundamentals of quantum circuit construction in QuantRS2. By the end, you'll understand how to build, analyze, and optimize quantum circuits for various applications.

## Circuit Basics

### Creating a Circuit

```python
import quantrs2
import numpy as np

# Create a 3-qubit circuit
circuit = quantrs2.Circuit(3)

print(f"Created circuit with {circuit.num_qubits} qubits")
print(f"Initial state: |{'0' * circuit.num_qubits}⟩")
```

Every qubit starts in the |0⟩ state. Let's visualize this:

```python
# Visualize initial circuit
quantrs2.visualize_circuit(circuit)
```

### Adding Gates

QuantRS2 provides all standard quantum gates:

```python
# Single-qubit gates
circuit.h(0)        # Hadamard on qubit 0
circuit.x(1)        # Pauli-X (bit flip) on qubit 1
circuit.y(2)        # Pauli-Y on qubit 2

# Rotation gates with parameters
circuit.rx(0, np.pi/2)    # X-rotation by π/2
circuit.ry(1, np.pi/4)    # Y-rotation by π/4
circuit.rz(2, np.pi/3)    # Z-rotation by π/3

# Phase gates
circuit.s(0)        # S gate (√Z)
circuit.t(1)        # T gate (√S)

# Two-qubit gates
circuit.cx(0, 1)    # CNOT (controlled-X)
circuit.cy(1, 2)    # Controlled-Y
circuit.cz(0, 2)    # Controlled-Z

print(f"Circuit now has {circuit.gate_count} gates")
```

## Understanding Gate Operations

### Single-Qubit Gates

Let's explore what each gate does:

```python
def demonstrate_single_qubit_gates():
    """Demonstrate the effect of single-qubit gates."""
    
    # Hadamard gate: creates superposition
    circuit_h = quantrs2.Circuit(1)
    circuit_h.h(0)
    circuit_h.measure_all()
    
    result_h = circuit_h.run()
    print("Hadamard gate result:")
    print(result_h.state_probabilities())
    # Output: {'0': 0.5, '1': 0.5}
    
    # Pauli-X gate: bit flip
    circuit_x = quantrs2.Circuit(1)
    circuit_x.x(0)  # |0⟩ → |1⟩
    circuit_x.measure_all()
    
    result_x = circuit_x.run()
    print("\nPauli-X gate result:")
    print(result_x.state_probabilities())
    # Output: {'1': 1.0}
    
    # Rotation gates: continuous rotations
    circuit_ry = quantrs2.Circuit(1)
    circuit_ry.ry(0, np.pi/3)  # 60-degree rotation
    circuit_ry.measure_all()
    
    result_ry = circuit_ry.run()
    print(f"\nRY(π/3) gate result:")
    print(result_ry.state_probabilities())

demonstrate_single_qubit_gates()
```

### Two-Qubit Gates and Entanglement

Two-qubit gates create correlations between qubits:

```python
def demonstrate_entanglement():
    """Show how CNOT creates entanglement."""
    
    # Bell state preparation
    bell_circuit = quantrs2.Circuit(2)
    bell_circuit.h(0)      # Create superposition
    bell_circuit.cx(0, 1)  # Entangle qubits
    bell_circuit.measure_all()
    
    result = bell_circuit.run()
    print("Bell state (entangled):")
    print(result.state_probabilities())
    # Output: {'00': 0.5, '11': 0.5}
    
    # Compare with separable state
    separable_circuit = quantrs2.Circuit(2)
    separable_circuit.h(0)  # Only qubit 0 in superposition
    separable_circuit.h(1)  # Only qubit 1 in superposition
    separable_circuit.measure_all()
    
    result_sep = separable_circuit.run()
    print("\nSeparable state (not entangled):")
    print(result_sep.state_probabilities())
    # Output: {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25}

demonstrate_entanglement()
```

## Circuit Construction Patterns

### Sequential Gate Application

```python
def create_ghz_state(num_qubits):
    """Create a GHZ (Greenberger-Horne-Zeilinger) state."""
    circuit = quantrs2.Circuit(num_qubits)
    
    # Start with Hadamard on first qubit
    circuit.h(0)
    
    # Apply CNOT gates sequentially
    for i in range(num_qubits - 1):
        circuit.cx(0, i + 1)
    
    return circuit

# Create 4-qubit GHZ state
ghz_circuit = create_ghz_state(4)
ghz_circuit.measure_all()

result = ghz_circuit.run()
print("4-qubit GHZ state:")
print(result.state_probabilities())
# Should show only |0000⟩ and |1111⟩ with equal probability
```

### Layered Circuit Architecture

```python
def create_ansatz_circuit(num_qubits, depth, parameters):
    """Create a variational ansatz with layered structure."""
    circuit = quantrs2.Circuit(num_qubits)
    
    param_idx = 0
    
    for layer in range(depth):
        # Rotation layer
        for qubit in range(num_qubits):
            circuit.ry(qubit, parameters[param_idx])
            param_idx += 1
        
        # Entangling layer
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        
        # Optional: Add Z-rotations
        for qubit in range(num_qubits):
            circuit.rz(qubit, parameters[param_idx])
            param_idx += 1
    
    return circuit, param_idx

# Create ansatz circuit
num_qubits = 4
depth = 3
num_params = 2 * num_qubits * depth  # RY + RZ for each qubit in each layer

parameters = np.random.uniform(0, 2*np.pi, num_params)
ansatz, params_used = create_ansatz_circuit(num_qubits, depth, parameters)

print(f"Created ansatz with {params_used} parameters")
print(f"Circuit depth: {ansatz.depth}")
```

## Advanced Circuit Techniques

### Conditional Operations

```python
def quantum_teleportation():
    """Implement quantum teleportation protocol."""
    circuit = quantrs2.Circuit(3)
    
    # Prepare state to teleport (example: |+⟩ state)
    circuit.h(0)
    
    # Create Bell pair between qubits 1 and 2
    circuit.h(1)
    circuit.cx(1, 2)
    
    # Bell measurement on qubits 0 and 1
    circuit.cx(0, 1)
    circuit.h(0)
    
    # Measure qubits 0 and 1
    circuit.measure(0)
    circuit.measure(1)
    
    # Conditional corrections on qubit 2
    # (In a real implementation, these would be conditional on measurement results)
    circuit.cx(1, 2)  # Conditional X
    circuit.cz(0, 2)  # Conditional Z
    
    # Measure final qubit
    circuit.measure(2)
    
    return circuit

teleport_circuit = quantum_teleportation()
result = teleport_circuit.run()
print("Quantum teleportation results:")
print(result.state_probabilities())
```

### Circuit Composition

```python
def compose_circuits():
    """Demonstrate circuit composition techniques."""
    
    # Create encoder circuit
    encoder = quantrs2.Circuit(3)
    encoder.h(0)
    encoder.cx(0, 1)
    encoder.cx(1, 2)
    
    # Create decoder circuit
    decoder = quantrs2.Circuit(3)
    decoder.cx(1, 2)
    decoder.cx(0, 1)
    decoder.h(0)
    
    # Method 1: Sequential composition
    full_circuit = quantrs2.Circuit(3)
    
    # Add encoder gates
    full_circuit.h(0)
    full_circuit.cx(0, 1)
    full_circuit.cx(1, 2)
    
    # Add some operations in between
    full_circuit.rz(1, np.pi/4)
    
    # Add decoder gates
    full_circuit.cx(1, 2)
    full_circuit.cx(0, 1)
    full_circuit.h(0)
    
    return full_circuit

composed = compose_circuits()
print(f"Composed circuit has {composed.gate_count} gates")
```

## Circuit Analysis and Optimization

### Basic Circuit Properties

```python
def analyze_circuit(circuit):
    """Analyze circuit properties."""
    print(f"Circuit Analysis:")
    print(f"- Number of qubits: {circuit.num_qubits}")
    print(f"- Gate count: {circuit.gate_count}")
    print(f"- Circuit depth: {circuit.depth}")
    
    # Count gates by type
    gate_counts = circuit.gate_type_counts()
    print(f"- Gate breakdown: {gate_counts}")
    
    # Estimate resources
    two_qubit_gates = gate_counts.get('cx', 0) + gate_counts.get('cy', 0) + gate_counts.get('cz', 0)
    print(f"- Two-qubit gates: {two_qubit_gates}")
    
    return {
        'qubits': circuit.num_qubits,
        'gates': circuit.gate_count,
        'depth': circuit.depth,
        'two_qubit_gates': two_qubit_gates
    }

# Analyze different circuits
test_circuit = create_ansatz_circuit(4, 3, np.random.random(24))[0]
analysis = analyze_circuit(test_circuit)
```

### Circuit Optimization

```python
from quantrs2.optimization import optimize_circuit

def demonstrate_optimization():
    """Show circuit optimization techniques."""
    
    # Create an inefficient circuit
    inefficient = quantrs2.Circuit(3)
    
    # Add redundant operations
    inefficient.h(0)
    inefficient.h(0)  # H·H = I (identity)
    inefficient.x(1)
    inefficient.x(1)  # X·X = I
    inefficient.z(2)
    inefficient.z(2)  # Z·Z = I
    
    # Add useful operations
    inefficient.cx(0, 1)
    inefficient.ry(2, np.pi/4)
    
    print("Before optimization:")
    analyze_circuit(inefficient)
    
    # Optimize the circuit
    optimized = optimize_circuit(inefficient)
    
    print("\nAfter optimization:")
    analyze_circuit(optimized)
    
    # Verify equivalence
    inefficient.measure_all()
    optimized.measure_all()
    
    result1 = inefficient.run()
    result2 = optimized.run()
    
    print(f"\nResults equivalent: {result1.state_probabilities() == result2.state_probabilities()}")

demonstrate_optimization()
```

## Debugging and Validation

### Step-by-Step Execution

```python
from quantrs2.debugging import QuantumDebugger

def debug_circuit():
    """Debug a circuit step by step."""
    
    # Create a circuit to debug
    debug_circuit = quantrs2.Circuit(2)
    debug_circuit.h(0)
    debug_circuit.cx(0, 1)
    debug_circuit.ry(1, np.pi/4)
    
    # Create debugger
    debugger = QuantumDebugger(debug_circuit)
    
    print("Step-by-step execution:")
    for step, state in debugger.step_through():
        print(f"Step {step}: {state.description}")
        print(f"  State vector: {state.amplitudes[:4]}...")  # Show first 4 components
        print(f"  Probabilities: {state.probabilities}")
        print()

debug_circuit()
```

### Circuit Validation

```python
def validate_circuit(circuit):
    """Validate circuit properties."""
    
    # Check if circuit is unitary (reversible)
    if circuit.is_unitary():
        print("✓ Circuit is unitary (reversible)")
    else:
        print("✗ Circuit is not unitary")
    
    # Check for common issues
    issues = circuit.validate()
    if issues:
        print("Circuit issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ No issues found")
    
    # Performance warnings
    if circuit.depth > 100:
        print("⚠ Circuit depth is high, may be slow on real hardware")
    
    if circuit.gate_count > 1000:
        print("⚠ High gate count, consider optimization")

# Validate our circuits
print("Validating GHZ circuit:")
validate_circuit(ghz_circuit)

print("\nValidating ansatz circuit:")
validate_circuit(ansatz)
```

## Best Practices

### 1. Circuit Design Principles

```python
def well_designed_circuit(data_qubits, ancilla_qubits=1):
    """Example of well-designed circuit structure."""
    total_qubits = data_qubits + ancilla_qubits
    circuit = quantrs2.Circuit(total_qubits)
    
    # Clear structure with comments
    # 1. Initialization
    for i in range(data_qubits):
        circuit.h(i)  # Initialize data qubits in superposition
    
    # 2. Main computation
    for i in range(data_qubits - 1):
        circuit.cx(i, i + 1)  # Create entanglement
    
    # 3. Ancilla operations (if needed)
    if ancilla_qubits > 0:
        ancilla_idx = data_qubits
        circuit.cx(0, ancilla_idx)  # Use ancilla for computation
    
    # 4. Cleanup and measurement
    circuit.measure_all()
    
    return circuit
```

### 2. Parameter Management

```python
class ParameterizedCircuit:
    """Class for managing parameterized circuits."""
    
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_parameters = self._calculate_parameter_count()
    
    def _calculate_parameter_count(self):
        """Calculate required number of parameters."""
        return self.num_qubits * self.num_layers  # One parameter per qubit per layer
    
    def build_circuit(self, parameters):
        """Build circuit with given parameters."""
        if len(parameters) != self.num_parameters:
            raise ValueError(f"Expected {self.num_parameters} parameters, got {len(parameters)}")
        
        circuit = quantrs2.Circuit(self.num_qubits)
        param_idx = 0
        
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                circuit.ry(qubit, parameters[param_idx])
                param_idx += 1
            
            # Add entangling layer
            for qubit in range(self.num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        
        return circuit
    
    def random_parameters(self):
        """Generate random parameters."""
        return np.random.uniform(0, 2*np.pi, self.num_parameters)

# Usage
param_circuit = ParameterizedCircuit(4, 3)
params = param_circuit.random_parameters()
circuit = param_circuit.build_circuit(params)
```

### 3. Error Handling

```python
def robust_circuit_execution(circuit, shots=1024):
    """Execute circuit with proper error handling."""
    try:
        # Validate circuit before execution
        issues = circuit.validate()
        if issues:
            print(f"Warning: Circuit issues detected: {issues}")
        
        # Run with multiple shots for statistical analysis
        results = []
        for _ in range(shots):
            result = circuit.run()
            results.append(result)
        
        # Aggregate results
        aggregated_probs = {}
        for result in results:
            probs = result.state_probabilities()
            for state, prob in probs.items():
                if state in aggregated_probs:
                    aggregated_probs[state] += prob
                else:
                    aggregated_probs[state] = prob
        
        # Normalize
        total = sum(aggregated_probs.values())
        for state in aggregated_probs:
            aggregated_probs[state] /= total
        
        return aggregated_probs
        
    except quantrs2.QuantumError as e:
        print(f"Quantum computation error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Robust execution example
safe_circuit = quantrs2.Circuit(2)
safe_circuit.h(0)
safe_circuit.cx(0, 1)
safe_circuit.measure_all()

result = robust_circuit_execution(safe_circuit)
if result:
    print("Robust execution successful:")
    print(result)
```

## Next Steps

Now that you understand circuit construction fundamentals:

1. **[Basic Examples](basic-examples.md)**: Explore practical circuit examples
2. **[Quantum Algorithms](../user-guide/quantum-algorithms.md)**: Learn algorithm implementation
3. **[Visualization](../user-guide/visualization.md)**: Master circuit visualization
4. **[Performance](../user-guide/performance.md)**: Optimize your circuits

## Common Pitfalls to Avoid

1. **Measurement Placement**: Remember that measurement destroys superposition
2. **Parameter Scaling**: Keep rotation angles in reasonable ranges (0 to 2π)
3. **Circuit Depth**: Deep circuits are more susceptible to noise
4. **Gate Order**: Quantum gates don't always commute - order matters
5. **Resource Management**: Large circuits consume exponential memory

**Ready to explore examples?** Continue to [Basic Examples](basic-examples.md)!