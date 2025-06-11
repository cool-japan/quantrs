# Core API Reference

The core QuantRS2 API provides the fundamental building blocks for quantum circuit construction and simulation.

## Circuit Class

::: quantrs2.Circuit
    options:
      show_root_heading: true
      show_source: false

The `Circuit` class is the primary interface for building quantum circuits.

### Constructor

```python
Circuit(num_qubits: int) -> Circuit
```

Create a new quantum circuit with the specified number of qubits.

**Parameters:**
- `num_qubits` (int): Number of qubits in the circuit

**Example:**
```python
# Create a 3-qubit circuit
circuit = quantrs2.Circuit(3)
```

### Properties

#### `num_qubits`
```python
@property
def num_qubits(self) -> int
```
Number of qubits in the circuit.

#### `gate_count`
```python
@property
def gate_count(self) -> int
```
Total number of gates in the circuit.

#### `depth`
```python
@property
def depth(self) -> int
```
Circuit depth (maximum number of gates on any qubit).

### Single-Qubit Gates

#### Pauli Gates

```python
def x(self, qubit: int) -> None
```
Apply Pauli-X gate (bit flip).

```python
def y(self, qubit: int) -> None
```
Apply Pauli-Y gate.

```python
def z(self, qubit: int) -> None
```
Apply Pauli-Z gate (phase flip).

**Parameters:**
- `qubit` (int): Target qubit index

**Example:**
```python
circuit.x(0)  # Apply X gate to qubit 0
circuit.y(1)  # Apply Y gate to qubit 1
circuit.z(2)  # Apply Z gate to qubit 2
```

#### Hadamard Gate

```python
def h(self, qubit: int) -> None
```
Apply Hadamard gate (creates superposition).

**Parameters:**
- `qubit` (int): Target qubit index

**Example:**
```python
circuit.h(0)  # Put qubit 0 in superposition
```

#### Phase Gates

```python
def s(self, qubit: int) -> None
```
Apply S gate (√Z gate).

```python
def t(self, qubit: int) -> None
```
Apply T gate (√S gate).

**Parameters:**
- `qubit` (int): Target qubit index

#### Rotation Gates

```python
def rx(self, qubit: int, angle: float) -> None
```
Apply rotation around X-axis.

```python
def ry(self, qubit: int, angle: float) -> None
```
Apply rotation around Y-axis.

```python
def rz(self, qubit: int, angle: float) -> None
```
Apply rotation around Z-axis.

**Parameters:**
- `qubit` (int): Target qubit index
- `angle` (float): Rotation angle in radians

**Example:**
```python
circuit.rx(0, np.pi/2)    # 90-degree X rotation
circuit.ry(1, np.pi/4)    # 45-degree Y rotation
circuit.rz(2, np.pi)      # 180-degree Z rotation
```

### Two-Qubit Gates

#### CNOT Gate

```python
def cx(self, control: int, target: int) -> None
```
Apply controlled-X (CNOT) gate.

**Parameters:**
- `control` (int): Control qubit index
- `target` (int): Target qubit index

**Example:**
```python
circuit.cx(0, 1)  # Control on qubit 0, target on qubit 1
```

#### Controlled Gates

```python
def cy(self, control: int, target: int) -> None
```
Apply controlled-Y gate.

```python
def cz(self, control: int, target: int) -> None
```
Apply controlled-Z gate.

```python
def crx(self, control: int, target: int, angle: float) -> None
```
Apply controlled rotation around X-axis.

```python
def cry(self, control: int, target: int, angle: float) -> None
```
Apply controlled rotation around Y-axis.

```python
def crz(self, control: int, target: int, angle: float) -> None
```
Apply controlled rotation around Z-axis.

**Parameters:**
- `control` (int): Control qubit index
- `target` (int): Target qubit index
- `angle` (float): Rotation angle in radians (for rotation gates)

#### SWAP Gate

```python
def swap(self, qubit1: int, qubit2: int) -> None
```
Apply SWAP gate (exchanges qubit states).

**Parameters:**
- `qubit1` (int): First qubit index
- `qubit2` (int): Second qubit index

### Multi-Qubit Gates

#### Toffoli Gate

```python
def ccx(self, control1: int, control2: int, target: int) -> None
```
Apply Toffoli (CCX) gate (controlled-controlled-X).

**Parameters:**
- `control1` (int): First control qubit
- `control2` (int): Second control qubit
- `target` (int): Target qubit

**Example:**
```python
circuit.ccx(0, 1, 2)  # Toffoli gate with controls on 0,1 and target on 2
```

### Measurement Operations

#### Single Qubit Measurement

```python
def measure(self, qubit: int) -> None
```
Measure a specific qubit.

**Parameters:**
- `qubit` (int): Qubit to measure

#### Measure All Qubits

```python
def measure_all(self) -> None
```
Measure all qubits in the circuit.

**Example:**
```python
circuit.measure(0)      # Measure only qubit 0
circuit.measure_all()   # Measure all qubits
```

### Circuit Execution

#### Run Simulation

```python
def run(self, use_gpu: bool = False) -> SimulationResult
```
Execute the circuit on a quantum simulator.

**Parameters:**
- `use_gpu` (bool): Whether to use GPU acceleration (if available)

**Returns:**
- `SimulationResult`: Results of the quantum simulation

**Example:**
```python
result = circuit.run()
print(result.state_probabilities())

# With GPU acceleration
result_gpu = circuit.run(use_gpu=True)
```

### Circuit Utilities

#### Copy Circuit

```python
def copy(self) -> 'Circuit'
```
Create a deep copy of the circuit.

**Returns:**
- `Circuit`: A copy of the current circuit

#### Validate Circuit

```python
def validate(self) -> List[str]
```
Validate the circuit and return any issues found.

**Returns:**
- `List[str]`: List of validation issues (empty if valid)

#### Check if Unitary

```python
def is_unitary(self) -> bool
```
Check if the circuit represents a unitary operation.

**Returns:**
- `bool`: True if the circuit is unitary

#### Gate Type Counts

```python
def gate_type_counts(self) -> Dict[str, int]
```
Get count of each gate type in the circuit.

**Returns:**
- `Dict[str, int]`: Dictionary mapping gate names to counts

## SimulationResult Class

::: quantrs2.SimulationResult
    options:
      show_root_heading: true
      show_source: false

The `SimulationResult` class contains the results of quantum circuit execution.

### Properties

#### `amplitudes`
```python
@property
def amplitudes(self) -> List[complex]
```
Complex amplitudes of the quantum state vector.

#### `n_qubits`
```python
@property
def n_qubits(self) -> int
```
Number of qubits in the simulated circuit.

### Methods

#### State Probabilities

```python
def state_probabilities(self) -> Dict[str, float]
```
Get probability distribution over computational basis states.

**Returns:**
- `Dict[str, float]`: Mapping from basis states (e.g., "001") to probabilities

**Example:**
```python
result = circuit.run()
probs = result.state_probabilities()
print(probs)  # {'00': 0.5, '11': 0.5} for Bell state
```

## Utility Functions

### Bell State Creation

```python
def create_bell_state() -> SimulationResult
```
Create a Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2.

**Returns:**
- `SimulationResult`: Bell state simulation result

### GHZ State Creation

```python
def create_ghz_state(num_qubits: int) -> SimulationResult
```
Create a GHZ state for n qubits.

**Parameters:**
- `num_qubits` (int): Number of qubits

**Returns:**
- `SimulationResult`: GHZ state simulation result

### W State Creation

```python
def create_w_state(num_qubits: int) -> SimulationResult
```
Create a W state for n qubits.

**Parameters:**
- `num_qubits` (int): Number of qubits

**Returns:**
- `SimulationResult`: W state simulation result

### Uniform Superposition

```python
def create_uniform_superposition(num_qubits: int) -> SimulationResult
```
Create uniform superposition of all basis states.

**Parameters:**
- `num_qubits` (int): Number of qubits

**Returns:**
- `SimulationResult`: Uniform superposition state

## Constants and Enums

### Gate Types

```python
class GateType(Enum):
    """Enumeration of available gate types."""
    X = "x"
    Y = "y"
    Z = "z"
    H = "h"
    S = "s"
    T = "t"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    SWAP = "swap"
    CCX = "ccx"
    MEASURE = "measure"
```

### Common Constants

```python
# Mathematical constants
PI = 3.141592653589793
TWO_PI = 2 * PI
PI_OVER_2 = PI / 2
PI_OVER_4 = PI / 4

# Common rotation angles
ROTATION_90 = PI_OVER_2      # 90 degrees
ROTATION_45 = PI_OVER_4      # 45 degrees
ROTATION_180 = PI            # 180 degrees
```

## Error Handling

### Quantum Exceptions

```python
class QuantumError(Exception):
    """Base exception for quantum computation errors."""
    pass

class CircuitError(QuantumError):
    """Exception for circuit construction errors."""
    pass

class SimulationError(QuantumError):
    """Exception for simulation errors."""
    pass

class GateError(QuantumError):
    """Exception for gate operation errors."""
    pass
```

### Common Error Scenarios

```python
try:
    circuit = quantrs2.Circuit(2)
    circuit.cx(0, 2)  # Invalid: qubit 2 doesn't exist
except CircuitError as e:
    print(f"Circuit error: {e}")

try:
    circuit = quantrs2.Circuit(20)  # Large circuit
    result = circuit.run()
except SimulationError as e:
    print(f"Simulation error: {e}")
```

## Performance Considerations

### Memory Usage

The memory required for quantum simulation scales exponentially with the number of qubits:

- **n qubits**: 2ⁿ complex numbers (16 × 2ⁿ bytes)
- **10 qubits**: ~16 KB
- **20 qubits**: ~16 MB
- **30 qubits**: ~16 GB

### GPU Acceleration

GPU acceleration is available for circuits with sufficient complexity:

```python
# Check GPU availability
if quantrs2.gpu_available():
    result = circuit.run(use_gpu=True)
else:
    result = circuit.run()
```

### Circuit Optimization

Optimize circuits before execution:

```python
from quantrs2.optimization import optimize_circuit

# Optimize circuit structure
optimized = optimize_circuit(circuit)
result = optimized.run()
```

## Type Hints

QuantRS2 provides comprehensive type hints for better development experience:

```python
from typing import List, Dict, Optional, Union
import quantrs2

def create_parameterized_circuit(
    num_qubits: int,
    parameters: List[float]
) -> quantrs2.Circuit:
    """Create a parameterized quantum circuit."""
    circuit = quantrs2.Circuit(num_qubits)
    
    for i, param in enumerate(parameters):
        circuit.ry(i % num_qubits, param)
    
    return circuit

def analyze_results(
    result: quantrs2.SimulationResult
) -> Dict[str, Union[float, int]]:
    """Analyze simulation results."""
    probs = result.state_probabilities()
    
    return {
        'num_states': len(probs),
        'max_probability': max(probs.values()),
        'entropy': calculate_entropy(probs)
    }
```

## Examples

### Basic Circuit Construction

```python
import quantrs2
import numpy as np

# Create a 3-qubit circuit
circuit = quantrs2.Circuit(3)

# Add gates
circuit.h(0)        # Hadamard on qubit 0
circuit.cx(0, 1)    # CNOT from 0 to 1
circuit.ry(2, np.pi/4)  # Y-rotation on qubit 2
circuit.ccx(0, 1, 2)    # Toffoli gate

# Measure and run
circuit.measure_all()
result = circuit.run()

print(f"Circuit has {circuit.gate_count} gates")
print(f"Circuit depth: {circuit.depth}")
print(f"Results: {result.state_probabilities()}")
```

### Advanced Circuit Patterns

```python
def create_ansatz_circuit(num_qubits: int, depth: int) -> quantrs2.Circuit:
    """Create variational ansatz circuit."""
    circuit = quantrs2.Circuit(num_qubits)
    
    for layer in range(depth):
        # Rotation layer
        for qubit in range(num_qubits):
            circuit.ry(qubit, np.random.random() * 2 * np.pi)
        
        # Entangling layer
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        
        # Ring connection for final layer
        if layer == depth - 1 and num_qubits > 2:
            circuit.cx(num_qubits - 1, 0)
    
    return circuit

# Create and analyze ansatz
ansatz = create_ansatz_circuit(4, 3)
print(f"Ansatz circuit properties:")
print(f"  Gates: {ansatz.gate_count}")
print(f"  Depth: {ansatz.depth}")
print(f"  Gate types: {ansatz.gate_type_counts()}")
```

---

**See also:**
- [Gates API](gates.md): Detailed gate operations
- [Algorithms API](algorithms.md): High-level algorithm implementations
- [Visualization API](visualization.md): Circuit and state visualization