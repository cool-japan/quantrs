# QuantRS2 Gates Module

The `quantrs2.gates` module provides comprehensive support for quantum gate operations, including standard gates, parameterized gates, multi-qubit gates, and custom gate creation.

## Features

### Standard Gates

#### Single-Qubit Gates
- **Pauli Gates**: `X`, `Y`, `Z` - Basic quantum gates
- **Hadamard**: `H` - Creates superposition states
- **Phase Gates**: `S`, `SDagger`, `T`, `TDagger` - Phase rotation gates
- **Square Root**: `SX`, `SXDagger` - Square root of X gate
- **Rotations**: `RX`, `RY`, `RZ` - Parameterized rotation gates

#### Two-Qubit Gates
- **CNOT/CX**: Controlled-X gate
- **CY, CZ**: Controlled Y and Z gates
- **CH, CS**: Controlled Hadamard and S gates
- **SWAP**: Exchange two qubits
- **CRX, CRY, CRZ**: Controlled rotation gates

#### Three-Qubit Gates
- **Toffoli/CCX**: Doubly-controlled NOT gate
- **Fredkin/CSWAP**: Controlled SWAP gate

### Parametric Gates

For variational quantum algorithms:
- `ParametricRX`, `ParametricRY`, `ParametricRZ`: Symbolic rotation gates
- `ParametricU`: General single-qubit unitary with 3 parameters
- Support for symbolic parameters and parameter binding

### Custom Gates

Create gates from arbitrary unitary matrices:
```python
matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
custom_gate = CustomGate("MyGate", [0], matrix)
```

## Usage Examples

### Basic Gates

```python
from quantrs2 import gates

# Create gates
h_gate = gates.H(0)  # Hadamard on qubit 0
cnot_gate = gates.CNOT(0, 1)  # CNOT with control=0, target=1

# Get gate properties
print(h_gate.name)  # "H"
print(h_gate.qubits)  # [0]
print(h_gate.is_parameterized)  # False

# Get gate matrix
matrix = h_gate.matrix()  # Returns numpy array
```

### Rotation Gates

```python
# Create rotation gates
rx_gate = gates.RX(0, np.pi/2)  # π/2 rotation around X
ry_gate = gates.RY(0, np.pi/4)  # π/4 rotation around Y
rz_gate = gates.RZ(0, np.pi/3)  # π/3 rotation around Z

# Controlled rotations
crx_gate = gates.CRX(0, 1, np.pi/2)
```

### Parametric Gates

```python
# Create parametric gates with symbolic parameters
param_rx = gates.ParametricRX(0, "theta1")
param_ry = gates.ParametricRY(0, "theta2")

# Get parameter information
print(param_rx.parameter_names())  # ["theta1"]

# Assign values to parameters
param_rx_bound = param_rx.assign({"theta1": np.pi/2})

# Create general U gate
param_u = gates.ParametricU(0, "alpha", "beta", "gamma")
param_u_bound = param_u.bind({
    "alpha": np.pi/2,
    "beta": np.pi/4, 
    "gamma": 0
})
```

### Custom Gates

```python
# Create a custom 2-qubit gate
iswap_matrix = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

iswap = gates.CustomGate("iSWAP", [0, 1], iswap_matrix)
```

### Using Gates in Circuits

```python
from quantrs2 import Circuit

circuit = Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)
result = circuit.run()
```

## Gate Properties

All gates support:
- `name`: Gate name
- `qubits`: List of qubit indices the gate acts on
- `num_qubits`: Number of qubits
- `is_parameterized`: Whether the gate has parameters
- `matrix()`: Get the unitary matrix representation

## Type Hints

The module includes comprehensive type hints for better IDE support:

```python
def rx(qubit: int, theta: float) -> RX:
    """Create an X-rotation gate."""
    ...
```

## Integration with NumPy

Gates seamlessly integrate with NumPy arrays:
- Gate matrices are returned as NumPy arrays
- Parameters accept NumPy scalars
- Custom gates accept NumPy arrays for matrix input

## Performance

- Gates are implemented in Rust for optimal performance
- Matrix operations use optimized linear algebra routines
- Minimal Python overhead for gate creation

## See Also

- [Examples](../../../examples/): Example scripts demonstrating gate usage
- [Circuit API](../README.md): Using gates in quantum circuits
- [Visualization](../visualization.py): Visualizing gate operations