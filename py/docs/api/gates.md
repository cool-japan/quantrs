# Gates API Reference

The gates module provides advanced gate operations, custom gate definitions, and gate manipulation utilities for quantum circuits.

## Standard Gate Library

### Single-Qubit Gates

All single-qubit gates operate on a target qubit and can be applied to any circuit.

#### Identity and Pauli Gates

```python
def identity(circuit: Circuit, qubit: int) -> None
```
Apply identity gate (no operation).

```python
def pauli_x(circuit: Circuit, qubit: int) -> None
```
Apply Pauli-X gate (bit flip): |0⟩ ↔ |1⟩.

```python
def pauli_y(circuit: Circuit, qubit: int) -> None
```
Apply Pauli-Y gate: |0⟩ → i|1⟩, |1⟩ → -i|0⟩.

```python
def pauli_z(circuit: Circuit, qubit: int) -> None
```
Apply Pauli-Z gate (phase flip): |1⟩ → -|1⟩.

**Example:**
```python
from quantrs2.gates import pauli_x, pauli_y, pauli_z

circuit = quantrs2.Circuit(3)
pauli_x(circuit, 0)  # Bit flip on qubit 0
pauli_y(circuit, 1)  # Y gate on qubit 1
pauli_z(circuit, 2)  # Phase flip on qubit 2
```

#### Hadamard Gate

```python
def hadamard(circuit: Circuit, qubit: int) -> None
```
Apply Hadamard gate (creates superposition): |0⟩ → (|0⟩ + |1⟩)/√2, |1⟩ → (|0⟩ - |1⟩)/√2.

**Matrix:**
```
H = (1/√2) * [ 1  1 ]
             [ 1 -1 ]
```

#### Phase Gates

```python
def s_gate(circuit: Circuit, qubit: int) -> None
```
Apply S gate (√Z gate): |1⟩ → i|1⟩.

```python
def s_dagger(circuit: Circuit, qubit: int) -> None
```
Apply S† gate (inverse S gate): |1⟩ → -i|1⟩.

```python
def t_gate(circuit: Circuit, qubit: int) -> None
```
Apply T gate (√S gate): |1⟩ → e^(iπ/4)|1⟩.

```python
def t_dagger(circuit: Circuit, qubit: int) -> None
```
Apply T† gate (inverse T gate): |1⟩ → e^(-iπ/4)|1⟩.

```python
def phase_gate(circuit: Circuit, qubit: int, phase: float) -> None
```
Apply phase gate with arbitrary phase: |1⟩ → e^(iφ)|1⟩.

**Parameters:**
- `phase` (float): Phase angle in radians

#### Rotation Gates

```python
def rx_gate(circuit: Circuit, qubit: int, angle: float) -> None
```
Rotation around X-axis by angle θ.

**Matrix:**
```
RX(θ) = [ cos(θ/2)  -i*sin(θ/2) ]
        [-i*sin(θ/2)  cos(θ/2)  ]
```

```python
def ry_gate(circuit: Circuit, qubit: int, angle: float) -> None
```
Rotation around Y-axis by angle θ.

**Matrix:**
```
RY(θ) = [ cos(θ/2)  -sin(θ/2) ]
        [ sin(θ/2)   cos(θ/2) ]
```

```python
def rz_gate(circuit: Circuit, qubit: int, angle: float) -> None
```
Rotation around Z-axis by angle θ.

**Matrix:**
```
RZ(θ) = [ e^(-iθ/2)      0     ]
        [     0      e^(iθ/2)  ]
```

**Example:**
```python
from quantrs2.gates import rx_gate, ry_gate, rz_gate
import numpy as np

circuit = quantrs2.Circuit(3)
rx_gate(circuit, 0, np.pi/2)    # 90° X rotation
ry_gate(circuit, 1, np.pi/4)    # 45° Y rotation
rz_gate(circuit, 2, np.pi)      # 180° Z rotation
```

#### Universal Single-Qubit Gates

```python
def u1_gate(circuit: Circuit, qubit: int, lambda_param: float) -> None
```
Single-parameter gate: diagonal phase gate.

```python
def u2_gate(circuit: Circuit, qubit: int, phi: float, lambda_param: float) -> None
```
Two-parameter gate for creating superposition with phase.

```python
def u3_gate(circuit: Circuit, qubit: int, theta: float, phi: float, lambda_param: float) -> None
```
Three-parameter universal single-qubit gate.

**Matrix:**
```
U3(θ,φ,λ) = [ cos(θ/2)           -e^(iλ)*sin(θ/2)        ]
            [ e^(iφ)*sin(θ/2)     e^(i(φ+λ))*cos(θ/2)    ]
```

### Two-Qubit Gates

#### Controlled Gates

```python
def cnot_gate(circuit: Circuit, control: int, target: int) -> None
```
Controlled-NOT (CNOT) gate: flips target if control is |1⟩.

```python
def controlled_y(circuit: Circuit, control: int, target: int) -> None
```
Controlled-Y gate.

```python
def controlled_z(circuit: Circuit, control: int, target: int) -> None
```
Controlled-Z gate.

```python
def controlled_hadamard(circuit: Circuit, control: int, target: int) -> None
```
Controlled Hadamard gate.

```python
def controlled_phase(circuit: Circuit, control: int, target: int, phase: float) -> None
```
Controlled phase gate with arbitrary phase.

#### Controlled Rotation Gates

```python
def controlled_rx(circuit: Circuit, control: int, target: int, angle: float) -> None
```
Controlled rotation around X-axis.

```python
def controlled_ry(circuit: Circuit, control: int, target: int, angle: float) -> None
```
Controlled rotation around Y-axis.

```python
def controlled_rz(circuit: Circuit, control: int, target: int, angle: float) -> None
```
Controlled rotation around Z-axis.

**Example:**
```python
from quantrs2.gates import cnot_gate, controlled_ry
import numpy as np

circuit = quantrs2.Circuit(4)
circuit.h(0)  # Create superposition
cnot_gate(circuit, 0, 1)  # Entangle qubits 0 and 1
controlled_ry(circuit, 0, 2, np.pi/3)  # Controlled Y rotation
```

#### SWAP Gates

```python
def swap_gate(circuit: Circuit, qubit1: int, qubit2: int) -> None
```
SWAP gate: exchanges states of two qubits.

```python
def iswap_gate(circuit: Circuit, qubit1: int, qubit2: int) -> None
```
iSWAP gate: SWAP with additional phase.

```python
def fredkin_gate(circuit: Circuit, control: int, target1: int, target2: int) -> None
```
Fredkin gate (controlled-SWAP): swaps targets if control is |1⟩.

#### Parametric Two-Qubit Gates

```python
def xx_gate(circuit: Circuit, qubit1: int, qubit2: int, angle: float) -> None
```
XX interaction gate: e^(-i*θ/2 * X⊗X).

```python
def yy_gate(circuit: Circuit, qubit1: int, qubit2: int, angle: float) -> None
```
YY interaction gate: e^(-i*θ/2 * Y⊗Y).

```python
def zz_gate(circuit: Circuit, qubit1: int, qubit2: int, angle: float) -> None
```
ZZ interaction gate: e^(-i*θ/2 * Z⊗Z).

### Multi-Qubit Gates

#### Toffoli Gates

```python
def toffoli_gate(circuit: Circuit, control1: int, control2: int, target: int) -> None
```
Toffoli (CCX) gate: flips target if both controls are |1⟩.

```python
def controlled_controlled_z(circuit: Circuit, control1: int, control2: int, target: int) -> None
```
Controlled-controlled-Z gate.

#### Multi-Controlled Gates

```python
def mcx_gate(circuit: Circuit, controls: List[int], target: int) -> None
```
Multi-controlled X gate with arbitrary number of controls.

```python
def mcy_gate(circuit: Circuit, controls: List[int], target: int) -> None
```
Multi-controlled Y gate.

```python
def mcz_gate(circuit: Circuit, controls: List[int], target: int) -> None
```
Multi-controlled Z gate.

**Example:**
```python
from quantrs2.gates import mcx_gate, mcy_gate

circuit = quantrs2.Circuit(5)
# Multi-controlled X with 3 controls
mcx_gate(circuit, [0, 1, 2], 3)
# Multi-controlled Y with 2 controls
mcy_gate(circuit, [0, 1], 4)
```

## Custom Gate Definition

### Custom Single-Qubit Gates

```python
def custom_single_qubit_gate(circuit: Circuit, qubit: int, matrix: np.ndarray) -> None
```
Apply custom single-qubit gate defined by 2×2 unitary matrix.

**Parameters:**
- `matrix` (np.ndarray): 2×2 complex unitary matrix

**Example:**
```python
import numpy as np
from quantrs2.gates import custom_single_qubit_gate

# Define custom gate matrix
custom_matrix = np.array([
    [1/np.sqrt(2), 1j/np.sqrt(2)],
    [1j/np.sqrt(2), 1/np.sqrt(2)]
], dtype=complex)

circuit = quantrs2.Circuit(2)
custom_single_qubit_gate(circuit, 0, custom_matrix)
```

### Custom Two-Qubit Gates

```python
def custom_two_qubit_gate(circuit: Circuit, qubit1: int, qubit2: int, matrix: np.ndarray) -> None
```
Apply custom two-qubit gate defined by 4×4 unitary matrix.

**Parameters:**
- `matrix` (np.ndarray): 4×4 complex unitary matrix

### Gate Decomposition

```python
def decompose_single_qubit_gate(matrix: np.ndarray) -> Tuple[float, float, float]
```
Decompose arbitrary single-qubit gate into U3 parameters.

**Returns:**
- `Tuple[float, float, float]`: (theta, phi, lambda) parameters for U3 gate

```python
def decompose_two_qubit_gate(matrix: np.ndarray) -> List[GateOperation]
```
Decompose arbitrary two-qubit gate into basic gates.

**Returns:**
- `List[GateOperation]`: Sequence of basic gate operations

## Gate Properties and Analysis

### Gate Matrices

```python
def get_gate_matrix(gate_name: str, *args) -> np.ndarray
```
Get the unitary matrix representation of a gate.

**Parameters:**
- `gate_name` (str): Name of the gate ('x', 'y', 'z', 'h', 'cnot', etc.)
- `*args`: Gate parameters (e.g., rotation angles)

**Returns:**
- `np.ndarray`: Unitary matrix representation

**Example:**
```python
from quantrs2.gates import get_gate_matrix
import numpy as np

# Get standard gate matrices
h_matrix = get_gate_matrix('h')
cnot_matrix = get_gate_matrix('cnot')
rx_matrix = get_gate_matrix('rx', np.pi/2)

print(f"Hadamard matrix:\n{h_matrix}")
```

### Gate Verification

```python
def is_unitary(matrix: np.ndarray, tolerance: float = 1e-10) -> bool
```
Check if a matrix is unitary.

```python
def gate_fidelity(gate1: np.ndarray, gate2: np.ndarray) -> float
```
Calculate fidelity between two quantum gates.

```python
def commutator(gate1: np.ndarray, gate2: np.ndarray) -> np.ndarray
```
Calculate commutator [A, B] = AB - BA of two gates.

```python
def anticommutator(gate1: np.ndarray, gate2: np.ndarray) -> np.ndarray
```
Calculate anticommutator {A, B} = AB + BA of two gates.

### Gate Properties

```python
def gate_eigenvalues(matrix: np.ndarray) -> np.ndarray
```
Get eigenvalues of a gate matrix.

```python
def gate_trace(matrix: np.ndarray) -> complex
```
Calculate trace of a gate matrix.

```python
def gate_determinant(matrix: np.ndarray) -> complex
```
Calculate determinant of a gate matrix.

## Gate Sequences and Circuits

### Gate Sequence Optimization

```python
def optimize_gate_sequence(gates: List[GateOperation]) -> List[GateOperation]
```
Optimize a sequence of gates by combining and canceling.

```python
def cancel_consecutive_gates(gates: List[GateOperation]) -> List[GateOperation]
```
Cancel consecutive identical gates that sum to identity.

```python
def combine_rotations(gates: List[GateOperation]) -> List[GateOperation]
```
Combine consecutive rotation gates on the same qubit.

### Gate Equivalences

```python
def are_gates_equivalent(gate1: np.ndarray, gate2: np.ndarray, tolerance: float = 1e-10) -> bool
```
Check if two gates are equivalent up to global phase.

```python
def find_equivalent_sequence(target_gate: np.ndarray, basis_gates: List[str]) -> List[GateOperation]
```
Find gate sequence equivalent to target gate using only basis gates.

## Special Gate Classes

### Parametric Gates

```python
class ParametricGate:
    """Gate with variable parameters."""
    
    def __init__(self, gate_function: Callable, parameter_names: List[str]):
        self.gate_function = gate_function
        self.parameter_names = parameter_names
    
    def apply(self, circuit: Circuit, qubits: List[int], parameters: Dict[str, float]):
        """Apply gate with specific parameter values."""
        pass

def rx_parametric(parameter_name: str = 'theta') -> ParametricGate:
    """Create parametric RX gate."""
    pass

def ry_parametric(parameter_name: str = 'theta') -> ParametricGate:
    """Create parametric RY gate."""
    pass
```

### Composite Gates

```python
class CompositeGate:
    """Gate composed of multiple basic gates."""
    
    def __init__(self, name: str):
        self.name = name
        self.gates = []
    
    def add_gate(self, gate_function: Callable, qubits: List[int], *args):
        """Add gate to composite."""
        self.gates.append((gate_function, qubits, args))
    
    def apply(self, circuit: Circuit, qubit_mapping: Dict[int, int]):
        """Apply composite gate to circuit."""
        pass

def qft_gate(num_qubits: int) -> CompositeGate:
    """Create Quantum Fourier Transform gate."""
    pass

def grover_oracle(marked_states: List[str]) -> CompositeGate:
    """Create Grover oracle gate."""
    pass
```

## Gate Visualization

### Circuit Diagram Symbols

```python
def get_gate_symbol(gate_name: str) -> str:
    """Get standard symbol for gate in circuit diagrams."""
    symbols = {
        'x': 'X', 'y': 'Y', 'z': 'Z',
        'h': 'H', 's': 'S', 't': 'T',
        'cnot': '⊕', 'swap': '×',
        'ccx': '⊕', 'measure': '📏'
    }
    return symbols.get(gate_name, gate_name.upper())

def gate_color(gate_name: str) -> str:
    """Get standard color for gate visualization."""
    pass
```

### Matrix Visualization

```python
def visualize_gate_matrix(matrix: np.ndarray, title: str = None):
    """Visualize gate matrix as heatmap."""
    pass

def plot_gate_eigenvalues(matrix: np.ndarray):
    """Plot eigenvalues of gate matrix in complex plane."""
    pass
```

## Advanced Gate Operations

### Gate Exponentiation

```python
def gate_power(matrix: np.ndarray, power: float) -> np.ndarray
```
Raise gate to arbitrary power: G^p.

```python
def gate_sqrt(matrix: np.ndarray) -> np.ndarray
```
Calculate square root of gate: G^(1/2).

```python
def gate_log(matrix: np.ndarray) -> np.ndarray
```
Calculate logarithm of gate.

### Gate Simulation

```python
def simulate_gate_noise(ideal_gate: np.ndarray, noise_model: str, **params) -> np.ndarray
```
Simulate noisy version of ideal gate.

```python
def add_gate_error(circuit: Circuit, error_probability: float, error_type: str = 'depolarizing'):
```
Add error to all gates in circuit.

## Examples

### Custom Gate Definition

```python
import numpy as np
from quantrs2.gates import custom_single_qubit_gate, get_gate_matrix

# Define custom √X gate
sqrt_x_matrix = np.array([
    [0.5+0.5j, 0.5-0.5j],
    [0.5-0.5j, 0.5+0.5j]
], dtype=complex)

circuit = quantrs2.Circuit(2)
custom_single_qubit_gate(circuit, 0, sqrt_x_matrix)

# Apply twice to verify it equals X gate
custom_single_qubit_gate(circuit, 0, sqrt_x_matrix)

# Should be equivalent to X gate
circuit.measure_all()
result = circuit.run()
```

### Gate Sequence Analysis

```python
from quantrs2.gates import are_gates_equivalent, get_gate_matrix
import numpy as np

# Check if HZH = X
h_matrix = get_gate_matrix('h')
z_matrix = get_gate_matrix('z')
x_matrix = get_gate_matrix('x')

hzh_matrix = h_matrix @ z_matrix @ h_matrix
equivalent = are_gates_equivalent(hzh_matrix, x_matrix)
print(f"HZH = X: {equivalent}")  # Should be True
```

### Parametric Gate Usage

```python
from quantrs2.gates import ParametricGate, ry_gate
import numpy as np

# Create variational circuit with parametric gates
def create_variational_circuit(parameters):
    circuit = quantrs2.Circuit(2)
    
    # Parametric rotations
    ry_gate(circuit, 0, parameters[0])
    ry_gate(circuit, 1, parameters[1])
    
    # Entangling gate
    circuit.cx(0, 1)
    
    # More parametric rotations
    ry_gate(circuit, 0, parameters[2])
    ry_gate(circuit, 1, parameters[3])
    
    return circuit

# Use with optimization
params = np.random.random(4) * 2 * np.pi
var_circuit = create_variational_circuit(params)
```

---

**See also:**
- [Core API](core.md): Basic circuit operations
- [Algorithms API](algorithms.md): High-level quantum algorithms
- [Optimization API](../user-guide/performance.md): Gate optimization techniques