# QuantRS2 Core Python Bindings

Python bindings for the QuantRS2 quantum computing framework.

## Installation

```bash
pip install quantrs2-core-extension
```

## Quick Start

```python
import quantrs2.core as qrs

# Create qubits
q0 = qrs.QubitId(0)
q1 = qrs.QubitId(1)

# Create quantum gates
h_gate = qrs.create_hadamard_gate(0)
cnot_gate = qrs.create_cnot_gate(0, 1)

# Create variational circuits
circuit = qrs.VariationalCircuit(4)
circuit.add_rotation_layer("x")
circuit.add_entangling_layer()
```

## Features

- Quantum gate implementations
- Quantum decomposition algorithms
- Variational quantum circuits
- Hardware-specific compilation
- Interactive Jupyter visualization

## Documentation

For full documentation, visit: https://github.com/cool-japan/quantrs

## License

MIT OR Apache-2.0