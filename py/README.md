# QuantRS2-Py: Python Bindings for QuantRS2

[![Crates.io](https://img.shields.io/crates/v/quantrs2-py.svg)](https://crates.io/crates/quantrs2-py)
[![PyPI version](https://badge.fury.io/py/quantrs2.svg)](https://badge.fury.io/py/quantrs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Py provides Python bindings for the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, allowing Python users to access the high-performance Rust implementation with a user-friendly Python API.

## Features

- **Seamless Python Integration**: Easy-to-use Python interface for QuantRS2
- **High Performance**: Leverages Rust's performance while providing Python's usability 
- **Complete Gate Set**: All quantum gates from the core library exposed to Python
- **Simulator Access**: Run circuits on state vector and other simulators
- **GPU Acceleration**: Optional GPU acceleration via feature flag
- **PyO3-Based**: Built using the robust PyO3 framework for Rust-Python interoperability

## Installation

### From PyPI

```bash
pip install quantrs2
```

### From Source (with GPU support)

```bash
pip install git+https://github.com/cool-japan/quantrs.git#subdirectory=py[gpu]
```

## Usage

### Creating a Bell State

```python
import quantrs2 as qr
import numpy as np

# Create a 2-qubit circuit
circuit = qr.PyCircuit(2)

# Build a Bell state
circuit.h(0)
circuit.cnot(0, 1)

# Run the simulation
result = circuit.run()

# Print the probabilities
probs = result.state_probabilities()
for state, prob in probs.items():
    print(f"|{state}⟩: {prob:.6f}")
```

### Using GPU Acceleration

```python
import quantrs2 as qr

# Create a circuit
circuit = qr.PyCircuit(10)  # 10 qubits

# Apply gates
for i in range(10):
    circuit.h(i)

# Run with GPU acceleration if available
try:
    result = circuit.run(use_gpu=True)
    print("GPU simulation successful!")
except ValueError as e:
    print(f"GPU simulation failed: {e}")
    print("Falling back to CPU...")
    result = circuit.run(use_gpu=False)

# Get results
probs = result.probabilities()
```

### Analyzing Results

```python
import quantrs2 as qr
import matplotlib.pyplot as plt

# Create and run a circuit
circuit = qr.PyCircuit(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.x(2)
result = circuit.run()

# Get the state probabilities
probs = result.state_probabilities()

# Plot the results
states = list(probs.keys())
values = list(probs.values())

plt.bar(states, values)
plt.xlabel('Basis State')
plt.ylabel('Probability')
plt.title('Quantum State Probabilities')
plt.show()
```

## API Overview

### Main Classes

- `PyCircuit`: Quantum circuit representation
- `PySimulationResult`: Results from circuit simulation

### Circuit Methods

- `PyCircuit(n_qubits)`: Create a new circuit with specified number of qubits
- `h(qubit)`: Apply Hadamard gate
- `x(qubit)`, `y(qubit)`, `z(qubit)`: Apply Pauli gates
- `s(qubit)`, `sdg(qubit)`: Apply S and S-dagger gates
- `t(qubit)`, `tdg(qubit)`: Apply T and T-dagger gates
- `rx(qubit, theta)`, `ry(qubit, theta)`, `rz(qubit, theta)`: Apply rotation gates
- `cnot(control, target)`: Apply CNOT gate
- `swap(qubit1, qubit2)`: Apply SWAP gate
- `run(use_gpu=False)`: Run the circuit on a simulator

### Result Methods

- `amplitudes()`: Get the state vector amplitudes
- `probabilities()`: Get the measurement probabilities for each basis state
- `state_probabilities()`: Get a dictionary mapping basis states to probabilities

## Limitations

- Current implementation supports circuits with 1, 2, 3, 4, 5, 8, 10, or 16 qubits only
- GPU acceleration requires the `gpu` feature flag during compilation

## Future Plans

See [TODO.md](TODO.md) for planned improvements and features.

## Integration with Python Ecosystem

QuantRS2 Python bindings work well with common Python scientific libraries:

- **NumPy**: Results can be easily converted to NumPy arrays
- **Matplotlib**: Visualization of quantum states and probabilities
- **Pandas**: Analysis of simulation results
- **IPython/Jupyter**: Interactive exploration of quantum circuits

## License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.