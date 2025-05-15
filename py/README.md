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
- **Machine Learning Integration**: Quantum Neural Networks and Variational Algorithms
- **Domain-specific ML Applications**: Tools for high-energy physics, cryptography, and more
- **Quantum Circuit Visualization**: Built-in tools for visualizing quantum circuits 

## Installation

### From PyPI

```bash
pip install quantrs2
```

### From Source (with GPU support)

```bash
pip install git+https://github.com/cool-japan/quantrs.git#subdirectory=py[gpu]
```

### With Machine Learning Support

```bash
pip install quantrs2[ml]
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

### Using Quantum Neural Networks for Classification

```python
import quantrs2 as qr
import numpy as np
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features
y = iris.target

# Normalize features
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Create a QNN with 4 qubits
qnn = qr.QNN(n_qubits=4, n_layers=2)

# Process a sample
sample = X[0]
output = qnn.forward(sample.reshape(1, -1))
predicted_class = np.argmax(output)
print(f"Predicted class: {predicted_class}")
```

### Applying Variational Quantum Algorithms

```python
import quantrs2 as qr
import numpy as np

# Create a VQE instance for a 2-qubit system
vqe = qr.VQE(n_qubits=2)

# Run optimization to find ground state energy
final_energy, optimal_params = vqe.optimize(max_iterations=100)
print(f"Optimal energy: {final_energy:.6f}")
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

# Plot the results using quantrs2's visualization tools
viz = qr.visualize_probabilities(result)
viz.plot()
plt.show()
```

## API Overview

### Main Classes

- `PyCircuit`, `Circuit`: Quantum circuit representation
- `PySimulationResult`, `SimulationResult`: Results from circuit simulation
- `QNN`: Quantum Neural Network for machine learning
- `VQE`: Variational Quantum Eigensolver for optimization problems
- `HEPClassifier`: Specialized classifier for high-energy physics data
- `QuantumGAN`: Quantum Generative Adversarial Network

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
- `draw()`, `draw_html()`: Visualize the circuit

### Result Methods

- `amplitudes()`: Get the state vector amplitudes
- `probabilities()`: Get the measurement probabilities for each basis state
- `state_probabilities()`: Get a dictionary mapping basis states to probabilities
- `expectation_value(operator)`: Calculate expectation value of a Pauli operator

### Utility Functions

- `create_bell_state()`, `create_ghz_state()`: Predefined quantum states
- `visualize_circuit(circuit)`: Create a circuit visualizer
- `visualize_probabilities(result)`: Visualize measurement probabilities

## Limitations

- Current implementation supports circuits with 1, 2, 3, 4, 5, 8, 10, or 16 qubits only
- GPU acceleration requires the `gpu` feature flag during compilation
- Machine learning features may require additional dependencies (install with `pip install quantrs2[ml]`)

## Future Plans

See [TODO.md](TODO.md) for planned improvements and features.

## Integration with Python Ecosystem

QuantRS2 Python bindings work well with common Python scientific libraries:

- **NumPy**: Results can be easily converted to NumPy arrays
- **Matplotlib**: Visualization of quantum states and probabilities
- **Pandas**: Analysis of simulation results
- **scikit-learn**: Integration with classical ML pipelines
- **IPython/Jupyter**: Interactive exploration of quantum circuits

## License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.