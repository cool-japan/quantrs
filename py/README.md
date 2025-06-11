# QuantRS2-Py: Python Bindings for QuantRS2

[![Crates.io](https://img.shields.io/crates/v/quantrs2-py.svg)](https://crates.io/crates/quantrs2-py)
[![PyPI version](https://badge.fury.io/py/quantrs2.svg)](https://badge.fury.io/py/quantrs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Py provides Python bindings for the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, allowing Python users to access the high-performance Rust implementation with a user-friendly Python API.

## Features

### Core Quantum Computing
- **Seamless Python Integration**: Easy-to-use Python interface for QuantRS2
- **High Performance**: Leverages Rust's performance while providing Python's usability 
- **Complete Gate Set**: All quantum gates from the core library exposed to Python
- **Simulator Access**: Run circuits on state vector and other simulators
- **GPU Acceleration**: Optional GPU acceleration via feature flag
- **PyO3-Based**: Built using the robust PyO3 framework for Rust-Python interoperability

### Advanced Features ✨ **NEW in v0.1.0a5**
- **🧠 Quantum Machine Learning**: 
  - Quantum Neural Networks (QNN) with parameter-shift rule gradients
  - Variational Quantum Eigensolver (VQE) with multiple ansätze
  - Hardware-efficient parameterized circuits
- **🛡️ Error Mitigation**: 
  - Zero-Noise Extrapolation (ZNE) with multiple extrapolation methods
  - Circuit folding for noise scaling
  - Observable expectation value calculation
- **🔥 Quantum Annealing**: 
  - QUBO and Ising model optimization
  - Simulated annealing solver
  - Graph embedding for quantum hardware
  - Penalty optimization for constrained problems
- **🎨 Visualization**: Interactive circuit diagrams and state visualization
- **🔐 Cryptography**: Quantum key distribution and digital signatures
- **💰 Finance**: Portfolio optimization and option pricing 

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

## Advanced Usage Examples ✨

### Quantum Machine Learning

#### Quantum Neural Network (QNN)
```python
from quantrs2.ml import QNN
import numpy as np

# Create and train a QNN
qnn = QNN(n_qubits=4, n_layers=3, activation="relu")

# Training data
X_train = np.random.random((100, 4))
y_train = np.random.random((100, 4))

# Train the model
losses = qnn.train(X_train, y_train, epochs=50, learning_rate=0.01)

# Make predictions
predictions = qnn.forward(X_train[:10])
print(f"Predictions shape: {predictions.shape}")
```

#### Variational Quantum Eigensolver (VQE)
```python
from quantrs2.ml import VQE
import numpy as np

# Create VQE instance for ground state finding
vqe = VQE(n_qubits=4, ansatz="hardware_efficient")

# Optimize to find ground state
ground_energy, ground_state = vqe.compute_ground_state()
print(f"Ground state energy: {ground_energy:.6f}")
```

### Error Mitigation

#### Zero-Noise Extrapolation
```python
from quantrs2.mitigation import ZeroNoiseExtrapolation, ZNEConfig, Observable
from quantrs2 import PyCircuit

# Configure ZNE
config = ZNEConfig(
    scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
    extrapolation_method="richardson"
)
zne = ZeroNoiseExtrapolation(config)

# Create noisy circuit
circuit = PyCircuit(2)
circuit.h(0)
circuit.cnot(0, 1)

# Define observable
observable = Observable.z(0)

# Mitigate errors
result = zne.mitigate_observable(circuit, observable)
print(f"Mitigated value: {result.mitigated_value:.6f} ± {result.error_estimate:.6f}")
```

### Quantum Annealing

#### QUBO Optimization
```python
from quantrs2.anneal import QuboModel, PenaltyOptimizer

# Create QUBO model
qubo = QuboModel(n_vars=4)
qubo.add_linear(0, 1.0)
qubo.add_linear(1, -2.0)
qubo.add_quadratic(0, 1, 3.0)
qubo.add_quadratic(1, 2, -1.0)

# Solve using simulated annealing
solution, energy = qubo.solve_simulated_annealing(max_iter=1000)
print(f"Best solution: {solution}")
print(f"Energy: {energy:.6f}")

# Convert to Ising model
ising = qubo.to_ising()
print(f"Ising model with {ising.n_spins} spins")
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

## What's New in v0.1.0a5

- **🧠 Advanced Machine Learning**: Full QNN implementation with gradient-based training
- **🛡️ Error Mitigation Suite**: ZNE, circuit folding, and observable measurements  
- **🔥 Quantum Annealing**: Complete QUBO/Ising optimization framework
- **⚡ Performance**: Enhanced algorithms with better convergence
- **📚 Documentation**: Comprehensive examples and API references

## API Reference

### Core Classes
- `PyCircuit`: Main circuit building and execution
- `PySimulationResult`: Results from quantum simulations

### Machine Learning (`quantrs2.ml`)
- `QNN`: Quantum Neural Networks
- `VQE`: Variational Quantum Eigensolver
- `HEPClassifier`: High-Energy Physics classifier
- `QuantumGAN`: Quantum Generative Adversarial Networks

### Error Mitigation (`quantrs2.mitigation`)
- `ZeroNoiseExtrapolation`: ZNE implementation
- `Observable`: Quantum observables
- `CircuitFolding`: Noise scaling utilities

### Quantum Annealing (`quantrs2.anneal`)
- `QuboModel`: QUBO problem formulation
- `IsingModel`: Ising model optimization
- `PenaltyOptimizer`: Constrained optimization

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT/Apache-2.0 dual license.

## Citation

If you use QuantRS2 in your research, please cite:

```bibtex
@software{quantrs2,
  title = {QuantRS2: High-Performance Quantum Computing Framework},
  author = {Team KitaSan},
  year = {2024},
  url = {https://github.com/cool-japan/quantrs}
}
