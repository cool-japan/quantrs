# QuantRS2-Py: Python Bindings for QuantRS2

[![Crates.io](https://img.shields.io/crates/v/quantrs2-py.svg)](https://crates.io/crates/quantrs2-py)
[![PyPI version](https://badge.fury.io/py/quantrs2.svg)](https://badge.fury.io/py/quantrs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Py provides Python bindings for the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, allowing Python users to access the high-performance Rust implementation with a user-friendly Python API.

## Version 0.1.0-beta.1

This beta release benefits from the deep [SciRS2](https://github.com/cool-japan/scirs2) v0.1.0-alpha.5 integration:
- Enhanced performance through SciRS2's parallel operations
- SIMD-accelerated quantum computations under the hood
- Memory-efficient handling of large quantum circuits
- Automatic optimization selection for best performance

## Features

### Core Quantum Computing
- **Seamless Python Integration**: Easy-to-use Python interface for QuantRS2
- **High Performance**: Leverages Rust's performance while providing Python's usability 
- **Complete Gate Set**: All quantum gates from the core library exposed to Python
- **Simulator Access**: Run circuits on state vector and other simulators
- **GPU Acceleration**: Optional GPU acceleration via feature flag
- **PyO3-Based**: Built using the robust PyO3 framework for Rust-Python interoperability

### Advanced Features

- **Quantum Machine Learning**: QNN, VQE, QAOA, and quantum classifiers
- **Dynamic Qubit Allocation**: Runtime resource management with efficient memory usage
- **Hardware Backend Integration**: Support for IBM Quantum, Google Quantum AI, and AWS Braket
- **Error Mitigation**: Zero-noise extrapolation and other mitigation techniques
- **Quantum Annealing**: QUBO/Ising model optimization framework
- **Cryptography Protocols**: BB84, E91, and quantum signature implementations
- **Development Tools**: Interactive circuit builders and debugging utilities

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

## Advanced Usage Examples

### Quantum Machine Learning

#### Quantum Neural Network (QNN)
```python
from quantrs2.ml import QNN

# Create a QNN with 4 qubits and 2 layers
qnn = QNN(n_qubits=4, n_layers=2)

# Train on quantum data
qnn.fit(X_train, y_train, epochs=100)

# Make predictions
predictions = qnn.predict(X_test)
```

#### Variational Quantum Eigensolver (VQE)
```python
from quantrs2.algorithms import VQE
from quantrs2.optimizers import COBYLA

# Define a Hamiltonian
hamiltonian = qr.Hamiltonian.from_string("ZZ + 0.5*XI + 0.5*IX")

# Create VQE instance
vqe = VQE(hamiltonian, ansatz='ry', optimizer=COBYLA())

# Find ground state
result = vqe.run()
print(f"Ground state energy: {result.eigenvalue}")
```

### Hardware Integration

```python
from quantrs2.hardware import IBMQuantumBackend

# Connect to IBM Quantum
backend = IBMQuantumBackend(api_token="your_token")

# Create and execute circuit
circuit = qr.PyCircuit(5)
circuit.h(0)
circuit.cnot(0, 1)

# Execute on real hardware
job = backend.execute(circuit, shots=1024)
result = job.result()
```

### Error Mitigation

```python
from quantrs2.mitigation import ZeroNoiseExtrapolation

# Create a noisy circuit
circuit = qr.PyCircuit(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)

# Apply zero-noise extrapolation
zne = ZeroNoiseExtrapolation(noise_factors=[1, 3, 5])
mitigated_result = zne.run(circuit)
```

### Quantum Annealing

```python
from quantrs2.anneal import QuboModel

# Define a QUBO problem
Q = {
    (0, 0): -1,
    (1, 1): -1,
    (0, 1): 2
}

# Create and solve
model = QuboModel(Q)
solution = model.solve(sampler='simulated_annealing')
print(f"Optimal solution: {solution.best_sample}")
```

### GPU Acceleration

```python
# Enable GPU acceleration for large circuits
circuit = qr.PyCircuit(20)
# Build your circuit...

# Run with GPU acceleration
result = circuit.run(use_gpu=True)

# Alternatively, check GPU availability
if qr.is_gpu_available():
    result = circuit.run(use_gpu=True)
else:
    result = circuit.run(use_gpu=False)

# Get results
probs = result.probabilities()
```

## API Reference

### Core Classes
- `PyCircuit`: Main circuit building and execution
- `PySimulationResult`: Results from quantum simulations

### Module Structure

#### Machine Learning (`quantrs2.ml`)
- `QNN`: Quantum Neural Networks with gradient computation
- `VQE`: Variational Quantum Eigensolver with multiple ansätze
- `QuantumGAN`: Quantum Generative Adversarial Networks
- `HEPClassifier`: High-Energy Physics quantum classifier

#### Dynamic Allocation (`quantrs2.dynamic_allocation`)
- `QubitAllocator`: Runtime qubit resource management
- `DynamicCircuit`: Thread-safe dynamic circuit construction
- `AllocationStrategy`: Multiple allocation optimization strategies

#### Advanced Algorithms (`quantrs2.advanced_algorithms`)
- `AdvancedVQE`: Enhanced VQE with multiple optimization methods
- `EnhancedQAOA`: Advanced QAOA with sophisticated optimization
- `QuantumWalk`: Comprehensive quantum walk implementations
- `QuantumErrorCorrection`: Error correction protocol suite

#### Hardware Backends (`quantrs2.hardware_backends`)
- `HardwareBackendManager`: Multi-provider backend management
- `IBMQuantumBackend`: IBM Quantum integration
- `GoogleQuantumBackend`: Google Quantum AI integration
- `AWSBraketBackend`: AWS Braket integration

#### Enhanced Compatibility
- `enhanced_qiskit_compatibility`: Advanced Qiskit integration
- `enhanced_pennylane_plugin`: Comprehensive PennyLane integration

#### Error Mitigation (`quantrs2.mitigation`)
- `ZeroNoiseExtrapolation`: Advanced ZNE implementation
- `Observable`: Quantum observables with enhanced measurement
- `CircuitFolding`: Sophisticated noise scaling utilities

#### Quantum Annealing (`quantrs2.anneal`)
- `QuboModel`: Advanced QUBO problem formulation
- `IsingModel`: Enhanced Ising model optimization
- `PenaltyOptimizer`: Sophisticated constrained optimization

## Performance

QuantRS2-Py is designed for high performance quantum simulation:

- Efficiently simulates up to 30+ qubits on standard hardware
- GPU acceleration available for large circuits
- Optimized memory usage through Rust's zero-cost abstractions
- Parallel execution capabilities
- Automatic circuit optimization

## Requirements

- Python 3.8 or higher
- NumPy
- Optional: CUDA toolkit for GPU support

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under either:

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.