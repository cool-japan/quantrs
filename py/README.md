# QuantRS2-Py: Python Bindings for QuantRS2

[![Crates.io](https://img.shields.io/crates/v/quantrs2-py.svg)](https://crates.io/crates/quantrs2-py)
[![PyPI version](https://badge.fury.io/py/quantrs2.svg)](https://badge.fury.io/py/quantrs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Py provides Python bindings for the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, allowing Python users to access the high-performance Rust implementation with a user-friendly Python API.

## Version 0.1.0-rc.2 🎉

**Production-Ready Beta Release!** This release benefits from refined [SciRS2](https://github.com/cool-japan/scirs2) v0.1.0-rc.2 integration and comprehensive platform support:

### 🖥️ Platform Support
- **macOS Optimization**: Native Apple Silicon and Intel Mac support with optimized performance
- **CUDA/Linux Support**: Full CUDA GPU acceleration on Linux platforms
- **Cross-Platform Compatibility**: Unified codebase supporting Windows, macOS, and Linux

### 🚀 SciRS2 Integration & Performance
- **Enhanced Performance**: SciRS2's parallel operations with automatic optimization
- **SIMD Acceleration**: Hardware-aware vectorized quantum operations
- **GPU Computing**: Complete GPU backend with CUDA support and memory optimization
- **Memory Management**: Advanced algorithms for 30+ qubit simulations
- **Automatic Backend Selection**: Intelligent selection based on problem characteristics

## Features

### Core Quantum Computing
- **Seamless Python Integration**: Easy-to-use Python interface for QuantRS2
- **High Performance**: Leverages Rust's performance while providing Python's usability 
- **Complete Gate Set**: All quantum gates from the core library exposed to Python
- **Simulator Access**: Run circuits on state vector and other simulators
- **GPU Acceleration**: Optional GPU acceleration via feature flag
- **PyO3-Based**: Built using the robust PyO3 framework for Rust-Python interoperability

### Advanced Features

#### 🤖 Quantum Machine Learning Suite
- **Autograd Quantum ML**: Automatic differentiation for quantum machine learning
- **Enhanced QGANs**: Improved Quantum Generative Adversarial Networks
- **Quantum CNNs**: Quantum Convolutional Neural Networks implementation
- **QNN Training**: Parameter-shift rule gradients with adaptive learning rates
- **VQE**: Multi-ansatz support with hardware-efficient circuits
- **QAOA**: Quantum Approximate Optimization Algorithm with MaxCut examples
- **Quantum PCA**: Principal Component Analysis using quantum computing

#### 🛠️ Developer Experience Tools
- **Circuit Optimizer**: Advanced quantum circuit optimization with ZX-calculus
- **Tensor Network Optimization**: High-performance tensor network simulations
- **Performance Profiler**: Comprehensive execution analysis and optimization recommendations
- **Enhanced Testing**: Expanded test coverage with GPU backend validation
- **Resource Estimator**: Advanced complexity and performance analysis

#### 🏭 Production Features
- **Dynamic Qubit Allocation**: Runtime resource management with efficient memory usage
- **Hardware Backend Integration**: Support for IBM Quantum, Google Quantum AI, and AWS Braket
- **Error Mitigation**: Zero-noise extrapolation and other mitigation techniques
- **Quantum Annealing**: QUBO/Ising model optimization framework
- **Cryptography Protocols**: BB84, E91, and quantum signature implementations
- **Comprehensive Examples**: 50+ working examples demonstrating all features

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

### Advanced GPU Linear Algebra

```python
from quantrs2.gpu import GPUBackend, GPULinearAlgebra

# Initialize GPU backend
gpu_backend = GPUBackend()

# Create GPU-accelerated linear algebra operations
gpu_linalg = GPULinearAlgebra(gpu_backend)

# Perform high-performance quantum state operations
large_state = qr.create_quantum_state(25)  # 25-qubit state
optimized_state = gpu_linalg.optimize_state(large_state)

# GPU-accelerated tensor network contractions
tensor_result = gpu_linalg.contract_tensor_network(quantum_circuit)
```

### Tensor Network Optimization

```python
from quantrs2.tensor_networks import TensorNetworkOptimizer

# Create and optimize tensor networks
optimizer = TensorNetworkOptimizer()

# Build a complex quantum circuit
circuit = qr.PyCircuit(30)
# Add many gates...

# Optimize using tensor network techniques
optimized_circuit = optimizer.optimize(circuit)
result = optimized_circuit.run()
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

QuantRS2-Py v0.1.0-rc.2 delivers exceptional performance for production quantum computing:

### Simulation Capabilities
- **Large-Scale Simulation**: Efficiently simulates 30+ qubits on standard hardware
- **GPU Acceleration**: Complete GPU backend with CUDA support for massive speedups
- **Memory Optimization**: Advanced SciRS2-powered algorithms for efficient memory usage
- **SIMD Vectorization**: Hardware-aware vectorized operations on all platforms

### Platform Optimization
- **macOS Native**: Optimized for Apple Silicon and Intel Macs
- **CUDA/Linux**: Full GPU acceleration on Linux with CUDA support
- **Cross-Platform**: Consistent performance across Windows, macOS, and Linux
- **Automatic Detection**: Smart hardware capability detection and optimization

### Advanced Features
- **Parallel Execution**: Automatic parallelization via SciRS2 parallel operations
- **Tensor Networks**: High-performance tensor network contractions
- **Circuit Optimization**: Automatic quantum circuit optimization
- **Backend Selection**: Intelligent backend selection based on problem characteristics

## Requirements

### Basic Requirements
- Python 3.8 or higher
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0 (for visualization)
- IPython >= 7.0.0 (for interactive features)

### Optional Dependencies
- **GPU Support**: CUDA toolkit 11.0+ for GPU acceleration on Linux
- **Machine Learning**: scikit-learn >= 1.0.0, scipy >= 1.7.0 (install with `pip install quantrs2[ml]`)
- **Development**: pytest, black, flake8 (install with `pip install quantrs2[dev]`)

### Platform Specific
- **macOS**: Optimized for macOS 10.15+ (both Intel and Apple Silicon)
- **Linux**: CUDA support requires compatible NVIDIA drivers
- **Windows**: Full feature support with optional GPU acceleration

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under either:

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.