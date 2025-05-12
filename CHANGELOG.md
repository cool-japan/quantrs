# Changelog

All notable changes to Quantrs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.1] - 2025-05-14

### Added

- Initial alpha release with the following crates:
  - `quantrs-core`: Core types and traits
  - `quantrs-circuit`: Circuit building functionality
  - `quantrs-sim`: Quantum simulators
  - `quantrs-anneal`: Quantum annealing support
  - `quantrs-device`: Hardware connectivity
  - `quantrs-py`: Python bindings

#### Core Features
- Type-safe quantum circuits with const generics for compile-time qubit count verification
- Comprehensive gate set including standard gates and extensions:
  - Hadamard, Pauli (X, Y, Z)
  - Phase (S, T) and their dagger versions (S†, T†)
  - Square root of X (√X) and its dagger
  - Controlled variants of all gates
  - Rotation gates (Rx, Ry, Rz)
  - Multi-qubit operations (CNOT, SWAP, Toffoli)
- Circuit builder with fluent interface

#### Simulation
- High-performance state vector simulator optimized for 30+ qubits
- GPU-accelerated simulation using WGPU
- SIMD-optimized operations for CPU simulation
- Memory-efficient tensor network simulator
- Specialized contraction path optimization for common circuits:
  - Quantum Fourier Transform (QFT)
  - Quantum Approximate Optimization Algorithm (QAOA)
  - Linear and star-shaped circuit patterns
- Noise models:
  - Bit flip channel
  - Phase flip channel
  - Depolarizing channel
  - Amplitude damping
  - Phase damping
  - IBM-specific T1/T2 relaxation

#### Quantum Error Correction
- Bit flip code
- Phase flip code
- Shor code for general qubit errors
- 5-qubit perfect code

#### Hardware Integration
- IBM Quantum API client:
  - Authentication and device management
  - Circuit transpilation for specific hardware
  - Job submission and result processing
  - Support for IBM Heron and Condor QPUs
- D-Wave quantum annealing interface

#### Quantum Algorithms
- Grover's search algorithm
- Quantum Fourier Transform (QFT)
- Quantum Phase Estimation (QPE)
- Simplified Shor's algorithm
- Variational algorithms (QAOA)

#### Python Bindings
- Complete Python API via PyO3
- Support for all core features
- GPU acceleration in Python

### Examples
- Bell state creation
- Quantum teleportation
- Extended gate set demonstration
- Noisy simulation examples
- Quantum error correction demonstrations
- Tensor network optimization benchmarks
- Quantum annealing problem formulations:
  - Maximum Cut (MaxCut)
  - Graph Coloring
  - Traveling Salesman Problem (TSP)
  - 3-Rooks problem

### Documentation
- API documentation
- User guides
- Algorithm-specific tutorials
- Interactive learning resources

### Fixed
- Compatibility issues with various operating systems
- Performance bottlenecks in simulator implementations
- Memory usage optimizations for large qubit simulations