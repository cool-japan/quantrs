# Changelog

All notable changes to the QuantRS2 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.5] - 2025-06-17

### Added
- **ZX-Calculus Optimization** - Graph-based quantum circuit optimization
  - Spider fusion and identity removal rules
  - Hadamard edge simplification
  - Phase gadget optimization
  - Circuit gate count reduction capabilities

- **GPU Acceleration** - Specialized kernels for quantum gates
  - CUDA tensor core support
  - WebGPU compute shaders for cross-platform compatibility
  - SIMD dispatch with AVX-512/AVX2/SSE4 support
  - Specialized kernels for quantum ML operations

- **Quantum Approximate Optimization Algorithm (QAOA)**
  - Support for MaxCut and TSP problems
  - Customizable mixer Hamiltonians (X-mixer, XY-mixer, custom)
  - Optimization with gradient descent and BFGS
  - Performance benchmarking capabilities

- **Quantum Machine Learning for NLP**
  - Quantum attention mechanisms
  - Quantum word embeddings with rotational encoding
  - Positional encoding using quantum phase
  - Integration with variational quantum circuits

- **Gate Compilation Caching**
  - Persistent storage with memory-mapped files
  - LRU eviction policy with configurable size limits
  - Zstd compression for disk storage
  - Async write queue for non-blocking persistence
  - Benchmarking suite

- **Adaptive SIMD Dispatch**
  - Runtime CPU feature detection
  - Support for AVX-512, AVX2, and SSE4
  - Automatic selection of optimal implementation
  - Performance monitoring and optimization

### Updated
- **Module Integration** - Complete integration across all QuantRS2 modules
  - GPU acceleration for simulation module
  - Hardware translation for device module
  - Quantum-classical hybrid learning for ML module
  - Universal gate compilation with cross-platform optimization

## [0.1.0-alpha.4] - 2025-06-11

### Added
- Code quality improvements with zero compiler warnings
- ML capabilities including continual learning and AutoML
- Device orchestration and cloud management
- Quantum error correction with adaptive algorithms
- Quantum annealing with hybrid solvers

## [0.1.0-alpha.3] - 2025-06-05

### Added
- Advanced gate technologies
- Holonomic quantum computing with non-Abelian geometric phases
- Post-quantum cryptography primitives
- High-fidelity gate synthesis with quantum optimal control

## [0.1.0-alpha.2] - 2025-05-15

### Added
- Traditional quantum gates
- Quantum circuit simulation
- Device connectivity
- Quantum ML foundation

## [0.1.0-alpha.1] - 2025-05-13

### Added
- Initial release of QuantRS2 framework
- Core quantum computing abstractions
- Basic gate operations
- State vector simulator