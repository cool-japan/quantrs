# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-01-21

### Fixed
- **Device crate**: Added missing `#[cfg(feature = "photonic")]` guard on photonic module re-exports
- **Cross-platform benchmarking**: Fixed conditional compilation for `aws`, `azure`, and `ibm` client imports and struct fields
- **Feature gating**: Improved conditional compilation to avoid compilation errors when cloud provider features are disabled

### Changed
- All workspace crate versions bumped from 0.1.0 to 0.1.1
- Updated workspace dependencies to use version 0.1.1

---

## [0.1.0] - 2026-01-20

### Added

#### Core Framework
- **QuantRS2 Quantum Computing Framework**: Complete modular quantum computing toolkit
- **quantrs2-core**: Core types, traits, and abstractions for quantum computing
- **quantrs2-circuit**: Quantum circuit representation with DSL and gate library
- **quantrs2-sim**: High-performance quantum simulators (state-vector, tensor-network, stabilizer)
- **quantrs2-device**: Remote quantum hardware integration (IBM Quantum, Azure Quantum, AWS Braket)
- **quantrs2-ml**: Quantum machine learning with QNNs, QGANs, and HEP classifiers
- **quantrs2-anneal**: Quantum annealing support with D-Wave integration
- **quantrs2-tytan**: High-level quantum annealing library
- **quantrs2-symengine-pure**: Pure Rust symbolic mathematics engine (100% Rust, no C++ dependencies)
- **quantrs2-py**: Python bindings via PyO3 for seamless Python integration

#### SciRS2 Integration
- Full integration with SciRS2 ecosystem (v0.1.2) for scientific computing
- Unified array operations via `scirs2-core::ndarray`
- Unified random number generation via `scirs2-core::random`
- Complex number support via `scirs2_core::{Complex64, Complex32}`
- SIMD-accelerated quantum operations via `scirs2-core::simd_ops`
- Parallel quantum circuit execution via `scirs2-core::parallel_ops`
- GPU acceleration support via `scirs2-core::gpu`

#### Quantum Algorithms
- **Grover's Algorithm**: Quantum search with amplitude amplification
- **Quantum Fourier Transform (QFT)**: Foundation for quantum algorithms
- **Variational Quantum Eigensolver (VQE)**: Quantum chemistry and optimization
- **Quantum Approximate Optimization Algorithm (QAOA)**: Combinatorial optimization
- **Shor's Algorithm**: Integer factorization (simulation)
- **Quantum Phase Estimation (QPE)**: Eigenvalue estimation
- **Quantum Machine Learning**: QNN, QGAN, quantum reservoirs, HEP classifiers

#### Quantum Simulators
- **State Vector Simulator**: Up to 30+ qubits with optimized complex arithmetic
- **Stabilizer Simulator**: Up to 50+ qubits using stabilizer formalism
- **Tensor Network Simulator**: Efficient simulation for circuits with limited entanglement
- **Density Matrix Simulator**: Mixed state and open quantum system support
- **Quantum Reservoir Computing**: Novel ML approach with quantum dynamics

#### Hardware Integration
- IBM Quantum platform integration with Qiskit compatibility
- Azure Quantum integration
- AWS Braket integration
- D-Wave quantum annealer support
- Error mitigation and measurement optimization

#### Performance Features
- SIMD vectorization for quantum gate operations
- Multi-threaded parallel execution for independent operations
- GPU acceleration for large-scale quantum simulation
- Sparse matrix representations for memory efficiency
- Adaptive chunking for tensor network contractions

#### Documentation
- Comprehensive API documentation with rustdoc
- Quantum algorithm examples and tutorials
- Integration guides for SciRS2 ecosystem
- Python binding examples
- Performance benchmarking suite

### Changed
- Migration from SymEngine C++ bindings to pure Rust implementation (`quantrs2-symengine-pure`)
- Unified dependency management via workspace inheritance
- Optimized memory layout for quantum states
- Enhanced error handling with detailed quantum-specific error types

### Fixed
- Rustdoc HTML tag warnings in symbolic mathematics module
- Clippy warnings across all workspace crates
- Feature flag dependencies for optional GPU and CUDA support
- Documentation generation for docs.rs

### Compatibility

#### Target Frameworks (99%+ Compatibility)
- **Stim**: Stabilizer circuit simulation (99%+ compatibility)
- **cuQuantum**: NVIDIA GPU quantum simulation (95%+ compatibility)
- **TorchQuantum**: PyTorch quantum ML integration (99%+ compatibility)
- **IBM Qiskit**: IBM quantum platform (90%+ compatibility)
- **Google Cirq**: Google quantum platform (90%+ compatibility)
- **PennyLane**: Quantum ML framework (85%+ compatibility)

#### Pure Rust Policy
- **100% Pure Rust** default features (no C/C++/Fortran dependencies)
- Optional C/C++ dependencies feature-gated (CUDA, MKL)
- Full compliance with COOLJAPAN Pure Rust Policy

#### SciRS2 Ecosystem
- **SciRS2**: v0.1.2 (scientific computing core)
- **NumRS2**: v0.1.2 (numerical computing)
- **OptiRS**: v0.1.0 (ML optimization algorithms)
- **OxiBLAS**: Pure Rust BLAS implementation
- **Oxicode**: Pure Rust binary encoding

### Security
- Memory-safe quantum state management via Rust ownership
- No unsafe code in default features
- Dependency audit passing
- Secure random number generation for quantum measurements

### Performance
- State vector simulation: 30+ qubits
- Stabilizer simulation: 50+ qubits
- Tensor network simulation: 50+ qubits (circuit-dependent)
- SIMD acceleration: 2-4x speedup on supported platforms
- GPU acceleration: 10-100x speedup for large circuits (optional)

### Platform Support
- Linux (x86_64, aarch64)
- macOS (x86_64, Apple Silicon)
- Windows (x86_64)
- WebAssembly (wasm32)

### License
- Dual licensed under MIT OR Apache-2.0

### Authors
- COOLJAPAN OU (Team Kitasan)

### Repository
- <https://github.com/cool-japan/quantrs>

### Documentation
- API Docs: <https://docs.rs/quantrs2>
- Examples: See `examples/` directory
- Integration Guide: See `SCIRS2_INTEGRATION_POLICY.md`

---

## [Unreleased]

No unreleased changes yet.

---

[0.1.1]: https://github.com/cool-japan/quantrs/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/quantrs/releases/tag/v0.1.0
