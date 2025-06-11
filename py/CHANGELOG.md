# Changelog

All notable changes to QuantRS2-Py will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a4] - 2025-06

### Added - Major Feature Release 🚀

#### 🧠 Advanced Quantum Machine Learning
- **Quantum Neural Networks (QNN)**: Complete implementation with parameter-shift rule gradients
- **Variational Quantum Eigensolver (VQE)**: Multi-ansatz support with hardware-efficient circuits
- **Training Algorithms**: Gradient-based optimization with adaptive learning rates
- **Batch Processing**: Efficient handling of multiple training samples
- **Multiple Activation Functions**: ReLU, tanh, sigmoid support

#### 🛡️ Error Mitigation Suite
- **Zero-Noise Extrapolation (ZNE)**: Complete implementation with multiple extrapolation methods
  - Richardson extrapolation (linear fit)
  - Exponential extrapolation
  - Polynomial extrapolation
- **Circuit Folding**: Global and local noise scaling techniques
- **Observable Framework**: Pauli operator expectation value calculations
- **Statistical Analysis**: Error estimation and fit quality metrics

#### 🔥 Quantum Annealing Framework
- **QUBO Model**: Quadratic Unconstrained Binary Optimization with energy calculation
- **Ising Model**: Complete Ising spin system implementation
- **Bidirectional Conversion**: Seamless QUBO ↔ Ising model transformation
- **Simulated Annealing**: Classical optimization solver for quantum problems
- **Penalty Optimization**: Constrained problem handling with penalty terms
- **Graph Embedding**: Chimera topology support for quantum annealer hardware

#### 📚 Enhanced Documentation
- **Comprehensive README**: Detailed usage examples for all new features
- **API Documentation**: Complete class and method documentation
- **Code Examples**: Working examples for ML, error mitigation, and annealing
- **Installation Guide**: Updated with new feature dependencies

### Improved
- **Package Structure**: Enhanced module organization with proper fallbacks
- **Error Handling**: Better error messages and graceful degradation
- **Performance**: Optimized algorithms for better convergence
- **Testing**: Comprehensive test coverage for all new features

### Technical Details
- **Parameter-Shift Rule**: Accurate gradient computation for QNN training
- **Hardware-Efficient Ansätze**: Optimized circuit layouts for real quantum hardware
- **Noise Modeling**: Realistic noise simulation for error mitigation testing
- **Energy Landscapes**: Proper QUBO/Ising energy function implementations

## [0.1.0a3] - 2025-05

### Added
- Basic quantum circuit functionality
- GPU acceleration support
- Initial ML framework
- Visualization tools

### Fixed
- Package installation issues
- Import path standardization

## [0.1.0a2] - 2025-05

### Added
- Core quantum gates
- Circuit simulation
- Python bindings

## [0.1.0a1] - 2025-05

### Added
- Initial alpha release
- Basic circuit building
- PyO3 integration