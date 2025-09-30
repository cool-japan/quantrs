# Changelog

All notable changes to QuantRS2-Py will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-beta.2] - 2025-09-30

### 🎯 Policy Refinement & Documentation Release

QuantRS2-Py v0.1.0-beta.2 focuses on refined SciRS2 integration patterns and improved developer experience through comprehensive documentation.

### Updated

#### 🔧 SciRS2 v0.1.0-beta.3 Integration
- **Refined Integration**: Updated to SciRS2 v0.1.0-beta.3 with unified import patterns
- **Enhanced Distributions**: Improved random number generation with `UnifiedNormal`, `UnifiedBeta`
- **Consistent API**: Standardized SciRS2 usage across all Python bindings

#### 📚 Documentation Improvements
- **Comprehensive Policy Documentation**: Added SCIRS2_INTEGRATION_POLICY.md reference
- **Development Guidelines**: Added CLAUDE.md for AI-assisted development
- **Updated Examples**: Refreshed all examples with refined patterns

### Notes
- **No Breaking Changes**: API remains fully compatible with beta.1
- **Migration**: No code changes required from beta.1
- **Performance**: Maintained exceptional performance with refined patterns

## [0.1.0-beta.1] - 2025-09-21

### 🎉 Beta Release - Production Ready!

QuantRS2-Py v0.1.0-beta.1 represents a major milestone, delivering a comprehensive production-ready quantum computing Python framework with advanced GPU acceleration, cross-platform support, and exceptional performance capabilities.

### Added - Major Features 🚀

#### 🖥️ Enhanced Platform Support
- **macOS Optimization**: Native Apple Silicon and Intel Mac support with optimized performance
- **CUDA/Linux Support**: Full CUDA GPU acceleration on Linux platforms
- **Cross-Platform Compatibility**: Unified codebase supporting Windows, macOS, and Linux

#### 🔧 Complete SciRS2 v0.1.0-beta.2 Integration
- **Deep SciRS2 Integration**: Full integration with Scientific Rust v0.1.0-beta.2 for optimal performance
- **SIMD Operations**: All operations leverage `scirs2_core::simd_ops` with hardware-aware optimization
- **Parallel Computing**: Automatic parallelization via `scirs2_core::parallel_ops`
- **Memory Management**: Advanced memory-efficient algorithms for 30+ qubit simulations

#### 🚀 GPU Acceleration Suite
- **GPU Backend**: Complete GPU backend implementation with CUDA support
- **GPU Linear Algebra**: High-performance GPU-accelerated linear algebra operations
- **Memory Optimization**: Efficient GPU memory management for large quantum states
- **Hardware Detection**: Automatic GPU capability detection and optimization

#### 🛠️ Developer Experience Tools
- **Circuit Optimizer**: Advanced quantum circuit optimization with ZX-calculus
- **Tensor Network Optimization**: High-performance tensor network simulations
- **Performance Profiler**: Comprehensive execution analysis and optimization recommendations
- **Enhanced Testing**: Expanded test coverage with GPU backend validation

#### 🤖 Advanced Quantum Algorithms
- **Autograd Quantum ML**: Automatic differentiation for quantum machine learning
- **Enhanced QGAN**: Improved Quantum Generative Adversarial Networks
- **Quantum CNNs**: Quantum Convolutional Neural Networks implementation
- **QAOA**: Quantum Approximate Optimization Algorithm with MaxCut examples
- **Quantum PCA**: Principal Component Analysis using quantum computing

#### 📦 Production Readiness
- **PyO3 Integration**: Enhanced Python bindings with improved performance
- **Package Structure**: Optimized package organization with proper error handling
- **GPU Dependencies**: New optional `gpu` dependency group for GPU acceleration
- **Comprehensive Examples**: 50+ working examples demonstrating all features

### Improved
- **Package Performance**: Significant performance improvements across all modules
- **Memory Efficiency**: Optimized memory usage patterns for large-scale simulations
- **Error Handling**: Better error messages and graceful degradation
- **Documentation**: Enhanced API documentation with usage examples

### Technical Enhancements
- **SIMD Vectorization**: Hardware-aware vectorized quantum operations
- **Automatic Backend Selection**: Intelligent backend selection based on problem characteristics
- **Platform Detection**: Smart capability detection for optimal performance
- **Resource Estimation**: Advanced complexity and performance analysis

### Breaking Changes
- **API Improvements**: Minor API updates for consistency and performance
- **Migration Guide**: See main project MIGRATION_GUIDE_ALPHA_TO_BETA.md for upgrade instructions

## [0.1.0a5] - 2025-06

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