# QuantRS2-Core: Advanced Quantum Computing Foundation

[![Crates.io](https://img.shields.io/crates/v/quantrs2-core.svg)](https://crates.io/crates/quantrs2-core)
[![Documentation](https://docs.rs/quantrs2-core/badge.svg)](https://docs.rs/quantrs2-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Core is the foundational library of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing a comprehensive suite of quantum computing primitives, algorithms, and optimizations that power the entire ecosystem.

## Version 0.1.1 🎉 PRODUCTION READY!

**✅ Core Module - Policy Refinement Release**

This release focuses on comprehensive documentation, refined SciRS2 integration patterns, and improved developer experience.

### Release Highlights ✅
- **🔧 Complete SciRS2 v0.1.0 Integration**: Enhanced scientific computing acceleration with unified patterns
- **⚡ Advanced SIMD Operations**: Hardware-optimized vectorized quantum operations via `scirs2_core::simd_ops`
- **🔄 Unified Parallel Processing**: Automatic parallelization via `scirs2_core::parallel_ops`
- **🎯 Intelligent Platform Detection**: Automatic CPU/GPU capability detection and optimization
- **🛠️ Developer Experience Tools**: Complete suite of debugging, profiling, and optimization tools
- **🤖 AutoOptimizer**: Intelligent automatic backend selection for optimal performance
- **💾 Advanced Memory Management**: Memory-efficient algorithms for large-scale quantum computing
- **🎮 GPU Acceleration**: Full GPU support with Metal backend (macOS) and cross-platform compatibility

### Production Features ✅
- 30+ qubit simulation capabilities
- Comprehensive gate decomposition algorithms
- Advanced error correction and fault-tolerance
- Complete quantum machine learning framework
- Hardware integration for real quantum devices
- Zero-warning code quality with extensive testing


## Core Features

### Basic Quantum Primitives
- **Type-Safe Qubit Identifiers**: Zero-cost abstractions with compile-time validation
- **Comprehensive Gate Library**: All standard quantum gates with optimized matrix representations
- **Quantum Registers**: Fixed-size register types using const generics
- **Robust Error Handling**: Comprehensive error types with detailed diagnostics

### Advanced Decomposition Algorithms
- **Solovay-Kitaev**: Universal gate set approximation with configurable precision
- **Clifford+T Decomposition**: T-count optimization for fault-tolerant computing
- **KAK Decomposition**: Multi-qubit gate decomposition with recursive algorithms
- **Shannon Decomposition**: Optimal gate count decomposition for arbitrary unitaries
- **Cartan Decomposition**: Two-qubit gate decomposition using Lie algebra

### Quantum Machine Learning
- **QML Layers**: Rotation, entangling, and hardware-efficient layers
- **Data Encoding**: Feature maps and data re-uploading strategies
- **Training Framework**: Optimizers, loss functions, and hyperparameter optimization
- **Variational Algorithms**: VQE, QAOA with automatic differentiation

### Error Correction & Fault Tolerance
- **Quantum Error Correction**: Surface, color, and Steane codes
- **Stabilizer Formalism**: Pauli strings and syndrome decoding
- **Noise Models**: Comprehensive quantum channel representations
- **Topological QC**: Anyonic models, braiding operations, and fusion trees

### Advanced Computing Paradigms
- **MBQC**: Measurement-based quantum computing with cluster states
- **Tensor Networks**: Efficient contraction algorithms and optimization
- **Fermionic Systems**: Jordan-Wigner and Bravyi-Kitaev transformations
- **Bosonic Systems**: Continuous variable quantum computing

### Performance & Optimization
- **SciRS2 Integration**: Leveraging SciRS2's high-performance computing capabilities
- **GPU Acceleration**: CUDA, OpenCL, and Metal (macOS) backend support
- **SIMD Operations**: Vectorized quantum state operations via `scirs2_core::simd_ops`
- **Batch Processing**: Parallel execution using `scirs2_core::parallel_ops`
- **Platform Detection**: Automatic CPU/GPU capability detection for optimal performance
- **ZX-Calculus**: Graph-based circuit optimization
- **Memory Efficiency**: Optimized state vector representations with SciRS2

## Usage

### Basic Quantum Operations

```rust
use quantrs2_core::prelude::*;

fn main() -> QuantRS2Result<()> {
    // Create qubits and apply basic gates
    let q0 = QubitId::new(0);
    let q1 = QubitId::new(1);
    
    // Standard gates
    let x_gate = XGate::new();
    let h_gate = HGate::new();
    let cnot = CXGate::new();
    
    // Parametric gates
    let rz = RZGate::new(std::f64::consts::PI / 4.0);
    
    // Controlled operations
    let controlled_x = make_controlled(XGate::new(), vec![q0]);
    
    Ok(())
}
```

### Gate Decomposition

```rust
use quantrs2_core::prelude::*;

fn decomposition_example() -> QuantRS2Result<()> {
    // Solovay-Kitaev decomposition
    let config = SolovayKitaevConfig::default();
    let sk = SolovayKitaev::new(config);
    let target_gate = RYGate::new(0.123);
    let sequence = sk.decompose(&target_gate, 10)?;
    
    // Clifford+T decomposition for fault-tolerant computing
    let decomposer = CliffordTDecomposer::new();
    let ct_sequence = decomposer.decompose_gate(&target_gate)?;
    let t_count = count_t_gates_in_sequence(&ct_sequence);
    
    // KAK decomposition for two-qubit gates
    let cnot = CXGate::new();
    let kak_decomp = decompose_two_qubit_kak(&cnot.matrix())?;
    
    Ok(())
}
```

### Quantum Machine Learning

```rust
use quantrs2_core::prelude::*;

fn qml_example() -> QuantRS2Result<()> {
    // Create QML layers
    let rotation_layer = RotationLayer::new(4, vec!['X', 'Y', 'Z']);
    let entangling_layer = EntanglingLayer::circular(4);
    
    // Build a QML circuit
    let mut circuit = QMLCircuit::new(4);
    circuit.add_layer(Box::new(rotation_layer));
    circuit.add_layer(Box::new(entangling_layer));
    
    // Training configuration
    let config = TrainingConfig {
        learning_rate: 0.01,
        epochs: 100,
        batch_size: 32,
        optimizer: Optimizer::Adam,
        loss_function: LossFunction::MeanSquaredError,
    };
    
    Ok(())
}
```

### Error Correction

```rust
use quantrs2_core::prelude::*;

fn error_correction_example() -> QuantRS2Result<()> {
    // Surface code for error correction
    let surface_code = SurfaceCode::new(3, 3)?;
    let stabilizers = surface_code.stabilizers();
    
    // Syndrome measurement and decoding
    let syndrome = vec![0, 1, 0, 1]; // Example syndrome
    let decoder = LookupDecoder::new(&surface_code);
    let correction = decoder.decode(&syndrome)?;
    
    // Quantum channels for noise modeling
    let depolarizing = QuantumChannels::depolarizing(0.01);
    let amplitude_damping = QuantumChannels::amplitude_damping(0.05);
    
    Ok(())
}
```

### Batch Operations & GPU Acceleration

```rust
use quantrs2_core::prelude::*;

fn batch_and_gpu_example() -> QuantRS2Result<()> {
    // Create batch state vectors for parallel processing
    let batch_config = BatchConfig::new(1000, 4); // 1000 states, 4 qubits each
    let mut batch = create_batch(&batch_config)?;
    
    // Apply gates to entire batch
    apply_single_qubit_gate_batch(&mut batch, 0, &HGate::new())?;
    apply_two_qubit_gate_batch(&mut batch, 0, 1, &CXGate::new())?;
    
    // GPU acceleration (if available)
    let gpu_config = GpuConfig::default();
    let gpu_backend = GpuBackendFactory::create(&gpu_config)?;
    let gpu_state = gpu_backend.create_state_vector(10)?;
    
    Ok(())
}
```

## Module Structure

### Core Foundations
- **error.rs**: Comprehensive error types and result wrappers
- **gate.rs**: Gate trait definitions and standard quantum gates
- **qubit.rs**: Type-safe qubit identifier with zero-cost abstractions
- **register.rs**: Quantum register types with const generics
- **complex_ext.rs**: Extended complex number operations for quantum states
- **matrix_ops.rs**: Efficient quantum matrix operations with SciRS2 integration
- **operations.rs**: Non-unitary quantum operations (measurements, reset, POVM)

### Decomposition & Synthesis
- **decomposition.rs**: Universal gate decomposition algorithms
- **decomposition/solovay_kitaev.rs**: Universal gate set approximation
- **decomposition/clifford_t.rs**: Fault-tolerant gate decomposition
- **synthesis.rs**: Unitary matrix synthesis into gate sequences
- **shannon.rs**: Quantum Shannon decomposition with optimal gate counts
- **cartan.rs**: Cartan (KAK) decomposition using Lie algebra
- **kak_multiqubit.rs**: Multi-qubit KAK decomposition with recursive algorithms

### Advanced Quantum Systems
- **fermionic.rs**: Fermionic operators with Jordan-Wigner transformations
- **bosonic.rs**: Bosonic operators for continuous variable quantum computing
- **topological.rs**: Anyonic models, braiding operations, and fusion trees
- **mbqc.rs**: Measurement-based quantum computing with cluster states
- **tensor_network.rs**: Tensor network representations and contraction algorithms

### Error Correction & Fault Tolerance
- **error_correction.rs**: Quantum error correction codes (surface, color, Steane)
- **quantum_channels.rs**: Quantum channel representations (Kraus, Choi, Stinespring)
- **characterization.rs**: Gate characterization and eigenstructure analysis

### Optimization & Performance
- **optimization/**: Circuit optimization passes and algorithms
  - **compression.rs**: Gate sequence compression with configurable strategies
  - **fusion.rs**: Gate fusion for reducing circuit depth
  - **peephole.rs**: Local optimization patterns and T-count reduction
  - **zx_optimizer.rs**: ZX-calculus based optimization passes
- **zx_calculus.rs**: ZX-diagram representation and manipulation
- **zx_extraction.rs**: Circuit extraction from ZX-diagrams
- **simd_ops.rs**: SIMD-accelerated quantum operations
- **memory_efficient.rs**: Memory-optimized state vector representations

### Variational & Machine Learning
- **variational.rs**: Variational quantum circuits with automatic differentiation
- **variational_optimization.rs**: Optimization algorithms for variational methods
- **qml/**: Quantum machine learning components
  - **layers.rs**: Parameterized quantum layers (rotation, entangling, pooling)
  - **encoding.rs**: Data encoding strategies and feature maps
  - **training.rs**: Training algorithms and hyperparameter optimization
- **parametric.rs**: Parametric gates with symbolic computation
- **qaoa.rs**: Quantum Approximate Optimization Algorithm
- **qpca.rs**: Quantum Principal Component Analysis

### Specialized Algorithms
- **hhl.rs**: HHL algorithm for linear systems
- **eigensolve.rs**: Quantum eigenvalue algorithms
- **quantum_counting.rs**: Quantum counting and amplitude estimation
- **quantum_walk.rs**: Discrete and continuous quantum walks

### Hardware & Acceleration
- **gpu/**: GPU acceleration support
  - **mod.rs**: GPU backend abstractions
  - **cpu_backend.rs**: CPU fallback support
  - **metal_backend_scirs2_ready.rs**: Metal GPU support for macOS (placeholder for SciRS2 integration)
  - **scirs2_adapter.rs**: Adapter layer for SciRS2 GPU migration
- **platform/**: Platform-specific optimizations
  - **mod.rs**: PlatformCapabilities detection and adaptive optimization
- **controlled.rs**: Efficient controlled gate operations
- **batch/**: Batch processing for parallel quantum computations
  - **operations.rs**: Batch gate operations
  - **execution.rs**: Batch circuit execution
  - **measurement.rs**: Batch measurement and tomography
  - **optimization.rs**: Batch parameter optimization

### Testing & Validation
- **testing.rs**: Quantum-specific testing utilities and assertions

## API Overview

### Core Types
- `QubitId`: Type-safe qubit identifier with zero-cost abstractions
- `Register<N>`: Fixed-size quantum register using const generics
- `QuantRS2Error`: Comprehensive error enumeration with detailed diagnostics
- `QuantRS2Result<T>`: Convenient result type alias for error handling

### Gate System
- `GateOp`: Core trait for quantum gates with matrix representations
- Standard gates: `XGate`, `YGate`, `ZGate`, `HGate`, `SGate`, `TGate`, `CXGate`, `CZGate`
- Parametric gates: `RXGate`, `RYGate`, `RZGate`, `U1Gate`, `U2Gate`, `U3Gate`
- Controlled operations: `ControlledGate`, `MultiControlledGate`, `ToffoliGate`, `FredkinGate`

### Decomposition & Synthesis
- `SolovayKitaev`: Universal gate set approximation with configurable precision
- `CliffordTDecomposer`: T-count optimization for fault-tolerant computing
- `KAKDecomposition`: Multi-qubit gate decomposition structures
- `ShannonDecomposer`: Optimal gate count decomposition algorithms
- `CartanDecomposer`: Two-qubit gate decomposition using Lie algebra

### Machine Learning
- `QMLLayer`: Base trait for quantum machine learning layers
- `QMLCircuit`: Parameterized quantum circuits for ML applications
- `TrainingConfig`: Configuration for quantum ML training
- `VariationalGate`: Gates with trainable parameters and autodiff support

### Error Correction
- `StabilizerCode`: Base trait for quantum error correction codes
- `SurfaceCode`, `ColorCode`: Specific error correction codes
- `QuantumChannel`: Noise models with Kraus, Choi, and Stinespring representations
- `SyndromeDecoder`: Syndrome decoding algorithms (lookup, MWPM)

### Advanced Computing
- `TensorNetwork`: Tensor network representations with contraction optimization
- `FermionOperator`: Fermionic system representations with transformations
- `BosonOperator`: Bosonic operators for continuous variable systems
- `AnyonType`: Topological quantum computing with anyonic models

### Performance & Optimization
- `BatchStateVector`: Parallel processing of multiple quantum states
- `GpuBackend`: GPU acceleration for quantum computations
- `ZXDiagram`: ZX-calculus based circuit optimization
- `OptimizationPass`: Circuit optimization pass framework

## Performance Features

### SIMD Acceleration
- Vectorized quantum state operations using platform-specific SIMD instructions
- Optimized inner products, normalizations, and phase applications
- Batch processing of expectation values

### Memory Optimization
- Efficient state vector representations with configurable precision
- Sparse matrix support through SciRS2 integration
- Memory-mapped storage for large gate databases

### GPU Computing
- CUDA, OpenCL, and Metal backend support for large-scale simulations
- Optimized kernels for common quantum operations
- Automatic fallback to CPU processing
- Metal GPU support for Apple Silicon (M1/M2/M3) with unified memory architecture
- Forward-compatible implementation ready for SciRS2 v0.1.0 Metal integration

### SciRS2 Integration
- Advanced linear algebra operations using SciRS2's optimized BLAS/LAPACK bindings
- Sparse matrix solvers for large quantum systems  
- Parallel algorithms for batch quantum computations via `scirs2_core::parallel_ops`
- SIMD operations for vectorized quantum state manipulations
- Memory-efficient algorithms for large-scale quantum simulations
- Platform-aware optimization with automatic backend selection

## Technical Details

- Zero-cost abstractions: `QubitId` uses `#[repr(transparent)]` for no runtime overhead
- Column-major storage: Gate matrices stored for optimal BLAS compatibility
- Const generics: Compile-time validation of quantum register sizes
- Trait specialization: Optimized handling for common gate patterns
- Error propagation: Comprehensive error handling with detailed context

## Integration with QuantRS2 Ecosystem

### Circuit Module Integration
- Provides foundational gate types for circuit construction
- Supplies optimization passes for circuit compilation
- Offers decomposition algorithms for gate translation

### Simulation Module Integration
- Supplies optimized matrix representations for quantum simulation
- Provides batch processing capabilities for parallel simulations
- Offers GPU acceleration for large-scale quantum computations

### Device Module Integration
- Provides gate calibration data structures for hardware backends
- Supplies noise models for realistic quantum device simulation
- Offers translation algorithms for device-specific gate sets

### ML Module Integration
- Provides QML layers and training frameworks
- Supplies variational optimization algorithms
- Offers automatic differentiation for quantum gradients

## License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.