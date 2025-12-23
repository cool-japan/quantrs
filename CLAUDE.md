# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## QuantRS2 Project Overview

QuantRS2 (`/kwɒntərz tu:/`) is a comprehensive Rust-based quantum computing framework that provides a modular, high-performance toolkit for quantum simulation, algorithm development, and hardware interaction. It leverages Rust's zero-cost abstractions, memory safety, and the SciRS2 ecosystem to deliver production-ready quantum computing capabilities.

## Architecture

The project uses a modular workspace structure with specialized crates:

- **quantrs2-core**: Core types, traits, and abstractions shared across the ecosystem
- **quantrs2-circuit**: Quantum circuit representation and DSL
- **quantrs2-sim**: Quantum simulators (state-vector, tensor-network, stabilizer)
- **quantrs2-device**: Remote quantum hardware connections (IBM, Azure, AWS)
- **quantrs2-ml**: Quantum machine learning including QNNs, GANs, and HEP classifiers
- **quantrs2-anneal**: Quantum annealing support and D-Wave integration
- **quantrs2-tytan**: High-level quantum annealing library
- **quantrs2-py**: Python bindings with PyO3

Key dependencies:
- scirs2 (v0.1.0-rc.4): Scientific computing primitives with quantum support
- scirs2-optimize (v0.1.0-rc.4): Optimization for VQE, QAOA
- optirs (v0.1.0-rc.2): Advanced ML optimization algorithms
- numrs2 (v0.1.0-rc.3): Numerical computing library

## Common Development Commands

### Building
```bash
# Build all packages
cargo build

# Build with release optimizations
cargo build --release

# Build specific crate
cargo build --package quantrs2-sim

# Clean build artifacts
cargo clean
```

### Testing
```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test --package quantrs2-circuit

# Run a specific test
cargo test --package quantrs2-sim test_name

# Run tests with output
cargo test -- --nocapture

# Run integration tests
cargo test --test integration
```

### Code Quality
```bash
# Format code
cargo fmt

# Run clippy lints
cargo clippy -- -D warnings

# Audit dependencies for security issues
cargo audit
```

### Documentation
```bash
# Build and open documentation
cargo doc --open

# Build docs without opening
cargo doc --no-deps
```

### Running Examples
```bash
# Run quantum algorithm examples
cargo run --example grover_search
cargo run --example qaoa_maxcut
cargo run --example quantum_fourier_transform
cargo run --example vqe_h2_molecule
```

### Benchmarking
```bash
# Run benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench quantum_simulation
```

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT - SciRS2 Policy

**QuantRS2 MUST use SciRS2 as its scientific computing foundation.**

### 📋 **Comprehensive Policy Document**
For complete SciRS2 integration guidelines, quantum-specific patterns, and detailed crate usage documentation, see:

**➡️ [SCIRS2_INTEGRATION_POLICY.md](./SCIRS2_INTEGRATION_POLICY.md)**

### 🎯 **Quick Reference - Key Quantum Computing Patterns**

#### **Complex Number Operations** (Quantum Amplitudes)
```rust
// ✅ CORRECT: SciRS2 complex types for quantum states (direct from scirs2-core root)
use scirs2_core::{Complex64, Complex32};
type QuantumState = Array1<Complex64>;
type DensityMatrix = Array2<Complex64>;

// ❌ NEVER: Direct num-complex usage
use num_complex::Complex64;  // Policy violation
```

#### **Array Operations** (State Vectors & Operators)
```rust
// ✅ CORRECT - Unified SciRS2 usage
use scirs2_core::ndarray::*;  // Complete unified access
// Or selective:
use scirs2_core::ndarray::{Array1, Array2, array, s, Axis};

// ❌ WRONG - Fragmented SciRS2 usage (no autograd)
use scirs2_autograd::ndarray::{Array2, array};  // Fragmented - DON'T USE

// ❌ NEVER: Direct ndarray usage
use ndarray::{Array, array};  // Policy violation
```

#### **Random Number Generation** (Quantum Measurements)
```rust
// ✅ CORRECT - Unified SciRS2 usage
use scirs2_core::random::prelude::*;  // Common distributions & RNG
// Or selective:
use scirs2_core::random::{thread_rng, Normal as RandNormal, RandBeta, StudentT};

// ✅ CORRECT - Enhanced unified interface
use scirs2_core::random::distributions_unified::{UnifiedNormal, UnifiedBeta};

// Example usage for reproducible quantum experiments
let mut rng = thread_rng();
let measurement: f64 = rng.gen();

// ❌ NEVER: Direct rand usage
use rand::{Rng, thread_rng};  // Policy violation
```

#### **SIMD Operations** (Performance Critical)
```rust
// ✅ SIMD-accelerated quantum operations
use scirs2_core::simd_ops::{SimdOps, PlatformCapabilities};

// Apply quantum gates with hardware acceleration
if PlatformCapabilities::current().has_avx2() {
    scirs2_core::simd_ops::vectorized_complex_multiply(
        &mut state_vector,
        &gate_matrix
    );
}
```

### 🚨 NO DIRECT EXTERNAL DEPENDENCIES POLICY

**CRITICAL**: QuantRS2 crates MUST NOT use external dependencies directly. All external dependencies MUST go through SciRS2-Core abstractions.

#### ❌ FORBIDDEN in Cargo.toml (POLICY VIOLATIONS)
```toml
[dependencies]
rand = { workspace = true }              # ❌ Use scirs2_core::random
rand_distr = { workspace = true }        # ❌ Use scirs2_core::random
ndarray = { workspace = true }           # ❌ Use scirs2_core::ndarray
num-complex = { workspace = true }       # ❌ Use scirs2_core
num-traits = { workspace = true }        # ❌ Use scirs2_core::numeric
rayon = { workspace = true }             # ❌ Use scirs2_core::parallel_ops
nalgebra = { workspace = true }          # ❌ Use scirs2_linalg
ndarray-linalg = { workspace = true }    # ❌ Use scirs2_linalg
```

#### ✅ REQUIRED in Cargo.toml
```toml
[dependencies]
scirs2-core = { workspace = true, features = ["array", "random", "parallel"] }
scirs2-linalg = { workspace = true }  # For linear algebra operations
scirs2-sparse = { workspace = true }  # For sparse matrices
# Other SciRS2 crates as needed
```

### FULL USE OF SciRS2-Core for Quantum Computing

QuantRS2 must make **FULL USE** of scirs2-core's quantum-relevant capabilities:

#### Core Quantum Operations
```rust
// Complex arithmetic for quantum amplitudes (direct from scirs2-core root)
use scirs2_core::{Complex64, Complex32};

// Array operations for quantum states (unified access)
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis, array, s};

// Random number generation for quantum measurements (unified interface)
use scirs2_core::random::prelude::*;

// Enhanced unified distributions
use scirs2_core::random::distributions_unified::{UnifiedNormal, UnifiedBeta};

// SIMD-accelerated quantum operations (check availability)
use scirs2_core::simd_ops::{SimdOps};

// Parallel quantum gate application
use scirs2_core::parallel_ops::{par_chunks, par_join};
```

#### Memory Management for Large Quantum Systems
```rust
// Memory-efficient quantum state representation
use scirs2_core::memory_efficient::{MemoryMappedArray, ChunkedArray};

// Sparse quantum operators
use scirs2_sparse::{CsrMatrix, CscMatrix};

// Tensor network operations
use scirs2_core::memory_efficient::AdaptiveChunking;
```

### QuantRS2 Module-Specific SciRS2 Usage

#### quantrs2-core
- Use `scirs2_core::{Complex64, Complex32}` for all complex number operations
- Use `scirs2_core::ndarray::{Array1, Array2, array, s}` for quantum state vectors (unified access)
- Use `scirs2_core::random::prelude::*` for quantum measurements and sampling
- Use `scirs2_core::simd_ops` for vectorized quantum operations
- Use `scirs2_core::parallel_ops` for parallel quantum circuit execution

#### quantrs2-sim
- Use `scirs2_linalg` for unitary matrix operations
- Use `scirs2_sparse` for sparse state representation
- Use `scirs2_fft` for Quantum Fourier Transform
- Use `scirs2_core::gpu` for GPU-accelerated simulation

#### quantrs2-ml
- Use `scirs2_neural` for quantum neural networks
- Use `scirs2_autograd` for gradient computation in VQE/QAOA
- Use `scirs2_optimize` with `optirs` for parameter optimization

#### quantrs2-circuit
- Use `scirs2_graph` for graph state preparation
- Use `scirs2_linalg::unitary` for gate validation
- Use `scirs2_core::complex::*` for gate matrix representation

#### quantrs2-anneal
- Use `scirs2_optimize` for annealing schedules
- Use `scirs2_stats` for statistical analysis
- Use `scirs2_core::random` for Monte Carlo sampling

#### quantrs2-device
- Use `scirs2_stats` for measurement statistics
- Use `scirs2_metrics` for fidelity calculations
- Use `scirs2_core::random` for noise modeling

### Current SciRS2 Version

**QuantRS2 uses SciRS2 v0.1.0-rc.4** (Release Candidate)
- NumRS2: v0.1.0-rc.3
- All dependencies MUST be compatible with this version

### Migration Checklist - Ensure Full SciRS2 Usage

When reviewing or writing QuantRS2 code, verify:

#### ✅ Complex Numbers and Quantum States
- [ ] NO direct `use num_complex::{...}`
- [ ] YES `use scirs2_core::complex::{Complex64, ComplexFloat}`
- [ ] YES proper quantum state types with SciRS2 complex numbers

#### ✅ Arrays and State Vectors
- [ ] NO direct `use ndarray::{...}`
- [ ] NO fragmented `use scirs2_autograd::ndarray::{...}` (deprecated)
- [ ] NO incomplete `use scirs2_core::ndarray_ext::{...}` (missing macros)
- [ ] YES `use scirs2_core::ndarray::*` for complete unified access
- [ ] YES `use scirs2_core::ndarray::{Array1, Array2, array, s, Axis}` for selective imports

#### ✅ Random Number Generation for Measurements
- [ ] NO direct `use rand::{...}`
- [ ] NO direct `use rand_distr::{...}`
- [ ] YES `use scirs2_core::random::prelude::*` for common distributions & RNG
- [ ] YES `use scirs2_core::random::{thread_rng, Normal as RandNormal, ...}` for selective imports
- [ ] YES `use scirs2_core::random::distributions_unified::{UnifiedNormal, UnifiedBeta}` for enhanced interface
- [ ] YES `thread_rng()` for fast quantum measurements, `seeded_rng(42)` for reproducible experiments

#### ✅ Performance Optimization
- [ ] YES use `scirs2_core::simd_ops` for vectorized quantum operations
- [ ] YES use `scirs2_core::parallel_ops` for parallel circuit execution
- [ ] YES use `scirs2_core::gpu` for GPU quantum simulation
- [ ] YES use `scirs2_sparse` for large sparse quantum systems

#### ✅ Quantum-Specific Features
- [ ] YES use `scirs2_fft` for Quantum Fourier Transform
- [ ] YES use `scirs2_linalg` for unitary operations
- [ ] YES use `scirs2_graph` for graph states
- [ ] YES use `scirs2_metrics` for quantum metrics

### Common Anti-Patterns to Avoid
```rust
// ❌ WRONG - Direct dependencies (POLICY VIOLATIONS)
use ndarray::{Array2, array, s};
use rand::{Rng, thread_rng};
use num_complex::Complex64;

// ❌ WRONG - Fragmented SciRS2 usage
use scirs2_autograd::ndarray::{Array2, array};  // Fragmented - DON'T USE
use scirs2_core::ndarray_ext::{ArrayView};      // Missing macros - DON'T USE

// ✅ CORRECT - Unified SciRS2 usage (PROVEN PATTERNS)
use scirs2_core::ndarray::*;  // Complete unified access
// Or selective:
use scirs2_core::ndarray::{Array2, array, s, Axis};

use scirs2_core::random::prelude::*;  // Common distributions & RNG
// Or selective:
use scirs2_core::random::{thread_rng, Normal as RandNormal, RandBeta, StudentT};

// ✅ CORRECT - Enhanced unified interface
use scirs2_core::random::distributions_unified::{UnifiedNormal, UnifiedBeta};

// ✅ CORRECT - Complex numbers
use scirs2_core::{Complex64, Complex32};
```

## Key Implementation Details

### Quantum State Representation
All quantum states are represented using SciRS2's complex number types and array structures. State vectors use `Array1<Complex64>` and density matrices use `Array2<Complex64>`.

### Quantum Circuit Execution
Quantum circuits are compiled to optimized gate sequences that leverage SciRS2's SIMD operations for complex number arithmetic and parallel execution for multi-qubit operations.

### Quantum Measurements
Measurement probabilities are computed using SciRS2's complex arithmetic, and sampling uses SciRS2's random number generation with proper seeding for reproducibility.

### Tensor Network Simulation
For circuits with limited entanglement, QuantRS2 uses SciRS2's sparse matrix operations and memory-efficient chunking to simulate quantum systems beyond the classical memory limit.

### Hardware Integration
Real quantum device connections use SciRS2's statistical analysis for error mitigation and measurement result processing.

## Development Best Practices

### Type Safety
- Use const generics for compile-time qubit count verification
- Leverage Rust's type system for quantum state validity
- Implement builder patterns for complex quantum algorithms

### Performance
- Always use SciRS2's SIMD operations for quantum gates
- Parallelize independent quantum operations
- Use sparse representations when applicable
- Profile with SciRS2's benchmarking tools

### Testing
- Unit tests for each quantum gate
- Integration tests for quantum algorithms
- Property-based testing for quantum circuit equivalence
- Benchmarks for simulation performance

### Documentation
- Document quantum algorithm implementations
- Provide examples for common quantum patterns
- Include mathematical descriptions in docstrings

## Known Issues and Limitations

- Maximum simulation capacity: 30+ qubits (state vector), 50+ qubits (stabilizer)
- GPU acceleration requires CUDA-capable hardware
- Some quantum algorithms require specific SciRS2 features to be enabled
- Tensor network simulation performance depends on circuit structure

## Future Roadmap

- Enhanced quantum error correction with SciRS2 error models
- Distributed quantum simulation using SciRS2's parallel capabilities
- Advanced VQE/QAOA optimizers through OptiRS integration
- Quantum machine learning expansion with SciRS2-neural

## Quick Start Example

```rust
use quantrs2_circuit::{Circuit, gates};
use quantrs2_sim::StateVectorSimulator;
use scirs2_core::{Complex64};  // Unified SciRS2 complex numbers
use scirs2_core::ndarray::{Array1, array};  // Unified SciRS2 arrays

// Create a Bell state circuit
let mut circuit = Circuit::<2>::new();
circuit.h(0);  // Hadamard on qubit 0
circuit.cnot(0, 1);  // CNOT from 0 to 1

// Simulate the circuit
let simulator = StateVectorSimulator::new();
let result = simulator.run(&circuit, 1000)?;

// Analyze results
println!("Bell state measurements: {:?}", result.counts());
```

## Support and Resources

- Documentation: [docs.rs/quantrs2](https://docs.rs/quantrs2)
- Examples: See the `examples/` directory
- Issues: Report bugs via GitHub issues
- SciRS2 Policy: See SCIRS2_INTEGRATION_POLICY.md

---

**Remember**: Always use unified SciRS2 patterns - `scirs2_core::ndarray::*` for arrays, `scirs2_core::random::prelude::*` for distributions, `scirs2_core::{Complex64, Complex32}` for complex numbers. Never use fragmented scirs2_autograd or direct ndarray/rand/num-complex imports.