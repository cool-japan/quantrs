# SciRS2 Integration Guide

This guide explains how QuantRS2 leverages SciRS2 (Scientific Rust) features for enhanced performance and capabilities.

## Overview

SciRS2 is integrated into QuantRS2 to provide:
- High-performance numerical computing
- Memory-efficient data structures
- SIMD-accelerated operations
- Advanced linear algebra via BLAS/LAPACK
- Scientific computing primitives

## Core Integration Points

### 1. Complex Number Extensions (`core/src/complex_ext.rs`)

The `QuantumComplexExt` trait extends SciRS2's complex numbers with quantum-specific operations:

```rust
use scirs2_core::types::Complex64;

pub trait QuantumComplexExt {
    fn probability(&self) -> f64;      // |z|²
    fn normalize(&self) -> Complex64;   // z/|z|
    fn fidelity(&self, other: &Complex64) -> f64;
}
```

**Usage in quantum operations:**
- Computing measurement probabilities
- Normalizing quantum states
- Calculating state fidelity

### 2. Memory-Efficient State Vectors (`core/src/memory_efficient.rs`)

Leverages SciRS2's memory management for large quantum states:

```rust
pub struct EfficientStateVector {
    num_qubits: usize,
    data: Vec<Complex64>,
    chunk_size: usize,
}
```

**Features:**
- Automatic chunking for states > 20 qubits
- Thread-safe buffer pools
- Reduced memory allocation overhead

### 3. SIMD Operations (`core/src/simd_ops.rs`)

Accelerated quantum operations using SciRS2's SIMD capabilities:

```rust
pub fn apply_phase_simd(state: &mut [Complex64], phase: f64);
pub fn normalize_simd(state: &mut [Complex64]) -> f64;
pub fn inner_product(state1: &[Complex64], state2: &[Complex64]) -> Complex64;
```

**Performance gains:**
- 2-4x speedup for phase operations
- 3-5x speedup for normalization
- Vectorized expectation value calculations

### 4. Linear Algebra Operations (`sim/src/linalg_ops.rs`)

Enhanced matrix operations using SciRS2's BLAS/LAPACK integration:

```rust
pub fn apply_unitary(state: &mut Vec<Complex64>, unitary: &Matrix<Complex64>, target_qubits: &[usize]);
pub fn tensor_product(a: &Matrix<Complex64>, b: &Matrix<Complex64>) -> Matrix<Complex64>;
```

**Capabilities:**
- Optimized unitary transformations
- Efficient tensor products
- Partial trace operations

## Using SciRS2 Features

### For Algorithm Developers

1. **Use enhanced complex operations:**
   ```rust
   use quantrs2_core::complex_ext::QuantumComplexExt;
   
   let amplitude = Complex64::new(0.6, 0.8);
   let prob = amplitude.probability(); // 1.0
   ```

2. **Leverage memory-efficient storage:**
   ```rust
   use quantrs2_core::memory_efficient::EfficientStateVector;
   
   let state = EfficientStateVector::new(25); // 25 qubits
   // Automatically uses chunked processing
   ```

3. **Apply SIMD operations:**
   ```rust
   use quantrs2_core::simd_ops::{apply_phase_simd, normalize_simd};
   
   apply_phase_simd(&mut state, std::f64::consts::PI / 4.0);
   let norm = normalize_simd(&mut state);
   ```

### For Simulator Developers

1. **Use the enhanced simulator:**
   ```rust
   use quantrs2_sim::enhanced_statevector::EnhancedStateVectorSimulator;
   
   let simulator = EnhancedStateVectorSimulator::new();
   // Automatically switches to memory-efficient mode for large circuits
   ```

2. **Implement custom gates with SIMD:**
   ```rust
   use quantrs2_sim::linalg_ops::apply_unitary;
   
   let custom_gate = matrix![/* your gate matrix */];
   apply_unitary(&mut state, &custom_gate, &[qubit_index]);
   ```

## Performance Considerations

### When to Use Each Feature

1. **Complex Extensions**: Always use for quantum amplitude operations
2. **Memory-Efficient Vectors**: Automatically activated for >20 qubits
3. **SIMD Operations**: Best for operations on full state vectors
4. **Linear Algebra**: Use for multi-qubit gates and tensor operations

### Benchmarking

Run the SciRS2 integration demo to see performance comparisons:

```bash
cargo run --release --bin scirs2_integration_demo --features simulation
```

## Future Enhancements

### Planned SciRS2 Integrations

1. **Sparse Matrix Support**
   - For circuits with many identity operations
   - Reduced memory usage for specific gate patterns

2. **GPU Acceleration**
   - CUDA/OpenCL kernels via SciRS2
   - Multi-GPU support for distributed simulation

3. **Arbitrary Precision**
   - High-precision quantum calculations
   - Error bound tracking

4. **Symbolic Computation**
   - Parametric circuit optimization
   - Analytical gradient computation

## Contributing

When adding new features that could benefit from SciRS2:

1. Check if SciRS2 provides relevant functionality
2. Benchmark native vs SciRS2 implementation
3. Add feature flags for optional dependencies
4. Document performance characteristics
5. Add integration tests

## Dependencies

Add SciRS2 features to your `Cargo.toml`:

```toml
[dependencies]
scirs2-core = { workspace = true, features = ["types", "memory_management", "simd"] }
```

## Examples

See `examples/src/bin/scirs2_integration_demo.rs` for a comprehensive demonstration of all integrated features.

## Troubleshooting

### Common Issues

1. **BLAS/LAPACK not found**
   - On macOS: `export OPENBLAS_SYSTEM=1`
   - On Linux: Install `libopenblas-dev`

2. **SIMD not supported**
   - Check CPU features: `cat /proc/cpuinfo | grep -E "avx|sse"`
   - Fallback to scalar operations is automatic

3. **Memory errors with large states**
   - Increase stack size: `ulimit -s unlimited`
   - Use `EfficientStateVector` for automatic chunking

## References

- [SciRS2 Documentation](https://github.com/scirs/scirs2)
- [QuantRS2 Architecture](./README.md)
- [Performance Benchmarks](./benchmarks/)