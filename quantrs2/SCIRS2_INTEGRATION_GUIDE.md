# QuantRS2 SciRS2 Integration Guide

This guide explains how the `quantrs2` facade crate integrates with the SciRS2 scientific computing ecosystem and how to use SciRS2 features in your quantum computing applications.

## Table of Contents

- [Overview](#overview)
- [Why SciRS2?](#why-scirs2)
- [SciRS2 Integration in QuantRS2](#scirs2-integration-in-quantrs2)
- [Using SciRS2 Features](#using-scirs2-features)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

**SciRS2** is a comprehensive scientific computing framework for Rust that provides:
- High-performance linear algebra
- Automatic differentiation
- Optimization algorithms
- Statistical analysis
- GPU acceleration
- SIMD vectorization

**QuantRS2** is built on top of SciRS2 to leverage these capabilities for quantum computing.

### Version Compatibility

```rust
use quantrs2::version;

println!("QuantRS2: {}", version::QUANTRS2_VERSION);
println!("SciRS2: {}", version::SCIRS2_VERSION);
```

**Current Requirements:**
- **QuantRS2**: v0.1.0-beta.3
- **SciRS2**: v0.1.0-rc.2
- **OptiRS**: v0.1.0-beta.2
- **NumRS2**: v0.1.0-beta.3

## Why SciRS2?

### Traditional Approach (Without SciRS2)
```rust
// ❌ Multiple disconnected dependencies
use ndarray::Array2;
use num_complex::Complex64;
use rand::thread_rng;
use nalgebra::DMatrix;

// Each library has its own types and conventions
// Difficult to interoperate between libraries
```

### SciRS2 Approach (QuantRS2 Way)
```rust
// ✅ Unified scientific computing stack
use quantrs2::prelude::simulation::*;

// All scientific computing needs met through SciRS2
// Consistent APIs, optimized integration
```

### Benefits

1. **Unified Type System**: All numerical types work together seamlessly
2. **Optimized Performance**: Cross-library optimizations, SIMD, GPU support
3. **Automatic Differentiation**: Built-in autodiff for VQE/QAOA
4. **Consistent APIs**: Learn once, use everywhere
5. **Memory Efficiency**: Shared memory pools, cache optimization

## SciRS2 Integration in QuantRS2

### Architecture Layers

```
┌─────────────────────────────────────┐
│     QuantRS2 Facade (quantrs2)      │
│  Configuration, Diagnostics, Utils  │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│   QuantRS2 Subcrates (circuit,      │
│   sim, ml, anneal, device, tytan)   │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│          SciRS2 Core                 │
│  Complex, Arrays, Random, SIMD       │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│      SciRS2 Specialized Crates       │
│  linalg, autograd, optimize, stats   │
└──────────────────────────────────────┘
```

### Key Integration Points

#### 1. Complex Numbers (Quantum Amplitudes)
```rust
use quantrs2::prelude::simulation::*;

// Complex numbers from SciRS2
let amplitude = Complex64::new(0.707, 0.0);
let conjugate = amplitude.conj();
let probability = (amplitude * conjugate).re;
```

**Behind the scenes:**
- `Complex64` comes from `scirs2_core::Complex64`
- Optimized complex arithmetic with SIMD
- Integrated with SciRS2 linear algebra

#### 2. State Vectors (Quantum States)
```rust
use quantrs2::prelude::simulation::*;
use scirs2_core::ndarray::Array1;

// State vector backed by SciRS2 arrays
let state = Array1::<Complex64>::zeros(8); // 3-qubit system
```

**Behind the scenes:**
- Uses `scirs2_core::ndarray` (not raw `ndarray`)
- Memory-aligned for SIMD operations
- Integrated with SciRS2 GPU backends

#### 3. Quantum Operators (Matrices)
```rust
use quantrs2::prelude::simulation::*;
use scirs2_core::ndarray::Array2;

// Unitary matrix for quantum gate
let hadamard: Array2<Complex64> = array![
    [Complex64::new(1.0/2.0_f64.sqrt(), 0.0), Complex64::new(1.0/2.0_f64.sqrt(), 0.0)],
    [Complex64::new(1.0/2.0_f64.sqrt(), 0.0), Complex64::new(-1.0/2.0_f64.sqrt(), 0.0)]
];
```

**Behind the scenes:**
- Uses `scirs2_linalg` for matrix operations
- Optimized matrix-vector multiplication
- LAPACK/BLAS integration through SciRS2

#### 4. Random Number Generation (Measurements)
```rust
use quantrs2::prelude::simulation::*;
use scirs2_core::random::prelude::*;

// Reproducible quantum measurements
let mut rng = thread_rng();
let measurement: f64 = rng.gen();
```

**Behind the scenes:**
- Uses `scirs2_core::random` (not raw `rand`)
- High-quality random number generation
- Reproducible seeds for testing

## Using SciRS2 Features

### Automatic Differentiation for VQE

```rust
use quantrs2::prelude::algorithms::*;
use scirs2_autograd::*;

// Define parameterized quantum circuit
let mut circuit = ParameterizedCircuit::new(4);
circuit.add_parameter("theta");
circuit.ry(0, "theta")?;

// Automatic gradient computation via SciRS2
let optimizer = Adam::default(); // From scirs2_optimize
let result = vqe.optimize_with_autodiff(circuit, hamiltonian)?;
```

**Benefits:**
- Automatic gradient computation
- Efficient backpropagation
- GPU-accelerated gradients

### GPU Acceleration

```rust
use quantrs2::prelude::simulation::*;
use quantrs2::config::Config;

// Enable GPU backend (powered by SciRS2)
let cfg = Config::global();
cfg.set_gpu_enabled(true);

// Simulations automatically use GPU when available
let simulator = StateVectorSimulator::new();
let result = simulator.run(&circuit)?;
```

**Behind the scenes:**
- SciRS2 manages GPU memory
- Automatic data transfer (CPU ↔ GPU)
- Metal (macOS), CUDA (Linux/Windows), WebGPU support

### SIMD Optimization

```rust
use quantrs2::prelude::simulation::*;
use quantrs2::config::Config;

// Enable SIMD (powered by SciRS2)
let cfg = Config::global();
cfg.set_simd_enabled(true);

// Gate applications use vectorized operations
circuit.h(0); // Uses AVX2/AVX-512 if available
```

**Benefits:**
- 2-4x speedup on modern CPUs
- Automatic detection (AVX2, AVX-512, NEON)
- No code changes required

### Statistical Analysis

```rust
use quantrs2::prelude::simulation::*;
use scirs2_stats::*;

// Run multiple shots
let shots = 1000;
let result = simulator.run_shots(&circuit, shots)?;

// Statistical analysis via SciRS2
let counts = result.counts();
let mean = counts.values().map(|&v| v as f64).sum::<f64>() / counts.len() as f64;
let variance = scirs2_stats::variance(&counts.values().collect::<Vec<_>>());
```

### Optimization Algorithms

```rust
use quantrs2::prelude::algorithms::*;
use scirs2_optimize::*;

// Use SciRS2 optimizers for quantum algorithms
let optimizer = AdamOptimizer::new()
    .learning_rate(0.01)
    .beta1(0.9)
    .beta2(0.999);

let result = qaoa.optimize_with(optimizer)?;
```

**Available Optimizers (via SciRS2):**
- Adam, AdamW, RMSprop
- L-BFGS-B, BFGS, CG
- Nelder-Mead, Powell
- COBYLA (constrained optimization)

## Performance Optimization

### 1. Enable All SciRS2 Features

```toml
[dependencies]
quantrs2 = { version = "0.1.0-beta.3", features = ["full"] }

# SciRS2 will automatically use:
# - SIMD (AVX2, AVX-512, NEON)
# - GPU (Metal, CUDA, WebGPU)
# - Parallel processing (Rayon)
```

### 2. Configure for Maximum Performance

```rust
use quantrs2::config::Config;

let cfg = Config::global();
cfg.set_num_threads(16);           // Parallel processing
cfg.set_gpu_enabled(true);         // GPU acceleration
cfg.set_simd_enabled(true);        // SIMD vectorization
cfg.set_memory_limit_gb(64);       // Large state vectors
```

### 3. Use SciRS2 Memory Pools

```rust
use quantrs2::prelude::simulation::*;

// SciRS2 automatically manages memory pools
// Reuses allocated buffers for better cache performance
let mut simulator = StateVectorSimulator::new();

// Multiple simulations reuse memory
for circuit in circuits.iter() {
    let result = simulator.run(circuit)?;
    // Memory is reused between runs
}
```

### 4. Leverage SciRS2 Sparse Matrices

```rust
use quantrs2::prelude::simulation::*;
use scirs2_sparse::CsrMatrix;

// For sparse Hamiltonians (many zeros)
let sparse_hamiltonian = CsrMatrix::from_dense(&hamiltonian);

// Much faster for large, sparse systems
let result = vqe.optimize_sparse(sparse_hamiltonian)?;
```

## Best Practices

### ✅ DO: Use SciRS2 Types Consistently

```rust
use quantrs2::prelude::simulation::*;
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;

// Consistent SciRS2 types throughout
let state: Array1<Complex64> = quantum_state();
let operator: Array2<Complex64> = quantum_gate();
```

### ❌ DON'T: Mix Raw Dependencies

```rust
// ❌ DON'T DO THIS
use ndarray::Array2;              // Direct dependency
use num_complex::Complex64;       // Bypasses SciRS2

// This creates type incompatibilities and misses optimizations
```

### ✅ DO: Check SciRS2 Capabilities

```rust
use quantrs2::diagnostics;

let report = diagnostics::run_diagnostics();

// Check what SciRS2 features are available
if report.capabilities.has_avx2 {
    println!("AVX2 SIMD available");
}
if report.capabilities.has_gpu {
    println!("GPU acceleration available");
}
```

### ✅ DO: Use Configuration for Performance

```rust
use quantrs2::config::Config;

// Configure SciRS2 backend
let cfg = Config::global();
cfg.set_default_backend(DefaultBackend::Auto); // Auto-select best backend
```

## Troubleshooting

### Issue: "Cannot find type in scirs2"

**Problem:**
```rust
error[E0433]: failed to resolve: could not find `Complex64` in `scirs2`
```

**Solution:**
Use the correct import path:
```rust
// ✅ Correct
use scirs2_core::Complex64;

// ❌ Wrong
use scirs2::Complex64;  // scirs2 is the umbrella crate
```

### Issue: Performance Not as Expected

**Diagnosis:**
```rust
use quantrs2::diagnostics;

let report = diagnostics::run_diagnostics();
println!("{}", report);

// Check:
// - Is SIMD enabled? (AVX2, AVX-512)
// - Is GPU available?
// - Are there enough threads?
```

**Solution:**
```rust
use quantrs2::config::Config;

let cfg = Config::global();
cfg.set_simd_enabled(true);
cfg.set_gpu_enabled(true);
cfg.set_num_threads(8);
```

### Issue: Memory Errors

**Problem:**
```
thread 'main' panicked at 'out of memory'
```

**Solution:**
```rust
use quantrs2::utils;

// Check memory requirements first
let qubits = 30;
let required = utils::estimate_statevector_memory(qubits);
let available = 16 * 1024 * 1024 * 1024; // 16 GB

if !utils::is_valid_qubit_count(qubits, available) {
    // Use tensor network or stabilizer simulation instead
    use_alternative_backend();
}
```

### Issue: Version Incompatibility

**Problem:**
```
error: version mismatch between scirs2-core and quantrs2
```

**Solution:**
```rust
use quantrs2::version;

// Check version compatibility
match version::check_compatibility() {
    Ok(()) => println!("Versions compatible"),
    Err(issues) => {
        for issue in issues {
            eprintln!("Compatibility issue: {}", issue);
        }
    }
}
```

## Performance Benchmarks

### SciRS2 Impact on QuantRS2 Performance

| Operation | Without SciRS2 | With SciRS2 | Speedup |
|-----------|----------------|-------------|---------|
| Matrix-vector (CPU) | 100 ms | 25 ms | 4x |
| Matrix-vector (GPU) | 100 ms | 5 ms | 20x |
| Complex arithmetic | 50 ms | 15 ms | 3.3x |
| Random sampling | 30 ms | 10 ms | 3x |
| VQE gradient | 200 ms | 40 ms | 5x |

*Benchmarks on Apple M1 Max, 32GB RAM, 10-core CPU*

### SIMD Speedups

```rust
use quantrs2::bench::Timer;

let timer = Timer::start();
// Apply 1000 Hadamard gates
for _ in 0..1000 {
    circuit.h(0);
}
let duration = timer.elapsed();

println!("Gate application: {:?}", duration);
// Without SIMD: ~50ms
// With AVX2: ~15ms (3.3x faster)
// With AVX-512: ~10ms (5x faster)
```

## Further Reading

- [SciRS2 Documentation](https://docs.rs/scirs2-core)
- [QuantRS2 SCIRS2_INTEGRATION_POLICY.md](../SCIRS2_INTEGRATION_POLICY.md)
- [SciRS2 Performance Guide](https://github.com/cool-japan/scirs/blob/master/PERFORMANCE.md)
- [OptiRS Documentation](https://docs.rs/optirs-core)

## Example: Complete SciRS2-Powered Quantum Application

```rust
use quantrs2::prelude::simulation::*;
use quantrs2::{config, diagnostics, version};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Validate SciRS2 integration
    version::check_compatibility()?;

    // Step 2: Check SciRS2 capabilities
    let report = diagnostics::run_diagnostics();
    println!("SciRS2 Capabilities:");
    println!("  - SIMD: {}", report.capabilities.has_avx2);
    println!("  - GPU: {}", report.capabilities.has_gpu);

    // Step 3: Configure SciRS2 backend
    let cfg = config::Config::global();
    cfg.set_gpu_enabled(report.capabilities.has_gpu);
    cfg.set_simd_enabled(report.capabilities.has_avx2);

    // Step 4: Create quantum circuit
    let mut circuit = Circuit::<3>::new();
    circuit.h(0);
    circuit.cnot(0, 1);
    circuit.cnot(1, 2);

    // Step 5: Run simulation (powered by SciRS2)
    let simulator = StateVectorSimulator::new();
    let result = simulator.run(&circuit)?;

    // Step 6: Analyze using SciRS2 statistics
    println!("GHZ state created!");
    println!("Probabilities: {:?}", result.probabilities());

    Ok(())
}
```

---

**Version:** 0.1.0-beta.3
**Last Updated:** 2025-12-04
**SciRS2 Version:** v0.1.0-rc.2
