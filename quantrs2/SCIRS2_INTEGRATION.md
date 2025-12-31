# SciRS2 Integration in QuantRS2 Facade

This document explains how QuantRS2 leverages SciRS2 (Scientific Computing in Rust) as its scientific computing foundation, with specific patterns and best practices for the facade crate.

## Overview

The QuantRS2 facade crate provides a unified entry point to the quantum computing framework while strictly adhering to the SciRS2 Policy. This ensures:

- **Consistency**: All subcrates use identical SciRS2 patterns
- **Performance**: Zero-cost abstractions with hardware-accelerated operations
- **Safety**: Type-safe quantum operations with Rust's guarantees
- **Maintainability**: Single source of truth for scientific computing

## SciRS2 Policy Summary

### ✅ REQUIRED Patterns

#### 1. Complex Numbers (Quantum Amplitudes)

```rust
// ✅ CORRECT: Use scirs2_core root exports
use scirs2_core::{Complex64, Complex32};

// Quantum state amplitudes
let alpha = Complex64::new(0.707, 0.0);
let beta = Complex64::new(0.0, 0.707);

// Complex arithmetic
let inner_product = alpha * beta.conj();
let magnitude = (alpha * alpha.conj()).re.sqrt();

// ❌ NEVER: Direct num-complex usage
use num_complex::Complex64;  // POLICY VIOLATION
```

#### 2. Array Operations (State Vectors & Operators)

```rust
// ✅ CORRECT: Unified ndarray access from scirs2_core
use scirs2_core::ndarray::*;  // Complete unified access
// Or selective:
use scirs2_core::ndarray::{Array1, Array2, array, s, Axis};

// Quantum state vector
let state: Array1<Complex64> = array![
    Complex64::new(0.707, 0.0),  // |0⟩
    Complex64::new(0.707, 0.0),  // |1⟩
];

// Quantum operator matrix
let hadamard: Array2<Complex64> = array![
    [Complex64::new(0.707, 0.0), Complex64::new(0.707, 0.0)],
    [Complex64::new(0.707, 0.0), Complex64::new(-0.707, 0.0)]
];

// Array operations
let slice = state.slice(s![0..1]);
let norm: f64 = state.iter().map(|c| (c * c.conj()).re).sum();

// ❌ WRONG: Fragmented SciRS2 usage
use scirs2_autograd::ndarray::{Array2, array};  // DON'T USE - autograd path

// ❌ NEVER: Direct ndarray usage
use ndarray::{Array1, array};  // POLICY VIOLATION
```

#### 3. Random Number Generation (Measurements)

```rust
// ✅ CORRECT: Unified random from scirs2_core
use scirs2_core::random::prelude::*;  // Common distributions & RNG
// Or selective:
use scirs2_core::random::{thread_rng, Normal as RandNormal, RandBeta, StudentT};

// Enhanced unified distributions
use scirs2_core::random::distributions_unified::{UnifiedNormal, UnifiedBeta};

// Fast measurements
let mut rng = thread_rng();
let measurement: f64 = rng.gen();

// Reproducible experiments
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
let mut seeded_rng = StdRng::seed_from_u64(42);

// Gaussian noise
let normal = RandNormal::new(0.0, 0.1).unwrap();
let noise = normal.sample(&mut rng);

// ❌ NEVER: Direct rand usage
use rand::{Rng, thread_rng};  // POLICY VIOLATION
use rand_distr::Normal;        // POLICY VIOLATION
```

#### 4. SIMD Operations (Performance)

```rust
// ✅ CORRECT: Use scirs2_core SIMD operations
use scirs2_core::simd_ops::{SimdOps, PlatformCapabilities};

// Check platform capabilities
let caps = PlatformCapabilities::current();
if caps.has_avx2() {
    // Use AVX2-accelerated quantum gates
    scirs2_core::simd_ops::vectorized_complex_multiply(&mut state, &gate);
}

// Platform-specific optimizations
match caps.optimal_vector_size() {
    32 => { /* AVX-512 path */ },
    16 => { /* AVX2 path */ },
    8 => { /* SSE2 path */ },
    _ => { /* Scalar path */ },
}
```

### ❌ FORBIDDEN Direct Dependencies

These dependencies MUST NOT appear in any QuantRS2 crate `Cargo.toml`:

```toml
# ❌ POLICY VIOLATIONS - DO NOT USE
[dependencies]
ndarray = "*"           # Use scirs2_core::ndarray
rand = "*"              # Use scirs2_core::random
rand_distr = "*"        # Use scirs2_core::random
num-complex = "*"       # Use scirs2_core::{Complex64, Complex32}
num-traits = "*"        # Use scirs2_core::numeric
rayon = "*"             # Use scirs2_core::parallel_ops
nalgebra = "*"          # Use scirs2_linalg
ndarray-linalg = "*"    # Use scirs2_linalg
```

## QuantRS2 Facade SciRS2 Integration

### Module-Specific Patterns

#### quantrs2::prelude

All preludes re-export SciRS2 types for convenience:

```rust
// Essentials prelude includes SciRS2 primitives
use quantrs2::prelude::essentials::*;
// Provides: Complex64, Array types (when features enabled)

// Simulation prelude includes SciRS2 operations
use quantrs2::prelude::simulation::*;
// Provides: SciRS2 linear algebra, FFT, random
```

#### quantrs2::utils

Utility functions leverage SciRS2:

```rust
use quantrs2::utils;

// Memory estimation uses SciRS2 types
let memory = utils::estimate_statevector_memory(30);  // Uses scirs2 arrays

// Quantum math uses SciRS2 numerics
let entropy = utils::entropy(&probabilities);  // Uses scirs2 stats
let fidelity = utils::classical_fidelity(&p1, &p2);  // Uses scirs2 math
```

#### quantrs2::config

Configuration for SciRS2 backends:

```rust
use quantrs2::config::Config;

Config::global()
    .set_gpu_enabled(true)     // Enables scirs2_gpu
    .set_simd_enabled(true)    // Enables scirs2 SIMD
    .set_num_threads(8);       // Configures scirs2 parallel
```

#### quantrs2::diagnostics

Hardware detection uses SciRS2:

```rust
use quantrs2::diagnostics;

let report = diagnostics::run_diagnostics();
// Detects: SIMD capabilities, GPU (Metal/CUDA), memory
// All through scirs2_core platform APIs
```

## Feature-Specific SciRS2 Usage

### Circuit Feature

```rust
#[cfg(feature = "circuit")]
use scirs2_core::ndarray::Array2;  // For gate matrices
use scirs2_linalg::unitary;        // For gate validation
```

### Simulation Feature

```rust
#[cfg(feature = "sim")]
use scirs2_linalg::*;              // Matrix operations
use scirs2_sparse::*;              // Sparse state vectors
use scirs2_fft::*;                 // Quantum Fourier Transform
```

### ML Feature

```rust
#[cfg(feature = "ml")]
use scirs2_autograd::*;            // Gradient computation
use scirs2_optimize::*;            // VQE/QAOA optimizers
use scirs2_neural::*;              // Quantum neural networks
```

### Annealing Feature

```rust
#[cfg(feature = "anneal")]
use scirs2_optimize::*;            // Annealing schedules
use scirs2_stats::*;               // Statistical analysis
use scirs2_core::random::*;       // Monte Carlo sampling
```

## Performance Optimization with SciRS2

### 1. SIMD-Accelerated Quantum Gates

```rust
use scirs2_core::simd_ops::PlatformCapabilities;

let caps = PlatformCapabilities::current();

// Automatic vectorization for quantum operations
if caps.has_avx2() {
    // 2-4x speedup for gate applications
    apply_gate_simd(&mut state, &gate);
} else {
    apply_gate_scalar(&mut state, &gate);
}
```

### 2. GPU Acceleration

```rust
#[cfg(feature = "sim")]
use scirs2_gpu::*;

if quantrs2::diagnostics::has_gpu() {
    // 10-100x speedup for large circuits
    let gpu_simulator = GpuStateVectorSimulator::new()?;
    gpu_simulator.run(&circuit)?;
}
```

### 3. Parallel Operations

```rust
use scirs2_core::parallel_ops::par_chunks;

// Parallel gate application to independent qubits
state.par_chunks_mut(chunk_size)
    .for_each(|chunk| apply_gate_to_chunk(chunk, &gate));
```

### 4. Memory-Efficient Representations

```rust
use scirs2_sparse::CsrMatrix;

// For circuits with limited entanglement
let sparse_state = CsrMatrix::<Complex64>::from_dense(&state);
// Can represent states with <1% memory of dense representation
```

## Testing with SciRS2

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::{Complex64};
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantum_state_norm() {
        let state = array![
            Complex64::new(0.707, 0.0),
            Complex64::new(0.707, 0.0)
        ];
        let norm: f64 = state.iter()
            .map(|c| (c * c.conj()).re)
            .sum();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
```

### Reproducible Tests

```rust
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;

#[test]
fn test_measurement_reproducibility() {
    let mut rng = StdRng::seed_from_u64(42);
    let result1 = quantum_algorithm(&mut rng);

    let mut rng = StdRng::seed_from_u64(42);
    let result2 = quantum_algorithm(&mut rng);

    assert_eq!(result1, result2);
}
```

## Common Patterns and Anti-Patterns

### ✅ Pattern: Unified Imports

```rust
// Import everything you need from scirs2_core
use scirs2_core::{Complex64, Complex32};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
```

### ❌ Anti-Pattern: Mixed Sources

```rust
// DON'T MIX SciRS2 and direct dependencies
use scirs2_core::Complex64;
use ndarray::Array1;  // ❌ Inconsistent!
use rand::thread_rng;  // ❌ Inconsistent!
```

### ✅ Pattern: Feature-Gated SciRS2 Extensions

```rust
#[cfg(feature = "ml")]
use scirs2_autograd::AutoDiff;

#[cfg(feature = "sim")]
use scirs2_linalg::LinAlg;
```

### ❌ Anti-Pattern: Hardcoded External Deps

```rust
// DON'T bypass SciRS2 even in feature-gated code
#[cfg(feature = "ml")]
use nalgebra::Matrix;  // ❌ Should use scirs2_linalg!
```

## Migration from Direct Dependencies

If you find code using direct dependencies, migrate as follows:

### Complex Numbers

```rust
// Before (WRONG):
use num_complex::Complex64;
let c = Complex64::new(1.0, 2.0);

// After (CORRECT):
use scirs2_core::Complex64;
let c = Complex64::new(1.0, 2.0);
```

### Arrays

```rust
// Before (WRONG):
use ndarray::{Array1, array};
let arr: Array1<f64> = array![1.0, 2.0, 3.0];

// After (CORRECT):
use scirs2_core::ndarray::{Array1, array};
let arr: Array1<f64> = array![1.0, 2.0, 3.0];
```

### Random Numbers

```rust
// Before (WRONG):
use rand::{thread_rng, Rng};
use rand_distr::Normal;
let mut rng = thread_rng();
let normal = Normal::new(0.0, 1.0).unwrap();

// After (CORRECT):
use scirs2_core::random::{thread_rng, Normal as RandNormal};
use scirs2_core::random::prelude::*;
let mut rng = thread_rng();
let normal = RandNormal::new(0.0, 1.0).unwrap();
```

## Version Compatibility

QuantRS2 v0.1.0-rc.2 requires:

- **SciRS2**: v0.1.1 (stable)
- **NumRS2**: v0.1.1 (stable)
- **OptiRS**: v0.1.0 (stable)

Version compatibility is automatically checked via:

```rust
use quantrs2::version;

version::check_compatibility()?;
// Validates SciRS2 and related versions
```

## Resources

- **Main Policy**: `../SCIRS2_INTEGRATION_POLICY.md`
- **SciRS2 Repository**: https://github.com/cool-japan/scirs
- **Example Code**: `examples/scirs2_integration.rs`
- **Unit Tests**: `tests/integration_cross_subcrate.rs` (symengine_integration module)

## Quick Reference Card

```rust
// ═══════════════════════════════════════════════════════════
//  QuantRS2 SciRS2 Integration - Quick Reference
// ═══════════════════════════════════════════════════════════

// Complex Numbers (Quantum Amplitudes)
use scirs2_core::{Complex64, Complex32};

// Arrays (State Vectors & Operators)
use scirs2_core::ndarray::*;  // Unified complete access
// Or selective:
use scirs2_core::ndarray::{Array1, Array2, array, s, Axis};

// Random Numbers (Measurements)
use scirs2_core::random::prelude::*;
use scirs2_core::random::{thread_rng, Normal as RandNormal};

// Enhanced Distributions
use scirs2_core::random::distributions_unified::{UnifiedNormal, UnifiedBeta};

// SIMD (Performance)
use scirs2_core::simd_ops::PlatformCapabilities;

// ❌ NEVER USE DIRECTLY:
// - ndarray, rand, rand_distr, num-complex, num-traits
// - nalgebra, ndarray-linalg, rayon
```

## Verification

To verify SciRS2 policy compliance:

```bash
# Check for policy violations
grep -r "use ndarray::" src/
grep -r "use rand::" src/
grep -r "use num_complex::" src/

# Should return no results in compliant code

# Check for correct patterns
grep -r "use scirs2_core::" src/
# Should find unified SciRS2 usage
```

## Support

For questions about SciRS2 integration:

1. Check this document first
2. Review `../SCIRS2_INTEGRATION_POLICY.md`
3. See example: `examples/scirs2_integration.rs`
4. Run diagnostics: `cargo run --example diagnostics`

**Remember**: SciRS2 compliance ensures consistency, performance, and maintainability across all QuantRS2 components.
