# SciRS2 Integration Policy for QuantRS2

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**QuantRS2 MUST use SciRS2 as its scientific computing foundation.** This document establishes the policy for proper, minimal, and effective integration of SciRS2 crates into QuantRS2.

## Core Integration Principles

### 1. **Foundation, Not Dependency Bloat**
- QuantRS2 extends SciRS2's capabilities with quantum computing specialization
- Use SciRS2 crates **only when actually needed** by QuantRS2 functionality
- **DO NOT** add SciRS2 crates "just in case" - add them when code requires them

### 2. **Evidence-Based Integration**
- Each SciRS2 crate must have **clear justification** based on QuantRS2 features
- Document **specific use cases** for each integrated SciRS2 crate
- Remove unused SciRS2 dependencies during code reviews

### 3. **Architectural Hierarchy**
```
QuantRS2 (Quantum Computing Framework)
    ↓ builds upon
OptiRS (ML Optimization Specialization)
    ↓ builds upon
SciRS2 (Scientific Computing Foundation)
    ↓ builds upon
ndarray, num-traits, etc. (Core Rust Scientific Stack)
```

## Required SciRS2 Crates Analysis

### **ESSENTIAL (Always Required)**

#### `scirs2-core` - FOUNDATION
- **Use Cases**: Core scientific primitives, ScientificNumber trait, random number generation, SIMD operations, parallel computing, complex number support
- **QuantRS2 Modules**: `quantrs2-core`, `quantrs2-sim`, all modules use core utilities
- **Features**: SIMD ops, parallel ops, GPU support, complex arithmetic
- **Status**: ✅ REQUIRED - Foundation crate

#### `scirs2` - MAIN INTEGRATION
- **Use Cases**: Scientific computing integration, linear algebra, complex operations
- **QuantRS2 Modules**: Core operations throughout
- **Features**: ["standard", "linalg", "complex"]
- **Status**: ✅ REQUIRED - Main integration crate

### **HIGHLY LIKELY REQUIRED**

#### `scirs2-autograd` - AUTOMATIC DIFFERENTIATION & ARRAYS
- **Use Cases**: Gradient computation for quantum machine learning, **array! macro access**, ndarray types
- **QuantRS2 Modules**: `quantrs2-ml`, test modules throughout
- **Status**: ✅ REQUIRED - Quantum ML gradients and array operations
- **Special Note**: Essential for `array!` macro in tests and QML gradient descent

#### `scirs2-linalg` - LINEAR ALGEBRA
- **Use Cases**: Matrix operations, unitary transformations, eigenvalue problems, quantum state representation
- **QuantRS2 Modules**: `quantrs2-core`, `quantrs2-sim`, `quantrs2-circuit`
- **Status**: ✅ REQUIRED - Quantum state manipulation

#### `scirs2-optimize` - OPTIMIZATION
- **Use Cases**: Variational quantum algorithms (VQE, QAOA), parameter optimization
- **QuantRS2 Modules**: `quantrs2-ml`, `quantrs2-anneal`, quantum algorithms
- **Status**: ✅ REQUIRED - Quantum optimization algorithms
- **Note**: Also integrates with OptiRS for advanced optimizers

#### `scirs2-special` - SPECIAL FUNCTIONS
- **Use Cases**: Quantum phase functions, error functions for quantum error correction
- **QuantRS2 Modules**: `quantrs2-core`, error correction modules
- **Status**: ✅ REQUIRED - Mathematical special functions

#### `scirs2-sparse` - SPARSE MATRICES
- **Use Cases**: Sparse quantum state representation, tensor network simulation
- **QuantRS2 Modules**: `quantrs2-sim` (tensor network backend)
- **Status**: ✅ REQUIRED - Memory-efficient simulation

#### `scirs2-fft` - FAST FOURIER TRANSFORM
- **Use Cases**: Quantum Fourier Transform (QFT), phase estimation, period finding
- **QuantRS2 Modules**: `quantrs2-circuit`, quantum algorithms
- **Status**: ✅ REQUIRED - QFT implementation

### **CONDITIONALLY REQUIRED**

#### `scirs2-neural` - NEURAL NETWORKS
- **Use Cases**: Quantum neural networks, hybrid quantum-classical models
- **QuantRS2 Modules**: `quantrs2-ml`
- **Status**: ✅ REQUIRED - Quantum machine learning

#### `scirs2-signal` - SIGNAL PROCESSING
- **Use Cases**: Quantum signal processing, quantum walk algorithms
- **QuantRS2 Modules**: `quantrs2-sim`, specialized algorithms
- **Status**: ✅ REQUIRED - Signal processing capabilities

#### `scirs2-metrics` - EVALUATION METRICS
- **Use Cases**: Quantum state fidelity, entanglement measures, quantum metrics
- **QuantRS2 Modules**: `quantrs2-sim`, benchmarking tools
- **Status**: ✅ REQUIRED - Quantum state analysis

#### `scirs2-stats` - STATISTICAL ANALYSIS
- **Use Cases**: Quantum state tomography, measurement statistics, error analysis
- **QuantRS2 Modules**: `quantrs2-device`, measurement analysis
- **Status**: ✅ REQUIRED - Statistical quantum analysis

#### `scirs2-cluster` - CLUSTERING ALGORITHMS
- **Use Cases**: Quantum clustering algorithms, quantum k-means
- **QuantRS2 Modules**: `quantrs2-ml` (quantum clustering)
- **Status**: ✅ REQUIRED - Quantum clustering algorithms

#### `scirs2-graph` - GRAPH ALGORITHMS
- **Use Cases**: Quantum walk on graphs, graph state preparation, QAOA
- **QuantRS2 Modules**: `quantrs2-circuit` (graph states), QAOA implementation
- **Status**: ✅ REQUIRED - Graph quantum algorithms

#### `scirs2-spatial` - SPATIAL DATA PROCESSING
- **Use Cases**: Quantum lattice models, spatial quantum simulations
- **QuantRS2 Modules**: `quantrs2-sim` (lattice simulations)
- **Status**: ✅ REQUIRED - Spatial quantum models

### **DOMAIN-SPECIFIC (Optional)**

#### `scirs2-series` - TIME SERIES ANALYSIS
- **Use Cases**: Quantum time evolution, dynamical simulations
- **QuantRS2 Modules**: `quantrs2-sim` (time evolution)
- **Status**: ⚠️ OPTIONAL - Only for time-dependent simulations

#### `scirs2-datasets` - DATA HANDLING
- **Use Cases**: Quantum benchmark datasets, test circuits
- **QuantRS2 Modules**: `quantrs2-circuit` (benchmark circuits)
- **Status**: ⚠️ OPTIONAL - For standardized test data

#### `scirs2-text` - TEXT PROCESSING
- **Status**: ❌ UNLIKELY - Not applicable to quantum computing

#### `scirs2-vision` - COMPUTER VISION
- **Status**: ❌ UNLIKELY - Not applicable to quantum computing

## 🚨 SciRS2 DEPENDENCY ABSTRACTION POLICY

### Core Principle: NO Direct External Dependencies in QuantRS2

**Applies to:** ALL QuantRS2 crates
- All QuantRS2 packages (quantrs2-core, quantrs2-sim, quantrs2-ml, etc.)
- All tests, examples, benchmarks in all crates
- All integration tests and documentation examples

#### Prohibited Direct Dependencies in Cargo.toml:
```toml
# ❌ FORBIDDEN in QuantRS2 crates (these are POLICY VIOLATIONS)
[dependencies]
rand = { workspace = true }              # ❌ Use scirs2-core instead
rand_distr = { workspace = true }        # ❌ Use scirs2-core instead
rand_core = { workspace = true }         # ❌ Use scirs2-core instead
rand_chacha = { workspace = true }       # ❌ Use scirs2-core instead
ndarray = { workspace = true }           # ❌ Use scirs2-core instead
ndarray-rand = { workspace = true }      # ❌ Use scirs2-core instead
ndarray-stats = { workspace = true }     # ❌ Use scirs2-core instead
num-traits = { workspace = true }        # ❌ Use scirs2-core instead
num-complex = { workspace = true }       # ❌ Use scirs2-core instead
num-integer = { workspace = true }       # ❌ Use scirs2-core instead
nalgebra = { workspace = true }          # ❌ Use scirs2-core instead
```

#### Required SciRS2 Dependencies:
```toml
# ✅ REQUIRED in QuantRS2 crates
[dependencies]
scirs2-core = { workspace = true, features = ["array", "random"] }
# Additional SciRS2 crates as needed per module requirements
```

## 🏆 PROVEN SUCCESSFUL PATTERNS

### Array Import Patterns - **CRITICAL GUIDANCE**

```rust
// ✅ CORRECT - Unified SciRS2 usage (PROVEN PATTERN - v0.1.0-beta.4+)
use scirs2_core::ndarray::*;  // Complete unified access
// Or selective:
use scirs2_core::ndarray::{Array1, Array2, array, s, Axis};

// ❌ WRONG - Fragmented SciRS2 usage (DEPRECATED - DO NOT USE)
use scirs2_autograd::ndarray::{Array2, array};  // Fragmented - DEPRECATED
use scirs2_core::ndarray_ext::{ArrayView};      // Missing macros - DEPRECATED

// ❌ POLICY VIOLATION: Never use ndarray directly
use ndarray::{Array, array};  // FORBIDDEN - Policy violation
```

**Complete Dependency Mapping:**
| External Crate | SciRS2-Core Module | Usage in QuantRS2 |
|----------------|-------------------|------------------|
| `ndarray` | `scirs2_core::ndarray` | All array operations |
| `ndarray-rand` | `scirs2_core::ndarray` | Via `array` feature |
| `ndarray-stats` | `scirs2_core::ndarray` | Via `array` feature |
| `rand` | `scirs2_core::random` | All RNG operations |
| `rand_distr` | `scirs2_core::random` | All distributions |
| `num-complex` | `scirs2_core` (root) | Complex numbers |
| `num-traits` | `scirs2_core::numeric` | All traits |

**Key Decision Points**:
- **ALL array operations** → Use `scirs2_core::ndarray::*` or selective imports from `scirs2_core::ndarray::{...}`
- **NEVER** use `scirs2_autograd::ndarray` (deprecated fragmented approach)
- **NEVER** use `scirs2_core::ndarray_ext` (deprecated, missing macros)
- **NEVER** import `ndarray`, `rand`, or `num-complex` directly

### Complex Number Integration - **QUANTUM-SPECIFIC**

```rust
// ✅ CORRECT: Complex numbers for quantum amplitudes (direct from scirs2-core root)
use scirs2_core::{Complex64, Complex32};  // Import from root (v0.1.0-beta.4+)
use scirs2_core::complex::ComplexFloat;   // Traits

// ✅ CORRECT: Quantum state representation
type QuantumState = Array1<Complex64>;
type DensityMatrix = Array2<Complex64>;

// ❌ WRONG: Using num-complex directly (POLICY VIOLATION)
use num_complex::Complex64;  // FORBIDDEN - Violates SciRS2 policy
```

### Random Number Generation - **QUANTUM MEASUREMENT**

```rust
// ✅ CORRECT - Unified SciRS2 usage (PROVEN PATTERN - v0.1.0-beta.4+)
use scirs2_core::random::prelude::*;  // Common distributions & RNG
// Or selective:
use scirs2_core::random::{thread_rng, Normal as RandNormal, RandBeta, StudentT};

// ✅ CORRECT - Enhanced unified interface
use scirs2_core::random::distributions_unified::{UnifiedNormal, UnifiedBeta};

// ✅ CLEAN TYPE DECLARATIONS
struct QuantumSimulator {
    rng: ThreadRng,  // Use thread_rng() for most cases
}

// ✅ PROPER INITIALIZATION
impl QuantumSimulator {
    pub fn new() -> Self {
        Self { rng: thread_rng() }  // Fast quantum measurements
    }

    pub fn with_seed(seed: u64) -> Self {
        Self { rng: seeded_rng(seed) }  // Reproducible if needed
    }
}

// ❌ WRONG: Direct rand usage (POLICY VIOLATION)
use rand::{Rng, thread_rng};  // FORBIDDEN
use rand_distr::Normal;       // FORBIDDEN
```

### SIMD Operations - **PERFORMANCE CRITICAL**

```rust
// ✅ CORRECT: SIMD-accelerated quantum operations
use scirs2_core::simd_ops::{SimdOps, PlatformCapabilities};

impl QuantumSimulator {
    fn apply_gate(&mut self, gate: &Gate) {
        if PlatformCapabilities::current().has_avx2() {
            // Use AVX2 optimized implementation
            scirs2_core::simd_ops::vectorized_complex_multiply(
                &mut self.state_vector,
                &gate.matrix
            );
        } else {
            // Fallback to standard implementation
            self.apply_gate_standard(gate);
        }
    }
}
```

### Parallel Computing - **MULTI-QUBIT OPERATIONS**

```rust
// ✅ CORRECT: Parallel quantum gate application
use scirs2_core::parallel_ops::{parallel_for, ParallelIterator};

impl QuantumCircuit {
    fn apply_parallel_gates(&mut self) {
        parallel_for(&mut self.qubits, |qubit| {
            // Apply single-qubit gates in parallel
            qubit.apply_gate(&gate);
        });
    }
}
```

## QuantRS2 Module-Specific SciRS2 Usage

### quantrs2-core
```rust
// Complex numbers (direct from root)
use scirs2_core::{Complex64, Complex32};
// Arrays (unified access)
use scirs2_core::ndarray::{Array1, Array2, array, s};
// Random (unified interface)
use scirs2_core::random::prelude::*;
// Performance
use scirs2_core::simd_ops;
use scirs2_core::parallel_ops;
```

### quantrs2-sim
```rust
use scirs2_linalg;                              // Matrix operations
use scirs2_sparse;                              // Sparse state representation
use scirs2_core::gpu;                           // GPU acceleration
use scirs2_fft;                                 // Quantum Fourier Transform
use scirs2_core::ndarray::{Array2, ArrayView2}; // Unified array access
```

### quantrs2-ml
```rust
use scirs2_neural;                              // Quantum neural networks
use scirs2_autograd;                            // Gradient computation (NOT for arrays)
use scirs2_optimize;                            // Parameter optimization
use optirs;                                     // Advanced optimizers
use scirs2_core::ndarray::{Array1, Array2};     // Unified array access
```

### quantrs2-circuit
```rust
use scirs2_graph;                               // Graph state circuits
use scirs2_core::{Complex64, Complex32};        // Gate matrices (from root)
use scirs2_linalg::unitary;                     // Unitary verification
use scirs2_core::ndarray::{Array2, array};      // Unified array access
```

### quantrs2-anneal
```rust
use scirs2_optimize;                            // Annealing optimization
use scirs2_stats;                               // Statistical analysis
use scirs2_core::random::prelude::*;            // Monte Carlo sampling (unified)
use scirs2_core::ndarray::{Array1, Array2};     // Unified array access
```

### quantrs2-device
```rust
use scirs2_stats;                               // Measurement statistics
use scirs2_metrics;                             // Fidelity metrics
use scirs2_core::random::prelude::*;            // Noise simulation (unified)
use scirs2_core::ndarray::Array1;               // Unified array access
```

## 📋 STANDARD RESOLUTION WORKFLOW

### Proven 5-Step Migration Process (SciRS2 v0.1.0-beta.4+ Compliance)

1. **Cargo.toml Cleanup**
   ```toml
   # ❌ REMOVE: All direct dependencies violating SciRS2 POLICY
   # rand = { workspace = true }           # REMOVED: Use scirs2_core::random (SciRS2 POLICY)
   # rand_distr = { workspace = true }     # REMOVED: Use scirs2_core::random (SciRS2 POLICY)
   # ndarray = { workspace = true }        # REMOVED: Use scirs2_core::ndarray (SciRS2 POLICY)
   # num-complex = { workspace = true }    # REMOVED: Use scirs2_core (root) (SciRS2 POLICY)
   # num-traits = { workspace = true }     # REMOVED: Use scirs2_core::numeric (SciRS2 POLICY)

   # ✅ SciRS2 POLICY COMPLIANT dependencies
   scirs2-core = { workspace = true, features = ["array", "random"] }
   # Additional SciRS2 crates as needed
   ```

2. **Import Path Migration**
   ```rust
   // ❌ OLD PATTERN (violated SciRS2 POLICY)
   use rand::{thread_rng, Rng};
   use rand_distr::Normal;
   use ndarray::{Array, Array1, Array2, array, s};
   use num_complex::Complex64;

   // ✅ NEW PATTERN (SciRS2 v0.1.0-beta.4+ compliant - UNIFIED APPROACH)
   use scirs2_core::random::prelude::*;                  // Unified random
   use scirs2_core::ndarray::{Array, Array1, Array2, array, s};  // Unified arrays
   use scirs2_core::{Complex64, Complex32};              // Complex from root
   ```

3. **Type Declaration Fixes**
   ```rust
   // ❌ OLD: Direct external types
   struct QuantumState {
       amplitudes: Array1<Complex<f64>>,
       rng: ThreadRng,
   }

   // ✅ NEW: SciRS2 unified types
   use scirs2_core::{Complex64};
   use scirs2_core::ndarray::Array1;
   use scirs2_core::random::ThreadRng;

   struct QuantumState {
       amplitudes: Array1<Complex64>,
       rng: ThreadRng,
   }
   ```

4. **Initialization Updates**
   ```rust
   // ❌ OLD: Direct external initialization
   Self { rng: rand::thread_rng() }

   // ✅ NEW: SciRS2 unified patterns
   use scirs2_core::random::{thread_rng, seeded_rng};

   Self { rng: thread_rng() }      // For fast, non-deterministic
   Self { rng: seeded_rng(42) }    // For reproducible behavior
   ```

5. **Compilation Validation**
   - Remove ALL direct rand/ndarray/num-complex/num-traits dependencies from Cargo.toml
   - Test individual package compilation: `cargo build --package <package-name>`
   - Verify unified patterns: `scirs2_core::ndarray::*`, `scirs2_core::random::*`, `scirs2_core::{Complex64, Complex32}`
   - Ensure consistent SciRS2 usage across all modules

## Current Workspace Integration

### SciRS2 Dependencies (v0.1.0-rc.4)
```toml
[workspace.dependencies]
# Essential SciRS2 dependencies for QuantRS2 - COMPREHENSIVE INTEGRATION
# Status: Production Ready with SciRS2 foundation
scirs2 = { version = "0.1.0-rc.4", features = ["standard", "linalg", "complex"], default-features = false }
scirs2-core = { version = "0.1.0-rc.4", default-features = false }

# HIGHLY LIKELY REQUIRED SciRS2 crates
scirs2-autograd = { version = "0.1.0-rc.4", default-features = false }  # Primary source for ndarray types
scirs2-linalg = { version = "0.1.0-rc.4", default-features = false }
scirs2-optimize = { version = "0.1.0-rc.4", default-features = false }
scirs2-special = { version = "0.1.0-rc.4", default-features = false }
scirs2-sparse = { version = "0.1.0-rc.4", default-features = false }
scirs2-fft = { version = "0.1.0-rc.4", default-features = false }

# CONDITIONALLY REQUIRED SciRS2 crates
scirs2-neural = { version = "0.1.0-rc.4", default-features = false }
scirs2-signal = { version = "0.1.0-rc.4", default-features = false }
scirs2-metrics = { version = "0.1.0-rc.4", default-features = false }
scirs2-stats = { version = "0.1.0-rc.4", default-features = false }
scirs2-cluster = { version = "0.1.0-rc.4", default-features = false }
scirs2-graph = { version = "0.1.0-rc.4", default-features = false }
scirs2-spatial = { version = "0.1.0-rc.4", default-features = false }

# DOMAIN-SPECIFIC (Optional)
scirs2-series = { version = "0.1.0-rc.4", default-features = false }
scirs2-datasets = { version = "0.1.0-rc.4", default-features = false }

# Python bindings support (REQUIRED for quantrs2-py)
scirs2-numpy = { version = "0.1.0-rc.4" }  # SciRS2-compatible numpy bindings with ndarray 0.17+ support

# OptiRS integration for advanced optimization (VQE, QAOA)
optirs = { path = "../optirs/optirs", default-features = false }
optirs-core = { path = "../optirs/optirs-core", default-features = false }
```

### Module-Specific Usage Examples
```toml
# quantrs2-core
scirs2-core = { workspace = true }
scirs2 = { workspace = true, features = ["complex"] }

# quantrs2-sim
scirs2-linalg = { workspace = true }
scirs2-sparse = { workspace = true }
scirs2-fft = { workspace = true }

# quantrs2-ml
scirs2-neural = { workspace = true }
scirs2-autograd = { workspace = true }
scirs2-optimize = { workspace = true }
optirs = { workspace = true }

# quantrs2-circuit
scirs2-graph = { workspace = true }
scirs2-linalg = { workspace = true }

# quantrs2-anneal
scirs2-optimize = { workspace = true }
scirs2-stats = { workspace = true }

# quantrs2-device
scirs2-stats = { workspace = true }
scirs2-metrics = { workspace = true }
```

## Best Practices

### Import Granularity
```rust
// ✅ GOOD - Specific imports
use scirs2_core::complex::Complex64;
use scirs2_core::simd_ops::vectorized_apply;

// ❌ BAD - Broad imports
use scirs2_core::*;
use scirs2::*;
```

### Complex Number Operations
```rust
// ✅ GOOD - SciRS2 complex types
use scirs2_core::complex::{Complex64, ComplexFloat};

// ❌ BAD - Direct num-complex
use num_complex::Complex64;
```

### Quantum-Specific Patterns
```rust
// ✅ GOOD - Quantum state initialization
use scirs2_autograd::ndarray::{Array1, array};
use scirs2_core::complex::Complex64;

let initial_state: Array1<Complex64> = array![
    Complex64::new(1.0, 0.0),
    Complex64::new(0.0, 0.0),
];

// ✅ GOOD - Measurement sampling
use scirs2_core::random::{CoreRandom, seeded_rng};
use scirs2_core::essentials::Uniform;

let mut rng = seeded_rng(42);
let measurement_prob: f64 = rng.sample(Uniform::new(0.0, 1.0));
```

## Anti-Patterns to Avoid

### ❌ Common Mistakes (POLICY VIOLATIONS)
```rust
// ❌ WRONG - Direct external dependencies (FORBIDDEN)
use ndarray::{Array2, array};           // Policy violation
use rand::{Rng, thread_rng};            // Policy violation
use rand_distr::Normal;                 // Policy violation
use num_complex::Complex64;             // Policy violation
use num_traits::Float;                  // Policy violation

// ❌ WRONG - Deprecated fragmented SciRS2 usage
use scirs2_autograd::ndarray::{Array2, array};  // Deprecated
use scirs2_core::ndarray_ext::{ArrayView};      // Deprecated

// ❌ WRONG - Incorrect complex number import
use scirs2_core::complex::{Complex64};  // Should use root import
```

### ✅ Correct Patterns (SciRS2 v0.1.0-beta.4+ Unified Approach)
```rust
// ✅ CORRECT - Unified SciRS2 usage (PROVEN PATTERNS)
use scirs2_core::ndarray::*;  // Complete unified access
// Or selective:
use scirs2_core::ndarray::{Array2, array, s, Axis};

use scirs2_core::random::prelude::*;  // Common distributions & RNG
// Or selective:
use scirs2_core::random::{thread_rng, Normal as RandNormal, RandBeta, StudentT};

// ✅ CORRECT - Enhanced unified interface
use scirs2_core::random::distributions_unified::{UnifiedNormal, UnifiedBeta};

// ✅ CORRECT - Complex numbers (from root)
use scirs2_core::{Complex64, Complex32};

// ✅ CORRECT - Clean types
use scirs2_core::random::ThreadRng;
let rng: ThreadRng = thread_rng();  // Or seeded_rng(42) when needed
```

## Enforcement & Quality Assurance

### Automated Checks
- CI pipeline checks for unused SciRS2 dependencies
- Documentation tests verify integration examples work
- Dependency graph analysis in builds
- Quantum algorithm benchmarks with SciRS2 performance metrics

### Manual Reviews
- All SciRS2 integration changes require code review
- Quarterly dependency audits
- Regular performance benchmarking

### Success Metrics
- **Compilation Success Rate**: Track package build success
- **Policy Compliance**: No direct rand/ndarray/num-complex violations
- **Pattern Consistency**: Uniform usage across codebase
- **Performance**: Maintain or improve quantum simulation performance
- **Memory Efficiency**: Support 30+ qubit simulations

## Future Considerations

### SciRS2 Version Management
- Track SciRS2 release cycle (currently on 0.1.0-beta.3)
- Test QuantRS2 against SciRS2 beta releases
- Coordinate breaking change migrations

### Quantum-Specific Extensions
- Work with SciRS2 team on quantum-specific features
- Contribute quantum algorithms back to SciRS2
- Develop quantum benchmarks for SciRS2

### Performance Optimization
- Leverage SciRS2 GPU capabilities for quantum simulation
- Optimize SIMD operations for complex arithmetic
- Implement custom quantum kernels with SciRS2 primitives

## Conclusion

This policy ensures QuantRS2 properly leverages SciRS2's scientific computing foundation while maintaining high-performance quantum computing capabilities.

### Core Policy Enforcement
1. **NO direct external dependencies** (rand, ndarray, num-complex, etc.) in QuantRS2 crates
2. **ONLY use SciRS2 abstractions** from scirs2-core and other SciRS2 crates
3. **Unified patterns ONLY**: `scirs2_core::ndarray::*`, `scirs2_core::random::*`, `scirs2_core::{Complex64, Complex32}`
4. **NEVER use deprecated patterns**: No scirs2_autograd::ndarray, no scirs2_core::ndarray_ext

### Benefits of Strict Policy Compliance
- **Type Safety**: Consistent types across the quantum computing ecosystem
- **Maintainability**: Single source of truth for scientific computing primitives
- **Performance**: Optimizations from SciRS2 automatically available
- **Version Control**: Simplified dependency management
- **Portability**: Platform-specific code isolated in SciRS2-core

---

**Document Version**: 2.0 - SciRS2 Dependency Abstraction Policy Enforcement
**Last Updated**: 2025-10-04
**Based on**: Official SciRS2 Policy v3.0.0 (~/work/scirs/SCIRS2_POLICY.md)
**SciRS2 Version**: v0.1.0-rc.4 (Release Candidate with unified patterns)
**NumRS2 Version**: v0.1.0-rc.3
**Next Review**: Q2 2026
**Owner**: QuantRS2 Architecture Team