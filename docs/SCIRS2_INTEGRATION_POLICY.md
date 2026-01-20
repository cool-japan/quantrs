# SciRS2 Integration Policy for QuantRS2

> **Note**: This document is copied from the official SciRS2 CORE_USAGE_POLICY.md
> to ensure all COOLJAPAN OU understand and follow the SciRS2 integration guidelines.
> **Last updated**: 2025-06-23

This document outlines the mandatory policies for using scirs2-core modules in QuantRS2. All contributors MUST adhere to these guidelines to ensure proper integration with the SciRS2 ecosystem.

## Table of Contents

1. [Overview](#overview)
2. [SIMD Operations Policy](#simd-operations-policy)
3. [GPU Operations Policy](#gpu-operations-policy)
4. [Parallel Processing Policy](#parallel-processing-policy)
5. [BLAS Operations Policy](#blas-operations-policy)
6. [Platform Detection Policy](#platform-detection-policy)
7. [Performance Optimization Policy](#performance-optimization-policy)
8. [Error Handling Policy](#error-handling-policy)
9. [Memory Management Policy](#memory-management-policy)
10. [Refactoring Guidelines](#refactoring-guidelines)
11. [Examples](#examples)

## Overview

The scirs2-core crate serves as the central hub for all common functionality, optimizations, and abstractions used across SciRS2 modules. This centralized approach ensures:

- **Consistency**: All modules use the same optimized implementations
- **Maintainability**: Updates and improvements are made in one place
- **Performance**: Optimizations are available to all modules
- **Portability**: Platform-specific code is isolated in core

## SIMD Operations Policy

### Mandatory Rules

1. **ALWAYS use `scirs2-core::simd_ops::SimdUnifiedOps` trait** for all SIMD operations
2. **NEVER implement custom SIMD** code in individual modules
3. **NEVER use direct SIMD libraries** (wide, packed_simd, std::arch) in modules
4. **ALWAYS provide scalar fallbacks** through the unified trait

### Required Usage Pattern

```rust
use scirs2_core::simd_ops::SimdUnifiedOps;

// CORRECT - Uses unified SIMD operations
let result = f32::simd_add(&a.view(), &b.view());
let dot_product = f64::simd_dot(&x.view(), &y.view());

// INCORRECT - Direct SIMD implementation
// use wide::f32x8;  // FORBIDDEN in modules
// let vec = f32x8::new(...);  // FORBIDDEN
```

### Available SIMD Operations

All operations are available through the `SimdUnifiedOps` trait:

- `simd_add`, `simd_sub`, `simd_mul`, `simd_div` - Element-wise operations
- `simd_dot` - Dot product
- `simd_gemv` - Matrix-vector multiplication
- `simd_gemm` - Matrix-matrix multiplication
- `simd_norm` - L2 norm
- `simd_max`, `simd_min` - Element-wise min/max
- `simd_scalar_mul` - Scalar multiplication
- `simd_sum`, `simd_mean` - Reductions
- `simd_fma` - Fused multiply-add
- `simd_transpose` - Matrix transpose
- `simd_abs`, `simd_sqrt` - Mathematical operations

## GPU Operations Policy

### Mandatory Rules

1. **ALWAYS use `scirs2-core::gpu` module** for all GPU operations
2. **NEVER implement direct CUDA/OpenCL/Metal kernels** in modules
3. **NEVER make direct GPU API calls** outside of core
4. **ALWAYS register GPU kernels** in the core GPU kernel registry

### GPU Backend Support

The core GPU module provides unified abstractions for:
- CUDA
- ROCm
- WebGPU
- Metal
- OpenCL

### Usage Pattern

```rust
use scirs2_core::gpu::{GpuDevice, GpuKernel};

// CORRECT - Uses core GPU abstractions
let device = GpuDevice::default()?;
let kernel = device.compile_kernel(KERNEL_SOURCE)?;

// INCORRECT - Direct CUDA usage
// use cuda_sys::*;  // FORBIDDEN in modules
```

## Parallel Processing Policy

### Mandatory Rules

1. **ALWAYS use `scirs2-core::parallel_ops`** for all parallel operations
2. **NEVER add direct `rayon` dependency** to module Cargo.toml files
3. **ALWAYS import via `use scirs2_core::parallel_ops::*`**
4. **NEVER use `rayon::prelude::*` directly** in modules

### Required Usage Pattern

```rust
// CORRECT - Uses core parallel abstractions
use scirs2_core::parallel_ops::*;

let results: Vec<i32> = (0..1000)
    .into_par_iter()
    .map(|x| x * x)
    .collect();

// INCORRECT - Direct Rayon usage
// use rayon::prelude::*;  // FORBIDDEN in modules
```

### Features Provided

The `parallel_ops` module provides:

- **Full Rayon functionality** when `parallel` feature is enabled
- **Sequential fallbacks** when `parallel` feature is disabled
- **Helper functions**:
  - `par_range(start, end)` - Create parallel iterator from range
  - `par_chunks(slice, size)` - Process slices in parallel chunks
  - `par_scope(closure)` - Execute in parallel scope
  - `par_join(a, b)` - Execute two closures in parallel
- **Runtime detection**:
  - `is_parallel_enabled()` - Check if parallel processing is available
  - `num_threads()` - Get number of threads for parallel operations

### Module Dependencies

```toml
# CORRECT - Module Cargo.toml
[dependencies]
scirs2-core = { workspace = true, features = ["parallel"] }

# INCORRECT - Direct Rayon dependency
# rayon = { workspace = true }  # FORBIDDEN
```

### Benefits

- **Unified behavior**: All modules respect the same feature flags
- **Graceful degradation**: Sequential execution when parallel is disabled
- **Zero overhead**: Direct re-export of Rayon when enabled
- **Testing flexibility**: Easy to test both parallel and sequential paths

## BLAS Operations Policy

### Mandatory Rules

1. **ALL BLAS operations go through `scirs2-core`**
2. **NEVER add direct BLAS dependencies** to individual modules
3. **Backend selection is handled by core's platform configuration**
4. **Use feature flags through core** for BLAS backend selection

### Supported BLAS Backends

- macOS: Accelerate Framework (default)
- Linux/Windows: OpenBLAS (default)
- Intel MKL (optional)
- Netlib (fallback)

### Module Dependencies

```toml
# CORRECT - Module Cargo.toml
[dependencies]
scirs2-core = { workspace = true, features = ["blas"] }

# INCORRECT - Direct BLAS dependency
# openblas-src = "0.10"  # FORBIDDEN
```

## Platform Detection Policy

### Mandatory Rules

1. **ALWAYS use `scirs2-core::simd_ops::PlatformCapabilities`** for capability detection
2. **NEVER implement custom CPU feature detection**
3. **NEVER duplicate platform detection code**

### Usage Pattern

```rust
use scirs2_core::simd_ops::PlatformCapabilities;

// CORRECT - Uses core platform detection
let caps = PlatformCapabilities::detect();
if caps.simd_available {
    // Use SIMD path
}

// INCORRECT - Custom detection
// if is_x86_feature_detected!("avx2") {  // FORBIDDEN
```

### Available Capabilities

- `simd_available` - SIMD support
- `gpu_available` - GPU support
- `cuda_available` - CUDA support
- `opencl_available` - OpenCL support
- `metal_available` - Metal support (macOS)
- `avx2_available` - AVX2 instructions
- `avx512_available` - AVX512 instructions
- `neon_available` - ARM NEON instructions

## Performance Optimization Policy

### Automatic Optimization Selection

Use `scirs2-core::simd_ops::AutoOptimizer` for automatic selection:

```rust
use scirs2_core::simd_ops::AutoOptimizer;

let optimizer = AutoOptimizer::new();

// Automatically selects best implementation based on problem size
if optimizer.should_use_gpu(problem_size) {
    // Use GPU implementation from core
} else if optimizer.should_use_simd(problem_size) {
    // Use SIMD implementation from core
} else {
    // Use scalar implementation
}
```

### Required Core Features

Each module should enable relevant core features:

```toml
[dependencies]
scirs2-core = { workspace = true, features = ["simd", "parallel", "gpu", "blas"] }
```

## Error Handling Policy

### Mandatory Rules

1. **Base all module errors on `scirs2-core::error`**
2. **Provide proper error conversions** to/from core errors
3. **Use core validation functions** for parameter checking

### Usage Pattern

```rust
use scirs2_core::error::CoreError;
use scirs2_core::validation::{check_positive, check_finite};

// Module-specific error should derive from core
#[derive(Debug, thiserror::Error)]
pub enum ModuleError {
    #[error(transparent)]
    Core(#[from] CoreError),
    // Module-specific variants...
}

// Use core validation
check_positive(value, "parameter_name")?;
check_finite(&array)?;
```

## Memory Management Policy

### Mandatory Rules

1. **Use `scirs2-core::memory_efficient` algorithms** for large data
2. **Use `scirs2-core::cache` for caching** instead of custom solutions
3. **Follow core memory pooling strategies** when available

### Available Memory-Efficient Operations

- `chunk_wise_op` - Process large arrays in chunks
- `streaming_op` - Stream processing for very large data
- Memory pools for temporary allocations

### Caching

```rust
use scirs2_core::cache::{CacheBuilder, TTLSizedCache};

// CORRECT - Uses core caching
let cache = CacheBuilder::new()
    .max_size(100)
    .ttl(Duration::from_secs(60))
    .build();

// INCORRECT - Custom caching
// let mut cache = HashMap::new();  // Don't implement custom caching
```

## Refactoring Guidelines

When encountering code that violates these policies, follow this priority order:

1. **SIMD implementations** - Replace all custom SIMD with `scirs2-core::simd_ops`
2. **GPU implementations** - Centralize all GPU kernels in `scirs2-core::gpu`
3. **Parallel operations** - Replace direct Rayon usage with `scirs2-core::parallel_ops`
4. **Platform detection** - Replace with `PlatformCapabilities::detect()`
5. **BLAS operations** - Ensure all go through core
6. **Caching mechanisms** - Replace custom caching with core implementations
7. **Error types** - Base on core error types
8. **Validation** - Use core validation functions

## Examples

### Example 1: Matrix Operations

```rust
use scirs2_core::simd_ops::SimdUnifiedOps;
use ndarray::{Array2, ArrayView2};

pub fn matrix_multiply(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros((a.nrows(), b.ncols()));
    
    // Use unified SIMD operations - no direct SIMD code
    f32::simd_gemm(1.0, a, b, 0.0, &mut result);
    
    result
}
```

### Example 2: Adaptive Implementation

```rust
use scirs2_core::simd_ops::{SimdUnifiedOps, AutoOptimizer};

pub fn process_data(data: &ArrayView1<f64>) -> f64 {
    let optimizer = AutoOptimizer::new();
    let size = data.len();
    
    if optimizer.should_use_simd(size) {
        // Automatically uses SIMD if available
        f64::simd_sum(data) / size as f64
    } else {
        // Falls back to scalar
        data.sum() / size as f64
    }
}
```

### Example 3: Platform-Aware Code

```rust
use scirs2_core::simd_ops::PlatformCapabilities;

pub fn get_optimization_info() -> String {
    let caps = PlatformCapabilities::detect();
    
    format!(
        "Available optimizations: {}",
        caps.summary()
    )
}
```

### Example 4: Parallel Processing

```rust
use scirs2_core::parallel_ops::*;
use ndarray::{Array1, ArrayView1};

pub fn parallel_distance_matrix(points: &ArrayView1<f64>) -> Array1<f64> {
    // Works with or without parallel feature
    let distances: Vec<f64> = (0..points.len())
        .into_par_iter()
        .map(|i| {
            // Complex computation for each point
            compute_distance(points[i])
        })
        .collect();
    
    Array1::from_vec(distances)
}

pub fn adaptive_processing(data: &[f64]) -> f64 {
    if is_parallel_enabled() && data.len() > 1000 {
        // Use parallel processing for large datasets
        data.into_par_iter()
            .map(|&x| x * x)
            .sum::<f64>()
    } else {
        // Use sequential for small datasets
        data.iter()
            .map(|&x| x * x)
            .sum()
    }
}
```

## Enforcement

- Code reviews MUST check for policy compliance
- CI/CD pipelines should include linting for direct SIMD/GPU usage
- Regular audits should identify and refactor non-compliant code
- New modules MUST follow these policies from the start

## Benefits

By following these policies, we achieve:

1. **Unified Performance**: All modules benefit from optimizations
2. **Easier Maintenance**: Updates in one place benefit all modules
3. **Consistent Behavior**: Same optimizations across the ecosystem
4. **Better Testing**: Centralized testing of critical operations
5. **Improved Portability**: Platform-specific code is isolated
6. **Reduced Duplication**: No repeated implementation of common operations

## Questions or Clarifications

If you have questions about these policies or need clarification on specific use cases, please:

1. Check the `scirs2-core` documentation
2. Review existing implementations in other modules
3. Open an issue for discussion
4. Consult with the core team

Remember: When in doubt, use the core abstractions!