# SciRS2 Integration Checklist for QuantRS2

This checklist tracks the integration of SciRS2 core features into QuantRS2 based on the CORE_USAGE_POLICY.

## Status Legend
- ✅ Completed
- 🚧 In Progress
- ❌ Not Started
- ⚠️ Needs Refactoring

## 1. Parallel Operations (`scirs2_core::parallel_ops`)

### Status: ✅ Completed
- [x] Removed direct `rayon` dependency from module imports
- [x] Added `scirs2_core::parallel_ops::*` imports
- [x] Enabled `parallel` feature in scirs2-core dependencies
- [x] Made scirs2-core non-optional in sim and tytan crates

### Files Updated:
- ✅ `/sim/src/statevector.rs`
- ✅ `/sim/src/concatenated_error_correction.rs`
- ✅ `/sim/src/distributed_gpu.rs`
- ✅ `/sim/src/enhanced_tensor_networks.rs`
- ✅ `/sim/src/memory_bandwidth_optimization.rs`
- ✅ `/sim/src/memory_prefetching_optimization.rs`
- ✅ `/sim/src/noise_extrapolation.rs`
- ✅ `/sim/src/open_quantum_systems.rs`
- ✅ `/sim/src/opencl_amd_backend.rs`
- ✅ `/sim/src/parallel_tensor_optimization.rs`
- ✅ `/tytan/src/performance_optimization.rs`
- ✅ `/tytan/src/problem_decomposition/domain_decomposer.rs`
- ✅ `/tytan/src/sampler/simulated_annealing.rs`
- ✅ `/core/src/batch/*.rs`

## 2. SIMD Operations (`scirs2_core::simd_ops`)

### Status: ✅ Completed (v0.1.0)

### Completed Actions:
- [x] Replaced all custom SIMD with `scirs2_core::simd_ops::SimdUnifiedOps`
- [x] Removed direct `wide` crate usage (no direct usage found)
- [x] Updated all SIMD operations to use unified trait methods
- [x] Migrated core module SIMD operations to SciRS2

### Files Updated:
- ✅ `/sim/src/optimized_simd.rs` - Updated to use SciRS2 SIMD implementations
- ✅ `/core/src/simd_ops.rs` - Fully migrated to scirs2_core::simd_ops
- ✅ `/core/src/gpu/adaptive_simd.rs` - Using SciRS2 SIMD abstractions
- ✅ `/sim/src/statevector.rs` - Already references optimized_simd module correctly
- ✅ `/tytan/src/performance_optimization.rs` - Updated to use SciRS2 SIMD

## 3. GPU Operations (`scirs2_core::gpu`)

### Status: 🚧 In Progress (Metal backend ready for v0.1.0)

### Current Issues:
- Direct WGPU usage in `/sim/src/gpu.rs`
- Custom GPU implementations scattered across modules
- Waiting for SciRS2 v0.1.0 for Metal support

### Completed Actions:
- [x] Created SciRS2 GPU adapter layer (`/core/src/gpu/scirs2_adapter.rs`)
- [x] Implemented Metal backend placeholder (`/core/src/gpu/metal_backend_scirs2_ready.rs`)
- [x] Written letter to SciRS2 team requesting Metal support
- [x] Created comprehensive GPU migration strategy

### Remaining Actions:
- [ ] Migrate all GPU operations to `scirs2_core::gpu` when Metal support available
- [ ] Register all GPU kernels in core GPU kernel registry
- [ ] Remove direct WGPU/CUDA dependencies from modules
- [ ] Complete migration to `GpuDevice` and `GpuKernel` abstractions

### Files Updated:
- ✅ `/core/src/gpu/scirs2_adapter.rs` - GPU adapter layer created
- ✅ `/core/src/gpu/metal_backend_scirs2_ready.rs` - Metal backend ready
- 🚧 `/sim/src/gpu.rs` - Awaiting full migration
- 🚧 `/sim/src/distributed_gpu.rs` - Awaiting full migration
- 🚧 `/sim/src/opencl_amd_backend.rs` - Awaiting full migration

## 4. Platform Detection (`PlatformCapabilities`)

### Status: ✅ Completed (v0.1.0)

### Completed Actions:
- [x] Replaced all custom CPU feature detection
- [x] Use `PlatformCapabilities::detect()` for all capability checks
- [x] Remove direct `is_x86_feature_detected!` usage
- [x] Implemented comprehensive platform detection system

### Files Updated:
- ✅ `/core/src/gpu/adaptive_simd.rs` - Now uses `PlatformCapabilities::detect()`
- ✅ `/core/src/platform/mod.rs` - Implements full platform detection with caching
- ✅ Platform detection integrated throughout core module
- ✅ Zero-warning implementation with OnceLock for thread safety

## 5. BLAS Operations

### Status: 🚧 In Progress

### Current Issues:
- Some modules may have direct BLAS dependencies
- Need to ensure all BLAS operations go through core

### Required Actions:
- [ ] Audit all BLAS usage
- [ ] Remove direct BLAS dependencies from modules
- [ ] Ensure all linear algebra uses scirs2-linalg

## 6. Memory Management

### Status: ❌ Not Started

### Required Actions:
- [ ] Use `scirs2_core::memory_efficient` for large data operations
- [ ] Replace custom caching with `scirs2_core::cache`
- [ ] Implement memory pooling strategies from core

## 7. Error Handling

### Status: 🚧 In Progress

### Required Actions:
- [ ] Base all module errors on `scirs2_core::error`
- [ ] Use core validation functions
- [ ] Ensure proper error conversions

## 8. Performance Optimization

### Status: ❌ Not Started

### Required Actions:
- [ ] Implement `AutoOptimizer` usage for automatic backend selection
- [ ] Add adaptive implementation selection based on problem size
- [ ] Use core performance profiling tools

## Priority Actions

1. **Immediate** (Blocking compilation):
   - ✅ Fix parallel_ops imports (COMPLETED)

2. **High Priority** (Performance critical):
   - [ ] Migrate SIMD operations to unified trait
   - [ ] Centralize GPU operations
   - [ ] Implement platform detection

3. **Medium Priority** (Consistency):
   - [ ] Unify error handling
   - [ ] Implement memory management policies
   - [ ] Add AutoOptimizer usage

4. **Low Priority** (Nice to have):
   - [ ] Add comprehensive examples
   - [ ] Update documentation
   - [ ] Add integration tests

## Next Steps

1. Start with SIMD migration as it's performance critical
2. Centralize GPU operations for better maintainability
3. Implement platform detection for adaptive optimization
4. Continue with remaining items based on priority

## Notes

- The SciRS2 team will handle complex refactoring as mentioned by the user
- Focus on getting basic integration working first
- Gradual migration is acceptable for non-critical paths