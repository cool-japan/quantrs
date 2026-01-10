# SciRS2 GPU Migration Strategy

## Overview

This document outlines the migration strategy for transitioning QuantRS2's GPU operations to use SciRS2's GPU abstractions. The migration is designed to be incremental and backward-compatible while leveraging SciRS2's unified GPU interface.

## Current Architecture

### Existing GPU Backend System

QuantRS2 currently implements multiple GPU backends:
- **CUDA Backend**: Direct CUDA API usage for NVIDIA GPUs
- **OpenCL Backend**: Cross-platform GPU support
- **Metal Backend**: macOS-specific GPU acceleration (placeholder)
- **CPU Backend**: Fallback for systems without GPU

### Key Abstractions

```rust
// Current QuantRS2 GPU traits
pub trait GpuBuffer: Send + Sync {
    fn size(&self) -> usize;
    fn upload(&mut self, data: &[Complex64]) -> QuantRS2Result<()>;
    fn download(&self, data: &mut [Complex64]) -> QuantRS2Result<()>;
    fn sync(&self) -> QuantRS2Result<()>;
}

pub trait GpuKernel: Send + Sync {
    fn apply_single_qubit_gate(...) -> QuantRS2Result<()>;
    fn apply_two_qubit_gate(...) -> QuantRS2Result<()>;
    fn measure_qubit(...) -> QuantRS2Result<(bool, f64)>;
}
```

## Migration Phases

### Phase 1: Adapter Layer (Current)

Create adapter types that bridge between QuantRS2 and SciRS2 APIs:

```rust
// scirs2_adapter.rs
pub struct SciRS2BufferAdapter {
    // Wraps SciRS2 GPU buffer
}

pub struct SciRS2KernelAdapter {
    // Wraps SciRS2 GPU kernel
}

pub fn get_scirs2_gpu_device() -> QuantRS2Result<GpuDevice> {
    // Use SciRS2 device selection
}
```

### Phase 2: Kernel Migration

Migrate quantum kernels to SciRS2 format:

1. **Register Kernels**: Add quantum operations to SciRS2 kernel registry
2. **Optimize Kernels**: Use SciRS2's optimization infrastructure
3. **Test Performance**: Ensure no regression in performance

```rust
// Example kernel registration
register_quantum_kernel(
    "apply_hadamard_gate",
    include_str!("kernels/hadamard.cl")
)?;
```

### Phase 3: Backend Unification

Replace backend-specific implementations with SciRS2 unified interface:

```rust
// Before
#[cfg(feature = "cuda")]
let backend = CudaBackend::new()?;

// After
let device = GpuDevice::new(GpuBackend::Auto, 0)?;
```

### Phase 4: Feature Deprecation

Gradually deprecate QuantRS2-specific GPU features:
1. Mark old APIs as deprecated
2. Provide migration guides
3. Remove legacy code in major version bump

## Implementation Details

### GPU Device Selection

```rust
pub fn select_best_gpu_device() -> QuantRS2Result<GpuDevice> {
    // Priority order for backend selection
    let backends = vec![
        GpuBackend::Cuda,      // NVIDIA GPUs
        #[cfg(target_os = "macos")]
        GpuBackend::Metal,     // Apple Silicon
        GpuBackend::OpenCL,    // Fallback
    ];
    
    for backend in backends {
        if let Ok(device) = GpuDevice::new(backend, 0) {
            return Ok(device);
        }
    }
    
    Err(QuantRS2Error::NoGpuAvailable)
}
```

### Quantum Kernel Implementation

```rust
// Quantum gate kernel using SciRS2 abstractions
pub fn create_hadamard_kernel(device: &GpuDevice) -> QuantRS2Result<GpuKernel> {
    let kernel_source = r#"
        __kernel void hadamard_gate(
            __global float2* state,
            const uint target_qubit,
            const uint num_qubits
        ) {
            const uint gid = get_global_id(0);
            const uint state_size = 1u << num_qubits;
            
            if (gid >= state_size / 2) return;
            
            // Quantum state indexing logic
            const uint mask = (1u << target_qubit) - 1u;
            const uint idx0 = ((gid & ~mask) << 1u) | (gid & mask);
            const uint idx1 = idx0 | (1u << target_qubit);
            
            // Apply Hadamard transformation
            const float sqrt2_inv = 0.7071067811865475f;
            float2 amp0 = state[idx0];
            float2 amp1 = state[idx1];
            
            state[idx0] = sqrt2_inv * (amp0 + amp1);
            state[idx1] = sqrt2_inv * (amp0 - amp1);
        }
    "#;
    
    device.compile_kernel("hadamard_gate", kernel_source)
}
```

### Memory Management

```rust
// Efficient state vector allocation
pub fn allocate_quantum_state(
    device: &GpuDevice,
    num_qubits: usize
) -> QuantRS2Result<GpuBuffer<Complex64>> {
    let state_size = 1 << num_qubits;
    let buffer = device.allocate_buffer::<Complex64>(state_size)?;
    
    // Initialize to |00...0⟩ state
    let mut initial_state = vec![Complex64::zero(); state_size];
    initial_state[0] = Complex64::one();
    buffer.upload(&initial_state)?;
    
    Ok(buffer)
}
```

## Platform-Specific Considerations

### Metal Integration

Special handling for Metal backend pending SciRS2 support:

```rust
#[cfg(target_os = "macos")]
pub fn create_metal_device() -> QuantRS2Result<Box<dyn GpuBackend>> {
    // Use placeholder implementation until SciRS2 v0.1.0-rc.2
    if metal_backend_scirs2_ready::is_metal_available() {
        Ok(Box::new(MetalPlaceholderBackend::new()?))
    } else {
        Err(QuantRS2Error::MetalNotAvailable)
    }
}
```

### CUDA Optimization

Leverage CUDA-specific features through SciRS2:

```rust
pub fn optimize_for_cuda(kernel: &mut GpuKernel) -> QuantRS2Result<()> {
    // Use tensor cores for complex number operations
    kernel.set_optimization_hint("use_tensor_cores", true)?;
    
    // Optimize thread block size for quantum operations
    kernel.set_block_size(256)?;
    
    Ok(())
}
```

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_scirs2_gpu_device_creation() {
    let device = get_scirs2_gpu_device();
    assert!(device.is_ok() || !is_gpu_available());
}

#[test]
fn test_quantum_kernel_compilation() {
    if let Ok(device) = get_scirs2_gpu_device() {
        let kernel = create_hadamard_kernel(&device);
        assert!(kernel.is_ok());
    }
}
```

### Performance Benchmarks

```rust
#[bench]
fn bench_hadamard_gate_scirs2(b: &mut Bencher) {
    let device = get_scirs2_gpu_device().unwrap();
    let mut state = allocate_quantum_state(&device, 20).unwrap();
    let kernel = create_hadamard_kernel(&device).unwrap();
    
    b.iter(|| {
        kernel.execute(&mut state, &[0]).unwrap();
    });
}
```

### Compatibility Tests

Ensure backward compatibility during migration:

```rust
#[test]
fn test_legacy_api_compatibility() {
    // Old API
    let backend = GpuBackendFactory::create_best_available().unwrap();
    let mut state = GpuStateVector::new(backend, 10).unwrap();
    
    // Should still work during migration
    state.apply_gate(&HGate::new(), &[QubitId(0)]).unwrap();
}
```

## Migration Timeline

### Q1 2024
- [x] Implement adapter layer
- [x] Create Metal placeholder implementation
- [x] Send letter to SciRS2 team for Metal support

### Q2 2024
- [ ] Migrate core quantum kernels to SciRS2
- [ ] Implement performance benchmarks
- [ ] Update documentation

### Q3 2024
- [ ] Complete backend unification
- [ ] Deprecate old GPU APIs
- [ ] Release beta with full SciRS2 GPU support

### Q4 2024
- [ ] Remove legacy GPU code
- [ ] Optimize for specific GPU architectures
- [ ] Release stable version

## Benefits of Migration

### Technical Benefits
1. **Unified API**: Single interface for all GPU backends
2. **Better Performance**: Leverage SciRS2's optimizations
3. **Reduced Maintenance**: Less backend-specific code
4. **Future-Proof**: Automatic support for new GPU architectures

### Development Benefits
1. **Simpler Testing**: One API to test instead of multiple
2. **Easier Debugging**: Centralized GPU error handling
3. **Better Documentation**: Single source of truth
4. **Community Support**: Leverage SciRS2 ecosystem

## Risk Mitigation

### Potential Risks
1. **Performance Regression**: Ensure thorough benchmarking
2. **API Breaking Changes**: Provide compatibility layer
3. **Platform Support**: Maintain fallbacks for unsupported platforms
4. **Migration Complexity**: Phase implementation carefully

### Mitigation Strategies
1. **Gradual Migration**: Implement in phases with fallbacks
2. **Extensive Testing**: Comprehensive test suite for each phase
3. **Performance Monitoring**: Continuous benchmarking
4. **Clear Documentation**: Migration guides and examples

## Conclusion

The migration to SciRS2 GPU abstractions will provide QuantRS2 with a more maintainable, performant, and future-proof GPU acceleration layer. By following this phased approach, we can ensure a smooth transition while maintaining backward compatibility and performance.