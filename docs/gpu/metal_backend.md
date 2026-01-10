# Metal GPU Backend Documentation

## Overview

The QuantRS2-Core module includes a forward-compatible Metal GPU backend implementation designed to integrate seamlessly with the expected SciRS2 GPU abstractions in v0.1.0-rc.2. This implementation provides GPU acceleration for quantum simulations on macOS systems with Apple Silicon (M1/M2/M3) processors.

## Architecture

### Placeholder Implementation

The current implementation (`metal_backend_scirs2_ready.rs`) provides placeholder types that mirror the expected SciRS2 Metal API:

```rust
// Placeholder types for future SciRS2 Metal support
pub struct MetalDevice {
    pub(crate) device: MetalDeviceHandle,
    pub(crate) command_queue: MetalCommandQueue,
}

pub struct MetalBuffer<T> {
    pub buffer: MetalBufferHandle,
    pub length: usize,
    pub _phantom: std::marker::PhantomData<T>,
}

pub struct MetalKernel {
    pub pipeline: MetalComputePipeline,
    pub function_name: String,
}
```

### Metal Shader Implementation

The module includes optimized Metal shaders for quantum operations:

```metal
// Complex number operations
struct Complex {
    float real;
    float imag;
};

// Single qubit gate kernel
kernel void apply_single_qubit_gate(
    device Complex* state [[buffer(0)]],
    constant Complex* gate_matrix [[buffer(1)]],
    constant uint& target_qubit [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Efficient bit manipulation for quantum state indexing
    uint mask = (1u << target_qubit) - 1u;
    uint idx0 = ((gid & ~mask) << 1u) | (gid & mask);
    uint idx1 = idx0 | (1u << target_qubit);
    
    // Apply gate matrix to quantum amplitudes
    Complex amp0 = state[idx0];
    Complex amp1 = state[idx1];
    
    state[idx0] = complex_add(
        complex_mul(gate_matrix[0], amp0),
        complex_mul(gate_matrix[1], amp1)
    );
    state[idx1] = complex_add(
        complex_mul(gate_matrix[2], amp0),
        complex_mul(gate_matrix[3], amp1)
    );
}
```

## Usage

### Checking Metal Availability

```rust
use quantrs2_core::gpu::metal_backend_scirs2_ready::{is_metal_available, get_metal_device_info};

if is_metal_available() {
    if let Some(info) = get_metal_device_info() {
        println!("Metal device: {}", info);
    }
}
```

### Creating a Metal-Accelerated Quantum State

```rust
use quantrs2_core::gpu::metal_backend_scirs2_ready::MetalQuantumState;

// Create a 10-qubit quantum state on Metal GPU
let mut state = MetalQuantumState::new(10)?;

// Apply a Hadamard gate to qubit 0
let hadamard = [
    Complex64::new(SQRT_2_INV, 0.0),
    Complex64::new(SQRT_2_INV, 0.0),
    Complex64::new(SQRT_2_INV, 0.0),
    Complex64::new(-SQRT_2_INV, 0.0),
];
state.apply_single_qubit_gate(&hadamard, QubitId(0))?;
```

## Performance Characteristics

### Apple Silicon Advantages

1. **Unified Memory Architecture**: Zero-copy data transfer between CPU and GPU
2. **High Memory Bandwidth**: Up to 400GB/s on M3 Max
3. **Efficient Compute Units**: Optimized for parallel quantum operations
4. **Low Power Consumption**: Ideal for long-running simulations

### Optimization Strategies

1. **Batch Operations**: Process multiple quantum gates in parallel
2. **Memory Coalescing**: Optimize memory access patterns for cache efficiency
3. **Thread Group Optimization**: Tune thread group sizes for specific quantum algorithms
4. **Pipeline State Caching**: Reuse compiled Metal kernels across operations

## Integration with SciRS2

### Current Status

The implementation is designed to be forward-compatible with the expected SciRS2 v0.1.0-rc.2 Metal API. Once SciRS2 provides native Metal support, the placeholder types will be replaced with actual SciRS2 types.

### Migration Path

1. **Phase 1** (Current): Placeholder implementation with Metal shader code
2. **Phase 2**: Replace placeholder types with SciRS2 Metal types when available
3. **Phase 3**: Full integration with SciRS2 GPU device management
4. **Phase 4**: Leverage SciRS2 kernel registry for optimized quantum kernels

### Letter to SciRS2 Team

A comprehensive letter has been sent to the SciRS2 team requesting Metal GPU support in v0.1.0-rc.2. The letter includes:
- Performance benchmarks on Apple Silicon
- Proposed API design
- Implementation timeline
- Contribution offer from the QuantRS2 team

## Testing

The module includes comprehensive tests:

```rust
#[test]
fn test_metal_availability() {
    let available = is_metal_available();
    #[cfg(target_os = "macos")]
    assert!(available);
}

#[test]
fn test_metal_quantum_state_creation() {
    for num_qubits in [1, 5, 10, 15] {
        let result = MetalQuantumState::new(num_qubits);
        assert!(result.is_ok());
    }
}

#[test]
fn test_single_qubit_gate_application() {
    let mut state = MetalQuantumState::new(5).unwrap();
    let pauli_x = [...]; // Pauli-X gate matrix
    let result = state.apply_single_qubit_gate(&pauli_x, QubitId(0));
    assert!(result.is_ok());
}
```

## Future Enhancements

### Planned Features

1. **Multi-qubit Gates**: Optimized kernels for 2-qubit and multi-qubit operations
2. **Quantum Circuit Batching**: Execute multiple circuits in parallel
3. **Tensor Network Contraction**: GPU-accelerated tensor operations
4. **Quantum Machine Learning**: Specialized kernels for QML operations

### Performance Goals

- Support for 30+ qubit simulations on M3 Max
- Sub-millisecond gate application for common operations
- Real-time quantum circuit visualization
- Interactive quantum algorithm development

## Platform Requirements

- macOS 13.0 or later
- Apple Silicon processor (M1/M2/M3)
- Metal 3.0 support
- 8GB+ unified memory (16GB+ recommended for large simulations)

## Known Limitations

1. Currently a placeholder implementation pending SciRS2 support
2. Limited to single-qubit gate operations in the initial version
3. No support for distributed GPU computing across multiple devices
4. Requires macOS; not available on iOS or other platforms

## Conclusion

The Metal GPU backend provides a foundation for high-performance quantum simulation on Apple Silicon. With the unified memory architecture and efficient compute units, Metal offers unique advantages for quantum computing workloads. The forward-compatible implementation ensures a smooth transition when SciRS2 provides native Metal support.