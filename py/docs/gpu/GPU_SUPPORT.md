# GPU Support in QuantRS2

This document explains how to enable and use GPU acceleration in QuantRS2 for quantum circuit simulations.

## Overview

QuantRS2 supports GPU-accelerated simulations using WebGPU (via the `wgpu` crate), which can significantly improve performance for larger quantum circuits (typically 10+ qubits). The GPU implementation uses compute shaders to parallelize the matrix operations required for quantum gate applications.

## Requirements

To use GPU acceleration, you need:

1. A compatible GPU:
   - On Windows/Linux: Any modern NVIDIA, AMD, or Intel GPU
   - On macOS: Any Apple Silicon (M1/M2/M3) Mac or supported Intel Mac

2. Required dependencies:
   - Rust toolchain with GPU feature flags enabled
   - Python with maturin installed (`pip install maturin`)

## Building with GPU Support

The GPU support is not enabled by default and must be explicitly included when compiling the package. Use one of the following methods:

### Method 1: Using the Provided Script

We've included a convenience script to build with GPU support:

```bash
# Make the script executable if needed
chmod +x build_with_gpu.sh

# Run the build script
./build_with_gpu.sh
```

### Method 2: Manual Build with Maturin

```bash
# Navigate to the Python bindings directory
cd py

# Build with GPU support
maturin develop --release --features gpu
```

## Testing GPU Support

After building with GPU support, you can verify it's working correctly:

```bash
python gpu_test.py
```

This script will:
1. Create a 10-qubit quantum circuit
2. Run the simulation on both CPU and GPU
3. Compare performance and verify the results match

## Using GPU Acceleration in Your Code

Once built with GPU support, you can enable GPU acceleration in your Python code by setting the `use_gpu` parameter to `True` when running a circuit:

```python
import quantrs2 as qr

# Create a quantum circuit
circuit = qr.PyCircuit(10)
circuit.h(0)
circuit.cnot(0, 1)
# Add more gates...

# Run on CPU (default)
cpu_result = circuit.run(use_gpu=False)

# Run on GPU
gpu_result = circuit.run(use_gpu=True)
```

## Performance Considerations

- For small circuits (less than 4 qubits), the GPU implementation automatically falls back to CPU, as the overhead of GPU data transfers outweighs the benefits
- GPU acceleration shows the most benefit for circuits with 10+ qubits
- The first GPU execution may be slower due to shader compilation and device initialization
- Memory usage increases exponentially with qubit count, as the state vector size is 2^n complex numbers

## Troubleshooting

### No GPU Acceleration

If the GPU acceleration is not working:

1. Verify you built with GPU support:
   ```bash
   # Check if GPU feature was included
   grep -r "gpu" $(python -c "import _quantrs2; print(_quantrs2.__file__)")
   ```

2. Check system compatibility:
   ```python
   import _quantrs2 as qr
   
   # Create a very simple circuit
   c = qr.PyCircuit(2)
   c.h(0)
   
   # Try running with GPU
   try:
       c.run(use_gpu=True)
       print("GPU is working")
   except Exception as e:
       print(f"GPU error: {e}")
   ```

3. Common issues:
   - Missing GPU drivers
   - Incompatible GPU hardware
   - WebGPU backend initialization failure

### Compilation Errors

If you encounter compilation errors:

1. Check that Cargo.toml features are configured correctly:
   ```toml
   [features]
   gpu = ["dep:wgpu", "dep:bytemuck", "dep:tokio"]
   ```

2. Make sure all dependencies are available:
   ```bash
   cargo check --features gpu
   ```

## Advanced: Custom GPU Configuration

For advanced users, you can modify GPU behavior in the Rust code (in `sim/src/gpu.rs`):

- Change the workgroup size
- Optimize memory usage
- Add custom kernels

After making changes, rebuild with:
```bash
maturin develop --release --features gpu
```