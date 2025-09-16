# QuantRS2 GPU Support Status

## Current Status

The GPU support in QuantRS2 is currently in **development state**. There are several issues with the GPU implementation that need to be addressed:

1. **API Compatibility**: The current GPU code is written for a newer version of wgpu (v25) than what is being selected by Cargo (v0.19.4).

2. **Dependency Resolution**: When building, Cargo downgrades wgpu from v25 to v0.19.4, causing incompatibilities in the API usage.

3. **Code Path Support**: While the actual GPU acceleration is not yet functional, we've implemented a code path that allows using the `use_gpu=True` flag in Python without errors.

## Options for Using GPU Features

### Option 1: Stub Implementation (Recommended for Now)

We've created a stub implementation that allows your code to use the `use_gpu=True` flag without errors, but it will actually use CPU for the simulation. This is useful for developing code that will eventually use GPU acceleration.

```bash
# Build with the stub GPU implementation
./build_with_gpu_stub.sh

# Test the GPU code path
python minimal_gpu.py
```

### Option 2: Wait for Full Implementation

The full GPU implementation requires more work to be compatible with the current wgpu version. We are working on:

1. Updating the GPU code to work with wgpu v0.19.4
2. Ensuring compatibility across different platforms
3. Optimizing the GPU implementation for large qubit counts

## Using the Stub GPU Support

After building with `build_with_gpu_stub.sh`, you can use the GPU flag in your code:

```python
import quantrs2 as qr

# Create a circuit
circuit = qr.PyCircuit(10)
circuit.h(0)
circuit.cnot(0, 1)
# Add more gates...

# This will use the CPU but through the GPU code path
result = circuit.run(use_gpu=True)

# Access the results as normal
probabilities = result.state_probabilities()
```

## Technical Details

The current implementation has these key issues:

1. wgpu API mismatches:
   - Recent wgpu versions have different struct fields and API methods
   - The current codebase attempts to use features from wgpu v25 while Cargo resolves to v0.19.4

2. Runtime dependencies:
   - The wgpu-related dependencies have complex version constraints
   - Some dependencies like tokio are needed for the async runtime

3. WebGPU shader compatibility:
   - The WGSL shaders need to be compatible with the actual wgpu version being used

## Future Work

1. Update the GPU implementation to work with wgpu v0.19.4 or lock to a specific wgpu version
2. Improve error handling and fallback mechanisms
3. Add performance benchmarks for different qubit counts
4. Optimize GPU shaders for specific hardware

## Contributing

If you're interested in contributing to the GPU implementation, the main file to work on is:
`$quantrs/sim/src/gpu.rs`

Key improvements needed:
1. Update wgpu API usage to match v0.19.4
2. Test on different platforms (Windows, Linux, macOS)
3. Add more optimized implementations for specific GPU types