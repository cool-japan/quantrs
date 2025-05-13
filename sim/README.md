# QuantRS2-Sim: High-Performance Quantum Simulators

[![Crates.io](https://img.shields.io/crates/v/quantrs2-sim.svg)](https://crates.io/crates/quantrs2-sim)
[![Documentation](https://docs.rs/quantrs2-sim/badge.svg)](https://docs.rs/quantrs2-sim)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Sim is part of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing high-performance simulators for quantum circuits that can handle 30+ qubits on standard hardware.

## Features

- **Multiple Simulation Backends**: State vector, tensor network, and optimized implementations
- **High Performance**: Leveraging SIMD, multi-threading, and memory-efficient algorithms
- **Optional GPU Acceleration**: WGPU-based state vector simulation
- **Realistic Noise Models**: Simulate quantum hardware with configurable noise channels
- **Tensor Network Simulation**: Memory-efficient simulation for circuits with limited entanglement
- **Benchmarking Tools**: Evaluate performance across different simulation strategies
- **Quantum Error Correction**: Codes and utilities for protected quantum information

## Usage

### Basic Simulation with State Vector

```rust
use quantrs2_circuit::builder::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() {
    // Create a Bell state circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).unwrap()
           .cnot(0, 1).unwrap();
    
    // Create and run the simulator
    let simulator = StateVectorSimulator::new();
    let result = circuit.run(simulator).unwrap();
    
    // Print probabilities
    for (i, prob) in result.probabilities().iter().enumerate() {
        println!("|{:02b}⟩: {:.6}", i, prob);
    }
}
```

### Simulation with Noise

```rust
use quantrs2_circuit::builder::Circuit;
use quantrs2_sim::{
    statevector::StateVectorSimulator,
    noise::{NoiseModel, NoiseChannel, NoiseParameters},
};

fn main() {
    // Create a circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).unwrap()
           .cnot(0, 1).unwrap();
    
    // Create a noise model with bit-flip errors
    let mut noise_model = NoiseModel::new();
    noise_model.add_channel(
        NoiseChannel::BitFlip,
        NoiseParameters::Probability(0.01),
    );
    
    // Create a noisy simulator
    let simulator = StateVectorSimulator::new_with_noise(noise_model);
    
    // Run the simulation
    let result = circuit.run(simulator).unwrap();
    
    // Print probabilities
    for (i, prob) in result.probabilities().iter().enumerate() {
        println!("|{:02b}⟩: {:.6}", i, prob);
    }
}
```

### GPU-Accelerated Simulation

```rust
#[cfg(feature = "gpu")]
use quantrs2_circuit::builder::Circuit;
#[cfg(feature = "gpu")]
use quantrs2_sim::gpu::GpuStateVectorSimulator;

#[cfg(feature = "gpu")]
async fn gpu_simulation() -> Result<(), Box<dyn std::error::Error>> {
    // Create a circuit
    let mut circuit = Circuit::<10>::new();
    for i in 0..10 {
        circuit.h(i)?;
    }
    
    // Check if GPU acceleration is available
    if GpuStateVectorSimulator::is_available() {
        // Create and run the GPU simulator
        let simulator = GpuStateVectorSimulator::new().await?;
        let result = circuit.run(simulator)?;
        
        // Process results
        println!("GPU simulation successful!");
    } else {
        println!("GPU acceleration not available");
    }
    
    Ok(())
}
```

## Module Structure

- **statevector.rs**: Standard state vector simulator
- **tensor.rs**: Tensor utilities for quantum simulation
- **tensor_network/**: Tensor network simulation for efficient memory usage
- **noise.rs**: Noise models and channels for realistic simulation
- **optimized_*.rs**: Optimized simulators using various techniques
- **error_correction/**: Quantum error correction implementations
- **benchmark.rs**: Performance benchmarking utilities
- **gpu.rs**: GPU-accelerated simulation (with feature flag)

## Feature Flags

- **default**: Includes optimized implementations and SIMD
- **gpu**: Enables GPU acceleration using WGPU
- **simd**: Uses SIMD instructions for improved performance
- **optimize**: Enables optimized implementations (included in default)
- **memory_efficient**: Enables algorithms optimized for large state vectors
- **advanced_math**: Enables advanced math using external libraries

## Performance

QuantRS2-Sim is designed for high-performance quantum simulation:

- Efficiently simulates up to 30+ qubits on standard hardware
- Parallel execution with Rayon for multi-threading
- SIMD optimizations for performance-critical operations
- Memory-efficient algorithms for large qubit counts
- Optional GPU acceleration for 10-100x speedups on large circuits

## Future Plans

See [TODO.md](TODO.md) for planned improvements and features.

## Integration with Other QuantRS2 Modules

This module is designed to work seamlessly with:
- [quantrs2-core](../core/README.md): Uses core types and operations
- [quantrs2-circuit](../circuit/README.md): Simulates circuits created with the circuit builder
- [quantrs2-device](../device/README.md): Simulates real hardware with accurate noise models

## License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.