# QuantRS2-Core Usage Guide

This guide provides comprehensive documentation on using QuantRS2-Core for quantum computing applications, with a focus on best practices, common patterns, and performance optimization.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Quantum Operations](#basic-quantum-operations)
3. [Variational Quantum Algorithms](#variational-quantum-algorithms)
4. [Quantum Machine Learning](#quantum-machine-learning)
5. [Error Correction](#error-correction)
6. [Performance Optimization](#performance-optimization)
7. [Benchmarking](#benchmarking)
8. [Hardware Integration](#hardware-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

Add QuantRS2-Core to your `Cargo.toml`:

```toml
[dependencies]
quantrs2-core = "0.1.2"
scirs2-core = "0.1.1"
```

### Basic Imports

```rust
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, array};
use scirs2_core::Complex64;
use scirs2_core::random::prelude::*;
```

### Creating Your First Quantum Circuit

```rust
use quantrs2_core::{
    gate::{HadamardGate, CNOTGate, GateOp},
    qubit::QubitId,
};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;

fn create_bell_state() -> Array1<Complex64> {
    let num_qubits = 2;
    let dim = 1 << num_qubits;

    // Initialize to |00⟩
    let mut state = Array1::zeros(dim);
    state[0] = Complex64::new(1.0, 0.0);

    // Apply H to qubit 0
    let h_gate = HadamardGate::new(QubitId(0));
    // Apply gate to state (implementation depends on your simulator)

    // Apply CNOT(0, 1)
    let cnot = CNOTGate::new(QubitId(0), QubitId(1));
    // Apply gate to state

    state // Returns |Φ+⟩ = (|00⟩ + |11⟩)/√2
}
```

## Basic Quantum Operations

### Single-Qubit Gates

```rust
use quantrs2_core::gate::*;
use quantrs2_core::qubit::QubitId;

// Pauli gates
let x_gate = PauliXGate::new(QubitId(0));
let y_gate = PauliYGate::new(QubitId(0));
let z_gate = PauliZGate::new(QubitId(0));

// Hadamard gate
let h_gate = HadamardGate::new(QubitId(0));

// Rotation gates
let rx_gate = RXGate::new(QubitId(0), std::f64::consts::PI / 4.0);
let ry_gate = RYGate::new(QubitId(0), std::f64::consts::PI / 4.0);
let rz_gate = RZGate::new(QubitId(0), std::f64::consts::PI / 4.0);

// Phase gates
let s_gate = SGate::new(QubitId(0));
let t_gate = TGate::new(QubitId(0));
```

### Two-Qubit Gates

```rust
// CNOT gate
let cnot = CNOTGate::new(QubitId(0), QubitId(1));

// CZ gate
let cz = CZGate::new(QubitId(0), QubitId(1));

// SWAP gate
let swap = SWAPGate::new(QubitId(0), QubitId(1));

// Controlled-U gate
use quantrs2_core::controlled::make_controlled;
let target_gate = RYGate::new(QubitId(1), 0.5);
let controlled_ry = make_controlled(Box::new(target_gate), QubitId(0))?;
```

### Batch Operations for Performance

```rust
use quantrs2_core::batch::{create_batch, BatchConfig};

let config = BatchConfig {
    batch_size: 100,
    num_qubits: 10,
    use_simd: true,
    parallel_execution: true,
};

let batch = create_batch(config)?;
// Process multiple quantum states in parallel
```

## Variational Quantum Algorithms

### Variational Quantum Eigensolver (VQE)

```rust
use quantrs2_core::{
    variational::{VariationalCircuit, VariationalGate},
    variational_optimization::{create_vqe_optimizer, OptimizationConfig, OptimizationMethod},
};

// Create variational circuit
let num_qubits = 4;
let mut circuit = VariationalCircuit::new(num_qubits);

// Add parametric gates
for q in 0..num_qubits {
    circuit.add_variational_gate(
        VariationalGate::new(QubitId(q as u32), "RY".to_string(), vec![0.1])
    );
}

// Configure optimizer
let config = OptimizationConfig {
    max_iterations: 100,
    convergence_threshold: 1e-6,
    learning_rate: 0.01,
    method: OptimizationMethod::Adam,
    ..Default::default()
};

// Create VQE optimizer
let optimizer = create_vqe_optimizer(config)?;

// Define Hamiltonian and run optimization
// let result = optimizer.optimize(&hamiltonian, &circuit)?;
```

### Quantum Approximate Optimization Algorithm (QAOA)

```rust
use quantrs2_core::{
    qaoa::{QAOACircuit, QAOAParams},
    variational_optimization::create_qaoa_optimizer,
};

// Define problem (e.g., MaxCut)
let num_qubits = 6;
let p = 3; // QAOA layers

// Create QAOA circuit
let params = QAOAParams::new(num_qubits, p);
let circuit = QAOACircuit::from_params(params);

// Optimize
let optimizer = create_qaoa_optimizer(config)?;
// let result = optimizer.optimize(&cost_hamiltonian, &mixer, &circuit)?;
```

## Quantum Machine Learning

### Quantum Kernel Methods

```rust
use quantrs2_core::qml::{
    advanced_algorithms::{QuantumKernel, QuantumKernelConfig, FeatureMapType},
    EntanglementPattern,
};

// Configure quantum kernel
let kernel_config = QuantumKernelConfig {
    num_qubits: 4,
    feature_map: FeatureMapType::ZZFeatureMap,
    reps: 2,
    entanglement: EntanglementPattern::Full,
    parameter_scaling: 2.0,
};

let mut kernel = QuantumKernel::new(kernel_config);

// Compute kernel matrix for training data
use scirs2_core::ndarray::Array2;
let training_data: Array2<f64> = /* ... */;
let kernel_matrix = kernel.kernel_matrix(&training_data)?;

// Use with quantum SVM
// let qsvm = QuantumSVM::new(kernel);
// qsvm.train(&training_data, &labels)?;
```

### Quantum Neural Networks

```rust
use quantrs2_core::qml::{
    layers::{RotationLayer, EntanglingLayer, Parameter},
    QMLCircuit, QMLConfig, EncodingStrategy,
};

// Create QML circuit
let config = QMLConfig {
    num_qubits: 4,
    num_layers: 3,
    encoding: EncodingStrategy::Angle,
    entanglement: EntanglementPattern::Linear,
    data_reuploading: true,
};

let mut qnn = QMLCircuit::new(config);

// Add layers
let rotation_layer = RotationLayer::new(4, vec![
    Parameter { value: 0.1, trainable: true },
    Parameter { value: 0.2, trainable: true },
    Parameter { value: 0.3, trainable: true },
    Parameter { value: 0.4, trainable: true },
]);

qnn.add_layer(Box::new(rotation_layer))?;

// Train using parameter shift rule or natural gradient
```

### Quantum Generative Adversarial Networks (QGANs)

```rust
use quantrs2_core::qml::generative_adversarial::{
    QGAN, QGANConfig, QuantumGenerator, QuantumDiscriminator,
};

let config = QGANConfig {
    num_qubits: 4,
    latent_dim: 2,
    learning_rate: 0.01,
    batch_size: 32,
    ..Default::default()
};

let qgan = QGAN::new(config);
// Training loop
// for epoch in 0..num_epochs {
//     let stats = qgan.train_epoch(&real_data)?;
//     println!("Epoch {}: G_loss={:.4}, D_loss={:.4}",
//              epoch, stats.generator_loss, stats.discriminator_loss);
// }
```

## Error Correction

### Surface Code

```rust
use quantrs2_core::error_correction::{SurfaceCode, Pauli, PauliString};

let distance = 5;
let surface_code = SurfaceCode::new(distance);

println!("Physical qubits: {}", surface_code.num_physical_qubits());
println!("Logical qubits: {}", surface_code.num_logical_qubits());

// Encode logical state
let logical_zero = surface_code.encode_zero()?;

// Simulate error
let error = PauliString::new(vec![
    (0, Pauli::X),
    (1, Pauli::Z),
]);

let corrupted_state = surface_code.apply_error(&logical_zero, &error)?;

// Measure syndrome
let syndrome = surface_code.measure_syndrome(&corrupted_state)?;

// Decode and correct
let correction = surface_code.decode(&syndrome)?;
let corrected_state = surface_code.apply_correction(&corrupted_state, &correction)?;
```

### Color Code

```rust
use quantrs2_core::error_correction::ColorCode;

let distance = 3;
let color_code = ColorCode::new(distance);

// Color codes support transversal Clifford gates
// and have better gate implementations than surface codes
```

### Real-Time Error Correction

```rust
use quantrs2_core::error_correction::realtime::{
    RealTimeErrorCorrection, RealTimeConfig,
};

let config = RealTimeConfig {
    syndrome_rate_hz: 100_000.0, // 100kHz syndrome measurement
    decoder: DecoderType::ParallelMWPM,
    latency_budget_us: 10.0, // 10μs latency budget
    ..Default::default()
};

let rtec = RealTimeErrorCorrection::new(config);
// rtec.start()?;
// Real-time syndrome processing and correction
```

## Performance Optimization

### SIMD Acceleration

```rust
use quantrs2_core::platform::detector::PlatformCapabilities;

// Detect platform capabilities
let caps = PlatformCapabilities::detect();

if caps.has_avx2() {
    println!("AVX2 acceleration available");
    // SIMD operations will be automatically used
}

// Force SIMD usage in batch operations
let config = BatchConfig {
    use_simd: true,
    parallel_execution: true,
    ..Default::default()
};
```

### GPU Acceleration

```rust
use quantrs2_core::gpu::{GpuBackendFactory, GpuConfig};

// Initialize GPU backend
let gpu_config = GpuConfig {
    device_id: 0,
    use_tensor_cores: true,
    memory_pool_size_gb: 4,
    ..Default::default()
};

if let Ok(gpu) = GpuBackendFactory::create_backend(gpu_config) {
    println!("GPU acceleration enabled");
    // Large-scale simulations will use GPU
}
```

### Adaptive Precision

```rust
use quantrs2_core::adaptive_precision::{
    AdaptivePrecisionConfig, AdaptivePrecisionSimulator,
};

let config = AdaptivePrecisionConfig {
    initial_precision: 1e-10,
    min_precision: 1e-12,
    max_precision: 1e-6,
    adaptation_rate: 0.9,
    monitor_interval: 10,
};

let simulator = AdaptivePrecisionSimulator::new(config);
// Automatically adjusts precision based on error accumulation
```

### Memory-Efficient Simulation

```rust
use quantrs2_core::memory_efficient::EfficientStateVector;

// For large quantum systems (>20 qubits)
let num_qubits = 25;
let state = EfficientStateVector::new(num_qubits);

// Uses sparse representation and memory mapping for large states
```

## Benchmarking

### Randomized Benchmarking

```rust
use quantrs2_core::noise_characterization::RandomizedBenchmarking;

let rb = RandomizedBenchmarking::new(num_qubits);
let sequence_lengths = vec![1, 5, 10, 20, 50, 100];

for &length in &sequence_lengths {
    let result = rb.run(length, num_sequences)?;
    println!("Length {}: fidelity = {:.6}", length, result.average_fidelity);
}

// Extract gate fidelity
let gate_fidelity = result.estimate_gate_fidelity()?;
```

### Quantum Volume

```rust
use quantrs2_core::quantum_volume_tomography::QuantumVolume;

let qv = QuantumVolume::new(num_qubits, depth);
let result = qv.measure(num_circuits)?;

if result.success_rate > 2.0/3.0 {
    println!("Quantum Volume {} achieved!", result.quantum_volume);
}
```

### Comprehensive Benchmarking Suite

```rust
use quantrs2_core::benchmarking_integration::ComprehensiveBenchmarkSuite;

let suite = ComprehensiveBenchmarkSuite::new();

// Run all benchmarks
let report = suite.run_all()?;

// View results
println!("Gate fidelity: {:.4}", report.gate_fidelity);
println!("Quantum volume: {}", report.quantum_volume);
println!("Error mitigation improvement: {:.2}x", report.mitigation_factor);
```

## Hardware Integration

### Neutral Atom Quantum Computers

```rust
use quantrs2_core::neutral_atom::{NeutralAtomQC, Position3D};

let mut qc = NeutralAtomQC::new(num_atoms);

// Position atoms
qc.move_atom(0, Position3D::new(0.0, 0.0, 0.0))?;
qc.move_atom(1, Position3D::new(5.0, 0.0, 0.0))?; // 5μm spacing

// Apply Rydberg gates
qc.rydberg_gate(0, 1, duration)?;
```

### Trapped Ion Systems

```rust
use quantrs2_core::trapped_ion::{TrappedIonSystem, IonSpecies};

let system = TrappedIonSystem::new(IonSpecies::Ytterbium171);

// Laser-based gates
system.apply_carrier_transition(0, frequency, duration)?;
system.apply_molmer_sorensen_gate(0, 1)?;
```

### Superconducting Qubits

```rust
use quantrs2_core::pulse::{PulseCompiler, Pulse, PulseEnvelope};

let compiler = PulseCompiler::new();

// Compile high-level gate to pulse sequence
let gate = RXGate::new(QubitId(0), PI / 2.0);
let pulse_sequence = compiler.compile(&gate)?;

// Optimize pulse for hardware
let optimized = compiler.optimize(pulse_sequence)?;
```

## Best Practices

### 1. Use SciRS2 Unified Patterns

✅ **Correct:**
```rust
use scirs2_core::ndarray::{Array1, Array2, array};
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex64, Complex32};
```

❌ **Incorrect:**
```rust
use ndarray::{Array1, Array2}; // Don't use direct ndarray
use rand::thread_rng; // Don't use direct rand
use num_complex::Complex64; // Don't use direct num-complex
```

### 2. Leverage Batch Processing

For processing multiple quantum states:
```rust
// Good: Batch processing with SIMD
let batch = create_batch(BatchConfig {
    batch_size: 100,
    use_simd: true,
    parallel_execution: true,
    ..Default::default()
})?;

// Avoid: Processing states individually in loops
for _ in 0..100 {
    // Individual state processing (slower)
}
```

### 3. Error Handling

Always handle quantum errors properly:
```rust
use quantrs2_core::error::QuantRS2Result;

fn my_quantum_algorithm() -> QuantRS2Result<f64> {
    let state = initialize_state()?;
    let result = apply_gates(&state)?;
    Ok(result)
}
```

### 4. Resource Estimation

Estimate resources before running large simulations:
```rust
use quantrs2_core::resource_estimator::estimate_resources;

let estimate = estimate_resources(&circuit)?;

if estimate.memory_gb > available_memory() {
    return Err("Insufficient memory for simulation");
}
```

### 5. Testing Quantum Circuits

```rust
use quantrs2_core::testing::{QuantumAssert, DEFAULT_TOLERANCE};

#[test]
fn test_bell_state() {
    let state = create_bell_state();

    // Assert state is properly normalized
    QuantumAssert::assert_normalized(&state, DEFAULT_TOLERANCE);

    // Assert entanglement
    QuantumAssert::assert_entangled(&state, &[0, 1], DEFAULT_TOLERANCE);
}
```

## Troubleshooting

### Memory Issues

If you encounter memory issues with large quantum systems:

1. **Use sparse representations:**
   ```rust
   use quantrs2_core::memory_efficient::EfficientStateVector;
   let state = EfficientStateVector::new(num_qubits);
   ```

2. **Enable adaptive precision:**
   ```rust
   let config = AdaptivePrecisionConfig::default();
   let sim = AdaptivePrecisionSimulator::new(config);
   ```

3. **Use GPU acceleration:**
   ```rust
   let gpu = GpuBackendFactory::create_backend(gpu_config)?;
   ```

### Performance Issues

If simulations are slow:

1. **Enable SIMD:**
   ```rust
   let config = BatchConfig { use_simd: true, ..Default::default() };
   ```

2. **Use parallel execution:**
   ```rust
   let config = BatchConfig { parallel_execution: true, ..Default::default() };
   ```

3. **Profile your code:**
   ```rust
   use quantrs2_core::profiling_advanced::QuantumProfiler;
   let profiler = QuantumProfiler::new();
   profiler.start();
   // ... your code ...
   let report = profiler.stop();
   ```

### Numerical Instability

If you see numerical errors:

1. **Increase precision:**
   ```rust
   let config = AdaptivePrecisionConfig {
       initial_precision: 1e-12,
       ..Default::default()
   };
   ```

2. **Use error correction:**
   ```rust
   let surface_code = SurfaceCode::new(distance);
   ```

3. **Apply error mitigation:**
   ```rust
   use quantrs2_core::noise_characterization::ZeroNoiseExtrapolation;
   let zne = ZeroNoiseExtrapolation::new(ExtrapolationMethod::Richardson);
   ```

## Additional Resources

- [API Documentation](https://docs.rs/quantrs2-core)
- [Examples Directory](./examples/)
- [Integration Policy](./SCIRS2_INTEGRATION_POLICY.md)
- [Development Guide](./CLAUDE.md)

## Contributing

See the main project repository for contribution guidelines.

## License

See LICENSE file in the root directory.
