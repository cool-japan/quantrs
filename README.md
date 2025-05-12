# Quantrs: Rust Quantum Computing Framework

Quantrs (`/kwɒntərz/`) is a comprehensive Rust-based quantum computing framework that provides a modular, high-performance toolkit for quantum simulation, algorithm development, and hardware interaction.

## Features

- **Type-Safe Quantum Circuits**: Using Rust's const generics for compile-time verification of qubit counts and operations
- **High Performance**: Leveraging SIMD, multi-threading, tensor networks, and optional GPU acceleration for efficient simulation
- **Multiple Paradigms**: Support for both gate-based quantum computing and quantum annealing
- **Hardware Connectivity**: Connect to real quantum devices from IBM, Azure Quantum, and other platforms
- **Comprehensive Gate Set**: Includes all standard gates plus S/T-dagger, Square root of X, and controlled variants
- **Realistic Noise Models**: Simulate quantum hardware with configurable noise channels (bit flip, phase flip, depolarizing, amplitude/phase damping) and IBM device-specific T1/T2 relaxation models
- **Quantum Error Correction**: Protect quantum information with error correction codes (bit flip code, phase flip code, Shor code, 5-qubit perfect code)
- **Tensor Network Simulation**: Memory-efficient simulation of quantum circuits with limited entanglement, featuring specialized contraction path optimization for QFT, QAOA, and other common circuit patterns with benchmarking tools to evaluate performance gains
- **IBM Quantum Integration**: Connect to real IBM quantum hardware with authentication, circuit transpilation, job submission, and result processing
- **Zero-Cost Abstractions**: Maintaining Rust's performance while providing intuitive quantum programming interfaces
- **Quality Code**: Follows modern Rust best practices with no compiler warnings or deprecation issues

## Project Structure

Quantrs is organized as a workspace with several crates:

- **quantrs-core**: Core types, traits, and abstractions shared across the ecosystem
- **quantrs-circuit**: Quantum circuit representation and DSL
- **quantrs-sim**: Quantum simulators (state-vector and tensor-network)
- **quantrs-anneal**: Quantum annealing support and D-Wave integration
- **quantrs-device**: Remote quantum hardware connections (IBM Quantum and other providers)
- **quantrs-py**: Python bindings with PyO3

## Getting Started

First, add Quantrs to your project:

```toml
[dependencies]
quantrs-core = "0.1"
quantrs-circuit = "0.1"
quantrs-sim = "0.1"
```

### Creating a Bell State

```rust
use quantrs_circuit::builder::Circuit;
use quantrs_sim::statevector::StateVectorSimulator;

fn main() {
    // Create a circuit with 2 qubits
    let mut circuit = Circuit::<2>::new();
    
    // Build a Bell state circuit: H(0) followed by CNOT(0, 1)
    circuit.h(0).unwrap()
           .cnot(0, 1).unwrap();
    
    // Run the circuit on the state vector simulator
    let simulator = StateVectorSimulator::new();
    let result = circuit.run(simulator).unwrap();
    
    // Print the resulting probabilities
    for (i, prob) in result.probabilities().iter().enumerate() {
        let bits = format!("{:02b}", i);
        println!("|{}⟩: {:.6}", bits, prob);
    }
}
```

### Creating a Superposition State

```rust
use quantrs_circuit::builder::Circuit;
use quantrs_sim::statevector::StateVectorSimulator;

fn main() {
    // Create a circuit with 3 qubits
    let mut circuit = Circuit::<3>::new();
    
    // Apply Hadamard gates to all qubits to create superposition
    circuit.h(0).unwrap()
           .h(1).unwrap()
           .h(2).unwrap();
    
    // Run the circuit
    let simulator = StateVectorSimulator::new();
    let result = circuit.run(simulator).unwrap();
    
    // Each basis state should have equal probability (1/8)
    for (i, prob) in result.probabilities().iter().enumerate() {
        let bits = format!("{:03b}", i);
        println!("|{}⟩: {:.6}", bits, prob);
    }
}
```

## Examples

Check out the `examples` directory for more quantum algorithms and demonstrations:

- Bell states and entanglement
- Quantum teleportation
- Extended gate set examples (S/T-dagger, √X, controlled gates)
- Noisy quantum simulation with various error channels
- Realistic IBM hardware noise models
- Advanced noise effects on complex quantum algorithms
- Quantum error correction (bit flip code, phase flip code, 5-qubit perfect code)
- Grover's search algorithm
- Quantum Fourier transform
- VQE (Variational Quantum Eigensolver)
- Tensor network optimization benchmarks (comparing different contraction strategies for QFT and QAOA)
- IBM Quantum hardware connectivity examples
- Quantum annealing optimization problems:
  - Maximum Cut (MaxCut)
  - Graph Coloring
  - Traveling Salesman Problem (TSP)
  - 3-Rooks problem

### Running Examples

The examples are divided into two categories:

1. **Quantum Annealing Examples** (no special dependencies):
   ```bash
   # Run MaxCut example
   cargo run --bin max_cut

   # Run Graph Coloring example
   cargo run --bin graph_coloring

   # Run Traveling Salesman example
   cargo run --bin traveling_salesman
   ```

2. **Quantum Simulation Examples** (require the `simulation` feature):
   ```bash
   # Run Bell State example
   cargo run --features simulation --bin bell_state

   # Run optimized simulator example
   cargo run --features simulation --bin optimized_sim_small

   # Run GPU-accelerated simulator example
   cargo run --features simulation,gpu --bin gpu_simulation

   # Run extended gates example
   cargo run --features simulation --bin extended_gates

   # Run noisy simulator examples
   cargo run --features simulation --bin noisy_simulator
   cargo run --features simulation --bin extended_gates_with_noise

   # Run IBM Quantum integration examples
   cargo run --features simulation,ibm --bin ibm_quantum_hello

   # Run quantum error correction examples
   cargo run --features simulation --bin error_correction
   cargo run --features simulation --bin phase_error_correction
   cargo run --features simulation --bin five_qubit_code
   cargo run --features simulation --bin error_correction_comparison

   # Run tensor network simulator examples
   cargo run --features simulation --bin tensor_network_sim
   cargo run --features simulation --bin tensor_network_optimization
   ```

Note: Simulation examples require additional dependencies including linear algebra libraries.

## Performance

Quantrs is designed for high performance quantum simulation:

- Efficiently simulates up to 30+ qubits on standard hardware
- Parallel execution with Rayon
- Optional GPU acceleration with WGPU
- Memory-efficient algorithms for large qubit counts (25+)
- Multiple simulation backends:
  - State vector simulator for general-purpose circuits
  - Tensor network simulator for circuits with limited entanglement
  - Automatic selection based on circuit structure
- Optimized contraction paths for tensor networks to minimize computational cost

## Roadmap

See [TODO.md](TODO.md) for the development roadmap and upcoming features.

## Development

### Code Quality

The Quantrs project maintains high code quality standards:

- All code compiles with zero warnings when using `cargo clippy -- -D warnings`
- Modern Rust APIs are used throughout (rand 0.9+, ndarray 0.15+)
- CI checks enforce compilation without warnings
- Dead code is appropriately marked with `#[allow(dead_code)]` for future API stability

### Build Requirements

For normal builds:
```bash
cargo build
```

For a completely warning-free build:
```bash
cargo clippy --all -- -D warnings
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Before submitting, please ensure:

1. Your code passes all tests and builds without warnings
2. You've added appropriate comments and documentation
3. You've updated any relevant examples

## Optional Features

Quantrs provides several optional features:

- **parallel**: Enables parallel execution using Rayon (enabled by default)
- **gpu**: Enables GPU acceleration using WGPU
- **ibm**: Enables IBM Quantum hardware integration
- **dwave**: Enables D-Wave quantum annealing integration using SymEngine (requires additional dependencies)

To use these features, add them to your dependencies:

```toml
[dependencies]
quantrs-sim = { version = "0.1", features = ["parallel", "gpu"] }
quantrs-device = { version = "0.1", features = ["ibm"] }
```

### GPU Acceleration

The `gpu` feature enables GPU-accelerated quantum simulation using WGPU:

```toml
[dependencies]
quantrs-sim = { version = "0.1", features = ["gpu"] }
```

This requires a WGPU-compatible GPU (most modern GPUs). The GPU acceleration implementation uses compute shaders to parallelize quantum operations, providing significant speedup for large qubit counts.

Use the `GpuStateVectorSimulator` to run circuits on the GPU:

```rust
use quantrs_circuit::prelude::*;
use quantrs_sim::gpu::GpuStateVectorSimulator;

async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if GPU acceleration is available
    if GpuStateVectorSimulator::is_available() {
        // Create a GPU simulator
        let simulator = GpuStateVectorSimulator::new().await?;

        // Create and run a circuit
        let circuit = Circuit::<10>::new().h(0).cnot(0, 1);
        let result = simulator.run(&circuit);

        // Process results
        println!("Result: {:?}", result);
    } else {
        println!("GPU acceleration not available");
    }

    Ok(())
}
```

For synchronous code, you can use the blocking constructor:

```rust
let simulator = GpuStateVectorSimulator::new_blocking()?;
```

The GPU simulator provides significant performance benefits for circuits with more than 10 qubits, often achieving 10-100x speedups over CPU simulation for large circuits (20+ qubits).

### IBM Quantum Integration

The `ibm` feature enables connection to IBM Quantum hardware:

```toml
[dependencies]
quantrs-device = { version = "0.1", features = ["ibm"] }
```

To use IBM Quantum, you'll need an IBM Quantum account and API token. Use the token to authenticate:

```rust
use quantrs_device::{create_ibm_client, create_ibm_device};

async fn run() {
    let token = "your_ibm_quantum_token";
    let device = create_ibm_device(token, "ibmq_qasm_simulator", None).await.unwrap();

    // Check device properties
    let properties = device.properties().await.unwrap();
    println!("Device properties: {:?}", properties);

    // Execute circuits...
}
```

### D-Wave Integration

The `dwave` feature enables symbolic problem formulation for quantum annealing:

```toml
[dependencies]
quantrs-tytan = { version = "0.1", features = ["dwave"] }
```

This requires the SymEngine library and its dependencies. See [TODO.md](TODO.md) for detailed setup instructions.

## License

This project is licensed under either:

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.