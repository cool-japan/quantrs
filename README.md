# Quantrs: Rust Quantum Computing Framework

Quantrs (`/kwɒntərz/`) is a comprehensive Rust-based quantum computing framework that provides a modular, high-performance toolkit for quantum simulation, algorithm development, and hardware interaction.

## Features

- **Type-Safe Quantum Circuits**: Using Rust's const generics for compile-time verification of qubit counts and operations
- **High Performance**: Leveraging SIMD, multi-threading, and optional GPU acceleration for efficient simulation
- **Multiple Paradigms**: Support for both gate-based quantum computing and quantum annealing
- **Hardware Connectivity**: Connect to real quantum devices from IBM, Azure Quantum, and other platforms
- **Zero-Cost Abstractions**: Maintaining Rust's performance while providing intuitive quantum programming interfaces

## Project Structure

Quantrs is organized as a workspace with several crates:

- **quantrs-core**: Core types, traits, and abstractions shared across the ecosystem
- **quantrs-circuit**: Quantum circuit representation and DSL
- **quantrs-sim**: Quantum simulators (state-vector and tensor-network)
- **quantrs-anneal**: Quantum annealing support and D-Wave integration
- **quantrs-device**: Remote quantum hardware connections
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
- Grover's search algorithm
- Quantum Fourier transform
- VQE (Variational Quantum Eigensolver)
- Quantum annealing

## Performance

Quantrs is designed for high performance quantum simulation:

- Efficiently simulates up to 30+ qubits on standard hardware
- Parallel execution with Rayon
- Optional GPU acceleration with WGPU
- Optimized tensor network contraction for certain circuit structures

## Roadmap

See [TODO.md](TODO.md) for the development roadmap and upcoming features.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.