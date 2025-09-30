#![doc = r#"
# QuantRS2 - High-Performance Quantum Computing Framework for Rust

<div align="center">

**QuantRS2** (`/kwɒntərz tu:/`) is a comprehensive, production-ready quantum computing framework
built on Rust's zero-cost abstractions and the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

[![Crates.io](https://img.shields.io/crates/v/quantrs2.svg)](https://crates.io/crates/quantrs2)
[![Documentation](https://docs.rs/quantrs2/badge.svg)](https://docs.rs/quantrs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

</div>

---

## 🌟 Overview

QuantRS2 provides a unified, modular toolkit for quantum computing that spans:
- **Quantum Circuit Design** with expressive DSLs and visual representations
- **Multiple Simulation Backends** (state-vector, tensor-network, stabilizer, GPU-accelerated)
- **Real Hardware Integration** (IBM Quantum, Azure Quantum, AWS Braket)
- **Quantum Machine Learning** (QNNs, QGANs, VQE, QAOA)
- **Quantum Annealing** (D-Wave integration, QUBO/Ising solvers)
- **Symbolic Quantum Computation** with SymEngine integration
- **Python Bindings** via PyO3 for seamless interoperability

Built on the [SciRS2 scientific computing foundation](https://github.com/cool-japan/scirs), QuantRS2
leverages battle-tested linear algebra, automatic differentiation, and optimization libraries,
ensuring both **correctness** and **performance** for quantum algorithm development.

---

## 📦 Installation

### Basic Installation

Add QuantRS2 to your `Cargo.toml`:

```toml
[dependencies]
quantrs2 = "0.1.0-beta.2"
```

### Feature Flags

Enable specific modules as needed:

```toml
# Full installation with all features
quantrs2 = { version = "0.1.0-beta.2", features = ["full"] }

# Selective installation
quantrs2 = { version = "0.1.0-beta.2", features = ["circuit", "sim", "ml"] }
```

**Available Features:**
- `core` (always enabled) - Core quantum types and traits
- `circuit` - Quantum circuit representation and DSL
- `sim` - Quantum simulators (state-vector, tensor-network, stabilizer)
- `device` - Real quantum hardware integration (IBM, Azure, AWS)
- `ml` - Quantum machine learning (QNNs, VQE, QAOA)
- `anneal` - Quantum annealing and optimization
- `tytan` - High-level annealing library (Tytan API)
- `symengine` - Symbolic computation with SymEngine
- `full` - All features enabled

---

## 🚀 Quick Start

### Example 1: Bell State Circuit

Create and simulate a Bell state (maximally entangled 2-qubit state):

```rust,ignore
// This example demonstrates basic quantum circuit creation and simulation
// Requires: quantrs2 = { version = "0.1.0-beta.2", features = ["circuit", "sim"] }

use quantrs2_circuit::Circuit;
use quantrs2_sim::StateVectorSimulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2-qubit circuit
    let mut circuit = Circuit::<2>::new();

    // Apply Hadamard gate to qubit 0
    circuit.h(0);

    // Apply CNOT gate (control: 0, target: 1)
    circuit.cnot(0, 1);

    // Simulate the circuit
    let simulator = StateVectorSimulator::new();
    let result = simulator.run(&circuit, 1000)?;

    // Print measurement statistics
    println!("Bell state measurements: {:?}", result.counts());
    // Expected: ~50% |00⟩, ~50% |11⟩

    Ok(())
}
```

### Example 2: Variational Quantum Eigensolver (VQE)

Compute the ground state energy of a molecular Hamiltonian:

```rust,ignore
// This example demonstrates VQE usage with the quantum ML module
// Requires: quantrs2 = { version = "0.1.0-beta.2", features = ["ml"] }

// Define H2 molecule Hamiltonian
let hamiltonian = MolecularHamiltonian::h2_sto3g(0.74)?;

// Create parameterized ansatz circuit
let ansatz = ParameterizedCircuit::hardware_efficient(4, 2);

// Configure VQE with Adam optimizer
let vqe = VQE::builder()
    .hamiltonian(hamiltonian)
    .ansatz(ansatz)
    .optimizer(Adam::default())
    .max_iterations(100)
    .build()?;

// Run optimization
let result = vqe.optimize()?;

println!("Ground state energy: {:.6} Ha", result.energy);
println!("Optimal parameters: {:?}", result.parameters);
```

### Example 3: Quantum Approximate Optimization Algorithm (QAOA)

Solve MaxCut problem on a graph:

```rust,ignore
// This example demonstrates QAOA usage with the quantum ML module
// Requires: quantrs2 = { version = "0.1.0-beta.2", features = ["ml"] }

// Define graph edges for MaxCut problem
let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
let graph = Graph::from_edges(&edges, 4);

// Configure QAOA with 3 layers
let qaoa = QAOA::new(graph, 3);

// Run optimization
let result = qaoa.optimize(500)?;

println!("MaxCut value: {}", result.objective);
println!("Optimal cut: {:?}", result.bitstring);
```

### Example 4: Quantum Annealing with D-Wave

```rust,ignore
// This example demonstrates quantum annealing with D-Wave integration
// Requires: quantrs2 = { version = "0.1.0-beta.2", features = ["anneal"] }

// Define QUBO problem
let mut qubo = QUBO::new(4);
qubo.add_term(0, 0, -1.0);
qubo.add_term(1, 1, -1.0);
qubo.add_term(0, 1, 2.0);

// Connect to D-Wave
let client = DWaveClient::from_env()?;

// Submit and solve
let result = client.sample_qubo(&qubo, 1000).await?;

println!("Best solution: {:?}", result.best_sample());
println!("Energy: {:.4}", result.best_energy());
```

---

## 🏗️ Architecture

QuantRS2 follows a modular, layered architecture:

```text
┌─────────────────────────────────────────────────────────────┐
│                     User Applications                       │
│  (Quantum Algorithms, ML Models, Research Experiments)      │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  High-Level Interfaces                      │
│  quantrs2-ml      quantrs2-anneal      quantrs2-tytan       │
│  (VQE, QAOA)      (QUBO, Ising)        (Annealing DSL)      │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Quantum Circuit & Simulation Layer             │
│  quantrs2-circuit            quantrs2-sim                   │
│  (Circuit DSL, Gates)        (Simulators, Backends)         │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Hardware & Device Integration                │
│  quantrs2-device           quantrs2-symengine               │
│  (IBM, Azure, AWS)         (Symbolic Computation)           │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core Foundation                         │
│  quantrs2-core   (Types, Traits, State Representation)      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  SciRS2 Ecosystem (scirs2-core, scirs2-linalg, etc.)  │  │
│  │  (Complex Numbers, Linear Algebra, FFT, Optimization) │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Crate Overview

| Crate | Purpose | Key Features |
|-------|---------|--------------|
| **`quantrs2-core`** | Foundational types and traits | Quantum states, gates, measurements, error types |
| **`quantrs2-circuit`** | Circuit construction and manipulation | DSL, gate definitions, circuit optimization |
| **`quantrs2-sim`** | Quantum simulation backends | State-vector, tensor-network, stabilizer, GPU |
| **`quantrs2-device`** | Real quantum hardware integration | IBM Quantum, Azure Quantum, AWS Braket |
| **`quantrs2-ml`** | Quantum machine learning | VQE, QAOA, QNNs, QGANs, gradient computation |
| **`quantrs2-anneal`** | Quantum annealing | QUBO/Ising solvers, D-Wave integration |
| **`quantrs2-tytan`** | High-level annealing DSL | Intuitive problem modeling, auto-compilation |
| **`quantrs2-symengine`** | Symbolic computation | Parametric gates, symbolic optimization |
| **`quantrs2-py`** | Python bindings | PyO3-based interoperability with NumPy |

---

## 🧪 Feature Comparison

### Simulation Backends

| Backend | Max Qubits | Memory | Speed | Use Case |
|---------|-----------|---------|-------|----------|
| **State Vector** | 30+ | O(2ⁿ) | Fast | General circuits, small-medium scale |
| **Tensor Network** | 50+ | O(poly) | Medium | Low-entanglement circuits |
| **Stabilizer** | 100+ | O(n²) | Very Fast | Clifford circuits, error correction |
| **GPU (CUDA)** | 35+ | O(2ⁿ) | Very Fast | Large-scale batch simulations |

### Hardware Providers

| Provider | Access Method | Supported Devices | Authentication |
|----------|---------------|-------------------|----------------|
| **IBM Quantum** | REST API + WebSocket | 100+ qubits (various topologies) | API Token |
| **Azure Quantum** | Azure SDK | IonQ, Rigetti, Honeywell | Azure Credentials |
| **AWS Braket** | Boto3 | IonQ, Rigetti, Oxford | AWS IAM |

---

## 🔬 Advanced Features

### Automatic Differentiation for Variational Algorithms

QuantRS2 integrates with SciRS2's automatic differentiation engine for efficient gradient computation:

```rust,ignore
// Define parametrized quantum circuit
let params = Variable::new(vec![0.5, 1.2, 0.8]);
let circuit = ParameterizedCircuit::from_parameters(&params);

// Compute expectation value
let energy = circuit.expectation_value(&hamiltonian)?;

// Automatic gradient computation
let gradients = backward(&energy)?;
```

### Circuit Optimization and Compilation

Automatic circuit optimization reduces gate count and depth:

```rust,ignore
let mut circuit = Circuit::<5>::new();
// ... build circuit ...

// Apply optimization passes
let optimized = optimize::pipeline()
    .add_pass(optimize::RemoveIdentityGates)
    .add_pass(optimize::MergeSingleQubitGates)
    .add_pass(optimize::CommuteCNOTs)
    .run(&circuit)?;

println!("Original depth: {}", circuit.depth());
println!("Optimized depth: {}", optimized.depth());
```

### Error Mitigation

Built-in error mitigation techniques for noisy quantum devices:

```rust,ignore
let result = client
    .run_circuit(&circuit)
    .with_error_mitigation(ErrorMitigation::ZNE { scale_factors: vec![1.0, 1.5, 2.0] })
    .shots(10000)
    .execute()
    .await?;
```

### Tensor Network Contraction

Efficient simulation of low-entanglement quantum circuits:

```rust,ignore
let simulator = TensorNetworkSimulator::builder()
    .strategy(ContractionStrategy::Greedy)
    .max_bond_dimension(128)
    .build();

let result = simulator.run(&circuit, 1000)?;
```

---

## 🔧 Dependencies and Integration

### SciRS2 Foundation

QuantRS2 is built on the [SciRS2 scientific computing ecosystem](https://github.com/cool-japan/scirs):

- **`scirs2-core`**: Complex numbers, random number generation, SIMD operations
- **`scirs2-linalg`**: Unitary matrix operations, eigenvalue solvers
- **`scirs2-autograd`**: Automatic differentiation for variational algorithms
- **`scirs2-optimize`**: Optimization algorithms (Adam, L-BFGS, COBYLA)
- **`scirs2-fft`**: Fast Fourier Transform for Quantum Fourier Transform
- **`scirs2-sparse`**: Sparse matrix operations for large Hamiltonians
- **`scirs2-neural`**: Neural network primitives for quantum ML

**Important**: QuantRS2 follows strict SciRS2 integration policies. All array operations use
`scirs2_core::ndarray`, all complex numbers use `scirs2_core::{Complex64, Complex32}`, and all
random number generation uses `scirs2_core::random`. See [`SCIRS2_INTEGRATION_POLICY.md`](https://github.com/cool-japan/quantrs/blob/master/SCIRS2_INTEGRATION_POLICY.md) for details.

### OptiRS Integration

Advanced optimization algorithms from [OptiRS](https://github.com/cool-japan/optirs):

```rust,ignore
let optimizer = DifferentialEvolution::default();
let result = vqe.with_optimizer(optimizer).optimize()?;
```

---

## 🎯 Use Cases

### Quantum Chemistry

```rust,ignore
// Define molecule geometry
let h2o = Molecule::new()
    .add_atom("O", [0.0, 0.0, 0.0])
    .add_atom("H", [0.0, 0.757, 0.586])
    .add_atom("H", [0.0, -0.757, 0.586])
    .build()?;

// Compute ground state energy
let hamiltonian = h2o.hamiltonian(BasisSet::STO3G)?;
let vqe = VQE::new(hamiltonian);
let energy = vqe.optimize()?.energy;
```

### Quantum Machine Learning

```rust,ignore
// Build hybrid quantum-classical neural network
let qnn = QuantumNeuralNetwork::builder()
    .add_layer(QuantumLayer::new(4, 2))  // 4 qubits, 2 layers
    .add_classical_layer(Dense::new(16, 10))
    .build()?;

// Train on dataset
qnn.fit(&train_data, &train_labels, epochs: 50)?;
```

### Combinatorial Optimization

```rust,ignore
// Solve traveling salesman problem
let distances = vec![vec![0.0, 2.0, 3.0], vec![2.0, 0.0, 1.0], vec![3.0, 1.0, 0.0]];
let tsp = TravelingSalesman::new(distances);
let solution = tsp.solve_quantum_annealing()?;

println!("Optimal route: {:?}", solution.route);
println!("Total distance: {:.2}", solution.distance);
```

### Quantum Cryptography

```rust,ignore
// Quantum key distribution
let alice = BB84::new_sender(1024);
let bob = BB84::new_receiver();

let (alice_key, bob_key) = BB84::exchange(&mut alice, &mut bob)?;
assert_eq!(alice_key, bob_key);
```

---

## 📊 Performance

### Benchmarks (Apple M2 Max, 64GB RAM)

| Operation | Time | Throughput |
|-----------|------|------------|
| 20-qubit state vector simulation | 1.2 ms | 830 circuits/s |
| 30-qubit state vector simulation | 1.8 s | 0.55 circuits/s |
| CNOT gate application (20 qubits) | 45 μs | 22k gates/s |
| VQE iteration (4 qubits, 10 params) | 3.2 ms | 310 iterations/s |
| Tensor network contraction (50 qubits) | 250 ms | 4 circuits/s |

### Memory Usage

| System Size | State Vector | Tensor Network | Stabilizer |
|-------------|--------------|----------------|------------|
| 20 qubits | 16 MB | 2 MB | 3 KB |
| 30 qubits | 16 GB | 8 MB | 7 KB |
| 50 qubits | 16 PB (infeasible) | 128 MB | 20 KB |
| 100 qubits | - | - | 80 KB |

---

## 🌍 Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux (x86_64)** | ✅ Full Support | Recommended for production |
| **macOS (Intel)** | ✅ Full Support | Native performance |
| **macOS (Apple Silicon)** | ✅ Full Support | Optimized for M-series chips |
| **Windows (x86_64)** | ✅ Full Support | Requires MSVC toolchain |
| **WebAssembly** | 🚧 Experimental | Limited feature set |
| **CUDA (NVIDIA GPUs)** | ✅ Full Support | Requires CUDA 11.8+ |
| **OpenCL** | 🚧 Experimental | Limited backend support |

---

## 🧰 Development Tools

### Command Line Tools

```bash
# Generate circuit visualization (requires graphviz)
quantrs2 viz circuit.json -o circuit.png

# Benchmark quantum algorithms
quantrs2 bench --algorithm vqe --qubits 4-12

# Validate circuit on hardware constraints
quantrs2 validate circuit.json --backend ibm_lagos

# Convert between circuit formats
quantrs2 convert circuit.qasm -o circuit.json
```

### Python Integration

```python
import quantrs2

# Use QuantRS2 from Python
circuit = quantrs2.Circuit(num_qubits=2)
circuit.h(0)
circuit.cnot(0, 1)

simulator = quantrs2.StateVectorSimulator()
result = simulator.run(circuit, shots=1000)

print(result.counts())  # {'00': 501, '11': 499}
```

---

## 📚 Resources

### Documentation
- **API Documentation**: [docs.rs/quantrs2](https://docs.rs/quantrs2)
- **User Guide**: [quantrs2.github.io/guide](https://quantrs2.github.io/guide)
- **Examples**: [github.com/cool-japan/quantrs/tree/master/examples](https://github.com/cool-japan/quantrs/tree/master/examples)
- **Integration Policy**: [SCIRS2_INTEGRATION_POLICY.md](https://github.com/cool-japan/quantrs/blob/master/SCIRS2_INTEGRATION_POLICY.md)

### Community
- **GitHub**: [github.com/cool-japan/quantrs](https://github.com/cool-japan/quantrs)
- **Issues**: [github.com/cool-japan/quantrs/issues](https://github.com/cool-japan/quantrs/issues)
- **Discussions**: [github.com/cool-japan/quantrs/discussions](https://github.com/cool-japan/quantrs/discussions)

### Related Projects
- **SciRS2**: Scientific computing foundation - [github.com/cool-japan/scirs](https://github.com/cool-japan/scirs)
- **OptiRS**: Advanced optimization algorithms - [github.com/cool-japan/optirs](https://github.com/cool-japan/optirs)
- **NumRS2**: Numerical computing library - [github.com/cool-japan/numrs](https://github.com/cool-japan/numrs)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/cool-japan/quantrs/blob/master/CONTRIBUTING.md) for guidelines.

Key areas for contribution:
- New quantum algorithms and applications
- Performance optimizations and benchmarks
- Hardware backend integrations
- Documentation and examples
- Bug reports and feature requests

---

## 📜 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## 🔖 Version Information

**Current Version**: `0.1.0-beta.2`

### Recent Updates (v0.1.0-beta.2)

- ✨ Refined SciRS2 v0.1.0-beta.3 integration with unified import patterns
- 📚 Comprehensive policy documentation ([SCIRS2_INTEGRATION_POLICY.md](https://github.com/cool-japan/quantrs/blob/master/SCIRS2_INTEGRATION_POLICY.md))
- 🛠️ Enhanced developer experience with [CLAUDE.md](https://github.com/cool-japan/quantrs/blob/master/CLAUDE.md) development guidelines
- 🔄 All subcrates updated with consistent SciRS2 usage patterns
- 📖 Significantly improved documentation across all modules
- 🚀 Performance optimizations in state-vector simulator
- 🐛 Bug fixes in circuit optimization passes

### Roadmap

- **v0.1.0-beta.3**: Distributed quantum simulation, enhanced error correction
- **v0.1.0**: Stable release with complete API stability guarantees
- **v0.2.0**: Advanced quantum ML models, quantum chemistry extensions

---

<div align="center">

**Built with ❤️ by the QuantRS2 Contributors**

*Empowering quantum computing research and development with Rust's performance and safety.*

</div>
"#]

pub use quantrs2_core as core;

#[cfg(feature = "circuit")]
pub use quantrs2_circuit as circuit;

#[cfg(feature = "sim")]
pub use quantrs2_sim as sim;

#[cfg(feature = "anneal")]
pub use quantrs2_anneal as anneal;

#[cfg(feature = "device")]
pub use quantrs2_device as device;

#[cfg(feature = "ml")]
pub use quantrs2_ml as ml;

#[cfg(feature = "tytan")]
pub use quantrs2_tytan as tytan;

#[cfg(feature = "symengine")]
pub use quantrs2_symengine as symengine;