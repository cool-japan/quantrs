#![doc = r#"
# `QuantRS2` - High-Performance Quantum Computing Framework for Rust

<div align="center">

**QuantRS2** (`/kwɒntərz tu:/`) is a comprehensive, production-ready quantum computing framework
built on Rust's zero-cost abstractions and the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

[![Crates.io](https://img.shields.io/crates/v/quantrs2.svg)](https://crates.io/crates/quantrs2)
[![Documentation](https://docs.rs/quantrs2/badge.svg)](https://docs.rs/quantrs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs/blob/master/LICENSE)

</div>

---

## 🌟 Overview

`QuantRS2` provides a unified, modular toolkit for quantum computing that spans:
- **Quantum Circuit Design** with expressive DSLs and visual representations
- **Multiple Simulation Backends** (state-vector, tensor-network, stabilizer, GPU-accelerated)
- **Real Hardware Integration** (IBM Quantum, Azure Quantum, AWS Braket)
- **Quantum Machine Learning** (QNNs, QGANs, VQE, QAOA)
- **Quantum Annealing** (D-Wave integration, QUBO/Ising solvers)
- **Symbolic Quantum Computation** with `SymEngine` integration
- **Python Bindings** via `PyO3` for seamless interoperability

Built on the [SciRS2 scientific computing foundation](https://github.com/cool-japan/scirs), `QuantRS2`
leverages battle-tested linear algebra, automatic differentiation, and optimization libraries,
ensuring both **correctness** and **performance** for quantum algorithm development.

---

## 📦 Installation

### Basic Installation

Add `QuantRS2` to your `Cargo.toml`:

```toml
[dependencies]
quantrs2 = "0.1.2"
```

### Feature Flags

Enable specific modules as needed:

```toml
# Full installation with all features
quantrs2 = { version = "0.1.2", features = ["full"] }

# Selective installation
quantrs2 = { version = "0.1.2", features = ["circuit", "sim", "ml"] }
```

**Available Features:**
- `core` (always enabled) - Core quantum types and traits
- `circuit` - Quantum circuit representation and DSL
- `sim` - Quantum simulators (state-vector, tensor-network, stabilizer)
- `device` - Real quantum hardware integration (IBM, Azure, AWS)
- `ml` - Quantum machine learning (QNNs, VQE, QAOA)
- `anneal` - Quantum annealing and optimization
- `tytan` - High-level annealing library (Tytan API)
- `symengine` - Symbolic computation with `SymEngine`
- `full` - All features enabled

---

## 🚀 Quick Start

### Example 1: Bell State Circuit

Create and simulate a Bell state (maximally entangled 2-qubit state):

```rust,ignore
// This example demonstrates basic quantum circuit creation and simulation
// Requires: quantrs2 = { version = "0.1.2", features = ["circuit", "sim"] }

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
// Requires: quantrs2 = { version = "0.1.2", features = ["ml"] }

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

Solve `MaxCut` problem on a graph:

```rust,ignore
// This example demonstrates QAOA usage with the quantum ML module
// Requires: quantrs2 = { version = "0.1.2", features = ["ml"] }

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
// Requires: quantrs2 = { version = "0.1.2", features = ["anneal"] }

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

### Example 5: Facade Features - Configuration, Diagnostics, and Utilities

The QuantRS2 facade provides powerful system management features:

```rust
use quantrs2::{config, diagnostics, utils, version};
use quantrs2::prelude::essentials::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Version information
    println!("QuantRS2 v{}", version::VERSION);

    // System diagnostics
    let report = diagnostics::run_diagnostics();
    if !report.is_ready() {
        eprintln!("System not ready: {}", report.summary());
        return Err("System check failed".into());
    }

    // Configuration management
    let cfg = config::Config::global();
    cfg.set_num_threads(8);
    cfg.set_memory_limit_gb(16);
    cfg.set_default_backend(config::DefaultBackend::Auto);

    // Memory estimation for quantum circuits
    let num_qubits = 25;
    let required_memory = utils::estimate_statevector_memory(num_qubits);
    println!(
        "{} qubits requires {}",
        num_qubits,
        utils::format_memory(required_memory)
    );

    // Calculate maximum qubits for available memory
    let available = 16 * 1024 * 1024 * 1024; // 16 GB
    let max_qubits = utils::max_qubits_for_memory(available);
    println!("Can simulate up to {} qubits with 16 GB", max_qubits);

    // Compatibility checking
    if let Err(issues) = version::check_compatibility() {
        eprintln!("Compatibility issues found:");
        for issue in &issues {
            eprintln!("  - {}", issue);
        }
        return Err("Compatibility check failed".into());
    }

    Ok(())
}
```

**Facade Capabilities:**
- **System Diagnostics**: Hardware detection, capability checking, compatibility validation
- **Global Configuration**: Thread pools, memory limits, backend selection, logging levels
- **Utility Functions**: Memory estimation, validation helpers, formatting utilities
- **Testing Helpers**: Assertion utilities, test data generation, temporary directories
- **Version Management**: Compatibility checking, version information, build metadata
- **Error Handling**: Categorized errors, context support, user-friendly messages

See the `examples/` directory for more comprehensive demonstrations.

---

## 🎨 Feature Combinations Guide

### Minimal Circuit Design (No Simulation)

For users who only need circuit construction and manipulation:

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["circuit"] }
```

```rust,ignore
use quantrs2::prelude::circuits::*;

// Create and manipulate quantum circuits
let mut circuit = Circuit::<3>::new();
circuit.h(0);
circuit.cnot(0, 1);
circuit.cnot(1, 2);

// Get circuit properties
println!("Circuit depth: {}", circuit.depth());
println!("Gate count: {}", circuit.gate_count());

// Optimize circuit
// let optimized = circuit.optimize();
```

**Use Cases**: Circuit design tools, QASM export, circuit analysis, educational tools

### Quantum Simulation

For quantum algorithm research and development:

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["sim"] }
```

```rust,ignore
use quantrs2::prelude::simulation::*;

// Create circuit
let mut circuit = Circuit::<4>::new();
circuit.h(0);
for i in 0..3 {
    circuit.cnot(i, i + 1);
}

// Simulate with multiple backends
let state_vector_sim = StateVectorSimulator::new();
let result1 = state_vector_sim.run(&circuit, 1000)?;

// For large circuits with low entanglement
// let tensor_network_sim = TensorNetworkSimulator::new();
// let result2 = tensor_network_sim.run(&circuit, 1000)?;
```

**Use Cases**: Algorithm development, quantum computing research, education, benchmarking

### Quantum Machine Learning

For variational algorithms and quantum ML:

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["ml"] }
```

```rust,ignore
use quantrs2::prelude::algorithms::*;

// VQE for quantum chemistry
// let hamiltonian = MolecularHamiltonian::h2_sto3g(0.74)?;
// let ansatz = ParameterizedCircuit::hardware_efficient(4, 2);
// let vqe = VQE::builder()
//     .hamiltonian(hamiltonian)
//     .ansatz(ansatz)
//     .optimizer(Adam::default())
//     .build()?;
// let result = vqe.optimize()?;

// QAOA for combinatorial optimization
// let graph = Graph::from_edges(&[(0,1), (1,2), (2,3), (3,0)], 4);
// let qaoa = QAOA::new(graph, 3);
// let solution = qaoa.optimize(500)?;
```

**Use Cases**: Quantum chemistry, combinatorial optimization, quantum neural networks

### Quantum Annealing

For QUBO/Ising model optimization:

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["tytan"] }
```

```rust,ignore
use quantrs2::prelude::tytan::*;

// Define optimization problem with high-level DSL
// let mut problem = Problem::new();
// problem.add_constraint(/* ... */);
//
// // Solve with quantum annealing
// let solver = TytanSolver::new();
// let solution = solver.solve(&problem)?;
```

**Use Cases**: Combinatorial optimization, scheduling, resource allocation, portfolio optimization

### Hardware Integration

For real quantum device execution:

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["device", "circuit"] }
```

```rust,ignore
use quantrs2::prelude::hardware::*;

// Connect to IBM Quantum
// let client = IBMClient::from_env()?;
// let backend = client.get_backend("ibm_lagos").await?;
//
// // Execute circuit on real hardware
// let mut circuit = Circuit::<2>::new();
// circuit.h(0);
// circuit.cnot(0, 1);
//
// let job = backend.run(&circuit, shots: 1024).await?;
// let result = job.wait_for_completion().await?;
```

**Use Cases**: Real quantum hardware experiments, noise characterization, benchmarking

### Full Installation

For comprehensive quantum computing capabilities:

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["full"] }
```

```rust,ignore
use quantrs2::prelude::full::*;

// Access all QuantRS2 capabilities:
// - Circuit design and optimization
// - Multiple simulation backends
// - Variational algorithms (VQE, QAOA)
// - Quantum annealing
// - Hardware integration
// - Symbolic computation
```

**Use Cases**: Research platforms, comprehensive quantum software, production applications

**Note**: `full` feature significantly increases compilation time (~2-3 minutes). Consider enabling only needed features for faster development cycles.

---

## ⚡ Performance & Compilation Trade-offs

### Compilation Time by Feature Set

| Feature Configuration | Compilation Time* | Binary Size* | Use Case |
|----------------------|-------------------|--------------|----------|
| `core` only | ~10s | ~2 MB | Type definitions, minimal usage |
| `circuit` | ~30s | ~8 MB | Circuit design only |
| `sim` | ~60s | ~25 MB | Simulation without ML |
| `ml` | ~90s | ~40 MB | Full algorithm development |
| `full` | ~120-180s | ~60 MB | Complete framework |

*Approximate values on Apple M2 Max, release mode, clean build

### Runtime Performance by Backend

| Backend | Qubits | Memory | Speed | Parallelization |
|---------|--------|--------|-------|----------------|
| **StateVector** | 1-30 | O(2^n) | Fast | ✅ Multi-threaded |
| **TensorNetwork** | 10-50+ | O(poly) | Medium | ✅ Multi-threaded |
| **Stabilizer** | 50-1000+ | O(n²) | Very Fast | ✅ Multi-threaded |
| **GPU** (CUDA) | 1-35 | O(2^n) | Very Fast | ✅ GPU parallel |

### SciRS2 Integration Performance

QuantRS2 uses **SciRS2** for all numerical operations, providing:
- **SIMD Acceleration**: 2-4× speedup on supported operations (AVX2, AVX-512, NEON)
- **GPU Offloading**: 10-50× speedup for large state vectors (CUDA/Metal)
- **Cache Optimization**: Efficient memory access patterns via `scirs2_core`
- **Zero-Cost Abstractions**: No runtime overhead from the facade layer

**Tip**: Enable SciRS2 SIMD features for maximum performance:
```toml
quantrs2 = { version = "0.1.2", features = ["sim"] }
# SciRS2 automatically uses SIMD when available
```

---

## 🔄 Migration Guide

### From Individual Crates to Facade

**Before** (Using individual crates):
```toml
[dependencies]
quantrs2-core = "0.1.2"
quantrs2-circuit = "0.1.2"
quantrs2-sim = "0.1.2"
```

**After** (Using facade):
```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["sim"] }
```

**Code Changes**:
```rust,ignore
// Before:
use quantrs2_circuit::Circuit;
use quantrs2_sim::StateVectorSimulator;

// After - Option 1 (via prelude):
use quantrs2::prelude::simulation::*;

// After - Option 2 (explicit modules):
use quantrs2::circuit::Circuit;
use quantrs2::sim::StateVectorSimulator;
```

**Benefits**:
- ✅ Single version management
- ✅ Automatic feature dependency resolution
- ✅ Unified error handling
- ✅ Access to facade utilities (diagnostics, config, etc.)
- ✅ Better compilation caching

### Updating Dependencies

**Update Cargo.toml**:
```bash
# Update to latest beta
cargo update -p quantrs2

# Or specify exact version
quantrs2 = "0.1.2"
```

**Verify Compatibility**:
```rust,ignore
use quantrs2::version;

fn main() {
    match version::check_compatibility() {
        Ok(()) => println!("✅ All dependencies compatible"),
        Err(issues) => {
            eprintln!("⚠️ Compatibility issues:");
            for issue in issues {
                eprintln!("  - {}", issue);
            }
        }
    }
}
```

---

## 🏗️ Architecture

`QuantRS2` follows a modular, layered architecture:

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
| **`quantrs2-py`** | Python bindings | PyO3-based interoperability with `NumPy` |

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
| **Azure Quantum** | Azure SDK | `IonQ`, Rigetti, Honeywell | Azure Credentials |
| **AWS Braket** | Boto3 | `IonQ`, Rigetti, Oxford | AWS IAM |

---

## 🔬 Advanced Features

### Automatic Differentiation for Variational Algorithms

`QuantRS2` integrates with `SciRS2`'s automatic differentiation engine for efficient gradient computation:

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

### `SciRS2` Foundation

`QuantRS2` is built on the [SciRS2 scientific computing ecosystem](https://github.com/cool-japan/scirs):

- **`scirs2-core`**: Complex numbers, random number generation, SIMD operations
- **`scirs2-linalg`**: Unitary matrix operations, eigenvalue solvers
- **`scirs2-autograd`**: Automatic differentiation for variational algorithms
- **`scirs2-optimize`**: Optimization algorithms (Adam, L-BFGS, COBYLA)
- **`scirs2-fft`**: Fast Fourier Transform for Quantum Fourier Transform
- **`scirs2-sparse`**: Sparse matrix operations for large Hamiltonians
- **`scirs2-neural`**: Neural network primitives for quantum ML

**Important**: `QuantRS2` follows strict `SciRS2` integration policies. All array operations use
`scirs2_core::ndarray`, all complex numbers use `scirs2_core::{Complex64, Complex32}`, and all
random number generation uses `scirs2_core::random`. See [`SCIRS2_INTEGRATION_POLICY.md`](https://github.com/cool-japan/quantrs/blob/master/SCIRS2_INTEGRATION_POLICY.md) for details.

### `OptiRS` Integration

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
| **Linux (`x86_64`)** | ✅ Full Support | Recommended for production |
| **macOS (Intel)** | ✅ Full Support | Native performance |
| **macOS (Apple Silicon)** | ✅ Full Support | Optimized for M-series chips |
| **Windows (`x86_64`)** | ✅ Full Support | Requires MSVC toolchain |
| **WebAssembly** | 🚧 Experimental | Limited feature set |
| **CUDA (NVIDIA GPUs)** | ✅ Full Support | Requires CUDA 11.8+ |
| **`OpenCL`** | 🚧 Experimental | Limited backend support |

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
- **`NumRS2`**: Numerical computing library - [github.com/cool-japan/numrs](https://github.com/cool-japan/numrs)

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
- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/cool-japan/quantrs/blob/master/LICENSE-APACHE))
- MIT License ([LICENSE-MIT](https://github.com/cool-japan/quantrs/blob/master/LICENSE-MIT))

at your option.

---

## 🔖 Version Information

**Current Version**: `0.1.2`

### Recent Updates (v0.1.2)

- 🎯 **Facade Enhancements**: Comprehensive system management features
  - System diagnostics module with hardware detection and compatibility checking
  - Global configuration management with builder pattern
  - Utility functions for memory estimation and validation
  - Testing helpers for quantum algorithm development
- 📊 **Testing Infrastructure**: 119+ tests including unit, integration, and doc tests
- 🔧 **Developer Experience**: Hierarchical prelude system, improved error handling
- 📝 **Documentation**: Enhanced examples and inline documentation
- 🛠️ **Code Quality**: All clippy warnings fixed, formatted with `cargo fmt`
- ✅ **SciRS2 v0.1.2 Stable Release with unified patterns
- ✨ Refined `SciRS2` integration with unified import patterns
- 📚 Comprehensive policy documentation
- 🛠️ Enhanced developer experience with CLAUDE.md development guidelines
- 🔄 All subcrates updated with consistent `SciRS2` usage patterns
- 🚀 Performance optimizations in state-vector simulator
- 🐛 Bug fixes in circuit optimization passes

### Roadmap

- **v0.1.2**: Current stable release with full API stability
- **v0.2.0**: Advanced quantum ML models, quantum chemistry extensions

---

<div align="center">

**Built with ❤️ by the `QuantRS2` Contributors**

*Empowering quantum computing research and development with Rust's performance and safety.*

</div>
"#]
// Allow pedantic lints that are too noisy for this crate
#![allow(clippy::doc_markdown)] // QuantRS2, SciRS2, etc. are proper names
#![allow(clippy::module_name_repetitions)] // Acceptable for a facade crate
#![allow(clippy::must_use_candidate)] // Most functions naturally return values

// Re-export subcrates as modules

/// Core quantum computing types, traits, and abstractions.
///
/// This module is **always available** and provides the fundamental building blocks
/// for quantum computing in QuantRS2:
///
/// - **Quantum Types**: `QubitId`, `GateMatrix`, `Pauli operators`
/// - **Traits**: `QuantumGate`, `QuantumState`, `Measurement`
/// - **Math Utilities**: Complex number operations, unitary validation
/// - **Error Handling**: Core error types and result types
///
/// # Examples
///
/// ```rust
/// use quantrs2::core::QubitId;
///
/// let qubit = QubitId::new(0);
/// assert_eq!(qubit.id(), 0);
/// ```
///
/// # When to Use
///
/// - Building custom quantum gates or operators
/// - Implementing new quantum algorithms
/// - Defining quantum state representations
/// - Working with quantum mathematical operations
///
/// # Related Modules
///
/// - `circuit` - Build quantum circuits using these core types (requires `circuit` feature)
/// - `sim` - Simulate quantum systems with these primitives (requires `sim` feature)
pub use quantrs2_core as core;

/// Quantum circuit construction, optimization, and representation.
///
/// This module provides tools for building and manipulating quantum circuits:
///
/// - **Circuit Builder**: Type-safe circuit construction with const generics
/// - **Gate Library**: Comprehensive quantum gate set (Pauli, Hadamard, CNOT, etc.)
/// - **Circuit Optimization**: Transpilation, gate fusion, routing
/// - **QASM Support**: Import/export OpenQASM 2.0 and 3.0
/// - **Pulse Control**: Low-level hardware pulse sequences
/// - **Visualization**: Circuit diagrams and analysis tools
///
/// Requires: `features = ["circuit"]`
///
/// # Examples
///
/// ```rust,ignore
/// use quantrs2::circuit::Circuit;
///
/// let mut circuit = Circuit::<2>::new();
/// circuit.h(0);  // Hadamard on qubit 0
/// circuit.cnot(0, 1);  // CNOT from 0 to 1
/// ```
///
/// # Key Types
///
/// - `Circuit<N>` - Type-safe quantum circuit with compile-time qubit count
/// - `Gate` - Quantum gate representation
/// - `CircuitOptimizer` - Circuit optimization and transpilation
/// - `QasmParser` - OpenQASM parser and exporter
///
/// # Performance
///
/// Circuit construction is zero-cost and happens at compile-time when possible.
/// Optimization passes use SciRS2's linear algebra for efficient gate decomposition.
///
/// # Related Modules
///
/// - `sim` - Execute circuits on quantum simulators (requires `sim` feature)
/// - `device` - Run circuits on real quantum hardware (requires `device` feature)
#[cfg(feature = "circuit")]
pub use quantrs2_circuit as circuit;

/// Quantum simulation backends and execution engines.
///
/// This module provides multiple simulation strategies for quantum circuits:
///
/// - **State Vector**: Full state simulation (2^n complex amplitudes)
/// - **Stabilizer**: Efficient Clifford circuit simulation
/// - **Tensor Network**: Low-entanglement circuit simulation
/// - **Matrix Product State (MPS)**: Efficient for 1D systems
/// - **GPU Acceleration**: Metal (macOS), CUDA (Linux/Windows)
/// - **Noise Models**: Depolarizing, amplitude damping, measurement errors
///
/// Requires: `features = ["sim"]` (automatically enables `circuit`)
///
/// # Examples
///
/// ```rust,ignore
/// use quantrs2::sim::StateVectorSimulator;
/// use quantrs2::circuit::Circuit;
///
/// let circuit = Circuit::<4>::new();
/// let simulator = StateVectorSimulator::new();
/// let result = simulator.run(&circuit, shots=1000)?;
/// ```
///
/// # Simulator Selection Guide
///
/// | Simulator | Best For | Max Qubits | Memory |
/// |-----------|----------|------------|--------|
/// | StateVector | General circuits | ~30 | O(2^n) |
/// | Stabilizer | Clifford circuits | 100+ | O(n^2) |
/// | TensorNetwork | Low entanglement | ~50 | O(n·χ^2) |
/// | MPS | 1D systems | ~100 | O(n·χ^2) |
///
/// # Performance Tips
///
/// - Use `StateVectorSimulator` for general circuits (up to ~25 qubits)
/// - Use `StabilizerSimulator` for Clifford circuits (very fast, any size)
/// - Use `TensorNetworkSimulator` for circuits with limited entanglement
/// - Enable GPU with `config::Config::global().set_gpu_enabled(true)`
///
/// # Related Modules
///
/// - [`circuit`] - Build circuits to simulate
/// - [`ml`] - Use simulators for quantum machine learning
#[cfg(feature = "sim")]
pub use quantrs2_sim as sim;

/// Quantum annealing algorithms and optimization solvers.
///
/// This module provides quantum annealing capabilities:
///
/// - **QUBO/Ising Models**: Define combinatorial optimization problems
/// - **Simulated Annealing**: Classical solver with quantum-inspired heuristics
/// - **Quantum Annealing**: Interface to quantum annealers (D-Wave)
/// - **Hybrid Solvers**: Combine classical and quantum approaches
/// - **Problem Decomposition**: Large problem partitioning
///
/// Requires: `features = ["anneal"]` (automatically enables `circuit`)
///
/// # Examples
///
/// ```rust,ignore
/// use quantrs2::anneal::{QUBO, SimulatedAnnealer};
///
/// let mut qubo = QUBO::new(4);
/// qubo.add_term(0, 0, -1.0);
/// qubo.add_term(0, 1, 2.0);
///
/// let solver = SimulatedAnnealer::default();
/// let solution = solver.solve(&qubo)?;
/// ```
///
/// # Problem Types
///
/// - **QUBO**: Quadratic Unconstrained Binary Optimization
/// - **Ising**: Spin glass models
/// - **Max-Cut**: Graph partitioning
/// - **TSP**: Traveling Salesman Problem
/// - **Knapsack**: Combinatorial optimization
///
/// # Related Modules
///
/// - [`tytan`] - High-level DSL for annealing problems
/// - [`ml`] - Use annealing for training quantum models
#[cfg(feature = "anneal")]
pub use quantrs2_anneal as anneal;

/// Real quantum hardware backends and device interfaces.
///
/// This module provides connections to real quantum computers:
///
/// - **IBM Quantum**: Access IBM quantum processors via Qiskit Runtime
/// - **Azure Quantum**: Microsoft's quantum cloud platform
/// - **AWS Braket**: Amazon's quantum computing service
/// - **Device Topology**: Hardware connectivity and gate sets
/// - **Error Mitigation**: Zero-noise extrapolation, readout correction
/// - **Transpilation**: Circuit optimization for hardware constraints
///
/// Requires: `features = ["device"]` (automatically enables `circuit`)
///
/// # Examples
///
/// ```rust,ignore
/// use quantrs2::device::IBMBackend;
///
/// let token = std::env::var("IBM_QUANTUM_TOKEN")?;
/// let backend = IBMBackend::new(&token, "ibm_kyoto")?;
///
/// let transpiled = backend.transpile(&circuit)?;
/// let job = backend.submit(&transpiled)?;
/// let result = job.wait_for_completion()?;
/// ```
///
/// # Supported Backends
///
/// - **IBM Quantum**: 7-127 qubit systems (Heron, Eagle processors)
/// - **Azure Quantum**: IonQ, Quantinuum, Rigetti
/// - **AWS Braket**: IonQ, Rigetti, Oxford Quantum Circuits
///
/// # Best Practices
///
/// - Always transpile circuits for hardware topology
/// - Use error mitigation for production jobs
/// - Batch multiple circuits to reduce queue time
/// - Test on simulators before submitting to hardware
///
/// # Related Modules
///
/// - `circuit` - Build circuits to run on hardware (requires `circuit` feature)
/// - `sim` - Test locally before hardware submission (requires `sim` feature)
#[cfg(feature = "device")]
pub use quantrs2_device as device;

/// Quantum machine learning algorithms and neural networks.
///
/// This module provides quantum ML capabilities:
///
/// - **Variational Quantum Eigensolver (VQE)**: Ground state energy computation
/// - **Quantum Approximate Optimization Algorithm (QAOA)**: Combinatorial optimization
/// - **Quantum Neural Networks (QNNs)**: Parameterized quantum circuits for ML
/// - **Quantum Generative Adversarial Networks (QGANs)**: Quantum data generation
/// - **Quantum Autoencoders**: Dimensionality reduction
/// - **Hybrid Training**: Combine quantum and classical layers
///
/// Requires: `features = ["ml"]` (automatically enables `sim` and `anneal`)
///
/// # Examples
///
/// ```rust,ignore
/// use quantrs2::ml::{VQE, ParameterizedCircuit};
///
/// let hamiltonian = create_h2_hamiltonian();
/// let ansatz = ParameterizedCircuit::hardware_efficient(4, 2);
///
/// let vqe = VQE::builder()
///     .hamiltonian(hamiltonian)
///     .ansatz(ansatz)
///     .optimizer(Adam::default())
///     .build()?;
///
/// let result = vqe.optimize()?;
/// println!("Ground state energy: {:.6}", result.energy);
/// ```
///
/// # Algorithms
///
/// - **VQE**: Quantum chemistry, material science
/// - **QAOA**: Graph problems, combinatorial optimization
/// - **QNNs**: Classification, regression, pattern recognition
/// - **QGANs**: Quantum state generation, data augmentation
///
/// # Integration with SciRS2
///
/// - Uses `scirs2_autograd` for automatic differentiation
/// - Leverages `scirs2_optimize` and `optirs` for parameter optimization
/// - Employs `scirs2_neural` for hybrid quantum-classical models
///
/// # Related Modules
///
/// - `sim` - Execute quantum ML algorithms (requires `sim` feature)
/// - `anneal` - Use annealing for optimization problems (requires `anneal` feature)
#[cfg(feature = "ml")]
pub use quantrs2_ml as ml;

/// High-level quantum annealing DSL inspired by TYTAN.
///
/// This module provides a user-friendly domain-specific language for
/// quantum annealing and combinatorial optimization:
///
/// - **Problem Definition**: Intuitive API for QUBO/Ising formulation
/// - **Constraint Handling**: Automatic penalty method generation
/// - **Multiple Solvers**: Simulated annealing, GPU, quantum hardware
/// - **Visualization**: Problem structure and solution analysis
/// - **Benchmarking**: Performance comparison across solvers
///
/// Requires: `features = ["tytan"]` (automatically enables `anneal`)
///
/// # Examples
///
/// ```rust,ignore
/// use quantrs2::tytan::Qubo;
///
/// let mut qubo = Qubo::new();
/// let x = qubo.add_binary_variables(3);
///
/// // Maximize sum of variables with constraint x[0] + x[1] <= 1
/// qubo.add_objective(x[0] + x[1] + x[2]);
/// qubo.add_constraint(x[0] + x[1] <= 1);
///
/// let solution = qubo.solve()?;
/// ```
///
/// # Key Features
///
/// - **Auto Array**: Automatically manage variable arrays
/// - **Constraint DSL**: Express constraints naturally
/// - **Multiple Solvers**: Switch between CPU, GPU, quantum hardware
/// - **Visualization**: Generate problem graphs and solution heatmaps
///
/// # Use Cases
///
/// - **3-SAT Problems**: Boolean satisfiability
/// - **Graph Coloring**: Resource allocation
/// - **Job Scheduling**: Task assignment optimization
/// - **Portfolio Optimization**: Financial applications
///
/// # Related Modules
///
/// - [`anneal`] - Lower-level annealing primitives
/// - [`ml`] - Use with quantum ML algorithms
#[cfg(feature = "tytan")]
pub use quantrs2_tytan as tytan;

/// Symbolic quantum computation with SymEngine integration.
///
/// This module provides symbolic manipulation of quantum expressions:
///
/// - **Symbolic Expressions**: Represent gate parameters symbolically
/// - **Symbolic Differentiation**: Compute gradients analytically
/// - **Circuit Optimization**: Symbolic circuit simplification
/// - **Parameter Binding**: Late binding of numerical values
/// - **Algebraic Manipulation**: Simplify quantum expressions
///
/// Requires: `features = ["symengine"]`
///
/// # Examples
///
/// ```rust,ignore
/// use quantrs2::symengine::{Expr, symbols};
///
/// let theta = symbols("theta");
/// let expr = theta * 2.0;
///
/// // Differentiate symbolically
/// let grad = expr.diff(&theta);
/// assert_eq!(grad.to_string(), "2.0");
/// ```
///
/// # Key Types
///
/// - `Expr` - Symbolic expression
/// - `Symbol` - Symbolic variable
/// - `SymbolicCircuit` - Circuit with symbolic parameters
///
/// # Performance
///
/// - Symbolic operations are computed at compile-time when possible
/// - C++ SymEngine backend provides fast symbolic manipulation
/// - Particularly useful for variational algorithms with many parameters
///
/// # Related Modules
///
/// - [`circuit`] - Use symbolic parameters in circuits
/// - [`ml`] - Symbolic gradients for VQE/QAOA
#[cfg(feature = "symengine")]
pub use quantrs2_symengine_pure as symengine;

/// Hierarchical prelude modules for convenient imports.
///
/// This module provides multiple levels of prelude imports, allowing you to
/// choose the right level of functionality for your use case:
///
/// - `essentials` - Core types only (fastest compile)
/// - `circuits` - Circuit construction
/// - `simulation` - Quantum simulation
/// - `algorithms` - ML algorithms (VQE, QAOA)
/// - `hardware` - Real quantum hardware
/// - `quantum_annealing` - Annealing primitives
/// - `tytan` - TYTAN DSL
/// - `full` - Everything (slowest compile)
///
/// # Examples
///
/// ```rust
/// // Import only essentials (fast compile)
/// use quantrs2::prelude::essentials::*;
/// # let _ = QubitId::new(0);
/// ```
///
/// ```rust,ignore
/// // Import circuit + simulation (moderate compile)
/// use quantrs2::prelude::simulation::*;
/// ```
///
/// ```rust
/// // Import everything (slow compile, full functionality)
/// use quantrs2::prelude::full::*;
/// # let _ = QubitId::new(0);
/// ```
///
/// # Compilation Times
///
/// | Prelude Level | Compile Time | Best For |
/// |---------------|--------------|----------|
/// | `essentials` | ~2s | Basic types, library code |
/// | `circuits` | ~8s | Circuit construction |
/// | `simulation` | ~15s | Running simulations |
/// | `algorithms` | ~30s | ML algorithm development |
/// | `hardware` | ~10s | Hardware integration |
/// | `quantum_annealing` | ~12s | Annealing problems |
/// | `tytan` | ~15s | TYTAN DSL usage |
/// | `full` | ~45s | Complete applications |
///
/// # Recommendation
///
/// Start with `essentials` and progressively add features as needed
/// to maintain fast compile times during development.
pub mod prelude;

/// Unified error handling and categorization.
///
/// This module provides a comprehensive error handling system:
///
/// - **Error Categories**: Core, Circuit, Simulation, Hardware, etc.
/// - **Error Context**: Add contextual information to errors
/// - **User-Friendly Messages**: Clear error descriptions
/// - **Recoverability**: Classify errors as recoverable or fatal
/// - **Error Conversion**: Automatic conversion between subcrate errors
///
/// # Examples
///
/// ```rust,ignore
/// use quantrs2::error::{QuantRS2Error, QuantRS2ErrorExt, with_context};
///
/// fn operation() -> Result<(), QuantRS2Error> {
///     let err = QuantRS2Error::NetworkError("timeout".into());
///
///     if err.is_recoverable() {
///         // Retry logic
///     }
///
///     Err(with_context(err, "during quantum job submission"))
/// }
/// ```
///
/// # Error Categories
///
/// - `Core` - Fundamental quantum errors (invalid qubits, gates)
/// - `Circuit` - Circuit construction and validation errors
/// - `Simulation` - Simulator runtime errors
/// - `Hardware` - Device connection and execution errors
/// - `Algorithm` - ML algorithm convergence issues
/// - `Annealing` - Annealing solver errors
/// - `Symbolic` - SymEngine expression errors
/// - `Runtime` - General runtime errors
///
/// # Best Practices
///
/// - Use `?` operator for error propagation
/// - Add context with `with_context()` for better debugging
/// - Check `is_recoverable()` before retrying operations
/// - Use `user_message()` for user-facing error display
pub mod error;

/// Version information and compatibility checking.
///
/// This module provides version management and compatibility validation:
///
/// - **Version Constants**: Current QuantRS2 and SciRS2 versions
/// - **Build Information**: Git commit, build timestamp, compiler version
/// - **Compatibility Checking**: Validate subcrate versions
/// - **Deprecation Warnings**: Alert on outdated API usage
///
/// # Examples
///
/// ```rust
/// use quantrs2::version;
///
/// println!("QuantRS2 v{}", version::VERSION);
/// println!("SciRS2 v{}", version::SCIRS2_VERSION);
///
/// // Check compatibility
/// if let Err(issues) = version::check_compatibility() {
///     for issue in issues {
///         eprintln!("Warning: {}", issue);
///     }
/// }
/// ```
///
/// # Version Constants
///
/// - `VERSION` - QuantRS2 version (e.g., "0.1.2")
/// - `SCIRS2_VERSION` - SciRS2 dependency version
/// - `BUILD_TIMESTAMP` - When this build was created
/// - `GIT_COMMIT_HASH` - Git commit SHA
/// - `RUST_VERSION` - Compiler version used
///
/// # Compatibility
///
/// QuantRS2 follows semantic versioning. Compatibility is guaranteed
/// within the same minor version (e.g., 0.1.x → 0.1.y).
pub mod version;

/// Global configuration management.
///
/// This module provides centralized configuration for all QuantRS2 components:
///
/// - **Thread Pool**: Configure parallelism (default: num CPUs)
/// - **Memory Limits**: Set maximum memory usage
/// - **Backend Selection**: Choose default simulator/hardware
/// - **Logging**: Configure log levels and output
/// - **Hardware Acceleration**: Enable/disable GPU, SIMD
/// - **Environment Variables**: `QUANTRS2_*` variable support
///
/// # Examples
///
/// ```rust
/// use quantrs2::config::Config;
///
/// // Configure via builder pattern
/// Config::builder()
///     .num_threads(8)
///     .memory_limit_gb(32)
///     .enable_gpu(true)
///     .apply();
///
/// // Or configure directly
/// let cfg = Config::global();
/// cfg.set_log_level(quantrs2::config::LogLevel::Debug);
/// ```
///
/// # Environment Variables
///
/// - `QUANTRS2_NUM_THREADS` - Number of threads (default: CPU count)
/// - `QUANTRS2_LOG_LEVEL` - Logging level (trace|debug|info|warn|error)
/// - `QUANTRS2_MEMORY_LIMIT_GB` - Memory limit in GB
/// - `QUANTRS2_BACKEND` - Default backend (cpu|gpu|auto)
/// - `QUANTRS2_GPU_ENABLED` - Enable GPU (true|false)
///
/// # Thread Safety
///
/// Configuration is stored in a global singleton with interior mutability.
/// All operations are thread-safe.
pub mod config;

/// System diagnostics and health checks.
///
/// This module provides comprehensive system validation:
///
/// - **Hardware Detection**: CPU cores, memory, GPU availability
/// - **SIMD Capabilities**: AVX2, AVX-512, NEON detection
/// - **Feature Detection**: Enabled cargo features
/// - **Compatibility Checks**: Validate SciRS2 integration
/// - **Performance Profiling**: Benchmark system capabilities
/// - **Issue Reporting**: Generate diagnostic reports
///
/// # Examples
///
/// ```rust
/// use quantrs2::diagnostics;
///
/// // Run full diagnostics
/// let report = diagnostics::run_diagnostics();
/// if !report.is_ready() {
///     eprintln!("{}", report);
///     return;
/// }
///
/// // Quick checks
/// if diagnostics::is_ready() {
///     // System ready for quantum simulation
/// }
///
/// // Validate at startup (panics if not ready)
/// diagnostics::validate_or_panic();
/// ```
///
/// # Diagnostic Report
///
/// The diagnostic report includes:
/// - CPU: Model, cores, frequency
/// - Memory: Total RAM, available
/// - GPU: Detected GPUs (Metal/CUDA)
/// - SIMD: AVX2, AVX-512, NEON support
/// - Features: Enabled cargo features
/// - Compatibility: SciRS2 version check
///
/// # Performance
///
/// - Diagnostics are cached after first run
/// - Minimal overhead (~1ms)
/// - Can be disabled for production builds
pub mod diagnostics;

/// Utility functions for quantum computing.
///
/// This module provides cross-cutting utilities:
///
/// - **Memory Estimation**: Calculate memory requirements
/// - **Qubit Validation**: Check qubit count feasibility
/// - **Formatting**: Human-readable sizes and durations
/// - **Quantum Math**: Probability normalization, fidelity, entropy
/// - **Hilbert Space**: Dimension calculations
///
/// # Examples
///
/// ```rust
/// use quantrs2::utils;
///
/// // Estimate memory for 30 qubits
/// let mem = utils::estimate_statevector_memory(30);
/// println!("Memory needed: {}", utils::format_memory(mem));
///
/// // Find max qubits for 16 GB
/// let max = utils::max_qubits_for_memory(16 * 1024 * 1024 * 1024);
/// assert_eq!(max, 30);
///
/// // Validate configuration
/// # let available_memory = 32 * 1024 * 1024 * 1024; // 32 GB
/// if utils::is_valid_qubit_count(25, available_memory) {
///     // Can simulate 25 qubits
/// }
/// ```
///
/// # Quantum Math Utilities
///
/// - `is_normalized()` - Check probability vector normalization
/// - `normalize_probabilities()` - Normalize probability distribution
/// - `classical_fidelity()` - Compute fidelity between distributions
/// - `entropy()` - Shannon entropy of probability distribution
/// - `hilbert_dim()` - Hilbert space dimension for N qubits
/// - `num_qubits_from_dim()` - Infer qubit count from dimension
///
/// # Constants
///
/// - `SQRT_2`, `INV_SQRT_2` - Common quantum constants
/// - `PI_OVER_2`, `PI_OVER_4` - Rotation angles
pub mod utils;

/// Testing utilities for quantum algorithm development.
///
/// This module provides helpers for testing quantum code:
///
/// - **Floating-Point Assertions**: Compare with tolerance
/// - **Vector Assertions**: Element-wise comparison
/// - **Measurement Assertions**: Stochastic test helpers
/// - **Test Data Generation**: Reproducible random data
/// - **Temporary Directories**: Clean test environments
///
/// # Examples
///
/// ```rust
/// use quantrs2::testing;
///
/// // Floating-point comparison
/// testing::assert_approx_eq(1.0, 1.0000001, 1e-5);
///
/// // Vector comparison
/// # let expected = vec![1.0, 2.0, 3.0];
/// # let actual = vec![1.0, 2.0, 3.0];
/// testing::assert_vec_approx_eq(&expected, &actual, 1e-6);
///
/// // Measurement counts (stochastic)
/// # use std::collections::HashMap;
/// # let mut counts = HashMap::new();
/// # counts.insert("00".to_string(), 500);
/// # counts.insert("11".to_string(), 500);
/// # let mut expected = HashMap::new();
/// # expected.insert("00".to_string(), 500);
/// # expected.insert("11".to_string(), 500);
/// testing::assert_measurement_counts_close(&counts, &expected, 0.05);
///
/// // Reproducible test data
/// let data = testing::generate_random_test_data(100, testing::test_seed());
/// ```
///
/// # Test Seeds
///
/// Use `test_seed()` to get a reproducible seed for tests.
/// This ensures tests are deterministic but different across test runs.
///
/// # Temporary Directories
///
/// Use `temp_test_dir()` to create isolated test environments that
/// are automatically cleaned up.
pub mod testing;

/// Benchmarking utilities for performance measurement.
///
/// This module provides tools for benchmarking quantum algorithms:
///
/// - **Timers**: High-precision timing with `Timer`
/// - **Statistics**: Mean, stddev, percentiles
/// - **Throughput**: Operations per second
/// - **Comparison**: Compare algorithm variants
///
/// # Examples
///
/// ```rust
/// use quantrs2::bench::{BenchmarkTimer, BenchmarkStats};
/// use std::time::Duration;
///
/// let timer = BenchmarkTimer::start();
/// # std::thread::sleep(Duration::from_micros(1)); // Simulate work
/// let duration = timer.stop();
///
/// println!("Completed in {}", quantrs2::utils::format_duration(duration));
///
/// // Aggregate statistics
/// let mut stats = BenchmarkStats::new("test_operation");
/// for _ in 0..10 {
///     let timer = BenchmarkTimer::start();
///     # std::thread::sleep(Duration::from_micros(1)); // Simulate work
///     stats.record(timer.stop());
/// }
///
/// println!("Mean: {:?}, StdDev: {:?}", stats.mean(), stats.std_dev());
/// ```
///
/// # Throughput Measurement
///
/// ```rust,ignore
/// use quantrs2::bench;
///
/// let throughput = bench::measure_throughput(|| {
///     simulate_circuit(&circuit)
/// }, iterations=1000);
///
/// println!("Throughput: {:.2} circuits/sec", throughput);
/// ```
///
/// # Performance Tips
///
/// - Run benchmarks in `--release` mode
/// - Warm up before measurement
/// - Use sufficient iterations for statistical significance
/// - Report mean and standard deviation
pub mod bench;

/// Deprecation tracking and migration guidance.
///
/// This module provides deprecation management:
///
/// - **Deprecation Status**: Track API stability
/// - **Migration Paths**: Clear upgrade instructions
/// - **Removal Timelines**: When deprecated APIs will be removed
/// - **Stability Levels**: Experimental, Unstable, Stable
///
/// # Examples
///
/// ```rust
/// use quantrs2::deprecation;
///
/// // Check if a feature is deprecated
/// if deprecation::is_deprecated("old_api") {
///     if let Some(info) = deprecation::get_migration_info("old_api") {
///         println!("Deprecated: {}", info.reason);
///         if let Some(alt) = &info.alternative {
///             println!("Use instead: {}", alt);
///         }
///     }
/// }
///
/// // Get module stability
/// let stability = deprecation::get_module_stability("circuit");
/// println!("Circuit module stability: {:?}", stability);
///
/// // Generate migration report
/// let report = deprecation::migration_report();
/// println!("{}", report);
/// ```
///
/// # Stability Levels
///
/// - `Experimental` - May change without notice
/// - `Unstable` - Subject to breaking changes in minor versions
/// - `Stable` - Follows semantic versioning guarantees
///
/// # Deprecation Process
///
/// 1. `PendingDeprecation` - Will be deprecated in next minor version
/// 2. `Deprecated` - Deprecated, but still functional
/// 3. `Removed` - Removed from API (breaking change)
pub mod deprecation;

// Version information constants
pub use version::{QUANTRS2_VERSION, VERSION};

// ================================================================================================
// Unit Tests for Feature Gates
// ================================================================================================

#[cfg(test)]
mod feature_gate_tests;
