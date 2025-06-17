# Release Notes for QuantRS2 v0.1.0-alpha.5

## Overview

QuantRS2 v0.1.0-alpha.5 is the final alpha release before the production release. This version includes comprehensive quantum computing capabilities with enhanced performance, stability, and feature completeness.

## Major Highlights

### 🚀 SciRS2 Integration
- **Performance Boost**: SIMD-accelerated quantum operations provide 2-5x speedup for common operations
- **Memory Efficiency**: Automatic chunking and buffer pools for simulating 30+ qubit quantum states
- **Enhanced Linear Algebra**: Leveraging BLAS/LAPACK through SciRS2 for optimized matrix operations
- **Thread-Safe Operations**: Improved parallelization with thread-safe buffer management

### 🧠 Advanced Quantum Algorithms
- **Quantum Approximate Optimization Algorithm (QAOA)**: Complete implementation for solving combinatorial optimization problems
- **Quantum Principal Component Analysis (QPCA)**: Full implementation with density matrix preparation and eigenvalue extraction
- **HHL Algorithm**: Harrow-Hassidim-Lloyd algorithm for solving linear systems with quantum speedup
- **Quantum Walk Algorithms**: Both discrete-time and continuous-time quantum walks for graph problems
- **Quantum Counting**: Amplitude estimation and counting for quantum search problems

### 🔧 Circuit Optimization
- **Graph-Based Optimizer**: Circuit representation as directed graphs with advanced optimization passes
- **Gate Fusion**: Automatic fusion of compatible gates for reduced circuit depth
- **Hardware-Aware Compilation**: Optimization considering hardware connectivity constraints
- **Peephole Optimization**: Pattern matching for common gate sequences

### 🖥️ Enhanced Simulation Capabilities
- **Stabilizer Simulator**: Efficient O(n²) simulation of Clifford circuits, ideal for error correction
- **Sparse Clifford Simulator**: Memory-efficient simulation of large Clifford circuits (100+ qubits)
- **GPU Linear Algebra**: Foundation for GPU-accelerated quantum operations
- **Hardware Topology Analysis**: Graph-based analysis of quantum hardware connectivity

### 🤖 Quantum Machine Learning
- **Quantum Support Vector Machines (QSVM)**: Full implementation with multiple kernel types
- **Quantum Convolutional Neural Networks (QCNN)**: Quantum filters and pooling layers
- **Quantum Variational Autoencoders (QVAE)**: Data compression and feature extraction
- **Enhanced Quantum GANs**: Improved generators and discriminators with Wasserstein loss
- **Barren Plateau Detection**: Analysis and mitigation strategies for vanishing gradients
- **Quantum Reinforcement Learning**: Policy gradient methods and Q-learning

### 🔌 Hardware Integration
- **Hardware Topology Mapping**: Optimal qubit subset selection for circuit compilation
- **Qubit Routing**: Algorithms for mapping logical to physical qubits
- **Support for Real Topologies**: IBM Heavy-Hex and Google Sycamore architectures
- **Critical Qubit Analysis**: Identification of essential connections in hardware

## Breaking Changes
None in this release - fully backward compatible with alpha.2

## Installation

```toml
[dependencies]
quantrs2-core = "0.1.0-alpha.3"
quantrs2-circuit = "0.1.0-alpha.3"
quantrs2-sim = "0.1.0-alpha.3"
quantrs2-ml = "0.1.0-alpha.3"
quantrs2-device = "0.1.0-alpha.3"
```

## What's Next
- Continued optimization of GPU acceleration
- Enhanced distributed quantum simulation
- More quantum algorithms and applications
- Improved hardware backend support

---

# Release Notes for QuantRS2 v0.1.0-alpha.2

We're excited to announce the second alpha release of QuantRS2, which transforms our Rust-based quantum computing framework into a comprehensive, state-of-the-art quantum computing ecosystem. This release significantly extends the capabilities of the framework, making it one of the most advanced quantum computing platforms available.

With over 30 major enhancements and innovative features, QuantRS2 now offers support for all major quantum computing paradigms, advanced simulation capabilities, comprehensive error correction, quantum machine learning, and much more. This release positions QuantRS2 as a cutting-edge platform for both current NISQ-era devices and future fault-tolerant quantum computers.

## Key Highlights

### Dynamic Qubit Count Support

One of the major improvements in this release is the new `DynamicCircuit` abstraction, which enables variable qubit count support without template specialization. This makes the framework more flexible and easier to use, especially from languages like Python.

```rust
// Create a dynamic circuit with 5 qubits
let mut circuit = DynamicCircuit::new(5)?;

// Circuit operations remain the same
circuit.h(0)?
       .cnot(0, 1)?;

// Use with any simulator
let simulator = StateVectorSimulator::new();
let result = circuit.run(&simulator)?;
```

In Python, this allows for a much more natural interface:

```python
# Create a circuit with any number of qubits
circuit = Circuit(10)
circuit.h(0)
circuit.cnot(0, 1)

# Run simulation
result = circuit.run()
```

### Realistic Noise Models

This release introduces advanced noise models that accurately simulate real quantum hardware, including:

- Two-qubit depolarizing noise channels for modeling cross-qubit errors
- Thermal relaxation (T1/T2) simulation for realistic decoherence effects
- Crosstalk noise between adjacent qubits
- Device-specific noise profiles for IBM and Rigetti quantum processors

These models allow for more accurate simulation of how algorithms would perform on real quantum hardware.

```rust
// Create a realistic IBM device noise model
let qubits: Vec<QubitId> = (0..5).map(|i| QubitId::new(i)).collect();
let noise_model = RealisticNoiseModelBuilder::new(true)
    .with_ibm_device_noise(&qubits, "ibmq_lima")
    .build();

// Create a noisy simulator
let mut simulator = StateVectorSimulator::new();
simulator.set_advanced_noise_model(noise_model);

// Run with noise
let result = circuit.run(&simulator)?;
```

In Python:

```python
# Create IBM device noise model
ibm_noise = RealisticNoiseModel.ibm_device("ibmq_lima")

# Run with noise
result = circuit.simulate_with_noise(ibm_noise)
```

### Enhanced GPU Acceleration

We've significantly improved GPU-based quantum simulation with:

- Optimized WGPU shaders for single and two-qubit gate operations
- Automatic device detection and capability checks
- Better memory management for large circuit simulation
- Automatic fallback to CPU for devices without GPU support

This enables efficient simulation of larger quantum circuits (20+ qubits) with significant performance improvements.

### Cloud Connectivity

Improved quantum hardware connectivity allows for better integration with real quantum devices:

- Added AWS Braket authentication (Signature V4) for secure API access
- Implemented proper AWS Braket IR format conversion from QuantRS2 circuits
- Enhanced IBM Quantum integration with better error handling
- Added Azure Quantum integration for circuit submission
- Integrated IonQ and Honeywell quantum hardware support

### Parametric Gates

This release adds support for symbolic parameters in quantum gates, enabling more flexible circuit creation and optimization:

- Implemented `Parameter` and `SymbolicParameter` traits for variable gate parameters
- Created specialized parametric implementations of rotation gates (Rx, Ry, Rz)
- Added parameter binding and gate transformation methods

```rust
// Define symbolic parameters
let theta = SymbolicParameter::new("theta");
let phi = SymbolicParameter::new("phi");

// Create a parameterized circuit
let mut circuit = Circuit::<2>::new();
circuit.h(0)?
       .rz(0, theta)?
       .rx(1, phi)?
       .cnot(0, 1)?;

// Bind parameters to specific values at runtime
let bindings = ParameterBindings::new()
    .bind("theta", 0.5)
    .bind("phi", 1.2);

let bound_circuit = circuit.bind_parameters(&bindings)?;
let result = simulator.run(&bound_circuit)?;
```

In Python:

```python
# Create a parameterized circuit
theta = Parameter("theta")
phi = Parameter("phi")

circuit = Circuit(2)
circuit.h(0)
circuit.rz(0, theta)
circuit.rx(1, phi)
circuit.cnot(0, 1)

# Bind parameters and run
result = circuit.run(parameters={"theta": 0.5, "phi": 1.2})
```

### Gate Composition and Decomposition

New capabilities for transforming complex gates into simpler ones and vice versa:

- Implemented `GateDecomposable` and `GateComposable` traits
- Added decomposition algorithms for complex gates (Toffoli, SWAP, controlled-rotations)
- Created utility functions for optimizing gate sequences

```rust
// Create a circuit with complex gates
let mut circuit = Circuit::<3>::new();
circuit.toffoli(0, 1, 2)?;

// Decompose complex gates into basic gates
let decomposed = circuit.decompose_to_basic_gates()?;

// Optimize the gate sequence
let optimized = decomposed.optimize()?;
```

In Python:

```python
# Create a circuit with a Toffoli gate
circuit = Circuit(3)
circuit.toffoli(0, 1, 2)

# Decompose to basic gates
basic_circuit = circuit.decompose_to_basic_gates()

# Optimize the circuit
optimized = basic_circuit.optimize()
```

### Tensor Network Contraction Optimization

Significantly improved tensor network simulation with optimized contraction paths:

- Created `PathOptimizer` with multiple optimization strategies
- Implemented specialized optimizations for different circuit topologies
- Added hybrid approach that selects best strategy based on circuit characteristics
- Implemented approximate tensor network simulation for very large systems

```rust
// Create a tensor network simulator with optimized contraction
let simulator = TensorNetworkSimulator::new()
    .with_contraction_optimizer(PathOptimizer::new(OptimizationStrategy::Hybrid));

// Run with optimized contraction
let result = simulator.run(&circuit)?;
```

### Circuit Visualization for Python

New visualization capabilities for Python users:

- ASCII text representation for terminal display
- HTML visualization for Jupyter notebooks
- Customizable gate styles and layout options
- Interactive circuit designer for Python/Jupyter environments

```python
# Create a circuit
circuit = Circuit(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.toffoli(0, 1, 2)

# Get ASCII text representation
print(circuit.draw())

# In Jupyter notebooks, HTML visualization is automatic
circuit  # displays an HTML circuit diagram
```

### Quantum Machine Learning

Added extensive support for quantum machine learning applications through the new `quantrs2-ml` crate, covering a wide range of use cases from financial modeling to cybersecurity:

- **Quantum Neural Networks**: Implemented parameterized quantum circuits with customizable architectures and training methods
- **Variational Quantum Algorithms**: Created hybrid quantum-classical optimization routines for QAOA and VQE
- **ML Applications**:
  - **High-Energy Physics**: Particle collision classification and feature extraction for HEP data analysis
  - **Generative Models**: Hybrid quantum-classical GANs for synthetic data generation
  - **Cybersecurity**: Quantum anomaly detection for intrusion detection and threat analysis
  - **Natural Language Processing**: Quantum language models for text classification and sentiment analysis
  - **Blockchain Technology**: Quantum-secure distributed ledgers and smart contracts
  - **Cryptography**: Post-quantum cryptographic protocols extending beyond BB84

```rust
// Quantum Neural Network for binary classification example
use quantrs2_ml::qnn::{QuantumNeuralNetwork, QNNLayer, Optimizer};

// Create a QNN with a multi-layer architecture
let layers = vec![
    QNNLayer::EncodingLayer { num_features: 4 },
    QNNLayer::VariationalLayer { num_params: 18 },
    QNNLayer::EntanglementLayer { connectivity: "full".to_string() },
    QNNLayer::VariationalLayer { num_params: 18 },
    QNNLayer::MeasurementLayer { measurement_basis: "computational".to_string() },
];

let mut qnn = QuantumNeuralNetwork::new(
    layers, 
    6,     // 6 qubits
    4,     // 4 input features
    2,     // 2 output classes
)?;

// Configure optimizer (supports Adam, SGD, SPSA, etc.)
let optimizer = Optimizer::Adam { learning_rate: 0.01 };

// Train on data
let training_result = qnn.train(&x_train, &y_train, optimizer, 100)?;
println!("Final loss: {:.6}", training_result.final_loss);

// Evaluate on test data
let metrics = qnn.evaluate(&x_test, &y_test)?;
println!("Test accuracy: {:.2}%", metrics.accuracy * 100.0);
```

The ML module includes specialized submodules for different domains:

```rust
// For high-energy physics analysis
use quantrs2_ml::hep::{HEPQuantumClassifier, HEPEncodingMethod};

// For quantum GANs
use quantrs2_ml::gan::{QuantumGAN, QuantumGenerator, QuantumDiscriminator};

// For quantum-enhanced cryptography
use quantrs2_ml::crypto::{QuantumKeyDistribution, ProtocolType};

// For quantum blockchain applications
use quantrs2_ml::blockchain::{QuantumBlockchain, ConsensusType};

// For quantum NLP applications
use quantrs2_ml::nlp::{QuantumLanguageModel, NLPTaskType};
```

### Fermionic Simulation

Added support for quantum chemistry applications:

- Implemented Jordan-Wigner and Bravyi-Kitaev transformations
- Added molecular Hamiltonian construction utilities
- Created VQE (Variational Quantum Eigensolver) implementation
- Added tools for electronic structure calculations
- Integrated with classical chemistry libraries for pre-processing

### Distributed Quantum Simulation

Enhanced scalability for large circuits:

- Implemented multi-node distribution for statevector simulation
- Added memory-efficient partitioning for large quantum states
- Created checkpoint mechanisms for long-running simulations
- Added automatic workload balancing across computing resources
- Provided GPU cluster support for massive parallelization

### Performance Benchmarking

Tools for quantum algorithm assessment:

- Implemented benchmark suites for standard quantum algorithms
- Added profiling tools for execution time and resource usage
- Created comparison utilities for different simulation backends
- Added visualization for performance metrics
- Implemented quantum volume and cycle benchmarking methods

### Advanced Error Correction

Comprehensive fault-tolerant computing support:

- Implemented surface code with arbitrary code distance
- Added real-time syndrome measurement and correction
- Created decoding algorithms including minimum-weight perfect matching
- Added fault-tolerant logical gate implementations
- Implemented magic state distillation protocols

### Quantum Cryptography

Security protocols for quantum networks:

- Implemented BB84 and E91 quantum key distribution
- Added quantum coin flipping and secret sharing
- Created quantum digital signatures
- Implemented quantum key recycling and authentication

### NISQ Optimization

Tools for near-term quantum devices:

- Created hardware-specific circuit optimizers for various QPUs
- Implemented noise-aware compilation strategies
- Added measurement error mitigation techniques
- Created zero-noise extrapolation and probabilistic error cancelation

### Quantum Development Tools

Advanced development environment:

- Implemented quantum algorithm design assistant with AI guidance
- Added quantum circuit verifier for logical correctness
- Created custom quantum intermediate representation (QIR)
- Added QASM 3.0 import/export support

## New Examples

This release includes several new examples demonstrating the advanced features:

- `quantum_phase_estimation.rs`: Shows QPE algorithm with noise analysis
- `grovers_algorithm_noisy.rs`: Demonstrates noise impact on algorithm performance
- AWS device integration examples showing circuit execution on AWS Braket
- `realistic_noise_example.py`: Python example showing device-specific noise simulation
- `parametric_circuit.rs`: Demonstrates symbolic parameter use and runtime binding
- `gate_decomposition.py`: Shows complex gate transformation to basic gates
- `circuit_visualization.ipynb`: Jupyter notebook with interactive circuit visualization
- `tensor_network_optimization.rs`: Benchmarks different contraction strategies
- `quantum_chemistry.rs`: Demonstrates molecular simulations
- `quantum_machine_learning.py`: Shows QCNN (Quantum Convolutional Neural Network) implementation
- `distributed_simulation.rs`: Demonstrates large-scale distributed quantum simulation
- `algorithm_benchmarks.ipynb`: Performance benchmarking for different algorithms
- `surface_code_ft.rs`: Surface code implementation with logical operations
- `quantum_reinforcement_learning.py`: Q-learning with quantum circuits
- `error_mitigation_demo.rs`: Zero-noise extrapolation techniques
- `quantum_key_distribution.rs`: BB84 protocol implementation
- `nisq_compilation.rs`: Hardware-aware circuit optimization
- `quantum_assistant_demo.py`: AI-assisted quantum algorithm design
- Comprehensive examples beyond basic Bell states

## Getting Started with This Release

```toml
# Cargo.toml
[dependencies]
quantrs2-core = "0.1.0-alpha.2"
quantrs2-circuit = "0.1.0-alpha.2"
quantrs2-sim = "0.1.0-alpha.2"
quantrs2-device = { version = "0.1.0-alpha.2", features = ["ibm", "aws"] }
```

Or for Python users:

```bash
pip install quantrs2==0.1.0a2
```

## Trying the New Features

### Advanced Error Correction

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::error_correction::{SurfaceCode, LogicalOperation, Decoder, DecodingStrategy};
use quantrs2_sim::noise_advanced::RealisticNoiseModelBuilder;
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 5x5 surface code (distance d=5)
    let mut surface_code = SurfaceCode::new(5);
    
    // Encode a logical qubit state |0⟩
    surface_code.initialize_logical_zero()?;
    
    // Apply logical X operation
    surface_code.apply_logical_operation(LogicalOperation::X)?;
    
    // Create a physical noise model
    let noise_model = RealisticNoiseModelBuilder::new(true)
        .with_depolarizing_probability(0.005) // 0.5% error rate
        .build();
    
    // Create simulator with noise
    let mut simulator = StateVectorSimulator::new();
    simulator.set_advanced_noise_model(noise_model);
    
    // Run multiple syndrome extraction and correction cycles
    for _ in 0..10 {
        // Extract error syndromes
        let syndromes = surface_code.extract_syndromes(&simulator)?;
        
        // Decode the syndromes
        let decoder = Decoder::MinimumWeightPerfectMatching;
        let correction_ops = decoder.decode(&syndromes)?;
        
        // Apply corrections
        surface_code.apply_corrections(&correction_ops)?;
        
        // Print logical state fidelity
        let logical_state = surface_code.measure_logical_state(&simulator)?;
        println!("Logical |1⟩ state fidelity: {:.6}", logical_state.fidelity_to_one());
    }
    
    // Apply fault-tolerant logical Hadamard
    surface_code.apply_logical_operation(LogicalOperation::H)?;
    
    // Apply logical T gate via magic state injection
    surface_code.apply_logical_operation_via_injection(LogicalOperation::T)?;
    
    // Demonstrate full quantum error correction code
    let mut circuit = Circuit::<2>::new();
    circuit.h(0)?.cnot(0, 1)?;
    
    // Encode each qubit into a logical qubit using surface code
    let logical_circuit = surface_code.encode_circuit(circuit)?;
    
    // Run encoded circuit with fault tolerance
    let result = surface_code.run_fault_tolerant(
        &logical_circuit,
        &simulator,
        DecodingStrategy::FastUnion,
        5 // Number of error correction cycles
    )?;
    
    println!("\nFinal logical state probabilities:");
    for (i, prob) in result.logical_probabilities().iter().enumerate() {
        println!("Logical state |{:b}⟩: {:.6}", i, prob);
    }
    
    Ok(())
}
```

### Quantum Cryptography

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::cryptography::{BB84Protocol, QuantumKeyDistribution, PrivacyAmplification};
use quantrs2_sim::statevector::StateVectorSimulator;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a BB84 quantum key distribution protocol
    let mut bb84 = BB84Protocol::new()
        .with_key_length(1024)      // Generate a 1024-bit key
        .with_error_rate(0.05)      // Simulated channel noise
        .with_eavesdropping(false); // No eavesdropper
    
    // Create simulator
    let simulator = StateVectorSimulator::new();
    
    // Alice prepares qubits in random bases and bit values
    let (alice_bits, alice_bases) = bb84.alice_prepare()?;
    
    // Simulate sending qubits through a quantum channel
    let quantum_states = bb84.quantum_transmission(&simulator)?;
    
    // Bob measures in random bases
    let (bob_bits, bob_bases) = bb84.bob_measure(quantum_states, &simulator)?;
    
    // Alice and Bob compare bases (public discussion)
    let shared_key = bb84.key_sifting(alice_bits, alice_bases, bob_bits, bob_bases)?;
    
    // Error estimation and detection
    let error_rate = bb84.estimate_error_rate()?;
    println!("Estimated error rate: {:.4}%", error_rate * 100.0);
    
    // Check if channel is secure
    if bb84.is_channel_secure(error_rate)? {
        println!("Channel is secure, proceeding with key generation");
        
        // Information reconciliation to correct errors
        let corrected_key = bb84.information_reconciliation()?;
        
        // Privacy amplification to remove any leaked information
        let final_key = bb84.privacy_amplification(corrected_key)?;
        
        // Use the final key (show only a sample)
        println!("\nGenerated secure key (first 32 bits): {:032b}", final_key[0]);
        
        // Example use of key for encryption
        let message = "Hello, quantum cryptography!";
        let encrypted = bb84.encrypt(message, &final_key)?;
        let decrypted = bb84.decrypt(&encrypted, &final_key)?;
        
        println!("\nMessage: {}", message);
        println!("Encrypted: {:?}", encrypted);
        println!("Decrypted: {}", decrypted);
    } else {
        println!("Possible eavesdropping detected, aborting key generation!");
    }
    
    Ok(())
}```

### Dynamic Qubit Count

```rust
use quantrs2_sim::dynamic::DynamicCircuit;
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a dynamic circuit with 10 qubits
    let mut circuit = DynamicCircuit::new(10)?;
    
    // Create superposition on all qubits
    for i in 0..10 {
        circuit.h(i)?;
    }
    
    // Run simulation (picks best backend automatically)
    let result = circuit.run_best()?;
    
    // Each state should have equal probability (1/1024)
    for (i, prob) in result.probabilities().iter().enumerate().take(5) {
        println!("State |{:010b}⟩: {:.10}", i, prob);
    }
    println!("...");
    
    Ok(())
}
```

### Realistic Noise Simulation

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::noise_advanced::RealisticNoiseModelBuilder;
use quantrs2_sim::statevector::StateVectorSimulator;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a Bell state circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(0)?.cnot(0, 1)?;
    
    // Create IBM device noise model
    let qubits = vec![QubitId::new(0), QubitId::new(1)];
    let noise_model = RealisticNoiseModelBuilder::new(true)
        .with_ibm_device_noise(&qubits, "ibmq_lima")
        .build();
    
    // Create noisy simulator
    let mut simulator = StateVectorSimulator::new();
    simulator.set_advanced_noise_model(noise_model);
    
    // Run with noise
    let result = simulator.run(&circuit)?;
    
    // Print results (will show deviation from perfect Bell state)
    for (i, prob) in result.probabilities().iter().enumerate() {
        println!("State |{:02b}⟩: {:.6}", i, prob);
    }
    
    Ok(())
}
```

### Parametric Gates

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::parametric::{SymbolicParameter, ParameterBindings};
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define symbolic parameters
    let theta = SymbolicParameter::new("theta");
    let phi = SymbolicParameter::new("phi");
    
    // Create parameterized circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(0)?
           .rz(0, theta)?
           .rx(1, phi)?
           .cnot(0, 1)?;
    
    // Create different parameter bindings
    let bindings1 = ParameterBindings::new()
        .bind("theta", 0.0)
        .bind("phi", 0.0);
        
    let bindings2 = ParameterBindings::new()
        .bind("theta", std::f64::consts::PI / 2.0)
        .bind("phi", std::f64::consts::PI / 4.0);
    
    // Run with different parameter values
    let simulator = StateVectorSimulator::new();
    
    let bound_circuit1 = circuit.bind_parameters(&bindings1)?;
    let result1 = simulator.run(&bound_circuit1)?;
    
    let bound_circuit2 = circuit.bind_parameters(&bindings2)?;
    let result2 = simulator.run(&bound_circuit2)?;
    
    // Compare results with different parameter values
    println!("Results with theta=0, phi=0:");
    for (i, prob) in result1.probabilities().iter().enumerate() {
        println!("State |{:02b}⟩: {:.6}", i, prob);
    }
    
    println!("\nResults with theta=π/2, phi=π/4:");
    for (i, prob) in result2.probabilities().iter().enumerate() {
        println!("State |{:02b}⟩: {:.6}", i, prob);
    }
    
    Ok(())
}
```

### Gate Decomposition

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_circuit::decomposition::GateDecomposable;
use quantrs2_sim::statevector::StateVectorSimulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a circuit with a Toffoli gate
    let mut original = Circuit::<3>::new();
    original.h(0)?
            .x(1)?
            .toffoli(0, 1, 2)?;
    
    // Decompose to basic gates
    let decomposed = original.decompose_to_basic_gates()?;
    
    // Verify equivalence
    let simulator = StateVectorSimulator::new();
    let result_original = simulator.run(&original)?;
    let result_decomposed = simulator.run(&decomposed)?;
    
    // Compare state vectors
    println!("Original circuit gate count: {}", original.gate_count());
    println!("Decomposed circuit gate count: {}", decomposed.gate_count());
    
    let difference = result_original.state_vector().distance(result_decomposed.state_vector());
    println!("State vector difference: {:.10e}", difference);
    
    Ok(())
}
```

### Tensor Network Optimization

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::tensor_network::{TensorNetworkSimulator, OptimizationStrategy};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a quantum Fourier transform circuit on 10 qubits
    let mut qft_circuit = Circuit::<10>::new();
    
    // Add QFT gates
    for i in 0..10 {
        qft_circuit.h(i)?;
        for j in (i+1)..10 {
            let angle = std::f64::consts::PI / (1 << (j-i));
            qft_circuit.cphase(i, j, angle)?;
        }
    }
    
    // Compare different optimization strategies
    let strategies = vec![
        ("None", OptimizationStrategy::None),
        ("Greedy", OptimizationStrategy::Greedy),
        ("DynamicProgramming", OptimizationStrategy::DynamicProgramming),
        ("Hybrid", OptimizationStrategy::Hybrid),
        ("Approximate", OptimizationStrategy::Approximate(0.001))
    ];
    
    for (name, strategy) in strategies {
        let simulator = TensorNetworkSimulator::new()
            .with_optimization_strategy(strategy);
        
        let start = Instant::now();
        let _ = simulator.run(&qft_circuit)?;
        let duration = start.elapsed();
        
        println!("{} strategy: {:?}", name, duration);
    }
    
    Ok(())
}
```

### Quantum Machine Learning

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::parametric::{SymbolicParameter, ParameterBindings};
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::ml::{QuantumNeuralNetwork, Optimizer, QuantumReinforcementLearning};
use quantrs2_sim::ml::environments::GridWorldEnvironment;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a parameterized circuit for QML
    let params = (0..6).map(|i| SymbolicParameter::new(format!("p{}", i))).collect::<Vec<_>>();
    
    // Build a 4-qubit Quantum Neural Network
    let mut circuit = Circuit::<4>::new();
    
    // Input layer
    for i in 0..4 {
        circuit.ry(i, params[i % params.len()])?;
    }
    
    // Entanglement layer
    for i in 0..3 {
        circuit.cnot(i, i+1)?;
    }
    circuit.cnot(3, 0)?;
    
    // Final rotation layer
    for i in 0..4 {
        let j = i + params.len() / 2;
        circuit.ry(i, params[j % params.len()])?;
    }
    
    // Create neural network
    let qnn = QuantumNeuralNetwork::new(circuit, (0..2).collect());
    
    // Define input data and expected outputs (XOR function)
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    // Train the model
    let optimizer = Optimizer::Adam { learning_rate: 0.01 };
    let trained_params = qnn.train(&training_data, optimizer, 100)?;
    
    // Test the trained model
    for (input, expected) in training_data {
        let output = qnn.predict(&input, &trained_params)?;
        println!("Input: {:?}, Predicted: {:.4}, Expected: {}", 
                 input, output[0], expected[0]);
    }
    
    // Quantum Reinforcement Learning Example
    println!("\n\nQuantum Reinforcement Learning Example\n");
    
    // Define a simple grid world environment (4x4)
    let grid_size = 4;
    let env = GridWorldEnvironment::new(grid_size, grid_size)
        .with_goal(3, 3)            // Goal in bottom-right corner
        .with_obstacles(vec![(1, 1), (2, 1), (1, 2)]);
    
    // Create a quantum reinforcement learning agent
    let qrl = QuantumReinforcementLearning::new()
        .with_state_dimension(2 * grid_size) // Encoding position
        .with_action_dimension(4)            // Up, Right, Down, Left
        .with_learning_rate(0.01)
        .with_discount_factor(0.95)
        .with_exploration_rate(0.1);
    
    // Train the agent
    println!("Training quantum reinforcement learning agent...");
    let trained_agent = qrl.train(&env, 1000)?;
    
    // Test the trained agent
    let total_reward = trained_agent.evaluate(&env, 100)?;
    println!("Average reward after training: {:.2}", total_reward / 100.0);
    
    // Show learned policy
    println!("\nLearned policy:");
    for y in 0..grid_size {
        for x in 0..grid_size {
            // Check if this is an obstacle
            if env.is_obstacle(x, y) {
                print!("🧱 ");
                continue;
            }
            
            // Check if this is the goal
            if env.is_goal(x, y) {
                print!("🎯 ");
                continue;
            }
            
            // Get the learned action
            let state = vec![x as f64 / grid_size as f64, y as f64 / grid_size as f64];
            let action = trained_agent.get_action(&state)?;
            let action_symbol = match action {
                0 => "⬆️",
                1 => "➡️",
                2 => "⬇️",
                3 => "⬅️",
                _ => "?",
            };
            print!("{} ", action_symbol);
        }
        println!();
    }
    
    Ok(())
}
```

### Quantum Chemistry

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::chemistry::{Molecule, ChemistryBuilder, FermionicTransform};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define a simple hydrogen molecule
    let h2 = Molecule::new()
        .add_atom("H", [0.0, 0.0, 0.0])
        .add_atom("H", [0.0, 0.0, 0.735]);
    
    // Build the molecular Hamiltonian
    let builder = ChemistryBuilder::new()
        .with_basis("sto-3g")
        .with_transformation(FermionicTransform::JordanWigner);
    
    // Generate a VQE circuit
    let (hamiltonian, vqe_circuit) = builder.build_vqe_for_molecule(&h2)?;
    
    // Create a simulator
    let simulator = StateVectorSimulator::new();
    
    // Run the VQE optimization
    let (optimal_params, ground_state_energy) = vqe_circuit.optimize(&hamiltonian, &simulator)?;
    
    println!("H2 ground state energy: {:.8} Hartree", ground_state_energy);
    println!("Reference energy: -1.13727 Hartree");
    println!("Optimal parameters: {:?}", optimal_params);
    
    Ok(())
}
```

### Distributed Quantum Simulation

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::distributed::{DistributedSimulator, NodeConfig, ClusterType};
use quantrs2_sim::distributed::{StatePartitioning, CommunicationStrategy};
use rayon::prelude::*;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a large quantum circuit (40 qubits)
    let mut large_circuit = Circuit::<40>::new();
    
    // Apply Hadamard to all qubits
    for i in 0..40 {
        large_circuit.h(i)?;
    }
    
    // Add entanglement layer
    for i in 0..39 {
        large_circuit.cnot(i, i+1)?;
    }
    
    // Configure distributed simulation with GPU cluster
    let node_configs = vec![
        NodeConfig::new("local", 8)
            .with_cluster_type(ClusterType::GPU), // Local GPU cluster
        NodeConfig::new("192.168.1.10:8000", 16)
            .with_cluster_type(ClusterType::GPU),
        NodeConfig::new("192.168.1.11:8000", 16)
            .with_cluster_type(ClusterType::GPU),
    ];
    
    // Create distributed simulator with advanced options
    let mut simulator = DistributedSimulator::new(node_configs)
        .with_checkpoint_interval(Some(Duration::from_secs(300)))
        .with_state_partitioning(StatePartitioning::Optimized)
        .with_communication_strategy(CommunicationStrategy::Adaptive)
        .with_fault_tolerance(true);
    
    // Run the simulation
    let result = simulator.run(&large_circuit)?;
    
    // Get results for a subset of states
    let states_of_interest = vec![0, 1, (1 << 20), (1 << 39)]; // Specific states to check
    for &state in &states_of_interest {
        let amplitude = result.amplitude(state)?;
        println!("State |{:040b}⟩: amplitude = {:.6}", state, amplitude.norm());
    }
    
    Ok(())
}
```

### NISQ Optimization

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::nisq::{ErrorMitigator, MitigationStrategy, HardwareOptimizer, DeviceProperties};
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_device::ibm::IBMQDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a circuit
    let mut circuit = Circuit::<5>::new();
    
    // Add some gates
    circuit.h(0)?
           .cnot(0, 1)?
           .cnot(1, 2)?
           .cnot(2, 3)?
           .cnot(3, 4)?;
    
    // Get IBM device properties
    let device = IBMQDevice::new("ibmq_lima")?;
    let device_props = device.get_properties()?;
    
    // Create hardware-aware optimizer
    let optimizer = HardwareOptimizer::new(&device_props)
        .with_noise_awareness(true)
        .with_gate_fidelity_weighting(true);
    
    // Optimize the circuit for the specific hardware
    let optimized_circuit = optimizer.optimize(circuit)?;
    println!("Original gate count: {}", circuit.gate_count());
    println!("Optimized gate count: {}", optimized_circuit.gate_count());
    
    // Create error mitigator
    let mitigator = ErrorMitigator::new(&device_props)
        .with_strategy(MitigationStrategy::ZeroNoiseExtrapolation)
        .with_measurement_error_correction(true);
    
    // Create noisy simulator based on device properties
    let mut simulator = StateVectorSimulator::new();
    simulator.set_noise_model_from_device(&device_props)?;
    
    // Run the circuits
    let unmitigated_result = simulator.run(&optimized_circuit)?;
    let mitigated_result = mitigator.apply(&optimized_circuit, &simulator)?;
    
    // Compare results
    println!("\nUnmitigated measurement probabilities:");
    for (i, prob) in unmitigated_result.probabilities().iter().enumerate().take(4) {
        println!("State |{:05b}⟩: {:.6}", i, prob);
    }
    
    println!("\nMitigated measurement probabilities:");
    for (i, prob) in mitigated_result.probabilities().iter().enumerate().take(4) {
        println!("State |{:05b}⟩: {:.6}", i, prob);
    }
    
    // Calculate the improvement
    let ideal_simulator = StateVectorSimulator::new();
    let ideal_result = ideal_simulator.run(&optimized_circuit)?;
    
    let unmitigated_fidelity = unmitigated_result.fidelity(&ideal_result);
    let mitigated_fidelity = mitigated_result.fidelity(&ideal_result);
    
    println!("\nUnmitigated fidelity: {:.6}", unmitigated_fidelity);
    println!("Mitigated fidelity: {:.6}", mitigated_fidelity);
    println!("Improvement: {:.2}%", (mitigated_fidelity - unmitigated_fidelity) * 100.0);
    
    Ok(())
}
```

### Quantum Development Tools

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::dev_tools::{CircuitVerifier, QASMConverter, QuantumAssistant};
use quantrs2_sim::dev_tools::qir::{QirCompiler, QirTarget};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a circuit
    let mut circuit = Circuit::<3>::new();
    circuit.h(0)?.cnot(0, 1)?.cnot(1, 2)?;
    
    // Verify circuit correctness
    let verifier = CircuitVerifier::new();
    let verification_result = verifier.verify(&circuit)?;
    
    if verification_result.is_valid() {
        println!("Circuit verification passed!");
    } else {
        println!("Circuit verification failed: {:?}", verification_result.errors());
        return Ok(());
    }
    
    // Export to QASM 3.0
    let qasm_converter = QASMConverter::new();
    let qasm_code = qasm_converter.to_qasm3(&circuit)?;
    println!("\nQASM 3.0 representation:\n{}", qasm_code);
    
    // Compile to Quantum Intermediate Representation (QIR)
    let qir_compiler = QirCompiler::new(QirTarget::LLVM);
    let qir_bitcode = qir_compiler.compile(&circuit)?;
    println!("\nSuccessfully compiled to QIR bitcode ({} bytes)", qir_bitcode.len());
    
    // Use the quantum assistant to design algorithms
    let assistant = QuantumAssistant::new();
    
    let user_prompt = "Generate a quantum circuit that creates a GHZ state on 5 qubits";
    println!("\nAsking assistant: {}", user_prompt);
    
    let response = assistant.query(user_prompt)?;
    let suggested_circuit = response.circuit();
    
    println!("Assistant generated a circuit with {} gates", suggested_circuit.gate_count());
    
    // Run the generated circuit
    let simulator = StateVectorSimulator::new();
    let result = simulator.run(&suggested_circuit)?;
    
    // Check if we have a proper GHZ state (|00000⟩ + |11111⟩)/√2
    println!("\nGenerated GHZ state probabilities:");
    for (i, prob) in result.probabilities().iter().enumerate() {
        if prob > &0.01 {
            println!("State |{:05b}⟩: {:.6}", i, prob);
        }
    }
    
    Ok(())
}
```

### Quantum Memory (QRAM)

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::qram::{QRAMBuilder, AddressingScheme, MemoryModel};
use quantrs2_core::qubit::QubitId;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a quantum circuit with QRAM access
    let mut circuit = Circuit::<10>::new();
    
    // Initialize address and data registers
    let address_qubits = vec![QubitId::new(0), QubitId::new(1), QubitId::new(2)]; // 3-bit address
    let data_qubits = vec![QubitId::new(3), QubitId::new(4)]; // 2-bit data values
    let target_qubits = vec![QubitId::new(8), QubitId::new(9)]; // 2-bit output register
    
    // Create a QRAM with bucket brigade addressing
    let mut qram = QRAMBuilder::new()
        .with_addressing_scheme(AddressingScheme::BucketBrigade)
        .with_memory_model(MemoryModel::Exponential)
        .with_address_size(3) // 8 memory locations
        .with_data_size(2)    // 2 bits per memory location
        .build()?;
    
    // Initialize QRAM with classical data
    let memory_data = vec![
        vec![false, false], // Location 0: 00
        vec![false, true],  // Location 1: 01
        vec![true, false],  // Location 2: 10
        vec![true, true],   // Location 3: 11
        vec![false, true],  // Location 4: 01
        vec![true, false],  // Location 5: 10
        vec![true, true],   // Location 6: 11
        vec![false, false], // Location 7: 00
    ];
    qram.initialize_memory(&memory_data)?;
    
    // Put address qubits in superposition (query all addresses simultaneously)
    for qubit in &address_qubits {
        circuit.h(*qubit)?;
    }
    
    // Perform quantum memory lookup
    circuit.qram_lookup(&qram, &address_qubits, &target_qubits)?;
    
    // Create simulator
    let simulator = StateVectorSimulator::new();
    
    // Run simulation
    let result = simulator.run(&circuit)?;
    
    // Analyze the result - we should see entanglement between address and data
    println!("QRAM lookup result probabilities:");
    for (i, prob) in result.probabilities().iter().enumerate() {
        if prob > &0.01 {
            // Extract the address and data bits
            let address = (i >> 7) & 0b111;
            let data = (i >> 0) & 0b11;
            
            println!("Address: {:03b}, Data: {:02b}, Probability: {:.4}", address, data, prob);
        }
    }
    
    // Now prepare a specific address state and query
    let mut precise_circuit = Circuit::<10>::new();
    
    // Address 3 (011)
    precise_circuit.x(0)?
                   .x(1)?;
    
    // Perform lookup
    precise_circuit.qram_lookup(&qram, &address_qubits, &target_qubits)?;
    
    // Run simulation
    let precise_result = simulator.run(&precise_circuit)?;
    
    // Analyze the result - we should see only data for address 3
    println!("\nSingle address QRAM lookup:");
    println!("Querying address: 011 (3)");
    println!("Expected data: 11");
    
    println!("\nMeasurement probabilities for target register:");
    let target_probs = precise_result.reduced_density_matrix(&target_qubits)?.probabilities();
    for (i, prob) in target_probs.iter().enumerate() {
        println!("Data: {:02b}, Probability: {:.4}", i, prob);
    }
    
    // Demonstrate quantum associative memory
    println!("\nQuantum Associative Memory Demo:");
    let mut associative_circuit = Circuit::<12>::new();
    
    // Create partial pattern to search (with don't cares)
    let pattern_qubits = vec![QubitId::new(0), QubitId::new(1), QubitId::new(2)];
    associative_circuit.x(0)?; // Only first bit is set, others are "don't care"
    
    // Add associative memory query
    let result_qubits = vec![QubitId::new(10), QubitId::new(11)];
    qram.associative_lookup(&mut associative_circuit, &pattern_qubits, &result_qubits)?;
    
    // Run and analyze
    let assoc_result = simulator.run(&associative_circuit)?;
    println!("Associative memory lookup with pattern 1**: (first bit is 1, others don't care)");
    
    let assoc_probs = assoc_result.reduced_density_matrix(&result_qubits)?.probabilities();
    for (i, prob) in assoc_probs.iter().enumerate() {
        if prob > &0.01 {
            println!("Result: {:02b}, Probability: {:.4}", i, prob);
        }
    }
    
    Ok(())
}
```

### Topological Quantum Computing

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::topological::{AnyonicSystem, BraidingOperation, MajoranaSystem};
use quantrs2_core::qubit::QubitId;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an anyonic system with 4 topological qubits
    let mut anyon_system = AnyonicSystem::new(4)?;
    
    // Initialize the system
    anyon_system.initialize()?;
    
    // Create braiding operations for a CNOT gate
    let braids = vec![
        BraidingOperation::new(1, 2, 3.0)?,   // σ₁²
        BraidingOperation::new(3, 4, -3.0)?,   // σ₃⁻²
        BraidingOperation::new(2, 3, 3.0)?,    // σ₂²
    ];
    
    // Apply the braiding sequence
    println!("Applying braiding sequence for topological CNOT:");
    for (i, braid) in braids.iter().enumerate() {
        println!("Braid {}: Moving anyon {} around anyon {}, angle: {:.1}π", 
                 i+1, braid.anyon1(), braid.anyon2(), braid.angle() / std::f64::consts::PI);
        anyon_system.apply_braid(braid)?;
    }
    
    // Convert to a quantum circuit representation
    let topological_circuit = anyon_system.to_circuit()?;
    
    // Run the circuit with a simulator
    let simulator = StateVectorSimulator::new();
    let result = simulator.run(&topological_circuit)?;
    
    // Analyze the result
    println!("\nTopological CNOT truth table:");
    for (i, prob) in result.probabilities().iter().enumerate() {
        if prob > &0.01 {
            println!("Input: {:04b}, Probability: {:.4}", i, prob);
        }
    }
    
    // Now let's try a Majorana fermion system (edge modes)
    println!("\nMajorana fermion system demonstration:");
    
    // Create a Majorana system with 4 pairs (4 logical qubits)
    let mut majorana_system = MajoranaSystem::new(4)?;
    
    // Initialize in a superposition state
    majorana_system.initialize_superposition(0)?;
    
    // Apply a braiding sequence for a Hadamard gate on qubit 0
    majorana_system.apply_hadamard(0)?;
    
    // Apply a Majorana CNOT: qubit 0 controls qubit 1
    majorana_system.apply_cnot(0, 1)?;
    
    // Convert to circuit
    let majorana_circuit = majorana_system.to_circuit()?;
    
    // Run the circuit
    let majorana_result = simulator.run(&majorana_circuit)?;
    
    // Show probabilities
    println!("Majorana system state probabilities:");
    for (i, prob) in majorana_result.probabilities().iter().enumerate() {
        if prob > &0.01 {
            println!("State: {:04b}, Probability: {:.4}", i, prob);
        }
    }
    
    // Demonstrate topological error correction
    println!("\nTopological error correction demonstration:");
    
    // Create a topological surface code with distance d=3
    let mut topo_code = anyon_system.create_topo_code(3)?;
    
    // Initialize the code
    topo_code.initialize_logical_zero()?;
    
    // Apply a logical X operation
    topo_code.apply_logical_x()?;
    
    // Introduce errors (simulate random noise)
    topo_code.apply_random_errors(0.05)?; // 5% error rate
    
    // Detect and correct errors
    let syndromes = topo_code.measure_syndromes()?;
    println!("Detected {} error syndromes", syndromes.len());
    
    let corrections = topo_code.decode_syndromes(&syndromes)?;
    topo_code.apply_corrections(&corrections)?;
    
    // Measure logical state
    let logical_state = topo_code.measure_logical_state()?;
    println!("Logical |1⟩ state fidelity after correction: {:.6}", logical_state.fidelity_to_one());
    
    Ok(())
}
```

### Quantum Networking

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::network::{QuantumNetwork, Node, Channel, EntanglementProtocol};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a quantum network with 4 nodes
    let mut network = QuantumNetwork::new();
    
    // Add nodes
    network.add_node(Node::new("Alice", 5)?)?; // Node with 5 qubits
    network.add_node(Node::new("Bob", 5)?)?;   // Node with 5 qubits
    network.add_node(Node::new("Charlie", 5)?)?; // Node with 5 qubits
    network.add_node(Node::new("Relay", 10)?)?; // Relay node with 10 qubits
    
    // Add quantum channels
    network.add_channel(Channel::new("Alice", "Relay", 0.95)?)?; // 95% fidelity
    network.add_channel(Channel::new("Bob", "Relay", 0.95)?)?;   // 95% fidelity
    network.add_channel(Channel::new("Charlie", "Relay", 0.95)?)?; // 95% fidelity
    
    // Set up noise model for the network
    network.set_noise_model(0.01, 0.005, 50.0)?; // decoherence, gate error, distance in km
    
    // Create entanglement between Alice and Bob through the relay
    println!("Creating long-distance entanglement between Alice and Bob via Relay...");
    
    let protocol = EntanglementProtocol::EntanglementSwapping;
    let result = network.create_entanglement("Alice", "Bob", protocol)?;
    
    println!("Initial entanglement fidelity: {:.4}", result.fidelity);
    
    // Perform entanglement purification
    let purified = network.purify_entanglement("Alice", "Bob", 3)?; // 3 rounds of purification
    
    println!("Fidelity after purification: {:.4}", purified.fidelity);
    
    // Create a quantum circuit to use the entanglement resource
    let (alice_circuit, bob_circuit) = network.create_teleportation_circuits("Alice", "Bob")?;
    
    // Set up local simulator
    let simulator = StateVectorSimulator::new();
    
    // Run teleportation protocol
    println!("\nPerforming quantum teleportation of |+⟩ state:");
    
    // Prepare state to teleport (|+⟩ state)
    network.prepare_qubit("Alice", 0, |circuit| {
        circuit.h(0)?;
        Ok(())
    })?;
    
    // Perform teleportation
    let teleport_result = network.run_teleportation("Alice", 0, "Bob", 0)?;
    
    println!("Teleportation success probability: {:.4}", teleport_result.success_probability);
    println!("Teleported state fidelity: {:.4}", teleport_result.fidelity);
    
    // Now let's set up a more complex quantum network and simulate a quantum internet protocol
    println!("\nQuantum Internet Simulation:");
    
    // Create a network with 6 nodes in a realistic topology
    let mut quantum_internet = QuantumNetwork::new();
    
    // Add nodes (quantum routers)
    for i in 1..=6 {
        quantum_internet.add_node(Node::new(format!("Router{}", i), 20)?)?;
    }
    
    // Add channels in a realistic network topology
    quantum_internet.add_channel(Channel::new("Router1", "Router2", 0.96)?)?;
    quantum_internet.add_channel(Channel::new("Router2", "Router3", 0.95)?)?;
    quantum_internet.add_channel(Channel::new("Router3", "Router4", 0.97)?)?;
    quantum_internet.add_channel(Channel::new("Router4", "Router5", 0.94)?)?;
    quantum_internet.add_channel(Channel::new("Router2", "Router5", 0.93)?)?;
    quantum_internet.add_channel(Channel::new("Router5", "Router6", 0.95)?)?;
    quantum_internet.add_channel(Channel::new("Router3", "Router6", 0.94)?)?;
    
    // Find the best path for entanglement distribution
    let path = quantum_internet.find_optimal_path("Router1", "Router6")?;
    
    println!("Optimal quantum routing path: {:?}", path);
    
    // Distribute entanglement along the path
    let dist_result = quantum_internet.distribute_entanglement("Router1", "Router6", &path)?;
    
    println!("End-to-end entanglement fidelity: {:.4}", dist_result.fidelity);
    println!("Resource qubits used: {}", dist_result.resources_used);
    println!("Time required: {:?}", dist_result.time_required);
    
    Ok(())
}
```

### Continuous-Variable Quantum Computing

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::cv::{GaussianState, CvOperation, CvMode, PhotonicCircuit};
use quantrs2_sim::cv::operations::{Displacement, Squeezing, Beamsplitter};
use quantrs2_sim::cv::measurement::{HomodyneMeasurement, HeterodyneMeasurement};
use quantrs2_core::complex::Complex64;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a continuous-variable quantum computing simulator
    println!("Continuous-Variable Quantum Computing Example\n");
    
    // Initialize a 4-mode Gaussian state (vacuum state)
    let mut state = GaussianState::vacuum(4)?;
    
    // Apply displacement to the first mode
    let alpha = Complex64::new(1.0, 0.0); // Displacement amplitude
    let d1 = Displacement::new(0, alpha)?;
    state.apply_operation(&d1)?;
    
    // Apply squeezing to the second mode
    let r = 0.5; // Squeezing parameter
    let phi = 0.0; // Squeezing angle
    let s1 = Squeezing::new(1, r, phi)?;
    state.apply_operation(&s1)?;
    
    // Apply a beamsplitter between modes 0 and 1
    let theta = PI / 4.0; // 50:50 beamsplitter
    let phi_bs = 0.0;
    let bs = Beamsplitter::new(0, 1, theta, phi_bs)?;
    state.apply_operation(&bs)?;
    
    // Perform homodyne measurement on mode 0
    let homodyne = HomodyneMeasurement::new(0, 0.0)?; // Measure x-quadrature
    let homodyne_result = state.measure(&homodyne)?;
    
    println!("Homodyne measurement result: {:.4}", homodyne_result.value);
    
    // Get the Wigner function of mode 1
    let wigner = state.wigner_function(1, -5.0, 5.0, 100, -5.0, 5.0, 100)?;
    println!("Peak Wigner function value: {:.4}", wigner.peak_value());
    
    // Now let's create a photonic circuit
    println!("\nPhotonic Circuit Example:");
    
    // Create a 6-mode photonic circuit
    let mut circuit = PhotonicCircuit::new(6)?;
    
    // Prepare input states
    circuit.prepare_squeezed_state(0, 0.5, 0.0)?; // Squeezed vacuum in mode 0
    circuit.prepare_coherent_state(1, Complex64::new(1.0, 0.0))?; // Coherent state in mode 1
    circuit.prepare_fock_state(2, 1)?; // Single photon in mode 2
    
    // Add gates to the circuit
    circuit.add_displacement(0, Complex64::new(0.5, 0.0))?;
    circuit.add_squeezing(1, 0.3, PI/2.0)?;
    circuit.add_beamsplitter(0, 1, PI/4.0, 0.0)?; // 50:50 beamsplitter
    circuit.add_beamsplitter(1, 2, PI/4.0, PI/2.0)?;
    circuit.add_mzi(3, 4, 0.3, 0.5)?; // Mach-Zehnder interferometer
    
    // Run the circuit
    let output_state = circuit.run()?;
    
    // Calculate number of photons in each mode
    println!("Average photon numbers in each mode:");
    for i in 0..6 {
        let n = output_state.average_photon_number(i)?;
        println!("Mode {}: {:.4} photons", i, n);
    }
    
    // Perform Gaussian Boson Sampling
    println!("\nGaussian Boson Sampling Example:");
    
    // Create a GBS circuit with 8 modes
    let mut gbs_circuit = PhotonicCircuit::new(8)?;
    
    // Prepare squeezed states in the first 4 modes
    for i in 0..4 {
        gbs_circuit.prepare_squeezed_state(i, 0.5, 0.0)?;
    }
    
    // Create a random interferometer
    gbs_circuit.add_random_interferometer(0.8)?;
    
    // Run the circuit
    let gbs_output = gbs_circuit.run()?;
    
    // Sample photon number patterns (maximum 20 samples)
    let samples = gbs_output.sample_photon_patterns(20)?;
    
    // Print the first 5 samples
    println!("Photon number samples from GBS:");
    for (i, sample) in samples.iter().enumerate().take(5) {
        println!("Sample {}: {:?}", i+1, sample);
    }
    
    // Now demonstrate measurement-based CVQC
    println!("\nMeasurement-Based Continuous-Variable Quantum Computing:");
    
    // Create a CV cluster state with 9 modes (3x3 grid)
    let mut cv_cluster = PhotonicCircuit::new(9)?;
    
    // Prepare 9 squeezed states
    for i in 0..9 {
        cv_cluster.prepare_squeezed_state(i, 0.8, 0.0)?; // Squeezed in p-quadrature
    }
    
    // Create the cluster state connections (CZ gates)
    // Connect in a grid pattern
    let grid_connections = [
        (0, 1), (0, 3), (1, 2), (1, 4), (2, 5),
        (3, 4), (3, 6), (4, 5), (4, 7), (5, 8),
        (6, 7), (7, 8)
    ];
    
    for (i, j) in &grid_connections {
        cv_cluster.add_controlled_phase(*i, *j, 1.0)?; // CZ gate with weight 1.0
    }
    
    // Run the circuit to create the cluster state
    let cluster_state = cv_cluster.run()?;
    
    // Perform measurement-based computation (teleportation of a Fourier transform)
    let measurements = vec![PI/2.0, 0.0, PI/2.0, 0.0, 0.0, 0.0, PI/2.0, 0.0, 0.0];
    let mbqc_result = cluster_state.perform_mbqc(&measurements)?;
    
    println!("MBQC teleported output state fidelity: {:.4}", mbqc_result.fidelity);
    
    // Show graphical representation of the circuit
    println!("\nCircuit representation:\n{}", cv_cluster.to_string());
    
    Ok(())
}
```

### Quantum Error Correction Benchmarking

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::error_correction::{QecCode, SurfaceCode, ColorCode, SteaneCssCode};
use quantrs2_sim::error_correction::benchmarking::{QecBenchmark, ThresholdEstimator};
use quantrs2_sim::noise_advanced::RealisticNoiseModelBuilder;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a QEC benchmarking framework
    println!("Quantum Error Correction Benchmarking\n");
    
    // Initialize the benchmark with multiple QEC codes
    let mut benchmark = QecBenchmark::new()?;
    
    // Add different QEC codes to compare
    benchmark.add_code(SurfaceCode::new(3).into_qec_code())?; // d=3 surface code
    benchmark.add_code(SurfaceCode::new(5).into_qec_code())?; // d=5 surface code
    benchmark.add_code(SurfaceCode::new(7).into_qec_code())?; // d=7 surface code
    benchmark.add_code(ColorCode::new(3).into_qec_code())?;   // d=3 color code
    benchmark.add_code(SteaneCssCode::new().into_qec_code())?; // Steane [[7,1,3]] code
    
    // Create a noise model sweep (physical error rates from 0.1% to 10%)
    let error_rates = vec![0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
    
    // Perform the benchmark for each code and error rate
    println!("Performing logical error rate estimation for various codes...");
    println!("This will run many Monte Carlo simulations and may take a while.\n");
    
    let trials_per_point = 100; // Use a small number for demonstration purposes
    let results = benchmark.evaluate_logical_error_rates(&error_rates, trials_per_point)?;
    
    // Print the results
    println!("Logical Error Rates (p_L) vs Physical Error Rates (p):\n");
    println!("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15}", 
             "p", "Surface-3", "Surface-5", "Surface-7", "Color-3", "Steane");
    
    for i in 0..error_rates.len() {
        print!("{:<10.4}", error_rates[i]);
        for j in 0..results.num_codes() {
            print!(" {:<15.6e}", results.logical_error_rate(j, i));
        }
        println!();
    }
    
    // Estimate threshold for surface codes of different distances
    let surface_code_results = results.filter_by_name("Surface")?
        .to_vec();
    
    let threshold_estimator = ThresholdEstimator::new();
    let threshold = threshold_estimator.estimate(&error_rates, &surface_code_results)?;
    
    println!("\nEstimated threshold for surface code: {:.4}%", threshold * 100.0);
    
    // Extrapolate code performance
    println!("\nExtrapolated logical error rates for large distances:");
    let p_physical = 0.001; // 0.1% physical error rate
    
    for d in [9, 11, 15, 21, 31, 51, 101].iter() {
        let p_logical = results.extrapolate_to_distance(*d, p_physical)?;
        println!("d = {:3}: p_L ≈ {:.6e}", d, p_logical);
    }
    
    // Calculate the required code distance for a target logical error rate
    let target_logical_error = 1e-15;
    let required_distance = results.required_distance_for_target(p_physical, target_logical_error)?;
    
    println!("\nTo achieve a logical error rate of {:.1e} with physical error rate {:.1}%:", 
             target_logical_error, p_physical * 100.0);
    println!("Required code distance: approximately d = {}", required_distance);
    
    // Memory lifetime calculation
    let cycle_time_ns = 1000.0; // 1 microsecond per QEC cycle
    let lifetime_seconds = results.calculate_memory_lifetime(15, p_physical, cycle_time_ns)?;
    
    println!("\nWith d = 15 and p = {:.1}%, quantum memory lifetime: {:.2} seconds", 
             p_physical * 100.0, lifetime_seconds);
    
    Ok(())
}
```

### Quantum Neural Differential Equations

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::parametric::{SymbolicParameter, ParameterBindings};
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::ml::{QuantumNeuralNetwork, Optimizer};
use quantrs2_sim::ml::qnode::{QuantumNeuralODE, DiffEquationType};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Quantum Neural Differential Equation Examples
    println!("Quantum Neural Differential Equations\n");
    
    // First, let's solve a simple ODE: dy/dt = -y with y(0) = 1.0
    println!("Example 1: Solving the ODE dy/dt = -y with y(0) = 1.0\n");
    
    // Initial condition
    let y0 = Array1::from_vec(vec![1.0]);
    
    // Define the linear ODE function
    let ode_function = |y: &Array1<f64>, _t: f64, _params: &Array1<f64>| -> Array1<f64> {
        -&y // dy/dt = -y
    };
    
    // Create a quantum neural ODE solver
    let mut qnode = QuantumNeuralODE::new(
        DiffEquationType::FirstOrder,
        1, // dimension of the ODE
        20 // number of time steps
    )?;
    
    // Set up the solver parameters
    qnode.with_time_range(0.0, 5.0)?
         .with_ode_function(ode_function.into())?
         .with_initial_condition(y0.clone())?;
    
    // Solve the ODE using quantum circuit
    let solution = qnode.solve()?;
    
    // Print solution at selected time points
    println!("Selected solution points (t, y):");
    for i in 0..solution.len() {
        if i % 4 == 0 || i == solution.len() - 1 {
            println!("t = {:.2}, y = {:.6}", solution.time_at(i), solution.value_at(i)[0]);
        }
    }
    
    // Compare with analytical solution y = e^(-t)
    println!("\nComparison with analytical solution y = e^(-t):");
    for i in 0..solution.len() {
        if i % 4 == 0 || i == solution.len() - 1 {
            let t = solution.time_at(i);
            let analytical = (-t).exp();
            let error = (solution.value_at(i)[0] - analytical).abs();
            println!("t = {:.2}, Numerical = {:.6}, Analytical = {:.6}, Error = {:.6e}", 
                     t, solution.value_at(i)[0], analytical, error);
        }
    }
    
    // Now let's create a parameterized ODE and learn the parameters
    println!("\nExample 2: Learning ODE parameters from data\n");
    
    // Generate some data from the system dy/dt = a*y + b with a=-0.5, b=1.0
    let true_a = -0.5;
    let true_b = 1.0;
    
    // True function
    let true_function = |y: &Array1<f64>, _t: f64, _params: &Array1<f64>| -> Array1<f64> {
        true_a * y + true_b
    };
    
    let mut data_generator = QuantumNeuralODE::new(
        DiffEquationType::FirstOrder,
        1,
        20
    )?;
    
    // Set up the generator
    data_generator.with_time_range(0.0, 5.0)?
                  .with_ode_function(true_function.into())?
                  .with_initial_condition(Array1::from_vec(vec![1.0]))?;
    
    // Generate data
    let true_solution = data_generator.solve()?;
    
    // Create a parameterized ODE with unknown parameters
    let parameterized_ode = |y: &Array1<f64>, _t: f64, params: &Array1<f64>| -> Array1<f64> {
        params[0] * y + params[1]
    };
    
    // Initial parameter guesses
    let initial_params = Array1::from_vec(vec![0.0, 0.0]); // Start with a=0, b=0
    
    // Create learnable ODE
    let mut learnable_ode = QuantumNeuralODE::new(
        DiffEquationType::FirstOrder,
        1,
        20
    )?;
    
    learnable_ode.with_time_range(0.0, 5.0)?
                 .with_ode_function(parameterized_ode.into())?
                 .with_initial_condition(Array1::from_vec(vec![1.0]))?;
    
    // Learn the parameters
    let learned_params = learnable_ode.learn_parameters(
        &true_solution, // target data
        initial_params,  // initial parameters
        1000,            // epochs
        0.01,            // learning rate
        1e-6             // convergence threshold
    )?;
    
    println!("Learning results:");
    println!("True parameters: a = {:.4}, b = {:.4}", true_a, true_b);
    println!("Learned parameters: a = {:.4}, b = {:.4}", learned_params[0], learned_params[1]);
    
    // Solve the ODE with the learned parameters
    learnable_ode.set_parameters(&learned_params)?;
    let learned_solution = learnable_ode.solve()?;
    
    // Compare solutions
    println!("\nComparison of true and learned solutions:");
    for i in 0..true_solution.len() {
        if i % 4 == 0 || i == true_solution.len() - 1 {
            let t = true_solution.time_at(i);
            println!("t = {:.2}, True = {:.6}, Learned = {:.6}, Error = {:.6e}", 
                     t, true_solution.value_at(i)[0], learned_solution.value_at(i)[0],
                     (true_solution.value_at(i)[0] - learned_solution.value_at(i)[0]).abs());
        }
    }
    
    // Example 3: Quantum PDE solver for heat equation
    println!("\nExample 3: Solving the 1D Heat Equation\n");
    
    // Heat equation: ∂u/∂t = α ∂²u/∂x²
    // Using quantum circuits to solve this PDE
    
    // Physical parameters
    let alpha = 0.01; // Thermal diffusivity
    let length = 1.0;  // Domain length [0, L]
    let time_end = 0.5; // End time
    
    // Discretization parameters
    let nx = 32; // Spatial points
    let nt = 20; // Time steps
    
    // Set up the problem with an initial temperature distribution
    // u(x, 0) = sin(πx/L)
    let dx = length / (nx as f64 - 1.0);
    let mut u0 = Array1::zeros(nx);
    for i in 0..nx {
        let x = i as f64 * dx;
        u0[i] = (PI * x / length).sin();
    }
    
    // Create a quantum PDE solver
    let mut qpde = quantrs2_sim::ml::qnode::QuantumPDE::new(
        DiffEquationType::Heat,
        nx,
        nt
    )?;
    
    qpde.with_spatial_range(0.0, length)?
        .with_time_range(0.0, time_end)?
        .with_diffusivity(alpha)?
        .with_boundary_conditions(0.0, 0.0)? // u(0,t) = u(L,t) = 0
        .with_initial_condition(u0)?;
    
    // Solve the PDE
    let solution_2d = qpde.solve()?;
    
    // Print solution at selected times
    println!("Heat equation solution at selected times and positions:");
    
    let t_indices = [0, nt/4, nt/2, nt-1]; // Times to output
    let x_indices = [0, nx/4, nx/2, 3*nx/4, nx-1]; // Positions to output
    
    print!("{:<10}", "t \ x");
    for &xi in &x_indices {
        print!("{:<15.6}", xi as f64 * dx);
    }
    println!();
    
    for &ti in &t_indices {
        let t = ti as f64 * time_end / (nt as f64 - 1.0);
        print!("{:<10.4}", t);
        
        for &xi in &x_indices {
            print!("{:<15.6}", solution_2d[(ti, xi)]);
        }
        println!();
    }
    
    // Compare with analytical solution
    // u(x,t) = e^(-α(π/L)²t) · sin(πx/L)
    println!("\nComparison with analytical solution at final time:");
    let t = time_end;
    let decay_factor = (-alpha * (PI / length).powi(2) * t).exp();
    
    println!("{:<10} {:<15} {:<15} {:<15}", "x", "Numerical", "Analytical", "Error");
    for &xi in &x_indices {
        let x = xi as f64 * dx;
        let analytical = decay_factor * (PI * x / length).sin();
        let numerical = solution_2d[(nt-1, xi)];
        let error = (numerical - analytical).abs();
        
        println!("{:<10.4} {:<15.6} {:<15.6} {:<15.6e}", 
                 x, numerical, analytical, error);
    }
    
    Ok(())
}
```

### Quantum Machine Learning for High-Energy Physics

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::parametric::{SymbolicParameter, ParameterBindings};
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::ml::{QuantumNeuralNetwork, Optimizer};
use quantrs2_sim::ml::physics::{ParticleCollisionClassifier, EventReconstructor};
use ndarray::{Array1, Array2, s};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Quantum Machine Learning for High-Energy Physics\n");
    
    // Load CERN particle collision dataset (we'll use a small synthetic example here)
    println!("Loading sample collision data...");
    
    // Create synthetic dataset of particle collision features
    // In a real application, this would be loaded from CERN data files
    let mut collision_data = Array2::zeros((1000, 28));
    let mut collision_labels = Array1::zeros(1000);
    
    // Fill with synthetic data (in reality, this would be real collision data)
    for i in 0..1000 {
        // Generate synthetic features that might represent energy, momentum, angles, etc.
        for j in 0..28 {
            collision_data[[i, j]] = rand::random::<f64>() * 2.0 - 1.0;
        }
        
        // Create synthetic labels (0 = background, 1 = Higgs boson event)
        // In this synthetic example, we'll use a simple rule: sum of first 10 features > 0
        let sum: f64 = (0..10).map(|j| collision_data[[i, j]]).sum();
        collision_labels[i] = if sum > 0.0 { 1.0 } else { 0.0 };
    }
    
    // Split data into training and testing sets
    let train_size = 800;
    let test_size = 200;
    
    let train_data = collision_data.slice(s![0..train_size, ..]).to_owned();
    let train_labels = collision_labels.slice(s![0..train_size]).to_owned();
    let test_data = collision_data.slice(s![train_size.., ..]).to_owned();
    let test_labels = collision_labels.slice(s![train_size..]).to_owned();
    
    // Create a quantum classifier for particle collision events
    println!("\nCreating quantum particle collision classifier...");
    let mut classifier = ParticleCollisionClassifier::new()
        .with_input_features(28)
        .with_quantum_layers(3)
        .with_measurement_qubits(2)?;
    
    // Train the model
    println!("Training classifier on collision data...");
    let training_result = classifier.train(
        &train_data,
        &train_labels,
        1000,  // epochs
        0.01   // learning rate
    )?;
    
    println!("Training complete! Final loss: {:.6}", training_result.final_loss);
    
    // Evaluate on test data
    println!("\nEvaluating on test data...");
    let metrics = classifier.evaluate(&test_data, &test_labels)?;
    
    println!("Test accuracy: {:.2}%", metrics.accuracy * 100.0);
    println!("Precision: {:.4}", metrics.precision);
    println!("Recall: {:.4}", metrics.recall);
    println!("F1 score: {:.4}", metrics.f1_score);
    
    // Now let's try a more complex example: event reconstruction
    println!("\nEvent Reconstruction Example\n");
    
    // In real applications, this would reconstruct particle trajectory and properties
    let mut reconstructor = EventReconstructor::new()
        .with_input_features(28)
        .with_output_features(10)  // reconstruct 10 physical properties
        .with_quantum_layers(4)?;
    
    // Generate synthetic data for demonstration
    let mut event_data = Array2::zeros((500, 28));
    let mut event_properties = Array2::zeros((500, 10));
    
    // Fill with synthetic event data
    for i in 0..500 {
        for j in 0..28 {
            event_data[[i, j]] = rand::random::<f64>() * 2.0 - 1.0;
        }
        
        // Target properties (in real world, these would be known physical properties)
        for j in 0..10 {
            let sum: f64 = (0..5).map(|k| event_data[[i, j*2 + k % 28]]).sum();
            event_properties[[i, j]] = sum.tanh();
        }
    }
    
    // Split for training/testing
    let train_events = event_data.slice(s![0..400, ..]).to_owned();
    let train_properties = event_properties.slice(s![0..400, ..]).to_owned();
    let test_events = event_data.slice(s![400.., ..]).to_owned();
    let test_properties = event_properties.slice(s![400.., ..]).to_owned();
    
    // Train the reconstruction model
    println!("Training event reconstructor...");
    let recon_result = reconstructor.train(
        &train_events,
        &train_properties,
        1000,  // epochs
        0.01   // learning rate
    )?;
    
    println!("Training complete! Final MSE: {:.6}", recon_result.final_loss);
    
    // Evaluate on test data
    println!("\nEvaluating event reconstruction...");
    let recon_metrics = reconstructor.evaluate(&test_events, &test_properties)?;
    
    println!("Test MSE: {:.6}", recon_metrics.mse);
    println!("MAE: {:.6}", recon_metrics.mae);
    println!("R² score: {:.4}", recon_metrics.r2_score);
    
    // Example reconstruction for a single event
    let event_idx = 450;
    let event = test_events.row(event_idx - 400).to_owned();
    let true_props = test_properties.row(event_idx - 400).to_owned();
    let predicted_props = reconstructor.predict(&event)?;
    
    println!("\nExample event reconstruction for event #{}:", event_idx);
    println!("{:<15} {:<15} {:<15}", "Property", "True Value", "Predicted");
    for i in 0..10 {
        println!("{:<15} {:<15.6} {:<15.6}", 
                 format!("Property {}", i+1), true_props[i], predicted_props[i]);
    }
    
    // Rare event detection example
    println!("\nRare Event Detection: Quantum Anomaly Detection\n");
    
    // In high-energy physics, finding rare events is critical
    let mut anomaly_detector = quantrs2_sim::ml::physics::AnomalyDetector::new()
        .with_features(28)
        .with_quantum_encoder(true)
        .with_kernel_method(quantrs2_sim::ml::physics::KernelMethod::QuantumKernel)?;
    
    // Generate synthetic data with rare events
    let mut normal_events = Array2::zeros((950, 28));
    let mut rare_events = Array2::zeros((50, 28));
    
    // Fill normal events
    for i in 0..950 {
        for j in 0..28 {
            normal_events[[i, j]] = rand::random::<f64>() * 0.5;
        }
    }
    
    // Fill rare events (different distribution)
    for i in 0..50 {
        for j in 0..28 {
            rare_events[[i, j]] = 0.5 + rand::random::<f64>() * 0.5;
        }
    }
    
    // Combine all events and create labels (0=normal, 1=rare)
    let mut all_events = Array2::zeros((1000, 28));
    let mut event_types = Array1::zeros(1000);
    
    for i in 0..950 {
        for j in 0..28 {
            all_events[[i, j]] = normal_events[[i, j]];
        }
        event_types[i] = 0.0; // Normal event
    }
    
    for i in 0..50 {
        for j in 0..28 {
            all_events[[i+950, j]] = rare_events[[i, j]];
        }
        event_types[i+950] = 1.0; // Rare event
    }
    
    // Train the anomaly detector (unsupervised)
    println!("Training quantum anomaly detector on collision data...");
    anomaly_detector.fit(&all_events)?;
    
    // Compute anomaly scores
    let anomaly_scores = anomaly_detector.score(&all_events)?;
    
    // Evaluate detector performance
    let auc = anomaly_detector.evaluate(&event_types, &anomaly_scores)?;
    
    println!("Anomaly detection AUC: {:.4}", auc);
    
    // Print a few examples
    println!("\nExample anomaly scores:");
    println!("{:<10} {:<15} {:<15}", "Index", "True Type", "Anomaly Score");
    
    // Print 5 normal and 5 rare events
    for i in 0..5 {
        println!("{:<10} {:<15} {:<15.6}", 
                 i, "Normal", anomaly_scores[i]);
    }
    
    for i in 0..5 {
        println!("{:<10} {:<15} {:<15.6}", 
                 i+950, "Rare", anomaly_scores[i+950]);
    }
    
    Ok(())
}
```

### Quantum Blockchain and Distributed Ledger

```rust
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::cryptography::{QuantumSecureHash, DigitalSignature};
use quantrs2_sim::blockchain::{QuantumBlock, QuantumBlockchain, ConsensusProtocol};
use quantrs2_sim::blockchain::mining::{QuantumMiner, MiningStrategy};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Quantum Blockchain and Distributed Ledger Technology\n");
    
    // Create a quantum-secured blockchain
    println!("Creating quantum-secured blockchain...");
    let mut blockchain = QuantumBlockchain::new()
        .with_block_size(1024)  // 1KB blocks
        .with_consensus(ConsensusProtocol::QuantumProofOfWork)
        .with_quantum_signatures(true)?;
    
    // Create a quantum secure hash function
    let hash_function = QuantumSecureHash::new()
        .with_output_bits(256)
        .with_quantum_resistance(true)?;
    
    // Generate quantum-secure digital signatures for participants
    println!("Generating quantum-secure digital signatures...");
    let alice_signature = DigitalSignature::new_quantum_resistant("Alice")?;
    let bob_signature = DigitalSignature::new_quantum_resistant("Bob")?;
    let charlie_signature = DigitalSignature::new_quantum_resistant("Charlie")?;
    
    // Create transactions
    println!("Creating and signing transactions...");
    let tx1 = blockchain.create_transaction("Alice", "Bob", 5.0)?;
    let tx2 = blockchain.create_transaction("Bob", "Charlie", 2.0)?;
    let tx3 = blockchain.create_transaction("Charlie", "Alice", 1.0)?;
    
    // Sign transactions with quantum-resistant signatures
    let signed_tx1 = alice_signature.sign_transaction(&tx1)?;
    let signed_tx2 = bob_signature.sign_transaction(&tx2)?;
    let signed_tx3 = charlie_signature.sign_transaction(&tx3)?;
    
    // Add transactions to the blockchain
    blockchain.add_transaction(signed_tx1)?;
    blockchain.add_transaction(signed_tx2)?;
    blockchain.add_transaction(signed_tx3)?;
    
    // Create a quantum miner
    println!("\nStarting quantum mining process...");
    let mut miner = QuantumMiner::new()
        .with_strategy(MiningStrategy::QuantumSpeedup)
        .with_hash_function(&hash_function)?;
    
    // Mine a block
    let start = Instant::now();
    let block = miner.mine_next_block(&blockchain)?;
    let duration = start.elapsed();
    
    println!("Block mined in {:?}", duration);
    println!("Block hash: {}", block.hash());
    println!("Transactions in block: {}", block.transaction_count());
    
    // Add the block to the blockchain
    blockchain.add_block(block)?;
    
    // Verify the blockchain
    println!("\nVerifying blockchain integrity...");
    let is_valid = blockchain.verify()?;
    println!("Blockchain valid: {}", is_valid);
    
    // Try to tamper with the blockchain
    println!("\nAttempting to tamper with the blockchain...");
    let tampered = blockchain.tamper_with_block(0, "AliceBobTampered", 100.0)?;
    
    // Re-verify the blockchain
    let is_still_valid = tampered.verify()?;
    println!("Tampered blockchain valid: {}", is_still_valid);
    
    // Demonstrate quantum-resistant consensus algorithm
    println!("\nQuantum-resistant consensus demonstration:");
    let nodes = vec!["Node1", "Node2", "Node3", "Node4", "Node5"];
    
    let mut network = quantrs2_sim::blockchain::BlockchainNetwork::new()
        .with_nodes(&nodes)
        .with_consensus(ConsensusProtocol::QuantumProofOfStake)
        .with_quantum_randomness(true)?;
    
    // Add the blockchain to the network
    network.add_blockchain(blockchain)?;
    
    // Run consensus
    println!("Running quantum-resistant consensus algorithm...");
    let consensus_result = network.run_consensus(10)?; // 10 rounds
    
    println!("Consensus achieved: {}", consensus_result.consensus_reached);
    println!("Number of rounds: {}", consensus_result.rounds);
    println!("Final chain length: {}", consensus_result.chain_length);
    
    // Demonstrate quantum blockchain applications
    println!("\nDemonstrating quantum blockchain applications:\n");
    
    // 1. Quantum random number generation for blockchain
    println!("1. Quantum random number generation");
    let qrng = quantrs2_sim::blockchain::utils::QuantumRNG::new()?;
    let random_values = qrng.generate_array(5)?;
    
    println!("Quantum random numbers: {:?}", random_values);
    
    // 2. Quantum multiparty computation for private transactions
    println!("\n2. Quantum multiparty computation for private transactions");
    let mut private_tx = quantrs2_sim::blockchain::privacy::PrivateTransaction::new()
        .with_participants(&["Alice", "Bob", "Charlie"])
        .with_quantum_encryption(true)?;
    
    private_tx.set_input("Alice", 100.0)?;
    private_tx.set_input("Bob", 50.0)?;
    private_tx.set_input("Charlie", 75.0)?;
    
    let sum = private_tx.compute_sum()?;
    println!("Sum computed without revealing individual values: {:.1}", sum);
    
    // 3. Quantum smart contracts
    println!("\n3. Quantum smart contracts");
    let mut smart_contract = quantrs2_sim::blockchain::contracts::QuantumSmartContract::new()
        .with_code("if (balance > 100) { transfer(amount * 0.95); } else { transfer(amount); }")
        .with_quantum_verification(true)?;
    
    let execution_result = smart_contract.execute(120.0, 50.0)?;
    println!("Smart contract execution result: {:.2}", execution_result);
    
    println!("\nBlockchain state verification complete.");
    
    Ok(())
}
```

## Feedback and Contribution

We welcome your feedback and contributions to the QuantRS2 project. Please file issues, feature requests, and pull requests on our GitHub repository.

Thank you for using QuantRS2!

---

# Release Notes for QuantRS2 v0.1.0-alpha.2

We're excited to announce the first alpha release of QuantRS2, a high-performance quantum computing framework written in Rust!

## Highlights

- **Type-Safe Quantum Circuits**: Build quantum circuits with compile-time safety using Rust's const generics
- **High-Performance Simulation**: Efficiently simulate 30+ qubits with optimized state vector and tensor network backends
- **GPU Acceleration**: Run simulations on GPU using WGPU for significant speedups on large qubit counts
- **Quantum Hardware Integration**: Connect to IBM Quantum and D-Wave quantum hardware
- **Advanced Noise Models**: Simulate realistic quantum hardware with configurable noise channels
- **Quantum Error Correction**: Protect quantum information with various error correction codes
- **Tensor Network Optimization**: Specialized contraction path optimization for common circuit structures

## System Requirements

- Rust 1.86.0 or newer
- Optional: WGPU-compatible GPU for GPU acceleration
- Optional: IBM Quantum account for hardware connectivity
- Optional: D-Wave account for quantum annealing hardware

## Installation

Add QuantRS2 to your Cargo project:

```toml
[dependencies]
quantrs2-core = "0.1.0-alpha.2"
quantrs2-circuit = "0.1.0-alpha.2"
quantrs2-sim = "0.1.0-alpha.2"
```

For optional features:

```toml
# GPU acceleration
quantrs2-sim = { version = "0.1.0-alpha.2", features = ["gpu"] }

# IBM Quantum integration
quantrs2-device = { version = "0.1.0-alpha.2", features = ["ibm"] }

# D-Wave quantum annealing
quantrs2-anneal = { version = "0.1.0-alpha.2", features = ["dwave"] }
```

## Known Issues

- The D-Wave integration requires additional setup for SymEngine on macOS. See the README.md for details.
- GPU acceleration is currently in beta and may have performance variability across different GPU models.

## Acknowledgments

We would like to thank all contributors to the QuantRS2 project, as well as the broader quantum computing and Rust communities for their support and feedback.

## Future Plans

This is an alpha release, and we're actively working on:

- Automated testing in CI pipeline
- Improved documentation
- Additional quantum hardware integrations
- Enhanced tensor network contraction algorithms
- Python package distribution via PyPI

For more details, see our [Roadmap](https://github.com/cool-japan/quantrs/blob/master/TODO.md).